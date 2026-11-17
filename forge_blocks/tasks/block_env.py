"""ForgeEnv with the two changes needed to match the FORGE paper. See README, "Differences".

1. Velocity observations (Isaac Lab issue #8874). Upstream differences the noisy pose against a
   history buffer that FactoryEnv has just overwritten with the current true pose, so ee_linvel is
   position noise / dt and carries no motion. ee_angvel has the same problem. Here both come from
   consecutive noisy samples, which is what App. A of the paper describes.

2. Controlled position-error radius. FORGE's obs_rand.fixed_asset_pos gives per-axis Gaussian
   sigmas. Setting obs_rand.error_radius_mode replaces them with a radius from obs_rand.error_radius:
     "gaussian"  upstream behaviour (sigma from fixed_asset_pos, the default)
     "xyz"       uniform radius, uniform direction on the sphere (the paper's evaluation)
     "xy"        uniform radius in the plane, z stays Gaussian (the project's ablation grid)

3. Force-sensing stress tests (evaluation only; both off by default). The observed force becomes
   ft_force_scale * force + bias, with |bias| = ft_force_bias_N in a random direction per episode.
   They stand in for the real Franka's force estimate: a wrong sign (scale -1), a wrong
   magnitude, or an offset left after taring.

4. Force-sensing dead band (off by default). The Franka's force estimate is unreliable below about
   5 N. With ft_force_deadband_N = d, the observed force shrinks toward zero by d and anything
   smaller reads 0: per component ("axis", loses small sideways forces entirely) or on the
   magnitude ("norm", keeps the direction of what's left).

5. Joint-friction leak (off by default). On the real Franka, moving in free space makes the force
   estimate read 1.5-2.4 N against the motion (session 1). With ft_force_leak_N = [lo, hi], each
   episode draws a size per axis from U(lo, hi), and the observed force gets
   -size * tanh(v / ft_force_leak_v_mps) from the true fingertip velocity v, smoothed like the force.
   It is added to the raw estimate, before the scale / bias / dead band above. FORGE's actuation
   dead zone already models the other half of joint friction (the hand moving less than commanded).

6. Pull-based friction leak (off by default), which fits the session-2 walks far better than 5. With
   ft_pull_leak_alpha = [lo, hi], each episode draws alpha per axis and the observed force gets
   -alpha * pull, where pull = task stiffness x (controller target - fingertip) is the spring force
   the controller commands. With ft_pull_leak_offset_N = [lo, hi] it also gets a constant offset per
   axis and episode. Smoothed and placed like 5, and the two can be combined. The pull only gets
   large if the hand lags its target, so pair it with a dead zone that matches the robot's friction
   (env.ctrl.default_dead_zone; the walks needed 5-9 N in x).

7. Task-space Coulomb friction at the fingertip (off by default). FORGE's dead zone removes part of
   the controller's push, but the frictionless sim arm then coasts instead of sticking, so it doesn't
   lag like the real Franka (session-2 replays). With ctrl.task_friction_lo_N / _hi_N, each episode
   draws a friction per axis and direction ([x+, x-, y+, y-, z+, z-]) and every physics step adds,
   through J^T, the force that holds the fingertip still if it is within those limits, else the
   limit against the motion (observation.coulomb_friction). Not applied during resets.
   ctrl.task_friction_ramp_steps = [start, end] scales it from 0 to 1 over those env steps (a training
   curriculum: from scratch with full friction the policy learns to avoid contact and never inserts).
   With ctrl.ramp_leak_and_dead_zone = True the same factor s also scales the force leaks (5, 6 and the
   offset) by s and the translational part of FORGE's dead zone (redrawn from default_dead_zone) by 1 - s,
   so training starts as plain FORGE and ends with the friction model (dead zone 0, as it should be).

8. Collapse check (off by default). From scratch, some seeds settle on hovering above the hole (reward
   ~77, success 0, no contact) within ~20 epochs and never recover. With collapse_check =
   [start, end, min_success] (env steps, fraction), the mean episode success rate over episodes ending
   in [start, end) is checked at the first episode end at or after end; below min_success, training
   stops with a RuntimeError (the job's log says why; no final checkpoint is written).
"""

import torch

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from forge_blocks.observation import (
    CollapseCheck,
    coulomb_friction,
    force_deadband,
    friction_leak,
    noisy_finite_difference,
    pull_leak,
    ramp_factor,
    sample_radius_error,
)


class ForgeBlocksEnv(ForgeEnv):
    _prev_noisy_pos = None
    _prev_noisy_quat = None
    _force_bias = None
    _leak_N = None  # (num_envs, 3) per-episode speed-leak size; None while that leak is off
    _pull_alpha = None  # (num_envs, 3) per-episode share of the pull that leaks; None while off
    _leak_offset = None  # (num_envs, 3) per-episode constant offset [N]; None while off
    _leak_smooth = None  # EMA of the sum of the leaks above; None while all are off
    _ctrl_target_pos = None  # (num_envs, 3) last controller position target
    _task_friction = None  # (num_envs, 6) friction [x+, x-, y+, y-, z+, z-] in N; None while off
    _in_reset = False
    _collapse_check = None  # CollapseCheck, built on first use; False while the check is off

    def generate_ctrl_signals(self, ctrl_target_fingertip_midpoint_pos, *args, **kwargs):
        # ForgeEnv computes the clipped target as a local; keep it for the pull-based leak.
        self._ctrl_target_pos = ctrl_target_fingertip_midpoint_pos.clone()
        saved = self.ramped_dead_zone()
        try:
            super().generate_ctrl_signals(ctrl_target_fingertip_midpoint_pos, *args, **kwargs)
        finally:
            if saved is not None:
                self.dead_zone_thresholds = saved
        if self._task_friction is not None and not self._in_reset:
            self._apply_task_friction()

    def task_friction_scale(self) -> float:
        """Curriculum factor on the friction, 0..1, from ctrl.task_friction_ramp_steps and the env step."""
        start, end = getattr(self.cfg.ctrl, "task_friction_ramp_steps", [0, 0])
        return ramp_factor(getattr(self, "common_step_counter", 0), start, end)

    def _ramps_leak_and_dead_zone(self) -> bool:
        return bool(getattr(getattr(self.cfg, "ctrl", None), "ramp_leak_and_dead_zone", False))

    def ramped_dead_zone(self):
        """With ramp_leak_and_dead_zone, swap in the dead zone with its translational part scaled by
        1 - task_friction_scale() for one controller call. Returns the unscaled thresholds to put back
        afterwards, or None if nothing was swapped. ForgeEnv redraws the thresholds at every reset and
        every ~2 s, so they are scaled where they are used rather than where they are drawn."""
        dz = getattr(self, "dead_zone_thresholds", None)
        if dz is None or not self._ramps_leak_and_dead_zone():
            return None
        scaled = dz.clone()
        scaled[:, 0:3] *= 1.0 - self.task_friction_scale()
        self.dead_zone_thresholds = scaled
        return dz

    def _apply_task_friction(self):
        """Add the fingertip friction to the torques FactoryEnv.generate_ctrl_signals just set."""
        scale = self.task_friction_scale()
        if scale == 0.0:
            return
        J = self.fingertip_midpoint_jacobian[:, 0:3, :]  # (n, 3, 7) translational
        Jt = J.transpose(1, 2)
        task_mass = torch.linalg.inv(J @ torch.linalg.inv(self.arm_mass_matrix) @ Jt)
        force = coulomb_friction(
            self.applied_wrench[:, 0:3], self.fingertip_midpoint_linvel, task_mass, self.physics_dt,
            scale * self._task_friction[:, 0::2], scale * self._task_friction[:, 1::2],
        )
        n = J.shape[2]
        self.joint_torque[:, 0:n] = self.joint_torque[:, 0:n] + (Jt @ force.unsqueeze(-1)).squeeze(-1)
        self._robot.set_joint_effort_target(self.joint_torque)

    def controller_pull(self) -> torch.Tensor:
        """Spring force the controller commands, stiffness x (target - fingertip), as on the robot
        (deploy/franka_bridge.py: pull = k * (target - tip)). Zero before the first control step."""
        if self._ctrl_target_pos is None:
            return torch.zeros_like(self.fingertip_midpoint_pos)
        return self.task_prop_gains[:, 0:3] * (self._ctrl_target_pos - self.fingertip_midpoint_pos)

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)

        if self._leak_smooth is not None:
            leak = torch.zeros_like(self.noisy_force)
            if self._leak_N is not None:
                leak = leak + friction_leak(
                    self.fingertip_midpoint_linvel, self._leak_N, self.cfg.obs_rand.ft_force_leak_v_mps
                )
            if self._pull_alpha is not None:
                leak = leak + pull_leak(self.controller_pull(), self._pull_alpha)
            if self._leak_offset is not None:
                leak = leak + self._leak_offset
            if self._ramps_leak_and_dead_zone():
                leak = self.task_friction_scale() * leak
            ema = self.cfg.ft_smoothing_factor  # the same EMA ForgeEnv applies to the sensed force
            self._leak_smooth = ema * leak + (1 - ema) * self._leak_smooth
            self.noisy_force = self.noisy_force + self._leak_smooth

        scale = getattr(self.cfg.obs_rand, "ft_force_scale", 1.0)
        if scale != 1.0:
            self.noisy_force = scale * self.noisy_force
        if self._force_bias is not None:
            self.noisy_force = self.noisy_force + self._force_bias
        deadband = getattr(self.cfg.obs_rand, "ft_force_deadband_N", 0.0)
        if deadband > 0.0:
            self.noisy_force = force_deadband(
                self.noisy_force, deadband, getattr(self.cfg.obs_rand, "ft_force_deadband_mode", "axis")
            )

        if self._prev_noisy_pos is None:
            self._remember_noisy_pose()
        self.ee_linvel_fd, self.ee_angvel_fd = noisy_finite_difference(
            self.noisy_fingertip_pos, self._prev_noisy_pos, self.noisy_fingertip_quat, self._prev_noisy_quat, dt
        )
        self._remember_noisy_pose()

    def randomize_initial_state(self, env_ids):
        # Factory's reset drives the hand to its start pose with generate_ctrl_signals; no friction there.
        self._in_reset = True
        try:
            super().randomize_initial_state(env_ids)
        finally:
            self._in_reset = False

        # Factory zeroes the velocities at the end of its reset; start the history from here so the
        # first step after a reset doesn't difference across the teleport.
        self._remember_noisy_pose()

        bias_n = getattr(self.cfg.obs_rand, "ft_force_bias_N", 0.0)
        if bias_n > 0.0:
            if self._force_bias is None:
                self._force_bias = torch.zeros((self.num_envs, 3), device=self.device)
            direction = torch.randn((len(env_ids), 3), device=self.device)
            self._force_bias[env_ids] = bias_n * direction / torch.linalg.norm(direction, dim=1, keepdim=True)

        obs_rand = self.cfg.obs_rand
        leak_lo, leak_hi = getattr(obs_rand, "ft_force_leak_N", [0.0, 0.0])
        if leak_hi > 0.0:
            self._leak_N = self._draw_per_axis(self._leak_N, env_ids, leak_lo, leak_hi)
        alpha_lo, alpha_hi = getattr(obs_rand, "ft_pull_leak_alpha", [0.0, 0.0])
        if alpha_hi > 0.0:
            self._pull_alpha = self._draw_per_axis(self._pull_alpha, env_ids, alpha_lo, alpha_hi)
        off_lo, off_hi = getattr(obs_rand, "ft_pull_leak_offset_N", [0.0, 0.0])
        if off_lo != 0.0 or off_hi != 0.0:
            self._leak_offset = self._draw_per_axis(self._leak_offset, env_ids, off_lo, off_hi)
        if self._leak_N is not None or self._pull_alpha is not None or self._leak_offset is not None:
            if self._leak_smooth is None:
                self._leak_smooth = torch.zeros((self.num_envs, 3), device=self.device)
            # The reset just moved the hand; start the new episode's smoothed leak from 0, as ForgeEnv
            # does for the force.
            self._leak_smooth[env_ids] = 0.0
        if self._ctrl_target_pos is not None:
            # The reset's last target belongs to the reset motion; no pull until the next control step.
            self._ctrl_target_pos[env_ids] = self.fingertip_midpoint_pos[env_ids]

        ctrl = getattr(self.cfg, "ctrl", None)
        fric_lo = getattr(ctrl, "task_friction_lo_N", [0.0] * 6)
        fric_hi = getattr(ctrl, "task_friction_hi_N", [0.0] * 6)
        if max(fric_hi) > 0.0:
            if self._task_friction is None:
                self._task_friction = torch.zeros((self.num_envs, 6), device=self.device)
            lo = torch.tensor(fric_lo, dtype=torch.float32, device=self.device)
            hi = torch.tensor(fric_hi, dtype=torch.float32, device=self.device)
            self._task_friction[env_ids] = lo + (hi - lo) * torch.rand((len(env_ids), 6), device=self.device)

        mode = getattr(self.cfg.obs_rand, "error_radius_mode", "gaussian")
        if mode == "gaussian":
            return
        # Safe to overwrite here: Factory only uses the noisy estimate after this point (FORGE's
        # initial-action computation in _reset_idx, observations and action frames).
        error = sample_radius_error(len(env_ids), self.cfg.obs_rand.error_radius, mode, self.device)
        if mode == "xy":
            self.init_fixed_pos_obs_noise[env_ids, 0:2] = error[:, 0:2]
        else:
            self.init_fixed_pos_obs_noise[env_ids] = error

    def _log_factory_metrics(self, rew_dict, curr_successes):
        super()._log_factory_metrics(rew_dict, curr_successes)
        if self._collapse_check is None:
            start, end, min_success = getattr(self.cfg, "collapse_check", [0, 0, 0.0])
            self._collapse_check = CollapseCheck(start, end, min_success) if end > start else False
        if self._collapse_check and bool(self.reset_buf.any()):
            rate = float(curr_successes.float().mean())
            verdict = self._collapse_check.update(self.common_step_counter, rate)
            if verdict:
                print(f"[collapse_check] {verdict}", flush=True)
                raise RuntimeError(f"collapse_check: {verdict}")

    def _draw_per_axis(self, buf, env_ids, lo, hi):
        """U(lo, hi) per axis for env_ids, allocating the (num_envs, 3) buffer on first use."""
        if buf is None:
            buf = torch.zeros((self.num_envs, 3), device=self.device)
        buf[env_ids] = lo + (hi - lo) * torch.rand((len(env_ids), 3), device=self.device)
        return buf

    def _remember_noisy_pose(self):
        self._prev_noisy_pos = self.noisy_fingertip_pos.clone()
        self._prev_noisy_quat = self.noisy_fingertip_quat.clone()
