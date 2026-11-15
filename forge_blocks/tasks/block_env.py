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
"""

import torch

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from forge_blocks.observation import force_deadband, friction_leak, noisy_finite_difference, sample_radius_error


class ForgeBlocksEnv(ForgeEnv):
    _prev_noisy_pos = None
    _prev_noisy_quat = None
    _force_bias = None
    _leak_N = None  # (num_envs, 3) per-episode leak size; None while the leak is off
    _leak_smooth = None

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)

        if self._leak_N is not None:
            leak = friction_leak(self.fingertip_midpoint_linvel, self._leak_N, self.cfg.obs_rand.ft_force_leak_v_mps)
            alpha = self.cfg.ft_smoothing_factor  # the same EMA ForgeEnv applies to the sensed force
            self._leak_smooth = alpha * leak + (1 - alpha) * self._leak_smooth
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
        super().randomize_initial_state(env_ids)

        # Factory zeroes the velocities at the end of its reset; start the history from here so the
        # first step after a reset doesn't difference across the teleport.
        self._remember_noisy_pose()

        bias_n = getattr(self.cfg.obs_rand, "ft_force_bias_N", 0.0)
        if bias_n > 0.0:
            if self._force_bias is None:
                self._force_bias = torch.zeros((self.num_envs, 3), device=self.device)
            direction = torch.randn((len(env_ids), 3), device=self.device)
            self._force_bias[env_ids] = bias_n * direction / torch.linalg.norm(direction, dim=1, keepdim=True)

        leak_lo, leak_hi = getattr(self.cfg.obs_rand, "ft_force_leak_N", [0.0, 0.0])
        if leak_hi > 0.0:
            if self._leak_N is None:
                self._leak_N = torch.zeros((self.num_envs, 3), device=self.device)
                self._leak_smooth = torch.zeros((self.num_envs, 3), device=self.device)
            self._leak_N[env_ids] = leak_lo + (leak_hi - leak_lo) * torch.rand((len(env_ids), 3), device=self.device)
            # The reset just moved the hand; start the new episode's smoothed leak from 0, as ForgeEnv
            # does for the force.
            self._leak_smooth[env_ids] = 0.0

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

    def _remember_noisy_pose(self):
        self._prev_noisy_pos = self.noisy_fingertip_pos.clone()
        self._prev_noisy_quat = self.noisy_fingertip_quat.clone()
