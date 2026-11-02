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
"""

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv

from forge_blocks.observation import noisy_finite_difference, sample_radius_error


class ForgeBlocksEnv(ForgeEnv):
    _prev_noisy_pos = None
    _prev_noisy_quat = None

    def _compute_intermediate_values(self, dt):
        super()._compute_intermediate_values(dt)

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
