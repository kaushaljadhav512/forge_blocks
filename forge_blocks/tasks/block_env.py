"""ForgeEnv plus one option: slot-position error with a controlled radius.

FORGE samples the slot-position estimate error as independent Gaussians per axis
(obs_rand.fixed_asset_pos = sigmas). The ablation grid is indexed by error *distance*
(0.5 / 1 / 2 / 3 cm), so we optionally replace the xy part with an offset of radius
U[r_min, r_max] in a uniformly random direction:
  * training "up to 1 cm":  env.obs_rand.use_xy_radius=true env.obs_rand.fixed_asset_pos_xy_radius=[0.0,0.01]
  * evaluating at 1 cm:     same flag with [0.01,0.01] (eval_sweep.py sets this itself)
The z part keeps FORGE's Gaussian (fixed_asset_pos[2]).
"""

import math

import torch

from isaaclab_tasks.direct.forge.forge_env import ForgeEnv


class ForgeBlocksEnv(ForgeEnv):
    def randomize_initial_state(self, env_ids):
        super().randomize_initial_state(env_ids)

        if not getattr(self.cfg.obs_rand, "use_xy_radius", False):
            return
        # Safe to overwrite here: Factory only uses the noisy estimate after this point (FORGE's
        # initial-action computation in _reset_idx, observations and action frames).
        r_min, r_max = self.cfg.obs_rand.fixed_asset_pos_xy_radius
        radius = r_min + (r_max - r_min) * torch.rand(len(env_ids), device=self.device)
        theta = 2 * math.pi * torch.rand(len(env_ids), device=self.device)
        self.init_fixed_pos_obs_noise[env_ids, 0] = radius * torch.cos(theta)
        self.init_fixed_pos_obs_noise[env_ids, 1] = radius * torch.sin(theta)
