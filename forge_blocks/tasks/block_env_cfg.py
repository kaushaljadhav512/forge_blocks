"""FORGE env configs for inserting a square block into a slot.

We subclass FORGE's peg-insert config and keep task.name == "peg_insert", so Factory's grasp
offset, keypoint reward and success check (all keyed on that string) work unchanged:
  * held asset origin at its bottom face, fixed asset origin at the slot floor;
  * success = block bottom within 2.5 mm (xy) of the floor centre and within
    success_threshold * SLOT_DEPTH of it in z.

Differences from the peg (the only things a square block changes):
  * Slot yaw is not randomised (FORGE spins the round hole 0-360 deg), because the policy never
    observes the slot's orientation.
  * Initial hand yaw noise defaults to 0 (option (a) in the plan). Set YAW_NOISE_DEG > 0 to make
    the policy correct yaw too; the real robot setup must then match.
"""

import copy
import math
import os

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass

from isaaclab_tasks.direct.factory.factory_tasks_cfg import FixedAssetCfg, HeldAssetCfg
from isaaclab_tasks.direct.forge.forge_env_cfg import ForgeCtrlCfg, ForgeEnvCfg, ForgeObsRandCfg, ForgeTaskPegInsertCfg
from isaaclab_tasks.direct.forge.forge_tasks_cfg import ForgePegInsert

from forge_blocks import geometry as g

ASSET_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "assets")

YAW_NOISE_DEG = 0.0


@configclass
class BlockObsRandCfg(ForgeObsRandCfg):
    # The paper trains with sigma = 2.5 mm (Table II); upstream defaults to 1 mm.
    fixed_asset_pos = [0.0025, 0.0025, 0.0025]
    # "gaussian" uses fixed_asset_pos. "xy" and "xyz" use a radius from U[error_radius] in metres;
    # see block_env.py. A string and list rather than None so Hydra overrides keep their types.
    error_radius_mode: str = "gaussian"
    error_radius: list = [0.0, 0.0]
    # Evaluation-only stress tests of the force observation (see block_env.py). Leave at
    # 1.0 / 0.0 for training: the exported runner_config.json records them.
    ft_force_scale: float = 1.0
    ft_force_bias_N: float = 0.0
    ft_force_deadband_N: float = 0.0
    ft_force_deadband_mode: str = "axis"
    # Joint-friction leak into the force estimate (see block_env.py): per-episode, per-axis size
    # range in N, and the speed in m/s at which it reaches ~76% of that size (tanh(1)).
    ft_force_leak_N: list = [0.0, 0.0]
    ft_force_leak_v_mps: float = 0.002
    # Friction leak tied to the controller's pull (see block_env.py; fitted on the session-2 walks):
    # fake force = offset - alpha * stiffness * (target - fingertip), per axis. Per episode and axis,
    # alpha ~ U(ft_pull_leak_alpha) and offset ~ U(ft_pull_leak_offset_N) in N (signed). Both off by
    # default. The walks gave alpha 0.25-0.43 and offsets near -1 N in x and z. Train with FORGE's
    # actuation dead zone widened in x (env.ctrl.default_dead_zone), or the sim hand follows its target
    # and there is little pull to leak.
    ft_pull_leak_alpha: list = [0.0, 0.0]
    ft_pull_leak_offset_N: list = [0.0, 0.0]


@configclass
class BlockCtrlCfg(ForgeCtrlCfg):
    # Task-space Coulomb friction at the fingertip (see block_env.py), standing in for the real Franka's
    # joint friction. Per episode, each of [x+, x-, y+, y-, z+, z-] (the friction when moving or pushed
    # that way) is drawn from U(lo, hi) in N. Off by default. The session-2 walks lagged as if
    # ~4.6 / 1.9 N in x, ~1.5 / 2.0 N in y, ~2 / 1 N in z. Set FORGE's dead zone (default_dead_zone)
    # to 0 when using it, since that was its stand-in.
    task_friction_lo_N: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    task_friction_hi_N: list = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]


@configclass
class ForgeBlockInsert(ForgePegInsert):
    name = "peg_insert"  # reuse Factory's peg_insert code paths; see module docstring
    asset_size = g.BLOCK_WIDTH * 1e3

    # Fingertip starts 5 mm (block bottom above rim) + noise above the slot opening.
    hand_init_pos: list = [0.0, 0.0, g.BLOCK_HEIGHT - g.FRANKA_FINGERPAD_LENGTH + 0.005]
    hand_init_orn: list = [math.pi, 0.0, 0.0]
    hand_init_orn_noise: list = [0.0, 0.0, math.radians(YAW_NOISE_DEG)]

    fixed_asset_init_orn_deg: float = 0.0
    fixed_asset_init_orn_range_deg: float = 0.0


def _make_task(clearance_mm: float) -> ForgeBlockInsert:
    task = ForgeBlockInsert()
    task.held_asset_cfg = HeldAssetCfg(
        usd_path=os.path.join(ASSET_DIR, g.BLOCK_USD_NAME),
        diameter=g.BLOCK_WIDTH,  # Factory sets the pre-grasp finger opening from this
        height=g.BLOCK_HEIGHT,
        mass=g.BLOCK_MASS,
    )
    task.fixed_asset_cfg = FixedAssetCfg(
        usd_path=os.path.join(ASSET_DIR, g.slot_usd_name(clearance_mm)),
        diameter=g.slot_inner_width(clearance_mm),
        height=g.SLOT_DEPTH,
        base_height=0.0,
        mass=g.SLOT_MASS,
    )

    # Same spawn settings as Factory's peg/hole; only the files and masses change.
    task.held_asset = copy.deepcopy(task.held_asset)
    task.held_asset.spawn.usd_path = task.held_asset_cfg.usd_path
    task.held_asset.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=g.BLOCK_MASS)

    task.fixed_asset = copy.deepcopy(task.fixed_asset)
    task.fixed_asset.spawn.usd_path = task.fixed_asset_cfg.usd_path
    task.fixed_asset.spawn.mass_props = sim_utils.MassPropertiesCfg(mass=g.SLOT_MASS)
    # Our USD is a free rigid body; pin it to the world like a slot screwed to the table.
    task.fixed_asset.spawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(fix_root_link=True)
    return task


@configclass
class ForgeBlockEnvCfg(ForgeEnvCfg):
    task_name = "peg_insert"
    episode_length_s = 10.0
    obs_rand: BlockObsRandCfg = BlockObsRandCfg()
    ctrl: BlockCtrlCfg = BlockCtrlCfg()
    task: ForgeBlockInsert = _make_task(2)


@configclass
class ForgeBlock5mmCfg(ForgeBlockEnvCfg):
    task: ForgeBlockInsert = _make_task(5)


@configclass
class ForgeBlock2mmCfg(ForgeBlockEnvCfg):
    task: ForgeBlockInsert = _make_task(2)


@configclass
class ForgeBlock1mmCfg(ForgeBlockEnvCfg):
    task: ForgeBlockInsert = _make_task(1)


@configclass
class ForgePegCfg(ForgeTaskPegInsertCfg):
    """FORGE's 8 mm peg with the paper's noise and fixed velocity observations.

    Friction is left at upstream's values: held 0.75 and fixed (0.25, 1.25) static average to the
    paper's [0.5, 1.0] under PhysX's default combine mode. Checkpoints are not interchangeable with
    Isaac-Forge-PegInsert-Direct-v0, because the velocity observations differ.
    """

    obs_rand: BlockObsRandCfg = BlockObsRandCfg()
    ctrl: BlockCtrlCfg = BlockCtrlCfg()
