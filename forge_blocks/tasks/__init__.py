"""Gym registrations for the block tasks.

Only string entry points here: importing this module must not pull in Isaac Sim, because our
script wrappers import it before Isaac Lab's train.py starts the app.
"""

import gymnasium as gym

from forge_blocks.geometry import CLEARANCES_MM

# FORGE's own PPO/LSTM config; scripts/train.py renames the run so logs don't mix with the peg.
_RL_GAMES_CFG = "isaaclab_tasks.direct.forge.agents:rl_games_ppo_cfg.yaml"

_ENV = f"{__name__}.block_env:ForgeBlocksEnv"


def _register(task_id: str, cfg_name: str) -> None:
    gym.register(
        id=task_id,
        entry_point=_ENV,
        disable_env_checker=True,
        kwargs={
            "env_cfg_entry_point": f"{__name__}.block_env_cfg:{cfg_name}",
            "rl_games_cfg_entry_point": _RL_GAMES_CFG,
        },
    )


# Phase 1 reference: FORGE's 8 mm peg, plus the radius-error option for eval_sweep.
_register("Isaac-ForgeBlocks-Peg8mm-Direct-v0", "ForgePegCfg")

for _c in CLEARANCES_MM:
    _register(f"Isaac-ForgeBlocks-Block{_c}mm-Direct-v0", f"ForgeBlock{_c}mmCfg")
