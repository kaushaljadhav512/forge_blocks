"""Train with Isaac Lab's rl_games train.py; accepts all of its arguments.

    ./isaaclab.sh -p ~/forge_blocks/scripts/train.py --task Isaac-ForgeBlocks-Block2mm-Direct-v0 \
        --headless --seed 0 env.obs_rand.fixed_asset_pos=[0.005,0.005,0.0] \
        agent.params.config.name=block2mm_err5mm
"""

from _isaaclab import run_upstream

run_upstream("train.py")
