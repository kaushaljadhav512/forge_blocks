"""Train with Isaac Lab's rl_games train.py; accepts all of its arguments.

    ./isaaclab.sh -p ~/forge_blocks/scripts/train.py --task Isaac-ForgeBlocks-Block2mm-Direct-v0 \
        --headless --seed 0 agent.params.config.name=block2mm \
        agent.params.config.full_experiment_name=seed0

Set full_experiment_name: upstream's default is "test", so runs with the same name overwrite each
other. scripts/train_seeds.sh does this for you.
"""

from _isaaclab import run_upstream

run_upstream("train.py")
