#!/usr/bin/env bash
# Train one task for several seeds, one after another. Run it inside tmux.
#
#   scripts/train_seeds.sh <task> <run_name> "<seeds>" [train.py arguments and Hydra overrides]
#   scripts/train_seeds.sh Isaac-ForgeBlocks-Peg8mm-Direct-v0 peg_forge "0 1 2"
#
# Checkpoints go to logs/rl_games/<run_name>/seed<N>/nn/ in this repo. A seed whose folder already
# exists is skipped, so rerunning the command after a crash continues with the missing seeds.
# Delete the folder of a seed that stopped partway through before rerunning.
set -euo pipefail

if [ "$#" -lt 3 ]; then
    sed -n 2,8p "$0"
    exit 2
fi
task=$1; name=$2; seeds=$3; shift 3

repo=$(cd "$(dirname "$0")/.." && pwd)
isaaclab=${ISAACLAB_PATH:-$HOME/IsaacLab}
[ -x "$isaaclab/isaaclab.sh" ] || { echo "No isaaclab.sh in $isaaclab; set ISAACLAB_PATH" >&2; exit 1; }

cd "$repo"
for seed in $seeds; do
    if [ -d "logs/rl_games/$name/seed$seed" ]; then
        echo "[train_seeds] skipping seed $seed: logs/rl_games/$name/seed$seed exists"
        continue
    fi
    echo "[train_seeds] $task seed $seed, started $(date)"
    "$isaaclab/isaaclab.sh" -p scripts/train.py --task "$task" --headless --seed "$seed" \
        "agent.params.config.name=$name" "agent.params.config.full_experiment_name=seed$seed" "$@"
done
echo "[train_seeds] finished $(date)"
