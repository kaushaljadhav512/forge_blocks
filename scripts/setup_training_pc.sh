#!/usr/bin/env bash
# Install Isaac Lab v2.3.2 with rl_games and this package. Isaac Sim 5.1 must already be available,
# either linked as IsaacLab/_isaac_sim or installed in the active conda or virtual environment:
# https://isaac-sim.github.io/IsaacLab/v2.3.2/source/setup/installation/index.html
#
#   scripts/setup_training_pc.sh [path to IsaacLab, default ~/IsaacLab]
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
isaaclab=${1:-$HOME/IsaacLab}

if [ ! -d "$isaaclab/.git" ]; then
    git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git "$isaaclab"
fi

version=$(git -C "$isaaclab" describe --tags --exact-match 2>/dev/null || echo unknown)
if [ "$version" != "v2.3.2" ]; then
    echo "$isaaclab is at '$version', not v2.3.2. Run: git -C $isaaclab checkout v2.3.2" >&2
    exit 1
fi

if [ ! -e "$isaaclab/_isaac_sim" ] && ! python -c "import isaacsim" >/dev/null 2>&1; then
    echo "Isaac Sim not found. Link it to $isaaclab/_isaac_sim or activate its environment." >&2
    exit 1
fi

cd "$isaaclab"
./isaaclab.sh --install rl_games
./isaaclab.sh -p -m pip install -e "$repo"
./isaaclab.sh -p -m unittest discover -s "$repo/tests"

echo
echo "Done. Next, from $repo:"
echo "  export ISAACLAB_PATH=$isaaclab"
echo "  \$ISAACLAB_PATH/isaaclab.sh -p scripts/preflight.py --headless"
