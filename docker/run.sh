#!/usr/bin/env bash
# Build the Isaac Lab v2.3.2 image, or run a command in a container with this repo mounted.
#
#   docker/run.sh build
#   docker/run.sh                 # interactive shell
#   docker/run.sh '$ISAACLAB_PATH/isaaclab.sh -p scripts/preflight.py --headless'
#   docker/run.sh 'scripts/train_seeds.sh Isaac-ForgeBlocks-Peg8mm-Direct-v0 peg_forge "0 1 2"'
#
# Don't use Isaac Lab's "container.py stop": it runs "docker compose down --volumes", which deletes
# the caches. The container here is removed on exit and the named volumes stay.
set -euo pipefail

repo=$(cd "$(dirname "$0")/.." && pwd)
isaaclab=${FORGE_ISAACLAB:-$(dirname "$repo")/IsaacLab}

version=$(git -C "$isaaclab" describe --tags --exact-match 2>/dev/null || echo unknown)
if [ "$version" != "v2.3.2" ]; then
    echo "$isaaclab is at '$version', not v2.3.2. Set FORGE_ISAACLAB or clone it there:" >&2
    echo "  git clone --branch v2.3.2 --depth 1 https://github.com/isaac-sim/IsaacLab.git $isaaclab" >&2
    exit 1
fi

# Upstream's compose file bind-mounts this file and expects it to exist.
touch "$isaaclab/docker/.isaac-lab-docker-history"

export FORGE_REPO=$repo
upstream=(docker compose --project-name forge --project-directory "$isaaclab/docker"
    --env-file "$isaaclab/docker/.env.base" --file "$isaaclab/docker/docker-compose.yaml" --profile base)
compose=("${upstream[@]}" --file "$repo/docker/compose.forge.yaml")

if [ "${1:-}" = build ]; then
    "${upstream[@]}" build isaac-lab-base
    exec docker build --tag forge-isaaclab --file "$repo/docker/Dockerfile.forge" "$repo/docker"
fi

tty=(); [ -t 0 ] && [ -t 1 ] || tty=(-T)
# The container runs as root, so hand everything it wrote back to the host user on the way out.
exec "${compose[@]}" run --rm "${tty[@]}" isaac-lab-base -c '
    if [ -n "$1" ]; then bash -c "$1"; else bash; fi
    status=$?
    chown -R "$2" /workspace/forge_blocks
    exit $status
' _ "${1:-}" "$(id -u):$(id -g)"
