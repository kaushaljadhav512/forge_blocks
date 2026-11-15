"""Run one of Isaac Lab's rl_games scripts with our tasks registered first.

Isaac Lab's train.py/play.py only know tasks registered by isaaclab_tasks. Registering ours
(string entry points only, no Isaac Sim imports) and then running their script in-process keeps
us on the exact upstream training code.
"""

import os
import runpy
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import forge_blocks.tasks  # noqa: E402,F401  (registers Isaac-ForgeBlocks-*)


def isaaclab_root() -> str:
    root = os.environ.get("ISAACLAB_PATH")
    if not root or not os.path.isdir(root):
        sys.exit("Set ISAACLAB_PATH to your IsaacLab checkout (v2.3.2), e.g. export ISAACLAB_PATH=~/IsaacLab")
    return root


def agent_device_overrides(argv: list[str]) -> list[str]:
    """Hydra overrides that put the policy on the --device GPU.

    Upstream train.py and play.py (v2.3.2) move only the sim to --device; the policy stays on the agent
    config's device (cuda:0) unless training is distributed. With --device cuda:1 the two differ and the
    first step fails with "Expected all tensors to be on the same device". Overrides already given on the
    command line win.
    """
    device = None
    for i, arg in enumerate(argv):
        if arg == "--device" and i + 1 < len(argv):
            device = argv[i + 1]
        elif arg.startswith("--device="):
            device = arg.split("=", 1)[1]
    if device is None:
        return []
    out = []
    for key in ("device", "device_name"):
        prefix = f"agent.params.config.{key}="
        if not any(a.startswith(prefix) for a in argv):
            out.append(prefix + device)
    return out


def run_upstream(script: str) -> None:
    path = os.path.join(isaaclab_root(), "scripts", "reinforcement_learning", "rl_games", script)
    sys.argv[0] = path
    sys.argv += agent_device_overrides(sys.argv[1:])
    runpy.run_path(path, run_name="__main__")
