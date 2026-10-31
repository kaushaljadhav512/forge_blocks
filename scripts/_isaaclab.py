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


def run_upstream(script: str) -> None:
    path = os.path.join(isaaclab_root(), "scripts", "reinforcement_learning", "rl_games", script)
    sys.argv[0] = path
    runpy.run_path(path, run_name="__main__")
