"""Check a task on the training PC before spending hours on it. Trains nothing.

    ./isaaclab.sh -p scripts/preflight.py --headless

Checks that the observation sizes and position noise are as expected, and that the velocity
observation follows real motion, which upstream's does not. Also prints the simulation speed, so
you can estimate how long training will take.
"""

import argparse
import math
import os
import sys
import time

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-ForgeBlocks-Peg8mm-Direct-v0")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--steps", type=int, default=64, help="policy steps; stay under one episode (150)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import forge_blocks.tasks  # noqa: E402,F401

EXPECTED_OBS_DIM = 24
EXPECTED_STATE_DIM = 61
EXPECTED_POS_SIGMA = 0.0025


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    # Without fingertip noise, a correct velocity follows the motion and upstream's is exactly zero.
    env_cfg.obs_rand.fingertip_pos = 0.0
    env_cfg.obs_rand.fingertip_rot_deg = 0.0
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped

    results = []

    def check(name, ok, detail):
        results.append(ok)
        print(f"[preflight] {'PASS' if ok else 'FAIL'}  {name}: {detail}")

    obs, _ = env.reset()
    obs_dim, state_dim = obs["policy"].shape[1], obs["critic"].shape[1]
    check("policy observation size", obs_dim == EXPECTED_OBS_DIM, f"{obs_dim}, expected {EXPECTED_OBS_DIM}")
    check("critic state size", state_dim == EXPECTED_STATE_DIM, f"{state_dim}, expected {EXPECTED_STATE_DIM}")

    sigma = list(env_cfg.obs_rand.fixed_asset_pos)
    check("position noise setting", sigma == [EXPECTED_POS_SIGMA] * 3, f"{sigma} m, expected {EXPECTED_POS_SIGMA} m")
    measured = u.init_fixed_pos_obs_noise.std().item()
    check("measured position noise", 0.5 * EXPECTED_POS_SIGMA < measured < 1.5 * EXPECTED_POS_SIGMA, f"{measured:.5f} m")

    fd, true = [], []
    actions = torch.zeros(u.num_envs, env_cfg.action_space, device=u.device)
    start = time.time()
    with torch.inference_mode():
        for t in range(args_cli.steps):
            actions[:, 0] = math.sin(t * 0.3)
            actions[:, 1] = math.cos(t * 0.3)
            env.step(actions)
            fd.append(u.ee_linvel_fd.clone())
            true.append(u.fingertip_midpoint_linvel.clone())
    elapsed = time.time() - start

    fd, true = torch.cat(fd).flatten(), torch.cat(true).flatten()
    corr = torch.corrcoef(torch.stack((fd, true)))[0, 1].item()
    check("velocity observation has signal", fd.std().item() > 1e-3, f"std {fd.std().item():.4f} m/s")
    check("velocity observation follows motion", corr > 0.9, f"correlation {corr:.3f} with true velocity")

    # An iteration collects 128 steps from every env (horizon_length), plus the network updates.
    collect_s = elapsed / args_cli.steps * 128
    print(f"[preflight] speed: about {collect_s:.0f} s of simulation per training iteration")
    print(f"[preflight] 200 iterations need at least {collect_s * 200 / 3600:.1f} h, excluding network updates")

    env.close()
    print(f"[preflight] {'ALL PASSED' if all(results) else 'FAILED'}")
    if not all(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
