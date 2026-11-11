"""Drive the sim through deploy/policy_runner.py and check it against ForgeEnv step by step.

    ./isaaclab.sh -p scripts/sim_bridge_test.py --task Isaac-ForgeBlocks-Block2mm-Direct-v0 \
        --export exports/block2mm_seed0 --episodes 2 --headless

Each env gets its own ForgePolicyRunner fed only quantities a real robot can provide (fingertip
pose, velocities, contact force, the *estimated* slot position). Every step we compare:
  * the runner's observation vs the env's policy observation,
  * the runner's EMA-smoothed actions vs env.actions,
  * the runner's pose target vs the target ForgeEnv sends to its controller (first substep).
Any mismatch is a frame/ordering/normaliser bug that would otherwise show up on the robot.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--export", required=True, help="folder written by export_policy.py")
parser.add_argument("--episodes", type=int, default=2)
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--atol", type=float, default=1e-4)
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
import forge_blocks.tasks  # noqa: E402,F401
from policy_runner import ForgePolicyRunner  # noqa: E402


def np_(t, i):
    return t[i].detach().cpu().numpy().astype(np.float64)


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    n = u.num_envs

    template = ForgePolicyRunner.from_export(args_cli.export, device=str(u.device))
    runners = []
    for _ in range(n):
        r = ForgePolicyRunner(copy.deepcopy(template.cfg), template.policy)
        runners.append(r)

    # Record the first controller target ForgeEnv computes in each policy step.
    captured = {}
    original = u.generate_ctrl_signals

    def capture(ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, **kw):
        if "pos" not in captured:
            captured["pos"] = ctrl_target_fingertip_midpoint_pos.clone()
            captured["quat"] = ctrl_target_fingertip_midpoint_quat.clone()
        return original(ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, **kw)

    u.generate_ctrl_signals = capture

    def reset_runners():
        slot_est = u.fixed_pos_obs_frame + u.init_fixed_pos_obs_noise
        for i, r in enumerate(runners):
            # Per-episode randomisations the policy is conditioned on: copy them so obs can match.
            r.cfg.ema_factor = float(u.ema_factor[i])
            r.cfg.force_threshold = float(u.contact_penalty_thresholds[i])
            r.cfg.pos_action_threshold = np_(u.pos_threshold, i).tolist()
            r.cfg.rot_action_threshold = np_(u.rot_threshold, i).tolist()
            r.reset(np_(u.fingertip_midpoint_pos, i), np_(u.fingertip_midpoint_quat, i), np_(slot_est, i))
            assert np.allclose(r.actions[[0, 1, 2, 5, 6]], np_(u.actions, i)[[0, 1, 2, 5, 6]], atol=args_cli.atol), (
                "reset actions differ",
                r.actions,
                np_(u.actions, i),
            )

    obs_dict, _ = env.reset()
    reset_runners()
    worst = {"obs": 0.0, "actions": 0.0, "target_pos": 0.0, "target_quat": 0.0}
    successes, episodes = [], 0

    while episodes < args_cli.episodes:
        env_obs = obs_dict["policy"]
        mus = torch.zeros(n, 7, device=u.device)
        targets = []
        for i, r in enumerate(runners):
            obs = r.build_obs(
                np_(u.noisy_fingertip_pos, i),
                np_(u.noisy_fingertip_quat, i),
                np_(u.ee_linvel_fd, i),
                np_(u.ee_angvel_fd, i),
                np_(u.noisy_force, i),
            )
            worst["obs"] = max(worst["obs"], float(np.abs(obs - np_(env_obs, i)).max()))
            mu = r._run_policy(obs)
            mus[i] = torch.as_tensor(mu, device=u.device)
            targets.append(r.apply_action(mu, np_(u.fingertip_midpoint_pos, i), np_(u.fingertip_midpoint_quat, i)))

        captured.clear()
        obs_dict, _, terminated, truncated, _ = env.step(torch.clamp(mus, -1.0, 1.0))

        done = terminated | truncated
        for i, (pos, quat) in enumerate(targets):
            # env.step has already reset finished envs, so u.actions belongs to the next episode.
            if not done[i]:
                worst["actions"] = max(worst["actions"], float(np.abs(runners[i].actions - np_(u.actions, i)).max()))
            worst["target_pos"] = max(worst["target_pos"], float(np.abs(pos - np_(captured["pos"], i)).max()))
            dot = abs(float(np.dot(quat, np_(captured["quat"], i))))
            worst["target_quat"] = max(worst["target_quat"], 1.0 - min(dot, 1.0))

        if done.any():
            successes.append(u.ep_succeeded.float().mean().item())
            episodes += 1
            reset_runners()
            print(f"[bridge] episode {episodes}: success {successes[-1]:.2f}, worst diffs so far {worst}")

    ok = all(v <= args_cli.atol for v in worst.values())
    print(f"[bridge] {'PASS' if ok else 'FAIL'}  worst diffs: {worst}  success: {np.mean(successes):.3f}")
    env.close()
    if not ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
    simulation_app.close()
