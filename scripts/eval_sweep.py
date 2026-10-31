"""Evaluate one checkpoint at several slot-position error distances (the sim half of the grid).

    ./isaaclab.sh -p scripts/eval_sweep.py --task Isaac-ForgeBlocks-Block2mm-Direct-v0 \
        --checkpoint logs/rl_games/<run>/<ts>/nn/<run>.pth --errors_mm 0 5 10 20 30 \
        --episodes 1000 --headless --out results/block2mm_seed0.csv

Error = xy offset between the true slot and the estimate the policy is given, at exactly that
distance in a random direction (see forge_blocks/tasks/block_env.py). Per episode it logs:
success, time to first success, peak contact force, and the policy's own success prediction.
Works for the peg too (Isaac-ForgeBlocks-Peg8mm-Direct-v0) with checkpoints from the upstream ID.
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--errors_mm", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0, 30.0])
parser.add_argument("--episodes", type=int, default=1000, help="per error level")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--pred_threshold", type=float, default=0.5, help="success-prediction cut-off")
parser.add_argument("--out", required=True, help="per-episode CSV; a .summary.json is written next to it")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
app_launcher = AppLauncher(args_cli)
simulation_app = app_launcher.app

import csv  # noqa: E402
import json  # noqa: E402
import math  # noqa: E402

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import forge_blocks.tasks  # noqa: E402,F401


def make_player(env, agent_cfg):
    """Same steps as Isaac Lab's rl_games play.py."""
    cfg = agent_cfg["params"]["config"]
    env_params = agent_cfg["params"]["env"]
    wrapped = RlGamesVecEnvWrapper(
        env,
        cfg["device"],
        env_params.get("clip_observations", math.inf),
        env_params.get("clip_actions", math.inf),
        env_params.get("obs_groups"),
        env_params.get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    cfg["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()
    player.is_deterministic = True  # evaluate the mean action, as on the robot
    return wrapped, player


def run_episode(env, wrapped, player, obs):
    """Run until the (synchronised) Factory episode ends. Returns per-env metric tensors and next obs."""
    u = env.unwrapped
    n = u.num_envs
    peak_raw = torch.zeros(n, device=u.device)
    peak_smooth = torch.zeros(n, device=u.device)
    first_pred = torch.full((n,), -1, dtype=torch.long, device=u.device)
    if player.is_rnn:
        player.init_rnn()
    t = 0
    while True:
        with torch.inference_mode():
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = wrapped.step(actions)
        t += 1
        peak_raw = torch.maximum(peak_raw, torch.linalg.norm(u.force_sensor_world[:, :3], dim=1))
        peak_smooth = torch.maximum(peak_smooth, torch.linalg.norm(u.force_sensor_smooth[:, :3], dim=1))
        pred = (u.actions[:, 6] + 1) / 2 > args_cli.pred_threshold
        first_pred = torch.where((first_pred < 0) & pred, torch.full_like(first_pred, t), first_pred)
        if dones.any():
            assert dones.all(), "Factory episodes should end together"
            # ep_succeeded/ep_success_times are only cleared on the next _pre_physics_step.
            out = {
                "success": u.ep_succeeded.clone().bool(),
                "success_step": u.ep_success_times.clone(),
                "peak_force_raw": peak_raw,
                "peak_force_smooth": peak_smooth,
                "first_pred_step": first_pred,
            }
            return out, obs


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if not hasattr(env_cfg.obs_rand, "use_xy_radius"):
        sys.exit(f"{args_cli.task} has no radius option; use an Isaac-ForgeBlocks-* task ID")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    wrapped, player = make_player(env, agent_cfg)
    step_dt = u.step_dt

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    rows, summary = [], {}
    obs = None
    for err_mm in args_cli.errors_mm:
        r = err_mm * 1e-3
        u.cfg.obs_rand.use_xy_radius = True
        u.cfg.obs_rand.fixed_asset_pos_xy_radius = [r, r]
        if obs is None:
            obs = wrapped.reset()
        else:
            # The episode in flight was reset with the previous level's error; run and discard it.
            _, obs = run_episode(env, wrapped, player, obs)
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = player.get_batch_size(obs, 1)

        level = []
        while len(level) < args_cli.episodes:
            m, obs = run_episode(env, wrapped, player, obs)
            for i in range(u.num_envs):
                succ = bool(m["success"][i])
                fp = int(m["first_pred_step"][i])
                level.append(
                    {
                        "error_mm": err_mm,
                        "success": int(succ),
                        "time_to_success_s": float(m["success_step"][i]) * step_dt if succ else "",
                        "peak_force_raw_N": float(m["peak_force_raw"][i]),
                        "peak_force_smooth_N": float(m["peak_force_smooth"][i]),
                        "predicted_success": int(fp >= 0),
                        "time_to_prediction_s": fp * step_dt if fp >= 0 else "",
                    }
                )
        level = level[: args_cli.episodes]
        rows += level

        n = len(level)
        succ = [e for e in level if e["success"]]
        pred = [e for e in level if e["predicted_success"]]
        times = sorted(e["time_to_success_s"] for e in succ)
        summary[str(err_mm)] = {
            "episodes": n,
            "success_rate": len(succ) / n,
            "median_time_to_success_s": times[len(times) // 2] if times else None,
            "mean_peak_force_smooth_N": sum(e["peak_force_smooth_N"] for e in level) / n,
            "pred_precision": sum(e["success"] for e in pred) / len(pred) if pred else None,
            "pred_recall": sum(e["predicted_success"] for e in succ) / len(succ) if succ else None,
        }
        print(f"[eval] error {err_mm:5.1f} mm: {summary[str(err_mm)]}")

    with open(args_cli.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.splitext(args_cli.out)[0] + ".summary.json"
    with open(summary_path, "w") as f:
        json.dump({"task": args_cli.task, "checkpoint": args_cli.checkpoint, "levels": summary}, f, indent=2)
    print(f"[eval] wrote {args_cli.out} and {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
