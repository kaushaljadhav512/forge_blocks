"""Evaluate one checkpoint at several slot-position error distances (the sim half of the grid).

    ./isaaclab.sh -p scripts/eval_sweep.py --task Isaac-ForgeBlocks-Block2mm-Direct-v0 \
        --checkpoint logs/rl_games/<run>/<ts>/nn/<run>.pth --errors_mm 0 5 10 20 30 \
        --episodes 1000 --headless --out results/block2mm_seed0.csv

Error = xy offset between the true slot and the estimate the policy is given, at exactly that
distance in a random direction (see forge_blocks/tasks/block_env.py). Per episode it logs:
success, time to first success, mean and peak contact force, the policy's success prediction, and per physics
step the peak fingertip speed (what deploy's 0.3 m/s safety limit checks) and peak controller pull
(stiffness x (target - fingertip), the spring force commanded; it exceeds the contact force by what friction holds).

For the paper's peg evaluation (3D error in its low/medium/high bands), use --paper_bins:

    ./isaaclab.sh -p scripts/eval_sweep.py --task Isaac-ForgeBlocks-Peg8mm-Direct-v0 --paper_bins \
        --checkpoint logs/rl_games/peg_forge/seed0/nn/<checkpoint>.pth --headless \
        --out results/peg_forge_seed0.csv
"""

import argparse
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--errors_mm", type=float, nargs="+", default=[0.0, 5.0, 10.0, 20.0, 30.0])
parser.add_argument("--error_mode", choices=["xy", "xyz"], default="xy", help="plane or sphere")
parser.add_argument("--paper_bins", action="store_true", help="xyz error in the paper's 0-1, 1-2.5, 2.5-5 mm bands")
parser.add_argument("--episodes", type=int, default=1000, help="per error level")
parser.add_argument("--num_envs", type=int, default=128)
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--force_threshold", type=float, default=7.5, help="N; the paper deploys at 7.5")
parser.add_argument("--pred_threshold", type=float, default=0.5, help="success-prediction cut-off")
# Stress tests for the gap to the real robot (defaults leave the sim unchanged).
parser.add_argument("--force_scale", type=float, default=1.0, help="multiply the observed force (-1: wrong sign)")
parser.add_argument("--force_bias_N", type=float, default=0.0, help="constant force offset, random direction per episode")
parser.add_argument("--action_delay", type=int, default=0, help="apply each action this many policy steps late")
parser.add_argument("--gain_scale", type=float, default=1.0, help="multiply the controller stiffness (before its noise)")
parser.add_argument("--force_deadband_N", type=float, default=0.0, help="observed force shrinks by this; smaller reads 0")
parser.add_argument("--force_deadband_mode", choices=["axis", "norm"], default="axis")
parser.add_argument("--friction_leak_N", type=float, nargs="+", default=[0.0],
                    help="fake force against the motion: one value (fixed) or LO HI (per episode and axis)")
parser.add_argument("--friction_leak_v_mm_s", type=float, default=2.0, help="speed where the leak is ~76%% of full size")
parser.add_argument("--pull_leak_alpha", type=float, nargs="+", default=[0.0],
                    help="fake force = -alpha x controller pull: one value (fixed) or LO HI (per episode and axis)")
parser.add_argument("--pull_leak_offset_N", type=float, nargs="+", default=[0.0],
                    help="constant force offset with the pull leak: one value or LO HI (per episode and axis, signed)")
parser.add_argument("--dead_zone_N", type=float, nargs=3, default=None, metavar=("X", "Y", "Z"),
                    help="FORGE's actuation dead-zone maximum per axis (each 2 s it draws U(0, max)); default 5 5 5")
parser.add_argument("--task_friction_N", type=float, nargs="+", default=[0.0],
                    help="fingertip Coulomb friction [x+ x- y+ y- z+ z-] in N: 6 values (fixed), or 12 "
                         "(6 LO then 6 HI, drawn per episode), or one value for all six")
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
    sum_smooth = torch.zeros(n, device=u.device)
    first_pred = torch.full((n,), -1, dtype=torch.long, device=u.device)
    u._eval_peak_speed = torch.zeros(n, device=u.device)  # filled per physics step by track_physics_step
    u._eval_peak_pull = torch.zeros(n, device=u.device)
    if player.is_rnn:
        player.init_rnn()
    # --action_delay: until the delayed actions arrive, send the reset actions, which hold the pose.
    queue = [u.actions.clone() for _ in range(args_cli.action_delay)]
    t = 0
    while True:
        with torch.inference_mode():
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            if queue:
                queue.append(actions)
                actions = queue.pop(0)
            obs, _, dones, _ = wrapped.step(actions)
        t += 1
        pred = (u.actions[:, 6] + 1) / 2 > args_cli.pred_threshold
        first_pred = torch.where((first_pred < 0) & pred, torch.full_like(first_pred, t), first_pred)
        if not dones.any():
            # Skipped on the last step: the env has already reset, so forces come from the new grasp.
            smooth = torch.linalg.norm(u.force_sensor_smooth[:, :3], dim=1)
            peak_raw = torch.maximum(peak_raw, torch.linalg.norm(u.force_sensor_world[:, :3], dim=1))
            peak_smooth = torch.maximum(peak_smooth, smooth)
            sum_smooth += smooth
        else:
            assert dones.all(), "Factory episodes should end together"
            # ep_succeeded/ep_success_times are only cleared on the next _pre_physics_step.
            out = {
                "success": u.ep_succeeded.clone().bool(),
                "success_step": u.ep_success_times.clone(),
                "peak_force_raw": peak_raw,
                "peak_force_smooth": peak_smooth,
                "mean_force_smooth": sum_smooth / max(t - 1, 1),
                "first_pred_step": first_pred,
                "peak_speed": u._eval_peak_speed.clone(),
                "peak_pull": u._eval_peak_pull.clone(),
            }
            return out, obs


def track_physics_step(u):
    """Wrap generate_ctrl_signals (called every physics step) to track peak fingertip speed and controller pull
    outside resets. Needs the ForgeBlocks env (controller_pull, _in_reset)."""
    orig = u.generate_ctrl_signals

    def hooked(*args, **kwargs):
        orig(*args, **kwargs)
        if getattr(u, "_eval_peak_speed", None) is None or u._in_reset:
            return
        speed = torch.linalg.norm(u.fingertip_midpoint_linvel, dim=1)
        pull = torch.linalg.norm(u.controller_pull(), dim=1)
        u._eval_peak_speed = torch.maximum(u._eval_peak_speed, speed)
        u._eval_peak_pull = torch.maximum(u._eval_peak_pull, pull)

    u.generate_ctrl_signals = hooked


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    env_cfg.seed = args_cli.seed
    if not hasattr(env_cfg.obs_rand, "error_radius_mode"):
        sys.exit(f"{args_cli.task} has no radius option; use an Isaac-ForgeBlocks-* task ID")
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")
    # Policy on the sim's GPU. The agent config says cuda:0, so with --device cuda:1 the two would differ.
    agent_cfg["params"]["config"]["device"] = agent_cfg["params"]["config"]["device_name"] = env_cfg.sim.device

    env_cfg.obs_rand.ft_force_scale = args_cli.force_scale
    env_cfg.obs_rand.ft_force_bias_N = args_cli.force_bias_N
    env_cfg.obs_rand.ft_force_deadband_N = args_cli.force_deadband_N
    env_cfg.obs_rand.ft_force_deadband_mode = args_cli.force_deadband_mode
    leak = args_cli.friction_leak_N
    if len(leak) not in (1, 2) or min(leak) < 0 or leak[0] > leak[-1]:
        sys.exit("--friction_leak_N takes one value or LO HI, both >= 0")
    env_cfg.obs_rand.ft_force_leak_N = [leak[0], leak[-1]]
    env_cfg.obs_rand.ft_force_leak_v_mps = args_cli.friction_leak_v_mm_s * 1e-3
    alpha, offset = args_cli.pull_leak_alpha, args_cli.pull_leak_offset_N
    if len(alpha) not in (1, 2) or min(alpha) < 0 or alpha[0] > alpha[-1]:
        sys.exit("--pull_leak_alpha takes one value or LO HI, both >= 0")
    if len(offset) not in (1, 2) or offset[0] > offset[-1]:
        sys.exit("--pull_leak_offset_N takes one value or LO HI")
    env_cfg.obs_rand.ft_pull_leak_alpha = [alpha[0], alpha[-1]]
    env_cfg.obs_rand.ft_pull_leak_offset_N = [offset[0], offset[-1]]
    if args_cli.dead_zone_N is not None:
        if min(args_cli.dead_zone_N) < 0:
            sys.exit("--dead_zone_N must be >= 0")
        env_cfg.ctrl.default_dead_zone = list(args_cli.dead_zone_N) + list(env_cfg.ctrl.default_dead_zone[3:])
    fr = args_cli.task_friction_N
    if len(fr) == 1:
        fr = fr * 6
    if len(fr) not in (6, 12) or min(fr) < 0:
        sys.exit("--task_friction_N takes 1, 6 or 12 values, all >= 0")
    env_cfg.ctrl.task_friction_lo_N, env_cfg.ctrl.task_friction_hi_N = list(fr[:6]), list(fr[-6:])

    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    u.default_gains *= args_cli.gain_scale  # ForgeEnv draws each episode's gains around these
    track_physics_step(u)
    wrapped, player = make_player(env, agent_cfg)
    step_dt = u.step_dt

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    rows, summary = [], {}
    obs = None
    if args_cli.paper_bins:
        mode, levels = "xyz", [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0)]
    else:
        mode, levels = args_cli.error_mode, [(e, e) for e in args_cli.errors_mm]
    u.cfg.obs_rand.error_radius_mode = mode
    u.cfg.task.contact_penalty_threshold_range = [args_cli.force_threshold, args_cli.force_threshold]
    for err_min, err_max in levels:
        u.cfg.obs_rand.error_radius = [err_min * 1e-3, err_max * 1e-3]
        label = f"{err_min:g}" if err_min == err_max else f"{err_min:g}-{err_max:g}"
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
                        "error_min_mm": err_min,
                        "error_max_mm": err_max,
                        "error_mode": mode,
                        "success": int(succ),
                        "time_to_success_s": float(m["success_step"][i]) * step_dt if succ else "",
                        "peak_force_raw_N": float(m["peak_force_raw"][i]),
                        "peak_force_smooth_N": float(m["peak_force_smooth"][i]),
                        "mean_force_smooth_N": float(m["mean_force_smooth"][i]),
                        "predicted_success": int(fp >= 0),
                        "time_to_prediction_s": fp * step_dt if fp >= 0 else "",
                        "peak_speed_m_s": float(m["peak_speed"][i]),
                        "peak_pull_N": float(m["peak_pull"][i]),
                    }
                )
        level = level[: args_cli.episodes]
        rows += level

        n = len(level)
        succ = [e for e in level if e["success"]]
        pred = [e for e in level if e["predicted_success"]]
        # Like ForgeEnv's logging: a prediction only counts if success had already happened.
        delays = [
            e["time_to_prediction_s"] - e["time_to_success_s"]
            for e in pred
            if e["success"] and e["time_to_prediction_s"] >= e["time_to_success_s"]
        ]
        times = sorted(e["time_to_success_s"] for e in succ)
        summary[label] = {
            "episodes": n,
            "success_rate": len(succ) / n,
            "mean_time_to_success_s": sum(times) / len(times) if times else None,
            "median_time_to_success_s": times[len(times) // 2] if times else None,
            "mean_force_smooth_N": sum(e["mean_force_smooth_N"] for e in level) / n,
            "mean_peak_force_raw_N": sum(e["peak_force_raw_N"] for e in level) / n,
            "mean_peak_force_smooth_N": sum(e["peak_force_smooth_N"] for e in level) / n,
            "mean_peak_speed_m_s": sum(e["peak_speed_m_s"] for e in level) / n,
            "max_peak_speed_m_s": max(e["peak_speed_m_s"] for e in level),
            "mean_peak_pull_N": sum(e["peak_pull_N"] for e in level) / n,
            "max_peak_pull_N": max(e["peak_pull_N"] for e in level),
            "pred_precision": len(delays) / len(pred) if pred else None,
            "pred_recall": len(delays) / len(succ) if succ else None,
            "mean_pred_delay_s": sum(delays) / len(delays) if delays else None,
        }
        print(f"[eval] {mode} error {label} mm: {summary[label]}")

    with open(args_cli.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summary_path = os.path.splitext(args_cli.out)[0] + ".summary.json"
    with open(summary_path, "w") as f:
        stress = {
            k: getattr(args_cli, k)
            for k in ("force_scale", "force_bias_N", "action_delay", "gain_scale", "force_deadband_N", "force_deadband_mode",
                      "friction_leak_N", "friction_leak_v_mm_s", "pull_leak_alpha", "pull_leak_offset_N", "task_friction_N")
        }
        stress["dead_zone_N"] = list(u.cfg.ctrl.default_dead_zone[:3])
        stress["force_threshold"] = args_cli.force_threshold
        json.dump(
            {
                "task": args_cli.task,
                "checkpoint": args_cli.checkpoint,
                "error_mode": mode,
                "stress": stress,
                "levels": summary,
            },
            f,
            indent=2,
        )
    print(f"[eval] wrote {args_cli.out} and {summary_path}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
