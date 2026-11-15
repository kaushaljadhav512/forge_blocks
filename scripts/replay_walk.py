"""Replay a friction walk in sim, without a policy, and compare it with the robot.

    ./isaaclab.sh -p scripts/replay_walk.py --walk deploy/runs/friction_walk_s0_cold_<time>.json --headless
    ./isaaclab.sh -p scripts/replay_walk.py --seed 0 --seconds 30 --headless     # sim only, no robot data

The sim fingertip is driven along the walk's targets through ForgeEnv's own controller (dead zone and
all), on every physics step, the way deploy/franka_bridge.py drives the robot. Each env is one sim
setting, every combination of --leak_N (the friction leak, the same on all axes; the walk's own fit is
added as a setting when the walk has one) and --dead_zone_N (FORGE's actuation dead zone, all three
axes; it trains on U(0, 5) per axis). Gains are at their defaults and the force noise is off, so the
settings differ only in friction.

Targets map to the sim relative to the hole rim: sim rim + (target - robot rim). The robot rim comes
from --hole (deploy/runs/hole.json), else the nominal (0.59, 0, 0.028) from deploy/README.md. Before
the walk, the sim hand moves from its reset pose to the walk's start and settles (--settle_s), as
friction_walk does with its goto and tare.

Prints, per setting: how far the sim fingertip is from the robot's (tracking: does the sim reproduce
the hand lagging and sticking?) and the force error against the robot's estimate, smoothed like the
policy's (does the sim reproduce what the policy would see?). The force error is given both raw and
with the robot's pose-dependent error removed, which the sim doesn't model. Writes <out>.json and
<out>.npz (all traces).
"""

import argparse
import json
import math
from dataclasses import asdict
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--walk", help="friction_walk_*.json from the robot")
parser.add_argument("--seed", type=int, default=0, help="without --walk: the WalkSpec seed")
parser.add_argument("--seconds", type=float, default=180.0, help="without --walk: walk length")
parser.add_argument("--task", default="Isaac-ForgeBlocks-Peg8mm-Direct-v0")
parser.add_argument("--leak_N", type=float, nargs="+", default=[0.0, 1.0, 2.0, 3.0])
parser.add_argument("--leak_v_mm_s", type=float, default=2.0, help="the leak's speed scale (training used 2)")
parser.add_argument("--dead_zone_N", type=float, nargs="+", default=[0.0, 2.5, 5.0])
parser.add_argument("--hole", help="hole.json from register-hole")
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--out", help="output prefix (default results/replay_<walk or seed>)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deploy"))
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import forge_blocks.tasks  # noqa: E402,F401
import franka_bridge as fb  # noqa: E402
from walk_compare import compare, robot_traces  # noqa: E402

NOMINAL_ROBOT_RIM = (0.59, 0.0, 0.028)  # deploy/README.md: hole at Lab2's cube spot, boss top ~0.028
DOWN_QUAT = (0.0, 1.0, 0.0, 0.0)  # w, x, y, z: roll pi, yaw 0 = fb.DOWN (test_down_with_yaw_matches_isaac_euler)


def load_walk():
    if args_cli.walk:
        walk = json.load(open(args_cli.walk))
        spec = fb.WalkSpec(**{k: tuple(v) if isinstance(v, list) else v for k, v in walk["spec"].items()})
        return walk, spec
    return None, fb.WalkSpec(seed=args_cli.seed, duration_s=args_cli.seconds)


def settings(walk):
    leaks = [(f"leak {v:g} N", [v, v, v]) for v in args_cli.leak_N]
    if walk and walk.get("fit"):
        ax = walk["fit"]["axes"]
        fitted = [max(0.0, (ax[a]["leak_plus_N"] + ax[a]["leak_minus_N"]) / 2) for a in "xyz"]
        leaks.append(("leak = walk fit " + "/".join(f"{v:.2f}" for v in fitted), fitted))
    return [(f"{name}, dead zone {dz:g} N", leak, dz) for name, leak in leaks for dz in args_cli.dead_zone_N]


def main():
    walk, spec = load_walk()
    traj = fb.walk_trajectory(spec)
    robot_rim = np.array(json.load(open(args_cli.hole))["rim"] if args_cli.hole else NOMINAL_ROBOT_RIM)
    cases = settings(walk)
    n = len(cases)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=n)
    env_cfg.episode_length_s = args_cli.settle_s + float(traj["t"][-1]) + 2.0  # no reset during the walk
    env_cfg.obs_rand.ft_force = 0.0  # no force noise: settings differ only in friction
    env_cfg.obs_rand.ft_force_leak_N = [0.0, 1.0]  # allocates the leak buffers; set per env below
    env_cfg.obs_rand.ft_force_leak_v_mps = args_cli.leak_v_mm_s / 1000.0
    # FORGE redraws the dead zone every 2 s (forge_env_cfg.py: EventTerm(randomize_dead_zone, mode="interval")),
    # which would overwrite the per-env settings below.
    env_cfg.events.dead_zone_thresholds = None
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device
    env.reset()

    # One setting per env, after the reset's randomisation.
    u._leak_N[:] = torch.tensor([c[1] for c in cases], dtype=torch.float32, device=dev)
    u._leak_smooth[:] = 0.0
    u.dead_zone_thresholds[:] = 0.0
    u.dead_zone_thresholds[:, :3] = torch.tensor([[c[2]] * 3 for c in cases], dtype=torch.float32, device=dev)
    u.task_prop_gains = u.default_gains.clone()
    u.task_deriv_gains = factory_utils.get_deriv_gains(u.task_prop_gains)

    rim = u.fixed_pos_obs_frame.clone()  # true rim, env frame
    targets = torch.tensor(traj["target"] - robot_rim, dtype=torch.float32, device=dev)  # (T, 3), rim-relative
    quat = torch.tensor(DOWN_QUAT, dtype=torch.float32, device=dev).repeat(n, 1)
    dt_phys = u.physics_dt
    n_settle = int(round(args_cli.settle_s / dt_phys))
    n_walk = int(math.floor(float(traj["t"][-1]) / dt_phys))
    rec = torch.zeros((n_walk, n, 12), device=dev)  # target, tip, observed force, true force (rim-relative)
    state = {"k": 0, "start": None}
    original_apply, original_ctrl = u._apply_action, u.generate_ctrl_signals

    def walk_target():
        k = state["k"]
        if state["start"] is None:
            state["start"] = u.fingertip_midpoint_pos.clone()
        if k < n_settle:  # reset pose -> walk start in the first 2/3, then hold
            s = min(1.0, k / (0.67 * n_settle))
            s = s * s * (3 - 2 * s)
            return state["start"] + s * (rim + targets[0] - state["start"])
        tw = (k - n_settle) * dt_phys
        i = min(int(tw / spec.dt), len(targets) - 1)
        j = min(i + 1, len(targets) - 1)
        a = min(max(tw / spec.dt - i, 0.0), 1.0)
        target = rim + targets[i] + a * (targets[j] - targets[i])
        w = k - n_settle
        if w < n_walk:
            rec[w, :, 0:3] = target - rim
            rec[w, :, 3:6] = u.fingertip_midpoint_pos - rim
            rec[w, :, 6:9] = u.noisy_force
            rec[w, :, 9:12] = u.force_sensor_smooth[:, 0:3]
        return target

    def ctrl(ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, **kw):
        # Called from ForgeEnv's _apply_action with the (zero-action) policy target: send the walk's instead.
        original_ctrl(ctrl_target_fingertip_midpoint_pos=walk_target(), ctrl_target_fingertip_midpoint_quat=quat, **kw)
        state["k"] += 1

    def apply_action():
        # ForgeEnv's own _apply_action runs (it updates the state and sets what the reward reads); only the
        # controller target is replaced. Reset code calls generate_ctrl_signals too, so swap it only here.
        u.generate_ctrl_signals = ctrl
        try:
            original_apply()
        finally:
            u.generate_ctrl_signals = original_ctrl

    u._apply_action = apply_action
    action = torch.zeros((n,) + env.action_space.shape[1:], device=dev)
    steps = math.ceil((n_settle + n_walk) / u.cfg.decimation) + 1
    print(f"[replay] {n} settings, {traj['t'][-1]:.0f} s walk, {steps} env steps")
    for step in range(steps):
        env.step(action)
        if u.reset_buf.any():
            raise RuntimeError("the env reset during the replay; episode_length_s too short?")
        if step % 500 == 0:
            print(f"[replay] step {step}/{steps}")

    R = rec.cpu().numpy().astype(np.float64)
    t = np.arange(n_walk) * dt_phys
    sim = {"t": t, "target": R[:, :, 0:3], "tip": R[:, :, 3:6], "force": R[:, :, 6:9]}
    name = os.path.splitext(os.path.basename(args_cli.walk))[0] if args_cli.walk else f"walk_s{spec.seed}"
    out = args_cli.out or os.path.join(ROOT, "results", f"replay_{name}")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    arrays = {"t": t, "sim_target": sim["target"], "sim_tip": sim["tip"], "sim_force": sim["force"],
              "sim_true_force": R[:, :, 9:12]}
    summary = {"walk": args_cli.walk, "spec": asdict(spec),
               "robot_rim": robot_rim.tolist(), "leak_v_mm_s": args_cli.leak_v_mm_s,
               "settings": [{"env": e, "name": c[0], "leak_N": c[1], "dead_zone_N": c[2]} for e, c in enumerate(cases)],
               "sim_lag_mm": [float(1000 * np.linalg.norm(sim["tip"][:, e] - sim["target"][:, e], axis=1).mean())
                              for e in range(n)]}

    if walk:
        robot = robot_traces(walk, robot_rim)
        rows = compare(robot, sim)
        for row, c in zip(rows, cases):
            row["name"] = c[0]
        summary["compare"] = rows
        for k in ("t", "tip", "target", "force"):
            arrays["robot_" + k] = robot[k]
        if "force_no_pose" in robot:
            arrays["robot_force_no_pose"] = robot["force_no_pose"]
        print(f"robot: mean lag behind the target {rows[0]['lag_robot_mm']:.2f} mm")
        print(f"{'setting':42s} {'lag (mm)':>8s} {'tip vs robot, x/y/z (mm)':>26s} {'force err x/y/z (N)':>22s} "
              f"{'w/o pose err (N)':>20s}")
        for r in rows:
            fmt = lambda v: "/".join(f"{x:.2f}" for x in v)  # noqa: E731
            print(f"{r['name']:42s} {r['lag_sim_mm']:8.2f} {fmt(r['tip_rms_mm']):>26s} {fmt(r['force_rms_N']):>22s} "
                  f"{fmt(r.get('force_rms_no_pose_N', [float('nan')] * 3)):>20s}")
        worst = max(r["target_mismatch_mm"] for r in rows)
        if worst > 0.5:
            print(f"WARNING: sim and robot targets differ by up to {worst:.1f} mm: wrong walk spec or time base")
    else:
        for e, c in enumerate(cases):
            print(f"{c[0]:42s} lag {summary['sim_lag_mm'][e]:.2f} mm, |force| mean "
                  f"{np.linalg.norm(sim['force'][:, e], axis=1).mean():.2f} N")

    np.savez_compressed(out + ".npz", **arrays)
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[replay] wrote {out}.json and {out}.npz")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
