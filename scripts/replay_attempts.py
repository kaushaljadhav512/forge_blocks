"""Re-run real policy attempts in sim from the same start poses, and compare.

    ./isaaclab.sh -p scripts/replay_attempts.py --export exports/peg_fric_curr_seed2 \
        --hole lab_runs/s3_runs/hole.json --attempts lab_runs/s3_runs/attempts/*_0-0_*.json --headless

Each real attempt (deploy/run_peg.py run, step log in deploy/runs/attempts/) starts from its first logged
fingertip pose relative to the registered rim, with its yaw. In sim the hand moves there from its reset pose
and settles (--settle_s), then the policy runs for the attempt's full 150 steps, driven through
deploy/policy_runner.py like the robot bridge: one runner per env, a policy step every 8 physics steps, and
runner.target() re-clipped around the fingertip on every physics step. The hole estimate is the true rim
(the real runs used --band 0 0).

Two controller variants, --copies envs each per attempt (they differ in the per-episode force-leak draws):
  sim:    as trained: rotation gains 28 Nm/rad, ForgeEnv's orientation rule (current tilt + at most 0.097 rad)
  deploy: as deployed since session 3: rotation 80 Nm/rad, orientation target held vertical
Friction is the `robot` eval setting (walk-fit task friction, pull leak alpha U(0.2, 0.45), offset U(-1.5, 0),
dead zone 0); gains are at their defaults (no per-episode gain noise).

Per env it logs, per policy step: fingertip and target relative to the rim, the force the policy sees, the true
contact force, tilt from vertical and the success prediction; and when the peg first seats (bridge's check).
The summary compares, per attempt: seated or not and when, the sideways target offset near the hole (the real
runs pushed ~13 mm in -x there), peak forces and tilt. Writes <out>.json and <out>.npz.
"""

import argparse
import glob
import json
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--export", required=True, help="exports/<run>: policy.pt + runner_config.json")
parser.add_argument("--hole", required=True, help="hole.json the real attempts were registered with")
parser.add_argument("--attempts", nargs="+", required=True, help="real attempt step logs (globs allowed)")
parser.add_argument("--task", default="Isaac-ForgeBlocks-Peg8mm-Direct-v0")
parser.add_argument("--copies", type=int, default=8, help="envs per attempt and variant")
parser.add_argument("--settle_s", type=float, default=3.0)
parser.add_argument("--steps", type=int, default=150, help="policy steps (the real timeout is 150)")
parser.add_argument("--out", default=None, help="output prefix (default results/replay_attempts)")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import copy  # noqa: E402

import gymnasium as gym  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.direct.factory import factory_utils  # noqa: E402
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

ROOT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "deploy"))
import forge_blocks.tasks  # noqa: E402,F401
import franka_bridge as fb  # noqa: E402
from policy_runner import ForgePolicyRunner  # noqa: E402

ROBOT_FRICTION_N = [4.6, 1.9, 1.5, 2.0, 2.1, 1.0]  # the `robot` eval: session-2 walk fit
VARIANTS = {"sim": (28.0, False), "deploy": (80.0, True)}  # rotation gain, hold_vertical
NEAR_HOLE_TIP_ABOVE_RIM = fb.GRASP_TIP_ABOVE_PEG_BOTTOM + 0.001  # peg bottom within 1 mm of the top face, or in


def real_attempt(path, rim):
    steps = json.load(open(path))["steps"]
    tip = np.array([s["tip"] for s in steps]) - rim
    target = np.array([s["target"] for s in steps]) - rim
    q = np.array([s["obs"][3:7] for s in steps])  # (0, qx, qy, 0) for a hand pointing down with yaw
    yaw = 2.0 * math.atan2(q[0, 2], q[0, 1])
    if "tilt_deg" in steps[0]:
        tilt = np.array([s["tilt_deg"] for s in steps])
    else:  # older logs: from the zeroed quat, |(qx, qy)| = cos(tilt / 2)
        tilt = np.degrees(2 * np.arccos(np.clip(np.linalg.norm(q[:, 1:3], axis=1), 0, 1)))
    seated = np.array([np.linalg.norm(t[:2]) < fb.SUCCESS_XY and t[2] < fb.SEATED_TIP_ABOVE_RIM + fb.SUCCESS_Z
                       for t in tip])
    return {"name": os.path.splitext(os.path.basename(path))[0], "start": tip[0], "yaw": yaw, "tip": tip,
            "target": target, "force": np.array([s["force_smooth"] for s in steps]), "tilt": tilt,
            "p": np.array([s["success_prob"] for s in steps]), "t": np.array([s["t"] for s in steps]),
            "seated_step": int(np.argmax(seated)) if seated.any() else None}


def near_hole_push(tip, target):
    """Sideways target offset while the peg bottom is at the top face or in the hole: mean x, min x, max |xy| (mm)."""
    m = tip[:, 2] < NEAR_HOLE_TIP_ABOVE_RIM
    if not m.any():
        return None
    d = (target - tip)[m] * 1000
    return {"steps": int(m.sum()), "mean_dx_mm": float(d[:, 0].mean()), "min_dx_mm": float(d[:, 0].min()),
            "max_dxy_mm": float(np.linalg.norm(d[:, :2], axis=1).max())}


def main():
    paths = sorted({p for pat in args_cli.attempts for p in glob.glob(pat)})
    if not paths:
        sys.exit("no attempt files matched")
    robot_rim = np.array(json.load(open(args_cli.hole))["rim"])
    real = [real_attempt(p, robot_rim) for p in paths]
    cases = [(a, v) for a in range(len(real)) for v in VARIANTS for _ in range(args_cli.copies)]
    n = len(cases)

    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=n)
    template = ForgePolicyRunner.from_export(args_cli.export, device=env_cfg.sim.device)
    policy_dt = template.cfg.policy_dt
    env_cfg.episode_length_s = args_cli.settle_s + args_cli.steps * policy_dt + 2.0  # no reset during a replay
    env_cfg.obs_rand.ft_pull_leak_alpha = [0.2, 0.45]
    env_cfg.obs_rand.ft_pull_leak_offset_N = [-1.5, 0.0]
    env_cfg.ctrl.task_friction_lo_N = list(ROBOT_FRICTION_N)
    env_cfg.ctrl.task_friction_hi_N = list(ROBOT_FRICTION_N)
    env_cfg.ctrl.default_dead_zone = [0.0, 0.0, 0.0] + list(env_cfg.ctrl.default_dead_zone[3:])
    env_cfg.events.dead_zone_thresholds = None  # FORGE redraws it every 2 s otherwise
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    dev = u.device
    env.reset()

    u.dead_zone_thresholds[:] = 0.0
    gains = u.default_gains.clone()
    for e, (_, v) in enumerate(cases):
        gains[e, 3:6] = VARIANTS[v][0]
    u.task_prop_gains = gains
    u.task_deriv_gains = factory_utils.get_deriv_gains(gains)

    runners = []
    for _, v in cases:
        cfg = copy.deepcopy(template.cfg)
        cfg.hold_vertical = VARIANTS[v][1]
        runners.append(ForgePolicyRunner(cfg, template.policy))

    rim = u.fixed_pos_obs_frame.clone()  # true sim rim, env frame (the estimate the policy gets, too)
    start = torch.tensor(np.array([real[a]["start"] for a, _ in cases]), dtype=torch.float32, device=dev) + rim
    yaws = torch.tensor([real[a]["yaw"] for a, _ in cases], dtype=torch.float32, device=dev)
    start_quat = torch.stack([torch.zeros_like(yaws), torch.cos(yaws / 2), torch.sin(yaws / 2), torch.zeros_like(yaws)], 1)
    seated_offset = torch.tensor([0.0, 0.0, fb.SEATED_TIP_ABOVE_RIM], device=dev)

    dt_phys = u.physics_dt
    n_settle = int(round(args_cli.settle_s / dt_phys))
    T = args_cli.steps
    rec = {k: np.zeros((T, n, 3)) for k in ("tip", "target", "force", "true_force")}
    rec["tilt"], rec["p"] = np.zeros((T, n)), np.zeros((T, n))
    seated_t = torch.full((n,), float("nan"), device=dev)
    state = {"k": 0, "from": None, "policy": False, "t0": None}
    original_apply, original_ctrl = u._apply_action, u.generate_ctrl_signals

    def np_(t, i):
        return t[i].detach().cpu().numpy().astype(np.float64)

    def target():
        k = state["k"]
        if not state["policy"]:  # reset pose -> attempt start in the first 2/3 of the settle, then hold
            if state["from"] is None:
                state["from"] = u.fingertip_midpoint_pos.clone()
            s = min(1.0, k / (0.67 * n_settle))
            s = s * s * (3 - 2 * s)
            return state["from"] + s * (start - state["from"]), start_quat
        pos, quat = [], []
        tips_np = u.fingertip_midpoint_pos.detach().cpu().numpy().astype(np.float64)
        quats_np = u.fingertip_midpoint_quat.detach().cpu().numpy().astype(np.float64)
        for i, r in enumerate(runners):
            p_, q_ = r.target(tips_np[i], quats_np[i])
            pos.append(p_), quat.append(q_)
        tip = u.fingertip_midpoint_pos - rim
        seated = (torch.linalg.norm(tip[:, :2], dim=1) < fb.SUCCESS_XY) & (tip[:, 2] < seated_offset[2] + fb.SUCCESS_Z)
        new = seated & torch.isnan(seated_t)
        seated_t[new] = (k - state["t0"]) * dt_phys
        return (torch.tensor(np.array(pos), dtype=torch.float32, device=dev),
                torch.tensor(np.array(quat), dtype=torch.float32, device=dev))

    def ctrl(ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, **kw):
        pos, quat = target()
        original_ctrl(ctrl_target_fingertip_midpoint_pos=pos, ctrl_target_fingertip_midpoint_quat=quat, **kw)
        state["k"] += 1

    def apply_action():
        u.generate_ctrl_signals = ctrl
        try:
            original_apply()
        finally:
            u.generate_ctrl_signals = original_ctrl

    u._apply_action = apply_action
    action = torch.zeros((n,) + env.action_space.shape[1:], device=dev)
    settle_steps = math.ceil(n_settle / u.cfg.decimation)
    print(f"[replay] {len(real)} attempts x {len(VARIANTS)} variants x {args_cli.copies} copies = {n} envs; "
          f"settle {settle_steps} env steps, then {T} policy steps")
    for _ in range(settle_steps):
        env.step(action)
    state["policy"], state["t0"] = True, state["k"]
    for i, r in enumerate(runners):
        r.reset(np_(u.fingertip_midpoint_pos, i), np_(u.fingertip_midpoint_quat, i), np_(rim, i))
    print("[replay] start error vs the real starts (mm): max "
          f"{1000 * float(torch.linalg.norm(u.fingertip_midpoint_pos - start, dim=1).max()):.2f}")

    for step in range(T):
        tip_t, quat_t = u.fingertip_midpoint_pos, u.fingertip_midpoint_quat
        w = quat_t[:, 0:4]
        r22 = 1.0 - 2.0 * (w[:, 1] ** 2 + w[:, 2] ** 2)  # fingertip z axis . world z
        tilt = torch.rad2deg(torch.arccos(torch.clamp(-r22, -1.0, 1.0)))
        cpu = {k: t.detach().cpu().numpy().astype(np.float64) for k, t in
               (("tip", tip_t), ("quat", quat_t), ("v", u.ee_linvel_fd), ("w", u.ee_angvel_fd), ("f", u.noisy_force),
                ("rim", rim))}
        for i, r in enumerate(runners):
            tgt, _, p, _ = r.step(cpu["tip"][i], cpu["quat"][i], cpu["v"][i], cpu["w"][i], cpu["f"][i])
            rec["target"][step, i] = tgt - cpu["rim"][i]
            rec["p"][step, i] = p
        rec["tip"][step] = (tip_t - rim).cpu().numpy()
        rec["force"][step] = u.noisy_force.cpu().numpy()
        rec["true_force"][step] = u.force_sensor_smooth[:, 0:3].cpu().numpy()
        rec["tilt"][step] = tilt.cpu().numpy()
        env.step(action)
        if u.reset_buf.any():
            raise RuntimeError("the env reset during the replay; episode_length_s too short?")
        if step % 30 == 0:
            print(f"[replay] policy step {step}/{T}")

    seated_s = seated_t.cpu().numpy()
    rows = []
    for e, (a, v) in enumerate(cases):
        tip, tgt = rec["tip"][:, e], rec["target"][:, e]
        pred = np.nonzero(rec["p"][:, e] > 0.5)[0]
        rows.append({"attempt": real[a]["name"], "variant": v,
                     "seated_s": None if math.isnan(seated_s[e]) else float(seated_s[e]),
                     "pred_s": float(pred[0] * policy_dt) if len(pred) else None,
                     "peak_true_force_N": float(np.linalg.norm(rec["true_force"][:, e], axis=1).max()),
                     "peak_obs_force_N": float(np.linalg.norm(rec["force"][:, e], axis=1).max()),
                     "max_tilt_deg": float(rec["tilt"][:, e].max()),
                     "near_hole": near_hole_push(tip, tgt)})

    summary = {"export": args_cli.export, "hole": args_cli.hole, "robot_rim": robot_rim.tolist(), "copies": args_cli.copies,
               "variants": {k: {"rot_gain": g, "hold_vertical": h} for k, (g, h) in VARIANTS.items()},
               "friction_N": ROBOT_FRICTION_N, "rows": rows, "real": []}
    print(f"\n{'attempt':28s} {'who':10s} {'seated':>9s} {'pred':>6s} {'peakF':>6s} {'tilt':>5s} "
          f"{'near-hole dx mean/min (mm)':>27s}")
    for a, r in enumerate(real):
        rp = near_hole_push(r["tip"], r["target"])
        rs = None if r["seated_step"] is None else float(r["t"][r["seated_step"]])
        rpred = np.nonzero(r["p"] > 0.5)[0]
        summary["real"].append({"attempt": r["name"], "seated_s": rs, "peak_obs_force_N":
                                float(np.linalg.norm(r["force"], axis=1).max()), "max_tilt_deg": float(r["tilt"].max()),
                                "near_hole": rp, "start_mm": (1000 * r["start"]).tolist(), "yaw_deg": math.degrees(r["yaw"])})
        fmt_push = lambda p: f"{p['mean_dx_mm']:7.1f} / {p['min_dx_mm']:6.1f}" if p else "      (never)"  # noqa: E731
        print(f"{r['name']:28s} {'REAL':10s} {('%.2f s' % rs) if rs is not None else 'no':>9s} "
              f"{('%.2f' % r['t'][rpred[0]]) if len(rpred) else '-':>6s} "
              f"{summary['real'][-1]['peak_obs_force_N']:6.1f} {r['tilt'].max():5.1f} {fmt_push(rp):>27s}")
        for v in VARIANTS:
            rs_ = [row for row in rows if row["attempt"] == r["name"] and row["variant"] == v]
            ok = [x["seated_s"] for x in rs_ if x["seated_s"] is not None]
            pushes = [x["near_hole"] for x in rs_ if x["near_hole"]]
            mean_dx = np.mean([p["mean_dx_mm"] for p in pushes]) if pushes else float("nan")
            min_dx = np.mean([p["min_dx_mm"] for p in pushes]) if pushes else float("nan")
            print(f"{'':28s} {'sim/' + v:10s} {len(ok):>3d}/{len(rs_):<2d}{(' %.1fs' % np.median(ok)) if ok else '':>4s} "
                  f"{'':>6s} {np.mean([x['peak_true_force_N'] for x in rs_]):6.1f} "
                  f"{np.mean([x['max_tilt_deg'] for x in rs_]):5.1f} {mean_dx:7.1f} / {min_dx:6.1f}")
    print("(sim: seated n/copies and median time; peakF = mean of the true contact force peaks; real peakF = the "
          "force the policy saw)")

    out = args_cli.out or os.path.join(ROOT, "results", "replay_attempts")
    os.makedirs(os.path.dirname(out), exist_ok=True)
    np.savez_compressed(out + ".npz", **{k: v for k, v in rec.items()}, seated_s=seated_s,
                        case_attempt=np.array([a for a, _ in cases]), case_variant=np.array([v for _, v in cases]))
    with open(out + ".json", "w") as f:
        json.dump(summary, f, indent=2)
    print(f"[replay] wrote {out}.json and {out}.npz")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
