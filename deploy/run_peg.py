#!/usr/bin/env python3
"""Peg insertion on the real Franka. Order of use (deploy/README.md, Phases 3-5):

    run_peg.py check                                   read-only: state, frames, force, gripper
    run_peg.py rate-test --export EXPORT --ack-safety  holds pose; measures stream/state rates, policy time
    run_peg.py grasp --ack-safety                      close on the peg (finger ends on the 32 mm line)
    run_peg.py press-test --ack-safety                 peg pressed on a free surface -> force_sign
    run_peg.py register-hole --ack-safety              hand-guide into the hole 5x -> hole.json
    run_peg.py shadow --export EXPORT                  guide mode; policy runs but commands nothing
    run_peg.py run --export EXPORT --air --trials 1 --ack-safety          estimate 10 cm high: no contact
    run_peg.py run --export EXPORT --band 0 0 --trials 3 --ack-safety     real insertions, no error
    run_peg.py run --export EXPORT --band 0 1 --trials 20 --ack-safety    evaluation bands: 0-1, 1-2.5, 2.5-5
    run_peg.py summary                                 per-band stats, like results/*.summary.json

State that later commands need (force_sign, hole) is kept in --dir (default deploy/runs/).
Every moving command needs --ack-safety and an E-stop operator, as in Lab2.
"""

import argparse
import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
import franka_bridge as fb  # noqa: E402
from policy_runner import ForgePolicyRunner  # noqa: E402


def confirm(msg):
    if input(msg + " Type YES to continue: ").strip() != "YES":
        raise fb.HardwareError("operator cancelled")


def log_event(path, **fields):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a") as f:
        f.write(json.dumps({"utc": datetime.now(timezone.utc).isoformat(), **fields}) + "\n")


def load_calib(d):
    p = Path(d) / "calibration.json"
    return json.loads(p.read_text()) if p.exists() else {}


def save_calib(d, **kw):
    calib = {**load_calib(d), **kw}
    Path(d).mkdir(parents=True, exist_ok=True)
    (Path(d) / "calibration.json").write_text(json.dumps(calib, indent=2))
    return calib


def bridge_cfg(args, calib):
    cfg = fb.BridgeConfig(control_hz=args.control_hz)
    if "quat_ref" in calib:
        cfg.quat_ref = tuple(calib["quat_ref"])
    return cfg


def make_runner(args, calib, need_sign=True):
    if need_sign and "force_sign" not in calib:
        raise SystemExit("no force_sign yet: run press-test first")
    return ForgePolicyRunner.from_export(args.export, force_sign=calib.get("force_sign", 1.0), force_threshold=args.force_threshold)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dir", type=Path, default=Path(__file__).resolve().parent / "runs")
    p.add_argument("--control_hz", type=float, default=200.0)
    sub = p.add_subparsers(dest="cmd", required=True)

    def add(name, moves=False, export=False):
        a = sub.add_parser(name)
        if moves:
            a.add_argument("--ack-safety", action="store_true", help="E-stop operator present; motion allowed")
        if export:
            a.add_argument("--export", required=True, help="exports/<run> folder: policy.pt + runner_config.json")
            a.add_argument("--force_threshold", type=float, default=7.5)
        return a

    add("check")
    a = add("rate-test", moves=True)
    a.add_argument("--export", help="also time the policy")
    a.add_argument("--force_threshold", type=float, default=7.5)
    a.add_argument("--seconds", type=float, default=5.0)
    add("grasp", moves=True)
    add("open", moves=True)
    add("press-test", moves=True)
    a = add("register-hole", moves=True)
    a.add_argument("--n", type=int, default=5)
    a = add("shadow", export=True)
    a.add_argument("--seconds", type=float, default=30.0)
    a = add("run", moves=True, export=True)
    band = a.add_mutually_exclusive_group(required=True)
    band.add_argument("--band", type=float, nargs=2, metavar=("MIN_MM", "MAX_MM"), help="estimate error radius range")
    band.add_argument("--air", action="store_true", help="estimate 10 cm above the hole: a no-contact dry run")
    a.add_argument("--trials", type=int, default=1)
    a.add_argument("--seed", type=int, default=0)
    a.add_argument("--auto", action="store_true", help="don't ask before each trial (only after supervised runs)")
    add("summary")
    args = p.parse_args(argv)

    if args.cmd == "summary":
        return summary(args.dir)
    if getattr(args, "ack_safety", True) is False:
        p.error(f"{args.cmd} moves the robot: an E-stop operator must be present; pass --ack-safety")

    calib = load_calib(args.dir)
    cfg = bridge_cfg(args, calib)
    arm = fb.FrankaPyArm()

    if args.cmd == "check":
        st = arm.read()
        tip, quat, R = fb.tip_pose(st, cfg)
        yaw_err = np.degrees(np.arccos(np.clip((np.trace(fb.DOWN.T @ R) - 1) / 2, -1, 1)))
        print("joints (rad):", np.round(st.q, 4).tolist())
        print("FrankaPy tool pos (m):", np.round(st.tool_pos, 4).tolist())
        print("sim fingertip pos (m):", np.round(tip, 4).tolist(), " quat (w,x,y,z):", np.round(quat, 4).tolist())
        print(f"angle from 'gripper down, yaw 0': {yaw_err:.1f} deg")
        print("raw force O_F_ext_hat_K[:3] (N):", np.round(st.force, 2).tolist())
        print("Jacobian from FrankaPy:", "yes" if st.J is not None else "no (velocities from pose differences)")
        print("gripper width (m):", round(float(arm.fa.get_gripper_width()), 4))
        print("calibration:", calib or "none yet")
        return 0

    if args.cmd == "rate-test":
        runner = make_runner(args, calib, need_sign=False) if args.export else None
        out = fb.rate_test(arm, cfg, args.seconds, runner)
        print(json.dumps(out, indent=2))
        log_event(args.dir / "events.jsonl", event="rate_test", control_hz=args.control_hz, **out)
        return 0

    if args.cmd == "grasp":
        confirm("Peg between the fingers, finger ends on the 32 mm line?")
        print("grasp width (m):", arm.grasp())
        return 0

    if args.cmd == "open":
        confirm("Opening the gripper drops the peg. Hand ready?")
        print("width (m):", arm.open())
        return 0

    if args.cmd == "press-test":
        confirm("Peg grasped, 1-5 cm above a FREE flat surface (not the hole), nothing else below?")
        out = fb.press_test(arm, cfg)
        print(json.dumps(out, indent=2))
        save_calib(args.dir, force_sign=out["force_sign"], press_test=out)
        log_event(args.dir / "events.jsonl", event="press_test", **out)
        print(f"force_sign = {out['force_sign']:+.0f} saved")
        return 0

    if args.cmd == "register-hole":
        confirm("Hole base clamped? Peg grasped at the 32 mm line?")
        hole = fb.register_hole(arm, n=args.n, confirm=input)
        hole.save(args.dir / "hole.json")
        log_event(args.dir / "events.jsonl", event="register_hole", seated_tip=hole.seated_tip, spread_m=hole.spread_m)
        if hole.spread_m > 0.0005:
            print("WARNING: spread above 0.5 mm; check clamping and repeat")
        return 0

    hole = fb.Hole.load(args.dir / "hole.json")
    runner = make_runner(args, calib)

    if args.cmd == "shadow":
        return shadow(arm, runner, hole, cfg, args.seconds)

    if args.cmd == "run":
        return run_trials(arm, runner, hole, cfg, args)
    return 1


def shadow(arm, runner, hole, cfg, seconds):
    """Guide mode: move the arm by hand; the policy runs on real observations and prints its targets."""
    print("Guide mode: move the peg around above the hole. Nothing is commanded.")
    arm.guide(seconds, block=False)
    st = arm.read()
    tip, quat, _ = fb.tip_pose(st, cfg)
    runner.reset(tip, quat, hole.rim)
    ff = fb.ForceFilter(runner.cfg.ft_smoothing_factor, runner.cfg.sim_dt)
    tare, prev, t0 = st.force.copy(), None, time.monotonic()
    while time.monotonic() - t0 < seconds:
        st = arm.read()
        fs = ff.update(st.force - tare, (st.t - prev.t) if prev else runner.cfg.policy_dt)
        tip, quat, _ = fb.tip_pose(st, cfg)
        v, w = fb.tip_velocity(st, prev, cfg)
        target, _, p, _ = runner.step(tip, quat, v, w, fs)
        print(f"tip-rim {np.round((tip - hole.rim) * 1000, 1).tolist()} mm  target-tip "
              f"{np.round((target - tip) * 1000, 1).tolist()} mm  F {np.round(fs, 1).tolist()} N  p_succ {p:.2f}")
        prev = st
        time.sleep(runner.cfg.policy_dt)
    arm.stop()
    return 0


def run_trials(arm, runner, hole, cfg, args):
    rng = np.random.default_rng(args.seed)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    label = "air" if args.air else f"{args.band[0]:g}-{args.band[1]:g}"
    results_path = args.dir / "results.jsonl"
    for k in range(args.trials):
        if args.air:
            err = np.array([0.0, 0.0, 0.10])
        else:
            err = fb.sample_error(args.band[0] / 1000, args.band[1] / 1000, rng)
        hole_est = hole.rim + err
        start_pos, start_R = fb.sample_start(hole, rng, cfg)
        if args.air:
            start_pos = start_pos + err  # start above the (raised) estimate
        st = arm.read()
        lift = st.tool_pos.copy()
        lift[2] = max(lift[2], start_pos[2] + 0.03)
        print(f"trial {k + 1}/{args.trials} [{label}] error {np.round(err * 1000, 2).tolist()} mm")
        if not args.auto:
            confirm("Peg grasped and out of the hole, workspace clear, E-stop ready?")
        arm.goto(lift, st.tool_R, 3.0)
        arm.goto(start_pos, start_R, 3.0)
        steps = []
        try:
            res = fb.run_attempt(arm, runner, hole, hole_est, cfg, fb.SafetyLimits(), log_steps=steps)
        except BaseException:
            arm.stop()
            raise
        finally:
            name = f"{stamp}_{label}_{k + 1:03d}"
            (args.dir / "attempts").mkdir(parents=True, exist_ok=True)
            (args.dir / "attempts" / f"{name}.json").write_text(json.dumps({"steps": steps}))
        res.update(trial=k + 1, band=label, error_m=err.tolist(), export=args.export, seed=args.seed, attempt=name)
        log_event(results_path, **res)
        print(f"  -> {res['outcome']}, success={res['success']}, t_success={res['time_to_success_s']}, "
              f"peak F={res['peak_force_smooth_N']:.1f} N" + (f", {res['error']}" if res["error"] else ""))
        st = arm.read()
        arm.goto(st.tool_pos + np.array([0.0, 0.0, 0.06]), st.tool_R, 3.0)  # pull the peg out
        if res["outcome"] == "safety_stop" and not args.air:
            print("Stopping the series after a safety stop. Inspect before continuing.")
            return 1
    return 0


def summary(d):
    rows = [json.loads(line) for line in (Path(d) / "results.jsonl").read_text().splitlines()]
    out = {}
    for band in sorted({r["band"] for r in rows}):
        rs = [r for r in rows if r["band"] == band]
        succ = [r for r in rs if r["success"]]
        pred = [r for r in rs if r["time_to_prediction_s"] is not None]
        good = [r for r in pred if r["success"] and r["time_to_success_s"] is not None
                and r["time_to_prediction_s"] >= r["time_to_success_s"]]
        times = sorted(r["time_to_success_s"] for r in succ if r["time_to_success_s"] is not None)
        out[band] = {
            "episodes": len(rs),
            "success_rate": len(succ) / len(rs),
            "median_time_to_success_s": times[len(times) // 2] if times else None,
            "mean_peak_force_smooth_N": float(np.mean([r["peak_force_smooth_N"] for r in rs])),
            "pred_precision": len(good) / len(pred) if pred else None,
            "pred_recall": len(good) / len(succ) if succ else None,
            "safety_stops": sum(r["outcome"] == "safety_stop" for r in rs),
        }
    print(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
