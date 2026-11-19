#!/usr/bin/env python3
"""Peg insertion on the real Franka. Order of use (deploy/README.md, Phases 3-5):

    run_peg.py check                                   read-only: state, frames, force, gripper
    run_peg.py home --ack-safety                       FrankaPy's reset_joints (gripper down, above the table)
    run_peg.py guide --seconds 20 --ack-safety         guide mode: move the arm by hand
    run_peg.py rate-test --export EXPORT --ack-safety  holds pose, then 2 mm up step: rates, latency, policy time
    run_peg.py grasp --ack-safety                      close on the peg (finger ends on the 32 mm line)
    run_peg.py press-test --ack-safety                 peg pressed on a free surface -> force_sign
    run_peg.py force-curve --ack-safety                force estimate vs commanded push, 0-15 N
    run_peg.py force-scale --ack-safety                press on a kitchen scale: commanded vs true vs estimate
    run_peg.py force-weights --ack-safety              hang known masses in free space: true vs estimate, sign
    run_peg.py force-drift --ack-safety                free space: drift over 15 s and error over a 4 cm move
    run_peg.py friction-breakaway --ack-safety         free space: pull needed to start moving, 6 directions
    run_peg.py friction-sweep --axis x --ack-safety    free space: friction and fake force vs speed, 3 cm and back
    run_peg.py friction-steps --ack-safety             free space: how far policy-sized target steps get
    run_peg.py friction-walk --seed 0 --ack-safety     free space: seeded random walk over the hole area, 3 min
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


def confirm(msg, ask=input):
    """Ask a yes/no question, no by default: y/yes continues; Enter, n/no or Ctrl-D cancels; anything
    else asks again."""
    while True:
        try:
            answer = ask(msg + " [y/N] ").strip().lower()
        except EOFError:
            answer = "n"
        if answer in ("y", "yes"):
            return
        if answer in ("", "n", "no"):
            raise fb.HardwareError("operator cancelled")
        print("Please answer y or n.")


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


def print_fit(label, fit):
    if fit["deadband_N"] is None:
        print(f"{label}: not enough points above 0.5 N to fit")
    else:
        print(f"{label}: responds from ~{fit['deadband_N']:.2f} N, slope {fit['slope']:.2f} ({fit['points']} points)")


def save_check(d, name, out):
    """Raw results to <dir>/<name>_<time>.json; the last summary of each check into calibration.json."""
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = Path(d) / f"{name}_{stamp}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    # time series stay in the raw file only
    summary = {k: v for k, v in out.items() if k not in ("rows", "joints") and not k.endswith("series")}
    save_calib(d, **{name: summary})
    log_event(Path(d) / "events.jsonl", event=name, file=path.name, **summary)
    print("saved", path)


def friction_force_sign(calib):
    """Sign for the friction checks' force readings, so they print the force on the hand as the policy sees
    it. The cold walk (runbook 2.1a) runs before 2.2's press-test re-check, so without a calibration this is
    session 1's -1. The raw estimate is saved either way."""
    return float(calib.get("force_sign", -1.0))


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
    add("home", moves=True)
    a = add("guide", moves=True)
    a.add_argument("--seconds", type=float, default=20.0)
    a = add("rate-test", moves=True)
    a.add_argument("--export", help="also time the policy")
    a.add_argument("--force_threshold", type=float, default=7.5)
    a.add_argument("--seconds", type=float, default=5.0)
    a = add("grasp", moves=True)
    a.add_argument("--force", type=float, default=fb.GRASP_FORCE_N, help=f"N, at most {fb.GRASP_FORCE_MAX_N:.0f}")
    add("open", moves=True)
    add("press-test", moves=True)
    add("force-curve", moves=True)
    a = add("force-scale", moves=True)
    a.add_argument("--levels", type=float, nargs="+", default=[1, 2, 3, 4, 5, 6, 8, 10], help="commanded pushes (N)")
    a = add("force-weights", moves=True)
    a.add_argument("--direction", type=float, nargs=3, default=[0.0, 0.0, -1.0],
                   help="way the string pulls the hand, base frame (0 0 -1: hanging; over a pulley: 1 0 0 etc.)")
    a = add("force-drift", moves=True)
    a.add_argument("--seconds", type=float, default=15.0)
    a.add_argument("--sweep_m", type=float, default=0.04)
    a = add("friction-breakaway", moves=True)
    a.add_argument("--max_N", type=float, default=8.0, help="give up at this spring pull")
    a = add("friction-sweep", moves=True)
    a.add_argument("--axis", choices=["x", "y", "z"], required=True)
    a.add_argument("--speeds", type=float, nargs="+", default=[2.0, 5.0, 10.0, 20.0], help="mm/s")
    a.add_argument("--sweep_mm", type=float, default=30.0)
    a = add("friction-steps", moves=True)
    a.add_argument("--steps_mm", type=float, nargs="+", default=[0.5, 1.0, 2.0, 4.0, 8.0])
    a.add_argument("--axes", default="xyz")
    a.add_argument("--stiffness_scale", type=float, default=1.0,
                   help="all six gains x this (0.7-1.3, the range the sim stress evals covered)")
    a = add("friction-walk", moves=True)
    a.add_argument("--seed", type=int, default=0, help="same seed, same walk: repeat it cold and warm, replay it in sim")
    a.add_argument("--seconds", type=float, default=180.0)
    a.add_argument("--max_speed", type=float, default=20.0, help="mm/s")
    a.add_argument("--centre", type=float, nargs=3, help="fingertip, base frame (default: above hole.json's rim, "
                   f"else {fb.WALK_CENTRE})")
    a.add_argument("--box", type=float, nargs=6, metavar=("XLO", "YLO", "ZLO", "XHI", "YHI", "ZHI"),
                   help=f"fingertip bounds, base frame (default {fb.WALK_BOX_LO} {fb.WALK_BOX_HI})")
    a.add_argument("--label", default="", help="e.g. cold, warm: added to the file name")
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
        cmp = arm.compare_with_service()  # arm still: should match to << 1 mm, << 0.1 deg
        print(f"state source: {cmp['source']}, age {cmp['state_age_ms']:.0f} ms; vs FrankaPy get_pose/get_joints: "
              f"pos {cmp['pos_diff_mm']:.3f} mm, rot {cmp['rot_diff_deg']:.3f} deg, joints {cmp['joint_diff_max_rad']:.5f} rad")
        return 0

    if args.cmd == "home":
        confirm("Arm moves to FrankaPy's home pose. Path clear?")
        arm.fa.reset_joints()
        return 0

    if args.cmd == "guide":
        confirm(f"Guide mode for {args.seconds:.0f} s: the arm goes limp under gravity compensation and you move it by "
                "hand. Ready to hold it?")
        arm.guide(args.seconds, block=True)
        return 0

    if args.cmd == "rate-test":
        runner = make_runner(args, calib, need_sign=False) if args.export else None
        confirm(f"Arm holds its pose for {args.seconds:g} s, then steps the target 2 mm up. Clear above the peg?")
        out = fb.rate_test(arm, cfg, args.seconds, runner)
        print(json.dumps(out, indent=2))
        log_event(args.dir / "events.jsonl", event="rate_test", control_hz=args.control_hz, **out)
        return 0

    if args.cmd == "grasp":
        confirm("Peg between the fingers, finger ends on the 32 mm line?")
        print(f"grasp width (m): {arm.grasp(force=args.force):.4f}  (force {args.force:.0f} N)")
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

    if args.cmd == "force-curve":
        confirm("Peg grasped, 1-5 cm above a FREE flat surface (not the hole), nothing else below?")
        out = fb.force_curve(arm, cfg)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        path = args.dir / f"force_curve_{stamp}.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(out))
        print("commanded push (N) -> mean |estimate z| (N):")
        for row in out["table"]:
            print(f"  {row['cmd_N']:5.1f} -> {row['mean_abs_fz_N']:5.2f}   (n={row['n']})")
        print("free-space std (N):", np.round(out["free_space_std_N"], 2).tolist() if out["free_space_std_N"] else None)
        print("saved", path)
        return 0

    if args.cmd == "force-scale":
        confirm("Kitchen scale on the table under the peg, ZEROED, peg grasped 1-5 cm above it?")
        out = fb.force_scale(arm, cfg, levels_N=args.levels)
        print("commanded (N)   scale (N)   |estimate z| (N)")
        for r in out["rows"]:
            print(f"  {r['cmd_N']:6.1f}     {r['scale_N']:6.2f}      {r['est_abs_z_N']:6.2f}")
        print_fit("actuation (commanded -> scale)", out["actuation"])
        print_fit("sensing   (scale -> estimate) ", out["sensing"])
        save_check(args.dir, "force_scale", out)
        return 0

    if args.cmd == "force-weights":
        if "force_sign" not in calib:
            print("no force_sign yet (press-test): the sign check is skipped")
        confirm("Arm in free space, peg grasped, string loop ready for the weights, NOTHING hanging now?")
        out = fb.force_weights(arm, cfg, direction=args.direction, force_sign=calib.get("force_sign"))
        print("mass (g)   true (N)   |estimate along| (N)   off-axis (N)")
        for r in out["rows"]:
            print(f"  {r['mass_g']:6.0f}    {r['true_N']:6.2f}      {r['est_along_abs_N']:6.2f}            "
                  f"{r['est_off_axis_N']:5.2f}")
        print_fit("sensing (true -> estimate)", out["sensing"])
        if "sign_ok" in out:
            print("sign agrees with press-test:", out["sign_ok"])
        save_check(args.dir, "force_weights", out)
        return 0

    if args.cmd == "force-drift":
        confirm(f"Arm in free space with at least {100 * args.sweep_m + 1:.0f} cm clear below the peg?")
        out = fb.force_drift(arm, cfg, hold_s=args.seconds, sweep_m=args.sweep_m)
        fmt = lambda v: "  ".join("   n/a" if x is None else f"{x:6.2f}" for x in v)
        print("                              x       y       z")
        print("hold: max |f| (N)          ", fmt(out["hold_max_abs_N"]))
        print("hold: last 1 s mean (N)    ", fmt(out["hold_end_mean_N"]))
        print("hold: drift (N/s)          ", fmt(out["hold_slope_N_per_s"]))
        print("hold: noise std (N)        ", fmt(out["hold_std_N"]))
        if "sweep_max_abs_N" in out:
            print("sweep: max |f| (N)         ", fmt(out["sweep_max_abs_N"]))
            print("sweep: change per cm (N/cm)", fmt(out["sweep_slope_N_per_cm"]))
        print("controller start shift (N) ", fmt(out["stream_start_shift_N"]), "  (removed by taring after it)")
        print(f"worst |f| anywhere: {out['worst_norm_N']:.2f} N")
        print("sim budget: under ~1 N of offset costs a few points; 2 N costs 10-20")
        save_check(args.dir, "force_drift", out)
        return 0

    if args.cmd == "friction-breakaway":
        if not 0.0 < args.max_N <= 8.0:
            raise SystemExit("--max_N must be in (0, 8]")
        confirm("Peg grasped, in free space, at least 10 cm clear in EVERY direction, nothing touching it?")
        out = fb.friction_breakaway(arm, cfg, max_pull_N=args.max_N, force_sign=friction_force_sign(calib))
        print("direction   breakaway pull (N)   estimate along it at the peak (N)")
        for r in out["rows"]:
            b = f"{r['breakaway_N']:6.2f}" if r["breakaway_N"] is not None else f">{r['max_pull_N']:5.1f} (no motion)"
            print(f"  {r['direction']}        {b:18s}   {r['est_along_N']:6.2f}")
        out["breakaway_N"] = {r["direction"]: r["breakaway_N"] for r in out["rows"]}
        out["est_along_N"] = {r["direction"]: r["est_along_N"] for r in out["rows"]}
        save_check(args.dir, "friction_breakaway", out)
        return 0

    if args.cmd == "friction-sweep":
        if not 0.0 < args.sweep_mm <= 30.0 or not all(0.5 <= v <= 20.0 for v in args.speeds):
            raise SystemExit("--sweep_mm must be in (0, 30] and --speeds in [0.5, 20] mm/s")
        confirm(f"Peg grasped, in free space, at least {args.sweep_mm / 10 + 5:.0f} cm clear along +{args.axis} "
                f"and 10 cm everywhere else, nothing touching it?")
        out = fb.friction_sweep(arm, cfg, axis=args.axis, speeds_m_s=[v / 1000 for v in args.speeds],
                                sweep_m=args.sweep_mm / 1000, force_sign=friction_force_sign(calib))
        i = "xyz".index(args.axis)
        print("speed (mm/s)   friction (N)   fake force along (N)   fake across (N)   pose error (N/cm)")
        for r in out["rows"]:
            print(f"  {r['speed_mm_s']:6.1f}        {r['friction_N']:6.2f}          {r['fake_along_N']:6.2f}"
                  f"              {r['fake_across_N']:6.2f}           {r['pose_error_N_per_cm'][i]:6.3f}")
        b = out["viscous_N_per_m_s"]
        print(f"fit: F = Fc + b*v, Fc = {out['coulomb_N']:.2f} N, b = " + ("n/a (one speed)" if b is None else f"{b:.0f} N/(m/s)"))
        print(f"force_sign {out['force_sign']:+g}; fake force along: < 0 opposes the motion. sim budget: under ~1 N of offset costs a few points; 2 N costs 7-20")
        out["table"] = [{k: r[k] for k in r if k not in ("series", "joints")} for r in out["rows"]]
        save_check(args.dir, f"friction_sweep_{args.axis}", out)
        return 0

    if args.cmd == "friction-steps":
        if not 0.7 <= args.stiffness_scale <= 1.3:
            raise SystemExit("--stiffness_scale must be in [0.7, 1.3] (what the sim stress evals covered)")
        if not all(0.0 < s <= 8.0 for s in args.steps_mm) or not set(args.axes) <= set("xyz"):
            raise SystemExit("--steps_mm must be in (0, 8] and --axes letters from xyz")
        cfg.stiffness = tuple(args.stiffness_scale * k for k in cfg.stiffness)
        confirm("Peg grasped, in free space, at least 10 cm clear in EVERY direction, nothing touching it?")
        out = fb.friction_steps(arm, cfg, steps_m=[s / 1000 for s in args.steps_mm], axes=args.axes)
        print(f"stiffness x {args.stiffness_scale:g}: distance moved (mm) after {out['hold_s']:g} s; 63% time (ms)")
        print("           " + "".join(f"{s:>12g}" for s in args.steps_mm) + "   (step, mm)")
        for ax in args.axes:
            for sgn in (1, -1):
                rs = [r for r in out["rows"] if r["axis"] == ax and r["sign"] == sgn]
                cells = []
                for r in rs:
                    t63 = "    -" if r["t63_ms"] is None else f"{r['t63_ms']:5.0f}"
                    cells.append(f"{r['moved_mm']:6.2f} {t63}")
                print(f"  {'+' if sgn > 0 else '-'}{ax}       " + "".join(cells))
        out["table"] = [{k: r[k] for k in r if k not in ("series", "joints")} for r in out["rows"]]
        save_check(args.dir, "friction_steps" if args.stiffness_scale == 1.0 else f"friction_steps_k{args.stiffness_scale:g}", out)
        return 0

    if args.cmd == "friction-walk":
        lo, hi = (tuple(args.box[:3]), tuple(args.box[3:])) if args.box else (fb.WALK_BOX_LO, fb.WALK_BOX_HI)
        centre = fb.WALK_CENTRE
        if args.centre:
            centre = tuple(args.centre)
        elif (args.dir / "hole.json").exists():
            rim = fb.Hole.load(args.dir / "hole.json").rim
            centre = (float(rim[0]), float(rim[1]), fb.WALK_CENTRE[2])
        spec = fb.WalkSpec(seed=args.seed, duration_s=args.seconds, box_lo=lo, box_hi=hi, centre=centre,
                           speed_mm_s=(1.0, args.max_speed))
        try:
            spec.check()
        except ValueError as e:
            raise SystemExit(str(e))
        size = 100 * (np.array(hi) - np.array(lo))
        peg_z = 100 * (np.array([lo[2], hi[2]]) - fb.GRASP_TIP_ABOVE_PEG_BOTTOM)
        print(f"Walk area: {size[0]:.0f} cm (x) by {size[1]:.0f} cm (y), x {100 * lo[0]:.0f}-{100 * hi[0]:.0f} cm in front of the "
              f"base, peg bottom {peg_z[0]:.0f}-{peg_z[1]:.0f} cm above the table; centre {np.round(centre, 3).tolist()}.")
        print("From `home` (or anywhere above the area, in front of the robot), the peg first moves across at its "
              "height to above the centre, then straight down to it.")
        confirm("Peg grasped, gripper pointing down, nothing in the area or on the way there, and clamps and hole "
                "base below the peg's lowest point?")
        force_sign = friction_force_sign(calib)
        out = fb.friction_walk(arm, cfg, spec, force_sign=force_sign)
        print(f"walked {out['seconds_done']:.0f} of {out['planned_s']:.0f} s, {len(out['series'])} states"
              + (f"; STOPPED: {out['stopped']}" if out["stopped"] else ""))
        fit = out["fit"]
        if fit:
            print(f"leak fit (force_sign {force_sign:+g}; v_s = {fit['v_s_mm_s']:g} mm/s, the sim uses 2): "
                  "fake force opposing the motion, per axis (> 0: opposes it, as in the sim)")
            print("axis   moving + (N)   moving - (N)   residual (N)   without friction terms (N)")
            for ax, r in fit["axes"].items():
                print(f"  {ax}       {r['leak_plus_N']:6.2f}         {r['leak_minus_N']:6.2f}         {r['rms_N']:5.2f}"
                      f"          {r['rms_no_friction_N']:5.2f}")
            print("sim training range: 0-3 N per axis")
        label = f"_{args.label}" if args.label else ""
        save_check(args.dir, f"friction_walk_s{args.seed}{label}", out)
        return 0 if not out["stopped"] else 1

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
