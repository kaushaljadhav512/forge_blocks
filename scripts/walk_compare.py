"""Compare a robot friction walk with its sim replay (numpy only; used by replay_walk.py, tested in
tests/test_walk_compare.py).

Both sides are expressed relative to the hole rim and as the force on the hand (the sim's and the
policy's convention: robot force = force_sign x the raw estimate). The robot force is smoothed with the
same EMA the policy sees (FORGE's ft_smoothing_factor per 1/120 s sim step, re-expressed for the robot's
sample times, as deploy/policy_runner.py's ForceFilter does), then both are compared on the sim's times.
"""

import numpy as np

SIM_DT = 1.0 / 120.0
FT_SMOOTHING = 0.25  # ForgeEnvCfg.ft_smoothing_factor


def ema(t, x, alpha_per_sim_step=FT_SMOOTHING, sim_dt=SIM_DT):
    """EMA of x (N, k) sampled at times t, with the time constant of alpha per sim step."""
    out = np.zeros_like(x, dtype=float)
    v = np.zeros(x.shape[1])
    prev = t[0]
    for i in range(len(t)):
        a = 1.0 - (1.0 - alpha_per_sim_step) ** (max(t[i] - prev, 0.0) / sim_dt) if i else alpha_per_sim_step
        v = a * x[i] + (1.0 - a) * v
        out[i] = v
        prev = t[i]
    return out


def robot_traces(walk: dict, robot_rim) -> dict:
    """From a friction_walk output: t, target and tip relative to the rim, and the force on the hand
    (raw x force_sign), smoothed like the policy's, plus the same with the fit's offset and pose-dependent
    error removed (what a sim without pose error can match)."""
    S = np.array([r[:10] for r in walk["series"]], dtype=float)
    rim = np.asarray(robot_rim, dtype=float)
    t, target, tip = S[:, 0], S[:, 1:4] - rim, S[:, 4:7] - rim
    f = walk.get("force_sign", 1.0) * S[:, 7:10]
    out = {"t": t, "target": target, "tip": tip, "force": ema(t, f)}
    fit = walk.get("fit")
    if fit:
        X = S[:, 4:7] - S[:, 4:7].mean(axis=0)
        pose = np.column_stack([fit["axes"][a]["offset_N"] + X @ (100.0 * np.array(fit["axes"][a]["pose_N_per_cm"]))
                                for a in "xyz"])
        out["force_no_pose"] = ema(t, f - pose)
    return out


def compare(robot: dict, sim: dict) -> list:
    """robot: robot_traces(); sim: t (M,), and per env (M, E, 3) target, tip, force (all rim-relative,
    force on the hand). Returns one dict per env: tracking and force errors on the common time span."""
    t = sim["t"]
    keep = (t >= robot["t"][0]) & (t <= robot["t"][-1])
    t = t[keep]

    def on_sim(x):
        return np.column_stack([np.interp(t, robot["t"], x[:, i]) for i in range(3)])

    r_tip, r_target, r_force = on_sim(robot["tip"]), on_sim(robot["target"]), on_sim(robot["force"])
    r_force_np = on_sim(robot["force_no_pose"]) if "force_no_pose" in robot else None
    rows = []
    for e in range(sim["tip"].shape[1]):
        s_tip, s_target, s_force = sim["tip"][keep, e], sim["target"][keep, e], sim["force"][keep, e]
        row = {
            "env": e,
            "target_mismatch_mm": float(1000 * np.abs(s_target - r_target).max()),  # replay sanity: ~0
            "tip_rms_mm": (1000 * np.sqrt(np.mean((s_tip - r_tip) ** 2, axis=0))).tolist(),
            "lag_robot_mm": float(1000 * np.linalg.norm(r_tip - r_target, axis=1).mean()),
            "lag_sim_mm": float(1000 * np.linalg.norm(s_tip - s_target, axis=1).mean()),
            "force_rms_N": np.sqrt(np.mean((s_force - r_force) ** 2, axis=0)).tolist(),
            "force_corr": [float(np.corrcoef(s_force[:, i], r_force[:, i])[0, 1]) if np.std(s_force[:, i]) > 1e-9 else None
                           for i in range(3)],
        }
        if r_force_np is not None:
            row["force_rms_no_pose_N"] = np.sqrt(np.mean((s_force - r_force_np) ** 2, axis=0)).tolist()
        rows.append(row)
    return rows
