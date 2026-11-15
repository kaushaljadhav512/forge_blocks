"""scripts/walk_compare.py: robot walk vs sim replay comparison (numpy only, no Isaac Sim)."""

import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
sys.path.insert(0, os.path.dirname(__file__))

import franka_bridge as fb  # noqa: E402
import walk_compare as wc  # noqa: E402
from policy_runner import ForceFilter  # noqa: E402
from test_franka_bridge import FakeArm  # noqa: E402

RIM = np.array([0.59, 0.0, 0.028])


def fake_walk(raw_sign=1.0, seconds=20.0):
    arm = FakeArm(vel_leak=(1.5, 0.002))
    arm.goto(fb.tip_to_tool(np.array([0.5, 0.05, 0.2]), fb.DOWN), fb.DOWN, 0.1)
    out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(seed=1, duration_s=seconds),
                           clock=arm.clock, sleep=arm.sleep)
    if raw_sign != 1.0:  # the lab robot's raw estimate has the opposite sign
        out["series"] = [r[:7] + [raw_sign * x for x in r[7:10]] + r[10:] for r in out["series"]]
        out["force_sign"] = raw_sign
    return out


def sim_from(robot, envs=2):
    t = np.arange(0.0, robot["t"][-1], wc.SIM_DT)
    on = lambda x: np.column_stack([np.interp(t, robot["t"], x[:, i]) for i in range(3)])  # noqa: E731
    stack = lambda x: np.repeat(on(x)[:, None, :], envs, axis=1)  # noqa: E731
    return {"t": t, "target": stack(robot["target"]), "tip": stack(robot["tip"]), "force": stack(robot["force"])}


class TestWalkCompare(unittest.TestCase):
    def test_ema_matches_the_policy_filter(self):
        rng = np.random.default_rng(0)
        t = np.cumsum(rng.uniform(0.005, 0.015, 300))
        x = rng.normal(size=(300, 3))
        flt = ForceFilter(wc.FT_SMOOTHING, wc.SIM_DT)
        want, prev = [], t[0]
        for i in range(len(t)):
            want.append(flt.update(x[i], t[i] - prev if i else wc.SIM_DT).copy())
            prev = t[i]
        np.testing.assert_allclose(wc.ema(t, x), np.array(want), atol=1e-12)

    def test_robot_traces_use_the_rim_and_force_sign(self):
        a, b = wc.robot_traces(fake_walk(), RIM), wc.robot_traces(fake_walk(raw_sign=-1.0), RIM)
        np.testing.assert_allclose(a["force"], b["force"], atol=1e-9)  # same force on the hand
        np.testing.assert_allclose(a["tip"][0] + RIM, fake_walk()["series"][0][4:7], atol=1e-9)
        self.assertIn("force_no_pose", a)

    def test_a_perfect_replay_has_no_error(self):
        robot = wc.robot_traces(fake_walk(), RIM)
        rows = wc.compare(robot, sim_from(robot))
        self.assertEqual(len(rows), 2)
        for r in rows:
            self.assertLess(r["target_mismatch_mm"], 0.2)  # interpolation only
            self.assertLess(max(r["tip_rms_mm"]), 0.05)
            self.assertLess(max(r["force_rms_N"]), 0.05)
            self.assertAlmostEqual(r["lag_sim_mm"], r["lag_robot_mm"], delta=0.05)

    def test_a_replay_without_the_leak_shows_the_leak_as_error(self):
        robot = wc.robot_traces(fake_walk(), RIM)
        sim = sim_from(robot)
        sim["force"][:, 1] = 0.0  # env 1: no leak in the sim
        rows = wc.compare(robot, sim)
        self.assertLess(max(rows[0]["force_rms_N"]), 0.05)
        self.assertGreater(min(rows[1]["force_rms_N"]), 0.3)


if __name__ == "__main__":
    unittest.main()
