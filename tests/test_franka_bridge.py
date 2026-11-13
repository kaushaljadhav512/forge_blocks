"""Checks for deploy/franka_bridge.py with a fake arm (numpy only): python -m unittest discover tests

The fake arm is a crude stand-in for FrankaPy: the fingertip follows the streamed target with a lag,
a peg sticks out GRASP below it, and a hole with a rim stops the peg unless it is centred. Its raw
force can be sign-flipped and biased like libfranka's estimate. This checks the bridge's plumbing
(frames, tare, sign, rates, success detection, safety, streaming lifecycle), not policy quality.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

import franka_bridge as fb  # noqa: E402
from policy_runner import ForgePolicyRunner, quat_from_euler_xyz  # noqa: E402
from test_policy_runner import make_cfg  # noqa: E402

RIM = np.array([0.59, 0.0, 0.028])


class FakeArm:
    def __init__(self, raw_sign=1.0, bias=(0.0, 0.0, 0.0), tau=0.1, R=fb.DOWN, spike_at=None):
        self.t = 0.0
        self.tip = RIM + np.array([0.0, 0.0, fb.START_ABOVE_RIM])
        self.R = np.array(R, dtype=float)
        self.target = None
        self.raw_sign, self.bias, self.tau, self.spike_at = raw_sign, np.array(bias), tau, spike_at
        self.streaming, self.sent, self.streams = False, 0, 0

    # virtual time
    def clock(self):
        return self.t

    def sleep(self, dt):
        steps = max(1, int(math.ceil(dt / 0.001)))
        for _ in range(steps):
            self._advance(dt / steps)

    def _floor_tip_z(self, tip):
        centred = np.linalg.norm(tip[:2] - RIM[:2]) < 0.0005  # ~1 mm diametral clearance
        bottom_floor = RIM[2] - (fb.HOLE_DEPTH if centred else 0.0)
        return bottom_floor + fb.GRASP_TIP_ABOVE_PEG_BOTTOM

    def _advance(self, dt):
        self.t += dt
        if self.target is not None:
            self.tip = self.tip + (self.target - self.tip) * min(1.0, dt / self.tau)
        self.tip[2] = max(self.tip[2], self._floor_tip_z(self.tip))

    def env_force(self):
        if self.target is None or self.tip[2] > self._floor_tip_z(self.tip) + 1e-9:
            return np.zeros(3)
        return np.array([0.0, 0.0, 565.0 * max(0.0, self.tip[2] - self.target[2])])  # surface pushes up

    def read(self):
        f = self.raw_sign * self.env_force() + self.bias
        if self.spike_at is not None and self.t > self.spike_at:
            f = f + np.array([0.0, 0.0, 100.0])
        return fb.RobotState(t=self.t, q=np.zeros(7), dq=np.zeros(7), tool_pos=fb.tip_to_tool(self.tip, self.R),
                             tool_R=self.R.copy(), force=f).check()

    def start_stream(self, pos, R, stiffness, max_duration):
        assert not self.streaming
        self.streaming, self.streams = True, self.streams + 1
        self.target = fb.tool_to_tip(pos, R)

    def send_target(self, pos, R):
        assert self.streaming, "target sent without a running stream"
        self.target = fb.tool_to_tip(pos, R)
        self.R = np.array(R, dtype=float)
        self.sent += 1

    def stop_stream(self):
        self.streaming = False

    def goto(self, pos, R, duration, stiffness=None):
        self.tip, self.R = fb.tool_to_tip(pos, R), np.array(R, dtype=float)
        self.target = None
        self._advance(duration)

    def stop(self):
        self.streaming = False


class ScriptedRunner(ForgePolicyRunner):
    """Pushes straight down onto the estimate; raises its success flag once seated (no torch)."""

    def _run_policy(self, obs):
        mu = np.zeros(7)
        mu[2] = -1.0
        mu[5] = self.actions[5]
        mu[6] = 1.0 if obs[2] < fb.SEATED_TIP_ABOVE_RIM + 0.002 else -1.0
        self.last_obs = obs
        return mu


def make_runner(force_sign=1.0):
    return ScriptedRunner(make_cfg(force_sign=force_sign, ema_factor=0.2))


def attempt(arm, runner, est=RIM, limits=None, **cfg_kw):
    cfg = fb.BridgeConfig(tare_s=0.05, **cfg_kw)
    steps = []
    res = fb.run_attempt(arm, runner, fb.Hole(seated_tip=(RIM + [0, 0, fb.SEATED_TIP_ABOVE_RIM]).tolist()), est, cfg,
                         limits or fb.SafetyLimits(), log_steps=steps, clock=arm.clock, sleep=arm.sleep)
    return res, steps


class TestFrames(unittest.TestCase):
    def test_tip_tool_roundtrip_and_offset(self):
        R = fb.down_with_yaw(0.3)
        tool = np.array([0.5, 0.1, 0.2])
        tip = fb.tool_to_tip(tool, R)
        np.testing.assert_allclose(tip - tool, [0, 0, -fb.TIP_OFFSET], atol=1e-12)  # pointing down: tip is lower
        np.testing.assert_allclose(fb.tip_to_tool(tip, R), tool, atol=1e-12)

    def test_quat_roundtrip(self):
        rng = np.random.default_rng(0)
        for _ in range(50):
            q = rng.normal(size=4)
            q /= np.linalg.norm(q)
            q2 = fb.mat_to_quat(fb.quat_to_mat(q))
            self.assertAlmostEqual(abs(float(np.dot(q, q2))), 1.0, places=9)

    def test_down_with_yaw_matches_isaac_euler(self):
        for yaw in (-0.7, 0.0, 0.4):
            q = quat_from_euler_xyz(math.pi, 0.0, yaw)  # the runner's copy of Isaac's helper
            np.testing.assert_allclose(fb.quat_to_mat(q), fb.down_with_yaw(yaw), atol=1e-12)

    def test_canonical_quat(self):
        q = fb.mat_to_quat(fb.DOWN)
        np.testing.assert_allclose(fb.canonical_quat(q), [0, 1, 0, 0], atol=1e-12)
        np.testing.assert_allclose(fb.canonical_quat(-q), [0, 1, 0, 0], atol=1e-12)

    def test_tip_velocity_lever_arm(self):
        cfg = fb.BridgeConfig()
        J = np.zeros((6, 7))
        J[3, 0] = 1.0  # joint 0 spins the tool about base x
        st = fb.RobotState(t=0, q=np.zeros(7), dq=np.r_[2.0, np.zeros(6)], tool_pos=np.zeros(3), tool_R=fb.DOWN,
                           force=np.zeros(3), J=J)
        v, w = fb.tip_velocity(st, None, cfg)
        np.testing.assert_allclose(w, [2.0, 0, 0])
        np.testing.assert_allclose(v, np.cross([2.0, 0, 0], [0, 0, -fb.TIP_OFFSET]), atol=1e-12)


class TestAttempt(unittest.TestCase):
    def test_inserts_with_flipped_biased_force(self):
        arm = FakeArm(raw_sign=-1.0, bias=(1.0, -2.0, 3.0))
        runner = make_runner(force_sign=-1.0)
        res, steps = attempt(arm, runner)
        self.assertEqual(res["outcome"], "predicted_success", res)
        self.assertTrue(res["success"])
        self.assertIsNotNone(res["time_to_success_s"])
        self.assertGreaterEqual(res["time_to_prediction_s"], res["time_to_success_s"])
        np.testing.assert_allclose(res["tare_N"], [1.0, -2.0, 3.0], atol=1e-9)
        self.assertFalse(arm.streaming)
        # While pressing on the hole floor, the observed force points up, as in sim.
        pressing = [s for s in steps if s["obs"][2] < fb.SEATED_TIP_ABOVE_RIM + 0.001]
        self.assertTrue(pressing and max(s["obs"][15] for s in pressing) > 1.0)
        # Targets go out much faster than policy steps (runner.target between steps).
        self.assertGreater(arm.sent, 5 * res["policy_steps"])
        # Policy steps at 15 Hz.
        ts = [s["t"] for s in steps]
        self.assertAlmostEqual(float(np.median(np.diff(ts))), runner.cfg.policy_dt, delta=0.006)

    def test_misaligned_estimate_rests_on_rim_and_times_out(self):
        arm = FakeArm()
        res, _ = attempt(arm, make_runner(), est=RIM + [0.004, 0.0, 0.0])
        self.assertEqual(res["outcome"], "timeout")
        self.assertFalse(res["success"])
        self.assertGreater(res["peak_force_smooth_N"], 3.0)  # pressing on the rim

    def test_safety_stop_on_force_spike(self):
        arm = FakeArm(spike_at=0.5)
        res, _ = attempt(arm, make_runner())
        self.assertEqual(res["outcome"], "safety_stop")
        self.assertIn("force", res["error"])
        self.assertFalse(arm.streaming)

    def test_safety_stop_outside_box(self):
        arm = FakeArm()
        res, _ = attempt(arm, make_runner(), est=RIM + [0.0, 0.0, 0.0], limits=fb.SafetyLimits(box_xy_m=-1.0))
        self.assertEqual(res["outcome"], "safety_stop")


class TestCalibration(unittest.TestCase):
    def test_press_test_sign(self):
        for raw_sign in (1.0, -1.0):
            arm = FakeArm(raw_sign=raw_sign, bias=(0.5, 0.5, -4.0))
            arm.tip = RIM + [0.03, 0.0, fb.GRASP_TIP_ABOVE_PEG_BOTTOM + 0.01]  # off the hole: rim is the surface
            out = fb.press_test(arm, fb.BridgeConfig(tare_s=0.05), descend_m_s=0.02, clock=arm.clock, sleep=arm.sleep)
            self.assertEqual(out["force_sign"], raw_sign)
            self.assertFalse(arm.streaming)

    def test_samplers(self):
        rng = np.random.default_rng(1)
        for _ in range(200):
            e = fb.sample_error(0.001, 0.0025, rng)
            self.assertTrue(0.001 - 1e-12 <= np.linalg.norm(e) <= 0.0025 + 1e-12)
        hole = fb.Hole(seated_tip=(RIM + [0, 0, fb.SEATED_TIP_ABOVE_RIM]).tolist())
        cfg = fb.BridgeConfig()
        for _ in range(200):
            pos, R = fb.sample_start(hole, rng, cfg)
            tip = fb.tool_to_tip(pos, R)
            d = tip - (hole.rim + [0, 0, fb.START_ABOVE_RIM])
            self.assertTrue(np.all(np.abs(d) <= np.array(fb.START_NOISE) + 1e-12))
            yaw = math.atan2(R[1, 0], R[0, 0])
            self.assertLessEqual(abs(yaw), fb.START_YAW_NOISE + 1e-9)

    def test_hole_rim(self):
        hole = fb.Hole(seated_tip=[0.5, 0.0, 0.035], samples=[[0.5, 0.0, 0.035], [0.5004, 0.0, 0.035]])
        np.testing.assert_allclose(hole.rim, [0.5, 0.0, 0.035 - fb.SEATED_TIP_ABOVE_RIM])
        self.assertAlmostEqual(hole.spread_m, 0.0002, places=9)


if __name__ == "__main__":
    unittest.main()
