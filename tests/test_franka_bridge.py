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
    def __init__(self, raw_sign=1.0, bias=(0.0, 0.0, 0.0), tau=0.1, R=fb.DOWN, spike_at=None, sense_deadband=0.0,
                 act_deadband=0.0, drift=(0.0, 0.0, 0.0), z_gain=(0.0, 0.0, 0.0)):
        self.t = 0.0
        self.tip = RIM + np.array([0.0, 0.0, fb.START_ABOVE_RIM])
        self.z0 = None  # z_gain reference: the first tip height read
        self.act_deadband = act_deadband  # joint friction: this much of the spring force never reaches the surface
        self.drift, self.z_gain = np.array(drift), np.array(z_gain)  # estimate error: N/s and N/m of tip z
        self.ext_load = np.zeros(3)  # e.g. a hanging mass: force of the environment on the hand
        self.R = np.array(R, dtype=float)
        self.target = None
        self.raw_sign, self.bias, self.tau, self.spike_at = raw_sign, np.array(bias), tau, spike_at
        self.sense_deadband = sense_deadband
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
        push = max(0.0, 565.0 * max(0.0, self.tip[2] - self.target[2]) - self.act_deadband)
        return np.array([0.0, 0.0, push])  # surface pushes up

    def read(self):
        if self.z0 is None:
            self.z0 = self.tip[2]
        f = self.env_force() + self.ext_load
        f = np.sign(f) * np.clip(np.abs(f) - self.sense_deadband, 0.0, None)  # unreliable small forces
        f = self.raw_sign * f + self.bias + self.drift * self.t + self.z_gain * (self.tip[2] - self.z0)
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

    def test_force_curve_shows_deadband(self):
        for band in (0.0, 5.0):
            arm = FakeArm(sense_deadband=band, bias=(0.3, 0.0, -2.0))
            arm.tip = RIM + [0.03, 0.0, fb.GRASP_TIP_ABOVE_PEG_BOTTOM + 0.01]
            out = fb.force_curve(arm, fb.BridgeConfig(tare_s=0.05), descend_m_s=0.02, clock=arm.clock, sleep=arm.sleep)
            self.assertFalse(arm.streaming)
            table = {round(r["cmd_N"] - 0.5): r["mean_abs_fz_N"] for r in out["table"]}
            self.assertAlmostEqual(table[10], max(10.5 - band, 0.0), delta=0.6)
            self.assertAlmostEqual(table[2], max(2.5 - band, 0.0), delta=0.6)

    def test_force_scale_separates_actuation_and_sensing(self):
        arm = FakeArm(act_deadband=1.5, sense_deadband=3.0, bias=(0.2, 0.0, -1.0))
        arm.tip = RIM + [0.03, 0.0, fb.GRASP_TIP_ABOVE_PEG_BOTTOM + 0.01]
        scale_g = lambda prompt: f"{arm.env_force()[2] / fb.G * 1000:.0f}"  # the scale sees the true push
        out = fb.force_scale(arm, fb.BridgeConfig(tare_s=0.05), read_scale=scale_g, descend_m_s=0.02,
                             settle_s=0.5, hold_s=0.3, clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        self.assertEqual(len(out["rows"]), 8)
        self.assertAlmostEqual(out["actuation"]["deadband_N"], 1.5, delta=0.2)
        self.assertAlmostEqual(out["sensing"]["deadband_N"], 3.0, delta=0.2)
        self.assertAlmostEqual(out["sensing"]["slope"], 1.0, delta=0.05)
        low = out["rows"][0]  # 1 N commanded: nothing reaches the scale, nothing sensed
        self.assertLess(low["scale_N"], 0.05)
        self.assertLess(low["est_abs_z_N"], 0.05)

    def test_force_weights_reads_masses_and_checks_sign(self):
        for raw_sign, band in ((1.0, 0.0), (-1.0, 2.0)):
            arm = FakeArm(raw_sign=raw_sign, sense_deadband=band, bias=(0.5, -0.3, 2.0))
            masses = iter(["0", "100", "250", "500", "1000", "0", ""])

            def hang(prompt):
                m = next(masses)
                arm.ext_load = np.array([0.0, 0.0, -float(m or 0) * fb.G / 1000])  # weight pulls the hand down
                return m

            # press-test would give force_sign = +1 for raw_sign +1 (pressing reads +z raw) and -1 otherwise
            out = fb.force_weights(arm, fb.BridgeConfig(), ask_mass=hang, force_sign=raw_sign, settle_s=0.2,
                                   hold_s=0.2, clock=arm.clock, sleep=arm.sleep)
            self.assertFalse(arm.streaming)
            self.assertEqual([r["mass_g"] for r in out["rows"]], [0, 100, 250, 500, 1000, 0])
            self.assertTrue(out["sign_ok"])
            heavy = out["rows"][4]
            self.assertAlmostEqual(heavy["est_along_abs_N"], 9.81 - band, delta=0.05)
            self.assertLess(heavy["est_off_axis_N"], 0.05)
            self.assertLess(out["rows"][-1]["est_along_abs_N"], 0.05)  # back to zero
            if band:
                self.assertAlmostEqual(out["sensing"]["deadband_N"], band, delta=0.1)
                self.assertLess(out["rows"][1]["est_along_abs_N"], 0.05)  # 100 g ~ 1 N: inside the dead band
        # a sign from press-test that disagrees with the weights is flagged
        arm = FakeArm()
        masses = iter(["500", ""])
        def hang(prompt):
            m = next(masses)
            arm.ext_load = np.array([0.0, 0.0, -float(m or 0) * fb.G / 1000])
            return m
        out = fb.force_weights(arm, fb.BridgeConfig(), ask_mass=hang, force_sign=-1.0, settle_s=0.2, hold_s=0.2,
                               clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(out["sign_ok"])

    def test_force_drift_measures_drift_and_pose_error(self):
        arm = FakeArm(bias=(1.0, 1.0, 1.0), drift=(0.0, 0.05, 0.0), z_gain=(0.0, 0.0, 20.0))  # 0.2 N per cm
        out = fb.force_drift(arm, fb.BridgeConfig(tare_s=0.05), hold_s=4.0, sweep_m=0.04, sweep_m_s=0.04,
                             clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        self.assertAlmostEqual(out["hold_slope_N_per_s"][1], 0.05, delta=0.005)
        self.assertAlmostEqual(out["hold_end_mean_N"][1], 0.05 * 3.5, delta=0.05)  # tare taken at t ~ 0
        self.assertAlmostEqual(out["sweep_slope_N_per_cm"][2], 0.2, delta=0.02)
        self.assertAlmostEqual(out["sweep_max_abs_N"][2], 0.8, delta=0.1)
        self.assertLess(abs(out["hold_end_mean_N"][0]), 0.01)  # constant bias is removed by the tare

    def test_force_drift_stops_on_contact(self):
        arm = FakeArm()
        arm.tip = RIM + [0.03, 0.0, fb.GRASP_TIP_ABOVE_PEG_BOTTOM + 0.01]  # surface 1 cm below, sweep 4 cm
        with self.assertRaises(fb.SafetyStop):
            fb.force_drift(arm, fb.BridgeConfig(tare_s=0.05), hold_s=0.2, sweep_m=0.04, sweep_m_s=0.04,
                           clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)

    def test_rate_test_reports_step_latency(self):
        arm = FakeArm(tau=0.05)
        out = fb.rate_test(arm, fb.BridgeConfig(), seconds=0.5, clock=arm.clock, sleep=arm.sleep)
        self.assertGreater(out["send_hz"], 150)
        self.assertLess(out["step_first_motion_ms"], 20)
        self.assertAlmostEqual(out["step_63pct_ms"], 50, delta=15)  # first-order lag with tau = 50 ms
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
