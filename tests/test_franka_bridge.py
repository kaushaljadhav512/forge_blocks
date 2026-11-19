"""Checks for deploy/franka_bridge.py with a fake arm (numpy only): python -m unittest discover tests

The fake arm is a crude stand-in for FrankaPy: the fingertip follows the streamed target with a lag,
a peg sticks out GRASP below it, and a hole with a rim stops the peg unless it is centred. Its raw
force can be sign-flipped and biased like libfranka's estimate. This checks the bridge's plumbing
(frames, tare, sign, rates, success detection, safety, streaming lifecycle), not policy quality.
"""

import json
import math
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

import franka_bridge as fb  # noqa: E402
from policy_runner import ForgePolicyRunner, quat_from_euler_xyz  # noqa: E402
from test_policy_runner import make_cfg  # noqa: E402

RIM = np.array([0.59, 0.0, 0.028])


class FakeArm:
    def __init__(self, raw_sign=1.0, bias=(0.0, 0.0, 0.0), tau=0.1, R=fb.DOWN, spike_at=None, sense_deadband=0.0,
                 act_deadband=0.0, drift=(0.0, 0.0, 0.0), z_gain=(0.0, 0.0, 0.0), stream_shift=(0.0, 0.0, 0.0),
                 fric_coulomb=0.0, fric_static=None, fric_leak=0.0, pos_gain=None, vel_leak=None):
        # Joint friction (per Cartesian axis, off unless fric_coulomb or fric_static is set): the hand
        # stays put until the spring pull exceeds fric_static, then moves at (pull - fric_coulomb) /
        # damping, and sticks again when the pull drops to fric_coulomb. damping = 565 * tau, so with no
        # friction it is the same first-order lag as below. fric_leak of the friction force on the hand
        # shows up in the raw estimate (the lab robot: a force opposing the motion).
        self.fric_coulomb = fric_coulomb
        self.fric_static = fric_coulomb if fric_static is None else fric_static
        self.fric_leak = fric_leak
        self.vel_leak = vel_leak  # (F, v_s): the sim's model, -F tanh(v / v_s) per axis on the tip velocity
        self.tip_vel = np.zeros(3)
        self.friction = fric_coulomb > 0 or self.fric_static > 0
        self.damping = 565.0 * tau
        self.k = np.full(3, 565.0)  # set from start_stream's stiffness
        self.stuck = np.ones(3, dtype=bool)
        self.fric_force = np.zeros(3)
        self.frozen_after = None  # sim time after which read() content stops changing
        self.pos_gain = None if pos_gain is None else np.array(pos_gain, dtype=float)  # N/m of tip displacement
        self.p0 = None
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
        self.stream_shift = np.array(stream_shift)  # estimate moves while the impedance skill runs (lab: ~1.7 N z)
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
        before = self.tip.copy()
        self.fric_force = np.zeros(3)
        if self.target is not None and self.friction:
            self._advance_friction(dt)
        elif self.target is not None:
            self.tip = self.tip + (self.target - self.tip) * min(1.0, dt / self.tau)
        self.tip[2] = max(self.tip[2], self._floor_tip_z(self.tip))
        self.tip_vel = (self.tip - before) / dt

    def _advance_friction(self, dt):
        gap = self.target - self.tip
        F = self.k * gap
        step = np.zeros(3)
        for i in range(3):
            if self.stuck[i] and abs(F[i]) > self.fric_static:
                self.stuck[i] = False
            if not self.stuck[i] and abs(F[i]) <= self.fric_coulomb:
                self.stuck[i] = True
            if self.stuck[i]:
                self.fric_force[i] = -F[i]  # static friction balances the pull
                continue
            self.fric_force[i] = -np.sign(F[i]) * self.fric_coulomb
            step[i] = np.sign(F[i]) * (abs(F[i]) - self.fric_coulomb) / self.damping * dt
            if abs(step[i]) > abs(gap[i]):
                step[i] = gap[i]
        self.tip = self.tip + step

    def env_force(self):
        if self.target is None or self.tip[2] > self._floor_tip_z(self.tip) + 1e-9:
            return np.zeros(3)
        push = max(0.0, 565.0 * max(0.0, self.tip[2] - self.target[2]) - self.act_deadband)
        return np.array([0.0, 0.0, push])  # surface pushes up

    def read(self):
        if self.z0 is None:
            self.z0 = self.tip[2]
        if self.p0 is None:
            self.p0 = self.tip.copy()
        f = self.env_force() + self.ext_load
        f = np.sign(f) * np.clip(np.abs(f) - self.sense_deadband, 0.0, None)  # unreliable small forces
        f = self.raw_sign * f + self.bias + self.drift * self.t + self.z_gain * (self.tip[2] - self.z0)
        if self.streaming:
            f = f + self.stream_shift
        f = f + self.fric_leak * self.fric_force
        if self.vel_leak is not None:
            f = f - self.vel_leak[0] * np.tanh(self.tip_vel / self.vel_leak[1])
        if self.pos_gain is not None:
            f = f + self.pos_gain @ (self.tip - self.p0)
        if self.spike_at is not None and self.t > self.spike_at:
            f = f + np.array([0.0, 0.0, 100.0])
        # q carries the clock: real joint readings are noisy and never repeat exactly (FreezeCheck relies on it)
        q = np.full(7, 1e-9 * self.t)
        if self.frozen_after is not None and self.t > self.frozen_after:
            q, f = np.zeros(7), self.__dict__.setdefault("_frozen_f", f)
        return fb.RobotState(t=self.t, q=q, dq=np.zeros(7), tool_pos=fb.tip_to_tool(self.tip, self.R),
                             tool_R=self.R.copy(), force=f).check()

    def start_stream(self, pos, R, stiffness, max_duration):
        assert not self.streaming
        self.streaming, self.streams = True, self.streams + 1
        self.target = fb.tool_to_tip(pos, R)
        self.k = np.array(stiffness[:3], dtype=float)

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
        self.stuck[:] = True
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
        res, steps = attempt(arm, runner, stop_after_seated_s=None, pred_threshold=0.5)
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

    def test_stops_soon_after_seating(self):
        arm = FakeArm()
        runner = make_runner()
        res, steps = attempt(arm, runner)  # the prediction doesn't end it by default
        self.assertEqual(res["outcome"], "seated", res)
        self.assertTrue(res["success"])
        self.assertTrue(res["seated_at_end"])
        self.assertLess(steps[-1]["t"] - res["time_to_success_s"], 0.3 + 0.07)

    def test_attempt_streams_stiffer_rotation_and_logs_orientation(self):
        arm = FakeArm()
        seen = []
        start = arm.start_stream
        arm.start_stream = lambda pos, R, stiffness, max_duration: (seen.append(stiffness),
                                                                     start(pos, R, stiffness, max_duration))[1]
        res, steps = attempt(arm, make_runner())
        self.assertEqual(list(seen[0]), [565.0, 565.0, 565.0, 80.0, 80.0, 80.0])
        self.assertEqual(list(fb.BridgeConfig().stiffness)[3:], [28.0, 28.0, 28.0])  # calibration commands unchanged
        np.testing.assert_allclose(steps[0]["tool_R"], arm.R, atol=1e-12)
        self.assertAlmostEqual(steps[0]["tilt_deg"], 0.0, places=6)

    def test_frozen_state_stops_the_attempt(self):
        # Session 3 trial 2: messages kept arriving but joints and force were bit-identical for seconds.
        arm = FakeArm()
        arm.frozen_after = arm.t + 1.0
        res, _ = attempt(arm, make_runner(), est=RIM + [0.004, 0.0, 0.0])
        self.assertEqual(res["outcome"], "safety_stop")
        self.assertIn("unchanged", res["error"])

    def test_tilt_stops_the_attempt(self):
        mon = fb.SafetyMonitor(fb.SafetyLimits(), RIM, RIM)
        tip = RIM + [0.0, 0.0, 0.02]
        c, s = math.cos(math.radians(12.0)), math.sin(math.radians(12.0))
        tilted = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ fb.DOWN
        mon.check(tip, 5.0, 0.01, fb.DOWN)  # upright: fine
        with self.assertRaises(fb.SafetyStop):
            mon.check(tip, 5.0, 0.01, tilted)

    def test_tilt_after_seating_ends_as_seated(self):
        c, s = math.cos(math.radians(12.0)), math.sin(math.radians(12.0))
        tilted = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ fb.DOWN

        class TipsWhenSeated(FakeArm):
            def read(self):
                if self.streaming and self.tip[2] < RIM[2] + fb.SEATED_TIP_ABOVE_RIM + 0.0005:
                    self.R = tilted
                return super().read()

        res, _ = attempt(TipsWhenSeated(), make_runner())
        self.assertEqual(res["outcome"], "seated", res)
        self.assertTrue(res["success"])
        self.assertIsNone(res["error"])
        self.assertIn("tilt", res["post_seat_stop"])

    def test_force_after_seating_ends_as_seated(self):
        class PushedWhenSeated(FakeArm):
            def read(self):
                if self.streaming and self.tip[2] < RIM[2] + fb.SEATED_TIP_ABOVE_RIM + 0.0005:
                    self.ext_load = np.array([30.0, 0.0, 0.0])
                return super().read()

        res, _ = attempt(PushedWhenSeated(), make_runner(), stop_after_seated_s=1.0)
        self.assertEqual(res["outcome"], "seated", res)
        self.assertIn("force", res["post_seat_stop"])

    def test_force_before_seating_is_a_safety_stop(self):
        class PushedHigh(FakeArm):
            def read(self):
                if self.streaming and self.tip[2] < RIM[2] + fb.START_ABOVE_RIM - 0.01:
                    self.ext_load = np.array([30.0, 0.0, 0.0])
                return super().read()

        res, _ = attempt(PushedHigh(), make_runner())
        self.assertEqual(res["outcome"], "safety_stop", res)
        self.assertEqual(res["stop_kind"], "force")

    def test_tilt_before_seating_is_a_safety_stop(self):
        c, s = math.cos(math.radians(12.0)), math.sin(math.radians(12.0))
        tilted = np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]]) @ fb.DOWN

        class TipsHigh(FakeArm):
            def read(self):
                if self.streaming and self.tip[2] < RIM[2] + fb.START_ABOVE_RIM - 0.01:
                    self.R = tilted
                return super().read()

        res, _ = attempt(TipsHigh(), make_runner())
        self.assertEqual(res["outcome"], "safety_stop", res)
        self.assertFalse(res["success"])
        self.assertIn("tilt", res["error"])
        self.assertEqual(res["stop_kind"], "tilt")

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

    def test_tare_after_stream_start_removes_controller_shift(self):
        shift = np.array([0.0, 0.0, -1.7])
        arm = FakeArm(stream_shift=shift)
        out = fb.force_drift(arm, fb.BridgeConfig(tare_s=0.05), hold_s=1.0, sweep_m=0.01, sweep_m_s=0.04,
                             clock=arm.clock, sleep=arm.sleep)
        np.testing.assert_allclose(out["stream_start_shift_N"], shift, atol=1e-9)
        np.testing.assert_allclose(out["hold_end_mean_N"], 0.0, atol=1e-9)  # tared before the stream: -1.7 N
        self.assertEqual(len(out["hold_series"][0]), 4)
        # The attempt loop tares the same way: no false force while hovering in free space.
        arm = FakeArm(stream_shift=shift)
        arm.tip = RIM + np.array([0.0, 0.0, 0.10])

        class HoverRunner(ForgePolicyRunner):  # target = the estimate = where the arm already is
            def _run_policy(self, obs):
                mu = np.zeros(7)
                mu[5], mu[6] = self.actions[5], -1.0
                return mu

        res, _ = attempt(arm, HoverRunner(make_cfg()), est=RIM + np.array([0.0, 0.0, 0.10]), timeout_s=0.5)
        np.testing.assert_allclose(res["tare_N"], shift, atol=1e-9)
        self.assertLess(res["peak_force_smooth_N"], 0.05)

    def test_force_drift_stops_on_contact(self):
        arm = FakeArm()
        arm.tip = RIM + [0.03, 0.0, fb.GRASP_TIP_ABOVE_PEG_BOTTOM + 0.01]  # surface 1 cm below, sweep 4 cm
        with self.assertRaises(fb.SafetyStop):
            fb.force_drift(arm, fb.BridgeConfig(tare_s=0.05), hold_s=0.2, sweep_m=0.04, sweep_m_s=0.04,
                           clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)

    def test_force_drift_hold_only(self):
        arm = FakeArm(drift=(0.0, 0.05, 0.0))
        out = fb.force_drift(arm, fb.BridgeConfig(tare_s=0.05), hold_s=2.0, sweep_m=0.0, clock=arm.clock, sleep=arm.sleep)
        self.assertNotIn("sweep_max_abs_N", out)
        self.assertAlmostEqual(out["hold_slope_N_per_s"][1], 0.05, delta=0.005)
        self.assertFalse(arm.streaming)

    def test_friction_breakaway_finds_stiction(self):
        cfg = fb.BridgeConfig(tare_s=0.05)
        arm = FakeArm(fric_static=3.0, fric_coulomb=2.0, fric_leak=0.5)
        out = fb.friction_breakaway(arm, cfg, directions=("+x", "-z"), ramp_m_s=0.002, clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        for r in out["rows"]:
            self.assertAlmostEqual(r["breakaway_N"], 3.0, delta=0.1)
            self.assertAlmostEqual(r["est_along_N"], -1.5, delta=0.1)  # stuck: static friction holds the pull
        arm = FakeArm()
        out = fb.friction_breakaway(arm, cfg, directions=("+y",), ramp_m_s=0.002, clock=arm.clock, sleep=arm.sleep)
        self.assertLess(out["rows"][0]["breakaway_N"], 0.3)  # no friction: only the lag
        arm = FakeArm(fric_static=9.0, fric_coulomb=2.0)
        out = fb.friction_breakaway(arm, cfg, directions=("+x",), ramp_m_s=0.004, clock=arm.clock, sleep=arm.sleep)
        self.assertIsNone(out["rows"][0]["breakaway_N"])  # gives up at max_pull_N
        self.assertAlmostEqual(out["rows"][0]["max_pull_N"], 8.0, delta=0.1)

    def test_friction_sweep_separates_friction_and_pose_error(self):
        gain = np.zeros((3, 3))
        gain[0, 0] = 20.0  # 0.2 N per cm of x
        arm = FakeArm(fric_coulomb=2.0, fric_leak=0.8, pos_gain=gain)
        out = fb.friction_sweep(arm, fb.BridgeConfig(tare_s=0.05), axis="x", speeds_m_s=(0.005, 0.02),
                                clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        self.assertAlmostEqual(out["coulomb_N"], 2.0, delta=0.1)
        self.assertAlmostEqual(out["viscous_N_per_m_s"], 565.0 * 0.1, delta=3.0)  # the fake's damping
        for r in out["rows"]:
            self.assertAlmostEqual(r["fake_along_N"], -1.6, delta=0.05)  # opposes the motion
            self.assertLess(r["fake_across_N"], 0.05)
        self.assertAlmostEqual(out["pose_error_N_per_cm"][0], 0.2, delta=0.02)

    def test_friction_checks_read_the_force_on_the_hand_for_either_raw_sign(self):
        class FlippedArm(FakeArm):  # raw estimate with the lab robot's sign (force_sign -1)
            def read(self):
                st = super().read()
                st.force = -st.force
                return st

        kw = dict(fric_coulomb=2.0, fric_leak=0.8)
        cfg = fb.BridgeConfig(tare_s=0.05)
        a, b = FakeArm(**kw), FlippedArm(**kw)
        sa = fb.friction_sweep(a, cfg, axis="x", speeds_m_s=(0.01,), clock=a.clock, sleep=a.sleep)
        sb = fb.friction_sweep(b, cfg, axis="x", speeds_m_s=(0.01,), force_sign=-1.0, clock=b.clock, sleep=b.sleep)
        self.assertAlmostEqual(sb["rows"][0]["fake_along_N"], sa["rows"][0]["fake_along_N"], places=6)
        self.assertLess(sb["rows"][0]["fake_along_N"], -1.0)  # opposes the motion
        a, b = FakeArm(fric_static=2.0, fric_coulomb=1.5), FlippedArm(fric_static=2.0, fric_coulomb=1.5, fric_leak=1.0)
        a.fric_leak = 1.0
        ba = fb.friction_breakaway(a, cfg, directions=("+x",), ramp_m_s=0.004, clock=a.clock, sleep=a.sleep)
        bb = fb.friction_breakaway(b, cfg, directions=("+x",), ramp_m_s=0.004, force_sign=-1.0, clock=b.clock, sleep=b.sleep)
        self.assertAlmostEqual(bb["rows"][0]["est_along_N"], ba["rows"][0]["est_along_N"], places=6)
        self.assertLess(bb["rows"][0]["est_along_N"], 0.0)

    def test_friction_sweep_stops_on_force(self):
        arm = FakeArm(spike_at=1.5)
        with self.assertRaises(fb.SafetyStop):
            fb.friction_sweep(arm, fb.BridgeConfig(tare_s=0.05), axis="y", speeds_m_s=(0.01,),
                              clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)

    def test_friction_steps_stop_short(self):
        steps = (0.002, 0.004)
        arm = FakeArm(fric_static=2.0, fric_coulomb=1.5)
        out = fb.friction_steps(arm, fb.BridgeConfig(tare_s=0.05), steps_m=steps, axes="x", hold_s=1.0,
                                clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        by = {(r["sign"], r["step_mm"]): r for r in out["rows"]}
        for sgn in (1, -1):
            self.assertAlmostEqual(by[(sgn, 2.0)]["moved_mm"], 0.0, delta=0.01)  # 1.1 N < 2 N breakaway
            self.assertIsNone(by[(sgn, 2.0)]["t63_ms"])
            self.assertAlmostEqual(by[(sgn, 4.0)]["moved_mm"], 4.0 - 1000 * 1.5 / 565, delta=0.05)
        stiff = fb.BridgeConfig(tare_s=0.05, stiffness=tuple(1.3 * k for k in fb.BridgeConfig().stiffness))
        arm = FakeArm(fric_static=2.0, fric_coulomb=1.5)
        out = fb.friction_steps(arm, stiff, steps_m=steps, axes="x", hold_s=1.0, clock=arm.clock, sleep=arm.sleep)
        r = next(r for r in out["rows"] if r["sign"] == 1 and r["step_mm"] == 4.0)
        self.assertAlmostEqual(r["moved_mm"], 4.0 - 1000 * 1.5 / (1.3 * 565), delta=0.05)
        arm = FakeArm()
        out = fb.friction_steps(arm, fb.BridgeConfig(tare_s=0.05), steps_m=(0.002,), axes="z", hold_s=1.0,
                                clock=arm.clock, sleep=arm.sleep)
        for r in out["rows"]:
            self.assertAlmostEqual(r["moved_mm"], 2.0, delta=0.01)
            self.assertAlmostEqual(r["t63_ms"], 100, delta=15)  # tau = 100 ms

    def test_joint_log_keeps_one_row_per_state(self):
        log = fb.JointLog()
        st = fb.RobotState(t=1.0, q=np.arange(7.0), dq=np.zeros(7), tool_pos=np.zeros(3), tool_R=np.eye(3),
                           force=np.zeros(3), tau_J=np.ones(7))
        log.add(0.0, st)
        log.add(0.005, st)  # the 200 Hz loop reads the same 100 Hz message twice
        self.assertEqual(len(log.rows), 1)
        row = log.rows[0]
        self.assertEqual(len(row), len(fb.JOINT_LOG_COLUMNS))
        self.assertEqual(row[1:8], list(range(7)))
        self.assertEqual(row[15:22], [1.0] * 7)
        self.assertEqual(row[22:], [None] * 7)  # no tau_ext from this source

    def test_friction_checks_log_joints(self):
        cfg = fb.BridgeConfig(tare_s=0.05)
        arm = FakeArm(fric_static=2.0, fric_coulomb=1.5)
        outs = [
            fb.friction_breakaway(arm, cfg, directions=("+x",), ramp_m_s=0.004, clock=arm.clock, sleep=arm.sleep),
            fb.friction_sweep(arm, cfg, axis="y", speeds_m_s=(0.02,), clock=arm.clock, sleep=arm.sleep),
            fb.friction_steps(arm, cfg, steps_m=(0.004,), axes="z", hold_s=0.5, clock=arm.clock, sleep=arm.sleep),
        ]
        for out in outs:
            self.assertEqual(out["joint_columns"], fb.JOINT_LOG_COLUMNS)
            for r in out["rows"]:
                self.assertGreater(len(r["joints"]), 10)
                self.assertTrue(all(len(j) == 29 for j in r["joints"]))
                t = [j[0] for j in r["joints"]]
                self.assertEqual(t, sorted(t))
        sweep = outs[1]["rows"][0]["joints"]
        self.assertGreater(sweep[-1][0], 2 * 0.03 / 0.02)  # covers both strokes, out and back

    def test_rate_test_reports_step_latency(self):
        arm = FakeArm(tau=0.05)
        out = fb.rate_test(arm, fb.BridgeConfig(), seconds=0.5, clock=arm.clock, sleep=arm.sleep)
        self.assertGreater(out["send_hz"], 150)
        self.assertLess(out["step_first_motion_ms"], 20)
        self.assertAlmostEqual(out["step_63pct_ms"], 50, delta=15)  # first-order lag with tau = 50 ms
        self.assertFalse(arm.streaming)

    def test_frankapy_read_sources_agree(self):
        # Raw end-effector pose O_T_EE and a non-identity tool delta. Tilted as well as yawed: any
        # gripper-straight-down rotation is a 180 deg turn about a horizontal axis, hence symmetric, so
        # with yaw alone a transposed rotation would go unnoticed.
        yaw, tilt = math.radians(30.0), math.radians(10.0)
        Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
        Rx = np.array([[1, 0, 0], [0, math.cos(tilt), -math.sin(tilt)], [0, math.sin(tilt), math.cos(tilt)]])
        B = np.eye(4)
        B[:3, :3] = fb.DOWN @ Rz @ Rx
        self.assertGreater(np.abs(B[:3, :3] - B[:3, :3].T).max(), 0.1)  # the test pose can catch a transpose
        B[:3, 3] = [0.3, -0.05, 0.48]
        D = np.eye(4)
        D[2, 3] = 0.01
        q, dq, ft = np.arange(7) * 0.1, np.ones(7) * 0.01, np.array([1.0, -0.3, 3.7, 0.0, 0.0, 0.0])

        class Pose:  # the parts of autolab_core.RigidTransform the bridge uses
            def __init__(self, T):
                self.matrix, self.translation, self.rotation = T, T[:3, 3], T[:3, :3]

        class StubFA:  # FrankaArm's getters, counting the ones that go to the Control PC
            def __init__(self):
                self.requests = 0
                self._tool_delta_pose = Pose(D)

            def _req(self, value):
                self.requests += 1
                return value

            def get_pose(self): return self._req(Pose(B @ D))  # FrankaPy applies the tool delta here
            def get_joints(self): return self._req(q)
            def get_joint_velocities(self): return self._req(dq)
            def get_ee_force_torque(self): return self._req(ft)
            def get_jacobian(self, q): return np.eye(6, 7)  # computed locally in FrankaPy

        class StubFACombined(StubFA):
            def get_robot_state(self):  # raw O_T_EE, no tool delta
                return self._req({"pose": Pose(B), "joints": q, "joint_velocities": dq, "ee_force_torque": ft})

        class Msg:  # franka_interface_msgs/RobotState: O_T_EE is column-major
            pass

        msg_in = Msg()
        msg_in.O_T_EE, msg_in.q, msg_in.dq, msg_in.O_F_ext_hat_K = tuple(B.T.reshape(-1)), tuple(q), tuple(dq), tuple(ft)
        tau_J, tau_ext = np.linspace(-3.0, 3.0, 7), np.linspace(0.1, 0.7, 7)
        msg_in.tau_J, msg_in.tau_ext_hat_filtered = tuple(tau_J), tuple(tau_ext)

        def make(fa, source):
            arm = fb.FrankaPyArm.__new__(fb.FrankaPyArm)  # skip __init__: it imports rospy and frankapy
            arm.fa = fa
            arm.has_jacobian = True
            arm.has_robot_state = hasattr(fa, "get_robot_state")
            arm._tool_delta = fa._tool_delta_pose.matrix
            arm.state_source, arm._latest = source, None
            return arm

        topic_arm = make(StubFACombined(), "topic")
        topic_arm._on_state(msg_in)
        states = {}
        for name, arm, want_requests in [("topic", topic_arm, 0),
                                         ("get_robot_state", make(StubFACombined(), "get_robot_state"), 1),
                                         ("separate calls", make(StubFA(), "separate calls"), 4)]:
            states[name] = arm.read()
            self.assertEqual(arm.fa.requests, want_requests, name)
        for name, st in states.items():
            np.testing.assert_allclose(st.tool_pos, (B @ D)[:3, 3], atol=1e-12, err_msg=name)
            np.testing.assert_allclose(st.tool_R, (B @ D)[:3, :3], atol=1e-12, err_msg=name)
            np.testing.assert_allclose(st.q, q, err_msg=name)
            np.testing.assert_allclose(st.dq, dq, err_msg=name)
            np.testing.assert_allclose(st.force, ft[:3], err_msg=name)
            self.assertEqual(st.J.shape, (6, 7))
        # Joint torques come only with the topic; the service paths leave them empty.
        np.testing.assert_allclose(states["topic"].tau_J, tau_J)
        np.testing.assert_allclose(states["topic"].tau_ext, tau_ext)
        for name in ("get_robot_state", "separate calls"):
            self.assertIsNone(states[name].tau_J, name)
            self.assertIsNone(states[name].tau_ext, name)
        del msg_in.tau_ext_hat_filtered  # an older message without the field still reads
        topic_arm._on_state(msg_in)
        self.assertIsNone(topic_arm.read().tau_ext)

        # Two reads of one message share its arrival time; a silent topic is a safety stop.
        self.assertEqual(topic_arm.read().t, topic_arm.read().t)
        msg, _ = topic_arm._latest
        topic_arm._latest = (msg, fb.time.monotonic() - 2 * fb.MAX_STATE_AGE_S)
        with self.assertRaises(fb.SafetyStop):
            topic_arm.read()

    def test_grasp_waits_for_the_fingers(self):
        # FrankaPy returns from goto_gripper after ~2 s while the fingers are still closing; the width
        # must be read only once they stop.
        clock = {"t": 0.0}

        class StubFA:
            def goto_gripper(self, **kw):
                self.t0, self.target = clock["t"], 0.0064  # the indented fingers stop at 6.4 mm on the peg

            def open_gripper(self, block=True):
                self.t0, self.target = clock["t"], 0.08

            def get_gripper_width(self):  # 72.5 mm at the start, 2 cm/s towards the target
                start = 0.0725 if self.target < 0.07 else 0.0064
                step = 0.02 * (clock["t"] - self.t0)
                return max(self.target, start - step) if self.target < start else min(self.target, start + step)

        arm = fb.FrankaPyArm.__new__(fb.FrankaPyArm)
        arm.fa = StubFA()
        fake_clock = lambda: clock["t"]  # noqa: E731

        def fake_sleep(dt):
            clock["t"] += dt

        def settled(**kw):
            return fb.FrankaPyArm.wait_gripper_settled(arm, clock=fake_clock, sleep=fake_sleep, **kw)

        arm.wait_gripper_settled = settled
        self.assertAlmostEqual(arm.grasp(), 0.0064)
        self.assertGreater(clock["t"], 3.3)  # (72.5 - 6.4) mm at 2 cm/s, then 0.5 s still
        self.assertAlmostEqual(arm.open(), 0.08)
        with self.assertRaises(fb.HardwareError):
            arm.fa.goto_gripper()
            settled(timeout_s=1.0)  # still closing after 1 s

    def _grasp_arm(self, stops_at):
        """FrankaPyArm whose fingers stop at `stops_at` (m) after closing; records goto_gripper's arguments."""

        class StubFA:
            def goto_gripper(self, **kw):
                self.kw = kw

            def get_gripper_width(self):
                return stops_at

        arm = fb.FrankaPyArm.__new__(fb.FrankaPyArm)
        arm.fa = StubFA()
        arm.wait_gripper_settled = lambda **kw: arm.fa.get_gripper_width()
        return arm

    def test_grasp_accepts_the_divots_down_to_3_5_mm(self):
        # The lab fingers' divots centre the 8 mm peg and close past it: 6.4 mm measured, ~4 mm possible.
        for width in (0.0064, 0.004, 0.0035, 0.010):
            arm = self._grasp_arm(width)
            self.assertAlmostEqual(arm.grasp(), width)
            kw = arm.fa.kw
            self.assertLessEqual(kw["width"] - kw["epsilon_inner"], 0.0035 + 1e-9)  # libfranka must accept it too
            self.assertAlmostEqual(kw["epsilon_outer"], 0.004)

    def test_grasp_refuses_a_missed_peg(self):
        for width in (0.0005, 0.003, 0.011):  # closed on nothing, or wider than the peg
            with self.assertRaises(fb.HardwareError, msg=width):
                self._grasp_arm(width).grasp()

    def test_grasp_force_default_and_limit(self):
        arm = self._grasp_arm(0.0064)
        arm.grasp()
        self.assertAlmostEqual(arm.fa.kw["force"], 60.0)  # 40 N let the peg slip in session 3
        arm.grasp(force=70.0)
        self.assertAlmostEqual(arm.fa.kw["force"], 70.0)
        for bad in (0.0, 71.0):
            with self.assertRaises(ValueError, msg=bad):
                arm.grasp(force=bad)

    @staticmethod
    def _register(seated, touch=None, measure_depth=True):
        """register_hole with a stub arm. seated: the 5 fingertips it reads after each seating guide (4 corners,
        then the main hole); touch: the fingertip after the depth-check guide."""
        clock, state, events, lines = {"t": 0.0}, {"guides": 0}, [], []
        R = np.diag([1.0, -1.0, -1.0])
        tips = {2 * k + 1: np.asarray(t, dtype=float) for k, t in enumerate(seated)}  # guides 1,3,..: seat; 2,4,..: lift
        if touch is not None:
            tips[11] = np.asarray(touch, dtype=float)

        class Arm:
            def guide(self, s, block=True):
                state["guides"] += 1
                events.append(("guide", s, block))
                self.until = clock["t"] + s

            def skill_done(self):
                return clock["t"] >= self.until

            def read(self):
                return types.SimpleNamespace(tool_pos=fb.tip_to_tool(tips[state["guides"]], R), tool_R=R)

        def sleep(dt):
            clock["t"] += dt

        hole = fb.register_hole(Arm(), confirm=lambda msg: events.append(("ask", msg)),
                                out=lambda *a, **k: lines.append(a[0] if a else ""), clock=lambda: clock["t"],
                                sleep=sleep, measure_depth=measure_depth)
        return hole, events, lines

    @staticmethod
    def _mount(centre, yaw_deg=12.0, corner_z=0.0288, main_z=0.0288, jitter=None, main_off=(0.0, 0.0)):
        """Seated fingertips for a mount centred at `centre`, rotated by yaw, corners in going-around order."""
        c, a = np.asarray(centre, dtype=float), math.radians(yaw_deg)
        h = fb.CORNER_PITCH_M / 2
        rot = np.array([[math.cos(a), -math.sin(a)], [math.sin(a), math.cos(a)]])
        corners = [np.r_[c + rot @ np.array(v) * h, corner_z] for v in ((1, 1), (-1, 1), (-1, -1), (1, -1))]
        if jitter is not None:
            corners = [p + np.r_[j, 0.0] for p, j in zip(corners, jitter)]
        return corners + [np.r_[c + np.asarray(main_off), main_z]]

    def test_register_hole_fits_the_main_hole_from_the_corners(self):
        centre = np.array([0.552, 0.004])
        jitter = [(0.00008, -0.00005), (-0.00006, 0.00008), (0.00005, 0.00007), (-0.00008, -0.00004)]  # in the play
        seated = self._mount(centre, jitter=jitter, main_off=(0.0004, -0.0002), main_z=0.0287)
        touch = np.r_[centre + [0.010, 0.0], 0.0287 + 0.0251]
        hole, events, lines = self._register(seated, touch)
        np.testing.assert_allclose(hole.seated_tip[:2], centre, atol=0.00005)  # corners pin it, not the 9 mm hole
        self.assertAlmostEqual(hole.seated_tip[2], 0.0287)  # z from the main-hole insertion
        self.assertLess(hole.square_error_m, 0.0002)
        self.assertAlmostEqual(hole.centre_offset_m, math.hypot(0.0004, 0.0002), delta=0.00005)
        self.assertAlmostEqual(hole.depth_m, 0.0251, places=6)
        self.assertFalse(any("WARNING" in ln for ln in lines), lines)
        guides = [e[1] for e in events if e[0] == "guide"]
        self.assertEqual(guides, [15.0, 10.0] * 5 + [10.0, 10.0])  # 4 corners + main, each seat and lift; touch, lift
        self.assertTrue(all(e[2] is False for e in events if e[0] == "guide"))  # non-blocking: we count down
        asks = [e[1] for e in events if e[0] == "ask"]
        self.assertEqual(sum("corner hole" in a for a in asks), 4)
        self.assertEqual(sum("main (middle) hole" in a for a in asks), 1)
        self.assertTrue(any(" 15 s left" in ln for ln in lines if "ARM COMPLIANT" in ln))
        self.assertEqual(sum("ARM STIFF again" in ln for ln in lines), 12)
        with tempfile.TemporaryDirectory() as d:  # survives save/load; old files without corners still load
            hole.save(Path(d) / "hole.json")
            back = fb.Hole.load(Path(d) / "hole.json")
            self.assertAlmostEqual(back.square_error_m, hole.square_error_m)
            self.assertAlmostEqual(back.depth_m, 0.0251, places=6)
            (Path(d) / "old.json").write_text(json.dumps({"seated_tip": [0.5, 0.0, 0.03], "samples": []}))
            old = fb.Hole.load(Path(d) / "old.json")
            self.assertIsNone(old.depth_m)
            self.assertIsNone(old.square_error_m)

    def test_register_hole_warns_on_a_bad_corner_or_main_hole(self):
        centre = np.array([0.552, 0.004])
        _, _, lines = self._register(self._mount(centre, jitter=[(0.001, 0.0), (0, 0), (0, 0), (0, 0)]),
                                     measure_depth=False)  # a corner 1 mm out: not seated / wrong hole
        self.assertTrue(any("WARNING: the corners don't form" in ln for ln in lines))
        _, _, lines = self._register(self._mount(centre, main_off=(0.0008, 0.0)), measure_depth=False)
        self.assertTrue(any("WARNING: the main hole isn't where" in ln for ln in lines))

    def test_register_hole_warns_on_wrong_depth_or_touch(self):
        centre = np.array([0.552, 0.004])
        _, _, lines = self._register(self._mount(centre), np.r_[centre + [0.010, 0.0], 0.0288 + 0.0235])
        self.assertTrue(any("WARNING: depth is -1.5 mm" in ln for ln in lines))
        _, _, lines = self._register(self._mount(centre), np.r_[centre + [0.004, 0.0], 0.0288])  # into the bore
        self.assertTrue(any("WARNING: touch should be" in ln for ln in lines))

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


class TestFrictionWalk(unittest.TestCase):
    SPEC = dict(seed=3, duration_s=40.0)

    def walk_arm(self, **kw):
        arm = FakeArm(**kw)
        arm.goto(fb.tip_to_tool(np.array([0.5, 0.05, 0.2]), fb.DOWN), fb.DOWN, 0.1)  # inside the box
        return arm

    def test_trajectory_is_seeded_and_stays_in_the_box(self):
        spec = fb.WalkSpec(**self.SPEC)
        a, b = fb.walk_trajectory(spec), fb.walk_trajectory(fb.WalkSpec(**self.SPEC))
        np.testing.assert_array_equal(a["target"], b["target"])
        c = fb.walk_trajectory(fb.WalkSpec(seed=4, duration_s=40.0))
        self.assertFalse(np.allclose(a["target"][: len(c["target"])][-100:], c["target"][: len(a["target"])][-100:]))
        tgt = a["target"]
        self.assertTrue((tgt >= np.array(spec.box_lo) - 1e-12).all() and (tgt <= np.array(spec.box_hi) + 1e-12).all())
        np.testing.assert_allclose(tgt[0], spec.centre)
        self.assertGreaterEqual(a["t"][-1], spec.duration_s - spec.dt)
        self.assertEqual(set(a["kind"].tolist()), set(fb.WALK_KINDS.values()))
        moving = a["kind"] == fb.WALK_KINDS["move"]
        speed = np.linalg.norm(np.diff(tgt, axis=0), axis=1) / spec.dt
        self.assertLessEqual(speed[moving[1:] & moving[:-1]].max(), spec.speed_mm_s[1] / 1000 + 1e-6)
        jumps = speed[a["kind"][1:] == fb.WALK_KINDS["step"]] * spec.dt
        self.assertLessEqual(jumps.max(), spec.step_mm[1] / 1000 + 1e-9)
        long = fb.walk_trajectory(fb.WalkSpec(seed=0, duration_s=600.0))["target"]
        self.assertLess(np.abs(long.mean(axis=0) - np.array(spec.centre)).max(), 0.04)  # pulled toward the centre

    def test_spec_refuses_boxes_beyond_the_table(self):
        with self.assertRaises(ValueError):
            fb.WalkSpec(box_lo=(0.40, -0.30, 0.12)).check()  # into the shelves
        with self.assertRaises(ValueError):
            fb.WalkSpec(box_lo=(0.40, -0.15, 0.02)).check()  # down to the table
        with self.assertRaises(ValueError):
            fb.WalkSpec(centre=(0.30, 0.0, 0.14)).check()  # centre outside the box

    def test_walk_recovers_the_sim_leak_model(self):
        arm = self.walk_arm(vel_leak=(1.5, 0.002))
        out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(**self.SPEC),
                               clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        self.assertIsNone(out["stopped"])
        self.assertAlmostEqual(out["seconds_done"], out["planned_s"], delta=0.05)
        self.assertEqual(len(out["series"][0]), len(fb.WALK_COLUMNS))
        self.assertEqual(len(out["joints"]), len(out["series"]))
        self.assertEqual(out["fit"]["v_s_mm_s"], 2.0)
        for ax, r in out["fit"]["axes"].items():
            for side in ("leak_plus_N", "leak_minus_N"):
                self.assertAlmostEqual(r[side], 1.5, delta=0.1, msg=f"{ax} {side}")
            self.assertLess(r["rms_N"], 0.3 * r["rms_no_friction_N"])

    def test_walk_fit_uses_the_force_sign(self):
        arm = self.walk_arm(vel_leak=(1.5, 0.002))
        out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(**self.SPEC),
                               clock=arm.clock, sleep=arm.sleep)
        flipped = [r[:7] + [-x for x in r[7:10]] + r[10:] for r in out["series"]]  # a robot whose raw sign is -1
        fit = fb.fit_walk_leak(flipped, force_sign=-1.0)
        for ax in "xyz":
            self.assertAlmostEqual(fit["axes"][ax]["leak_plus_N"], out["fit"]["axes"][ax]["leak_plus_N"], places=6)
        self.assertLess(fb.fit_walk_leak(flipped)["axes"]["x"]["leak_plus_N"], -1.0)  # wrong sign: reads as negative

    def test_walk_with_stiction_reads_a_smaller_leak(self):
        # Coulomb friction that also holds a force at rest after the hand sticks (0.8 x 1.5 = 1.2 N while
        # moving, up to that at rest). The tanh model has no term for the force at rest: over 3 min seeds 0-3
        # give 0.9-1.6 N per side, and the two sides of this symmetric fake differ by up to 0.5 N. The robot
        # may do this too, so compare repeats before reading a +/- difference as real.
        arm = self.walk_arm(fric_coulomb=1.5, fric_leak=0.8)
        out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(seed=3, duration_s=180.0),
                               clock=arm.clock, sleep=arm.sleep)
        for ax, r in out["fit"]["axes"].items():
            for side in ("leak_plus_N", "leak_minus_N"):
                self.assertTrue(0.7 < r[side] < 1.7, msg=f"{ax} {side}: {r[side]}")
            self.assertLess(r["rms_N"], r["rms_no_friction_N"])

    def test_walk_without_friction_finds_pose_error_only(self):
        gain = np.zeros((3, 3))
        gain[2, 0] = 10.0  # 0.1 N per cm of x shows up in z
        arm = self.walk_arm(pos_gain=gain)
        out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(**self.SPEC),
                               clock=arm.clock, sleep=arm.sleep)
        for ax, r in out["fit"]["axes"].items():
            self.assertLess(abs(r["leak_plus_N"]) + abs(r["leak_minus_N"]), 0.1, msg=ax)
        self.assertAlmostEqual(out["fit"]["axes"]["z"]["pose_N_per_cm"][0], 0.1, delta=0.01)

    def test_walk_stops_on_contact_and_keeps_its_data(self):
        arm = self.walk_arm(spike_at=9.0)  # the walk starts after ~6.7 s: two gotos + tare
        out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(**self.SPEC),
                               clock=arm.clock, sleep=arm.sleep)
        self.assertFalse(arm.streaming)
        self.assertIn("force", out["stopped"])
        self.assertGreater(len(out["series"]), 100)
        self.assertLess(out["seconds_done"], 10.0)

    def test_walk_refuses_to_start_outside_the_box(self):
        arm = FakeArm()  # starts 4.7 cm above the rim, below the box
        with self.assertRaises(fb.HardwareError):
            fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(**self.SPEC), clock=arm.clock, sleep=arm.sleep)
        self.assertEqual(arm.streams, 0)
        tilted = np.array([[1.0, 0.0, 0.0], [0.0, -math.cos(0.35), -math.sin(0.35)], [0.0, math.sin(0.35), -math.cos(0.35)]])
        arm = self.walk_arm()
        arm.R = tilted @ np.eye(3)  # 20 deg from gripper-down
        with self.assertRaises(fb.HardwareError):
            fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), fb.WalkSpec(**self.SPEC), clock=arm.clock, sleep=arm.sleep)
        self.assertEqual(arm.streams, 0)

    def test_walk_starts_from_home(self):
        arm = FakeArm()
        home_tip = np.array([0.31, 0.0, 0.48])  # FrankaPy's reset_joints pose
        arm.goto(fb.tip_to_tool(home_tip, fb.DOWN), fb.DOWN, 0.1)
        gotos = []
        goto = arm.goto
        arm.goto = lambda pos, R, duration, stiffness=None: (gotos.append(fb.tool_to_tip(pos, R)), goto(pos, R, duration))
        spec = fb.WalkSpec(seed=0, duration_s=2.0)
        out = fb.friction_walk(arm, fb.BridgeConfig(tare_s=0.05), spec, clock=arm.clock, sleep=arm.sleep)
        np.testing.assert_allclose(gotos[0], [spec.centre[0], spec.centre[1], 0.48], atol=1e-9)  # across, at home height
        np.testing.assert_allclose(gotos[1], spec.centre, atol=1e-9)  # then down
        self.assertEqual(len(out["approach"]), 2)
        self.assertIsNone(out["stopped"])
        self.assertIsNone(fb.walk_approach([0.31, 0.30, 0.48], spec))  # beside the shelves
        self.assertIsNone(fb.walk_approach([0.31, 0.0, 0.70], spec))  # above the corridor
        self.assertEqual(len(fb.walk_approach(spec.centre, spec)), 1)

    def test_confirm_is_yes_no_with_no_by_default(self):
        import run_peg

        def asker(*answers):
            it = iter(answers)
            asked = []

            def ask(prompt):
                asked.append(prompt)
                a = next(it)
                if a is EOFError:
                    raise EOFError
                return a
            return ask, asked

        for yes in ("y", "Y", "yes", " YES "):
            ask, asked = asker(yes)
            run_peg.confirm("Path clear?", ask)
            self.assertEqual(asked, ["Path clear? [y/N] "])
        for no in ("", "n", "No", EOFError):
            with self.assertRaises(fb.HardwareError):
                run_peg.confirm("Path clear?", asker(no)[0])
        ask, asked = asker("maybe", "yse", "y")  # invalid answers ask again
        run_peg.confirm("Path clear?", ask)
        self.assertEqual(len(asked), 3)

    def test_save_check_keeps_time_series_out_of_calibration(self):
        import tempfile
        from pathlib import Path

        import run_peg

        with tempfile.TemporaryDirectory() as d:
            run_peg.save_check(d, "friction_walk_s0", {"fit": {"v_s_mm_s": 2.0}, "series": [[0.0]], "joints": [[0.0]],
                                                       "hold_series": [[1.0]], "rows": [1]})
            calib = json.loads((Path(d) / "calibration.json").read_text())["friction_walk_s0"]
            self.assertEqual(set(calib), {"fit"})
            raw = json.loads(next(Path(d).glob("friction_walk_s0_*.json")).read_text())
            self.assertIn("series", raw)


if __name__ == "__main__":
    unittest.main()
