"""Checks for deploy/policy_runner.py that need only numpy: python -m unittest discover tests

These check internal consistency. Agreement with the real ForgeEnv is checked separately by
scripts/sim_bridge_test.py on the training PC.
"""

import math
import os
import sys
import unittest

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))

from policy_runner import (  # noqa: E402
    ForceFilter,
    ForgePolicyRunner,
    RunnerConfig,
    get_euler_xyz,
    quat_from_euler_xyz,
)

FORGE_OBS_ORDER = [
    "fingertip_pos_rel_fixed",
    "fingertip_quat",
    "ee_linvel",
    "ee_angvel",
    "ft_force",
    "force_threshold",
]


def make_cfg(**kw):
    d = dict(
        obs_order=FORGE_OBS_ORDER,
        pos_action_bounds=[0.05, 0.05, 0.05],
        pos_action_threshold=[0.02, 0.02, 0.02],
        rot_action_threshold=[0.097, 0.097, 0.097],
        ema_factor=0.0625,
        force_threshold=7.5,
        ft_smoothing_factor=0.25,
        sim_dt=1 / 120,
        policy_dt=8 / 120,
        rnn_layers=2,
        rnn_units=1024,
    )
    d.update(kw)
    return RunnerConfig(**d)


def hand_down(yaw):
    return quat_from_euler_xyz(math.pi, 0.0, yaw)


def yaw_diff(a, b):
    return (a - b + math.pi) % (2 * math.pi) - math.pi


class TestRotations(unittest.TestCase):
    def test_euler_roundtrip(self):
        for r, p, y in [(0.1, 0.2, 0.3), (math.pi, 0.0, 1.0), (3.0, -0.4, -2.0)]:
            r2, p2, y2 = get_euler_xyz(quat_from_euler_xyz(r, p, y))
            q1, q2 = quat_from_euler_xyz(r, p, y), quat_from_euler_xyz(r2, p2, y2)
            self.assertAlmostEqual(abs(np.dot(q1, q2)), 1.0, places=9)
            for v in (r2, p2, y2):
                self.assertTrue(0.0 <= v <= 2 * math.pi)  # -1e-17 % 2pi rounds to exactly 2pi


class TestRunner(unittest.TestCase):
    def setUp(self):
        self.runner = ForgePolicyRunner(make_cfg())
        self.slot = np.array([0.6, 0.0, 0.07])

    def test_obs_layout(self):
        pos, quat = self.slot + [0.01, -0.02, 0.03], hand_down(0.3)
        self.runner.reset(pos, quat, self.slot)
        obs = self.runner.build_obs(pos, quat, [0.1, 0.2, 0.3], [0.4, 0.5, 0.6], [1.0, 2.0, 3.0])
        self.assertEqual(obs.shape, (24,))  # 3+4+3+3+3+1 + 7 prev actions
        np.testing.assert_allclose(obs[0:3], [0.01, -0.02, 0.03], atol=1e-6)
        self.assertEqual(obs[3], 0.0)  # quat w zeroed
        self.assertEqual(obs[6], 0.0)  # quat z zeroed
        np.testing.assert_allclose(obs[7:10], [0.1, 0.2, 0.3], atol=1e-6)
        np.testing.assert_allclose(obs[10:13], [0.0, 0.0, 0.6], atol=1e-6)  # angvel z only
        np.testing.assert_allclose(obs[13:16], [1.0, 2.0, 3.0], atol=1e-6)
        self.assertAlmostEqual(obs[16], 7.5, places=5)
        np.testing.assert_allclose(obs[17:20], [0.2, -0.4, 0.6], atol=1e-6)  # reset actions
        self.assertEqual(obs[20:22].tolist(), [0.0, 0.0])  # roll/pitch actions zeroed
        self.assertEqual(obs[23], -1.0)  # success prediction starts at "no"

    def test_holding_reset_action_keeps_pose(self):
        """Re-issuing the reset action must command the current pose (what FORGE's reset intends)."""
        # Yaw action covers hand yaw in about [-90, 180] deg (joint-7 range); stay inside it.
        for yaw in [-1.5, -0.5, 0.0, 0.7, 1.5, 2.5, 3.1]:
            pos, quat = self.slot + [0.01, 0.005, 0.02], hand_down(yaw)
            self.runner.reset(pos, quat, self.slot)
            target_pos, target_quat = self.runner.apply_action(self.runner.actions.copy(), pos, quat)
            np.testing.assert_allclose(target_pos, pos, atol=1e-9)
            r, p, y = get_euler_xyz(target_quat)
            self.assertAlmostEqual(abs(yaw_diff(y, get_euler_xyz(quat)[2])), 0.0, places=6, msg=f"yaw={yaw}")
            self.assertAlmostEqual(abs(yaw_diff(r, math.pi)), 0.0, places=6)
            self.assertAlmostEqual(abs(yaw_diff(p, 0.0)), 0.0, places=6)

    def test_step_is_clipped(self):
        pos, quat = self.slot + [0.0, 0.0, 0.04], hand_down(0.0)
        cfg = make_cfg(ema_factor=1.0)  # no smoothing so the raw action reaches the target
        runner = ForgePolicyRunner(cfg)
        runner.reset(pos, quat, self.slot)
        mu = np.array([1.0, -1.0, -1.0, 0.0, 0.0, runner.actions[5], -1.0])
        target_pos, _ = runner.apply_action(mu, pos, quat)
        np.testing.assert_allclose(target_pos - pos, [0.02, -0.02, -0.02], atol=1e-9)

    def test_target_follows_pose_between_steps(self):
        """target() re-clips around each new pose (ForgeEnv does this every physics step)."""
        cfg = make_cfg(ema_factor=1.0)
        runner = ForgePolicyRunner(cfg)
        pos, quat = self.slot + [0.0, 0.0, 0.04], hand_down(0.0)
        runner.reset(pos, quat, self.slot)
        runner.update_actions(np.array([0.0, 0.0, -1.0, 0.0, 0.0, runner.actions[5], -1.0]))
        actions = runner.actions.copy()
        t0, _ = runner.target(pos, quat)
        moved = pos + [0.0, 0.0, -0.005]
        t1, _ = runner.target(moved, quat)
        np.testing.assert_allclose(t0 - pos, [0.0, 0.0, -0.02], atol=1e-9)
        np.testing.assert_allclose(t1 - moved, [0.0, 0.0, -0.02], atol=1e-9)  # clip moves with the hand
        np.testing.assert_array_equal(runner.actions, actions)  # target() doesn't touch the actions
        np.testing.assert_allclose(runner.target(moved, quat)[0], t1)

    def test_hold_vertical_aims_straight_down(self):
        """ForgeEnv's rule moves at most rot_action_threshold from the current tilt; hold_vertical aims at down."""
        pos = self.slot + [0.0, 0.0, 0.04]
        tilted = quat_from_euler_xyz(math.pi + 0.5, 0.0, 0.0)  # ~29 deg off vertical
        for hold, want in ((False, 0.5 - 0.097), (True, 0.0)):
            runner = ForgePolicyRunner(make_cfg(ema_factor=1.0, hold_vertical=hold))
            runner.reset(pos, tilted, self.slot)
            _, q = runner.target(pos, tilted)
            roll, pitch, _ = get_euler_xyz(q)
            self.assertAlmostEqual(abs(roll - math.pi), want, delta=1e-6, msg=hold)
            self.assertAlmostEqual(math.remainder(pitch, 2 * math.pi), 0.0, delta=1e-6, msg=hold)

    def test_ema(self):
        pos, quat = self.slot + [0.0, 0.0, 0.02], hand_down(0.0)
        self.runner.reset(pos, quat, self.slot)
        before = self.runner.actions.copy()
        self.runner.apply_action(np.ones(7), pos, quat)
        np.testing.assert_allclose(self.runner.actions, 0.0625 * 1.0 + 0.9375 * before)


class TestForceFilter(unittest.TestCase):
    def test_rate_invariance(self):
        sim = ForceFilter(0.25, 1 / 120)
        for _ in range(8):
            sim.update([10.0, 0.0, -5.0], 1 / 120)
        robot = ForceFilter(0.25, 1 / 120)
        for _ in range(67):  # ~1 kHz for one 15 Hz step
            robot.update([10.0, 0.0, -5.0], 1 / 1000 * (8 / 120) / (67 / 1000))
        np.testing.assert_allclose(sim.value, robot.value, rtol=1e-9)


if __name__ == "__main__":
    unittest.main()
