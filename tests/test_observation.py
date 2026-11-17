"""Checks for the changes to ForgeEnv's observations. Needs torch and gymnasium, not Isaac Sim.

The upstream classes are replaced by a stand-in that copies their history-buffer statements, so
these tests cover the order of updates. scripts/preflight.py checks the real env on the training PC.
"""

import importlib
import math
import os
import sys
import types
import unittest
from unittest import mock

try:
    import torch
except ImportError:
    torch = None

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

DT = 1.0 / 120.0
DOWN = (0.0, 1.0, 0.0, 0.0)  # hand pointing down at yaw 0; w and z are already zero, like ForgeEnv


def yaw_quat(yaw):
    # DOWN rotated about world z by yaw.
    return torch.tensor([[0.0, math.cos(yaw / 2), math.sin(yaw / 2), 0.0]])


def make_upstream_stub():
    class ForgeEnv:
        """The bookkeeping from FactoryEnv and ForgeEnv._compute_intermediate_values @ v2.3.2."""

        def __init__(self):
            self.prev_fingertip_pos = torch.zeros(1, 3)
            self.true_pos = torch.zeros(1, 3)
            self.quat = torch.tensor([DOWN])
            self.noise = torch.zeros(1, 3)

        def _compute_intermediate_values(self, dt):
            self.prev_fingertip_pos = self.true_pos.clone()  # FactoryEnv
            self.noisy_fingertip_pos = self.true_pos + self.noise  # ForgeEnv
            self.noisy_fingertip_quat = self.quat.clone()
            self.ee_linvel_fd = (self.noisy_fingertip_pos - self.prev_fingertip_pos) / dt
            self.prev_fingertip_pos = self.noisy_fingertip_pos.clone()
            self.ee_angvel_fd = torch.zeros(1, 3)

        def generate_ctrl_signals(
            self, ctrl_target_fingertip_midpoint_pos, ctrl_target_fingertip_midpoint_quat, ctrl_target_gripper_dof_pos
        ):
            self.ctrl_calls = getattr(self, "ctrl_calls", 0) + 1
            # FactoryEnv: the controller's wrench and torques, then set_joint_effort_target.
            self.applied_wrench = getattr(self, "stub_wrench", torch.zeros(1, 6))
            self.joint_torque = torch.zeros(1, 9)
            if hasattr(self, "_robot"):
                self._robot.set_joint_effort_target(self.joint_torque)

        def randomize_initial_state(self, env_ids):
            if hasattr(self, "_robot"):  # Factory moves the hand to its start pose with the controller
                self.generate_ctrl_signals(self.true_pos, self.quat, 0.0)
            self.init_fixed_pos_obs_noise = torch.full((1, 3), 0.0025)
            self.ee_linvel_fd = torch.zeros(1, 3)
            self.ee_angvel_fd = torch.zeros(1, 3)

    return ForgeEnv


@unittest.skipIf(torch is None, "needs torch")
class TestForgeBlocksEnv(unittest.TestCase):
    def setUp(self):
        upstream = types.ModuleType("isaaclab_tasks.direct.forge.forge_env")
        upstream.ForgeEnv = make_upstream_stub()
        self.upstream = upstream.ForgeEnv
        names = ["isaaclab_tasks", "isaaclab_tasks.direct", "isaaclab_tasks.direct.forge"]
        modules = {n: types.ModuleType(n) for n in names}
        modules[upstream.__name__] = upstream
        patcher = mock.patch.dict(sys.modules, modules)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("forge_blocks.tasks.block_env", None)
        self.addCleanup(sys.modules.pop, "forge_blocks.tasks.block_env", None)
        block_env = importlib.import_module("forge_blocks.tasks.block_env")

        class Env(block_env.ForgeBlocksEnv):
            device = "cpu"
            cfg = types.SimpleNamespace(obs_rand=types.SimpleNamespace(error_radius_mode="gaussian"))

        self.env = Env()

    def test_upstream_stub_shows_the_bug(self):
        env = self.upstream()
        env._compute_intermediate_values(DT)
        env.true_pos = torch.tensor([[0.001, 0.0, 0.0]])
        env._compute_intermediate_values(DT)
        self.assertEqual(env.ee_linvel_fd.abs().max().item(), 0.0)

    def test_linvel_follows_motion(self):
        env = self.env
        env._compute_intermediate_values(DT)
        for step in range(1, 4):
            env.true_pos = torch.tensor([[step * 0.001, 0.0, 0.0]])
            env._compute_intermediate_values(DT)
            self.assertAlmostEqual(env.ee_linvel_fd[0, 0].item(), 0.001 / DT, places=3)

    def test_linvel_includes_noise_from_both_samples(self):
        env = self.env
        env.noise = torch.tensor([[0.0002, 0.0, 0.0]])
        env._compute_intermediate_values(DT)
        env.noise = torch.tensor([[-0.0001, 0.0, 0.0]])
        env.true_pos = torch.tensor([[0.001, 0.0, 0.0]])
        env._compute_intermediate_values(DT)
        self.assertAlmostEqual(env.ee_linvel_fd[0, 0].item(), (0.001 - 0.0003) / DT, places=3)

    def test_angvel_follows_yaw_and_keeps_only_z(self):
        env = self.env
        env.quat = yaw_quat(0.1)
        env._compute_intermediate_values(DT)
        env.quat = yaw_quat(0.11)
        env._compute_intermediate_values(DT)
        self.assertAlmostEqual(env.ee_angvel_fd[0, 2].item(), 0.01 / DT, places=2)
        self.assertEqual(env.ee_angvel_fd[0, :2].abs().max().item(), 0.0)

    def test_angvel_ignores_quaternion_sign_flip(self):
        env = self.env
        env.quat = yaw_quat(0.1)
        env._compute_intermediate_values(DT)
        env.quat = -yaw_quat(0.11)  # ForgeEnv flips the sign per episode
        env._compute_intermediate_values(DT)
        self.assertAlmostEqual(env.ee_angvel_fd[0, 2].item(), 0.01 / DT, places=2)

    def test_reset_does_not_difference_across_the_teleport(self):
        env = self.env
        env._compute_intermediate_values(DT)
        env.true_pos = torch.tensor([[0.3, 0.0, 0.0]])  # the reset moves the hand
        env._compute_intermediate_values(DT)
        env.randomize_initial_state(torch.tensor([0]))
        env._compute_intermediate_values(DT)
        self.assertEqual(env.ee_linvel_fd.abs().max().item(), 0.0)

    def test_gaussian_mode_keeps_upstream_noise(self):
        env = self.env
        env._compute_intermediate_values(DT)
        env.randomize_initial_state(torch.tensor([0]))
        self.assertTrue(torch.equal(env.init_fixed_pos_obs_noise, torch.full((1, 3), 0.0025)))

    def test_radius_modes_overwrite_the_right_axes(self):
        env = self.env
        env.cfg.obs_rand.error_radius = [0.01, 0.01]
        env._compute_intermediate_values(DT)

        env.cfg.obs_rand.error_radius_mode = "xy"
        env.randomize_initial_state(torch.tensor([0]))
        noise = env.init_fixed_pos_obs_noise[0]
        self.assertAlmostEqual(torch.linalg.norm(noise[:2]).item(), 0.01, places=6)
        self.assertAlmostEqual(noise[2].item(), 0.0025, places=6)

        env.cfg.obs_rand.error_radius_mode = "xyz"
        env.randomize_initial_state(torch.tensor([0]))
        self.assertAlmostEqual(torch.linalg.norm(env.init_fixed_pos_obs_noise[0]).item(), 0.01, places=6)

    def _leak_env(self, leak_N, alpha=1.0):
        env = self.env
        env.num_envs = 1
        env.cfg.ft_smoothing_factor = alpha
        env.cfg.obs_rand.ft_force_leak_N = leak_N
        env.cfg.obs_rand.ft_force_leak_v_mps = 0.002
        env.fingertip_midpoint_linvel = torch.zeros(1, 3)
        env.noisy_force = torch.zeros(1, 3)
        orig = env._compute_intermediate_values

        def compute(dt):  # the stub has no force pipeline: start each call from a zero sensed force
            env.noisy_force = torch.zeros(1, 3)
            orig(dt)

        env._compute_intermediate_values = compute
        compute(DT)  # randomize_initial_state needs a noisy pose to remember
        return env

    def test_leak_is_off_by_default(self):
        env = self._leak_env([0.0, 0.0])
        env.randomize_initial_state(torch.tensor([0]))
        env.fingertip_midpoint_linvel = torch.tensor([[0.0, 0.0, -0.01]])
        env._compute_intermediate_values(DT)
        self.assertEqual(env.noisy_force.abs().max().item(), 0.0)

    def test_leak_opposes_motion_and_vanishes_at_rest(self):
        env = self._leak_env([2.0, 2.0])
        env.randomize_initial_state(torch.tensor([0]))
        env.fingertip_midpoint_linvel = torch.tensor([[0.02, 0.0, -0.02]])  # 20 mm/s: tanh(10) ~ 1
        env._compute_intermediate_values(DT)
        torch.testing.assert_close(env.noisy_force, torch.tensor([[-2.0, 0.0, 2.0]]), atol=1e-4, rtol=0)
        env.fingertip_midpoint_linvel = torch.zeros(1, 3)
        env._compute_intermediate_values(DT)
        self.assertEqual(env.noisy_force.abs().max().item(), 0.0)

    def test_leak_is_smoothed_and_reset(self):
        env = self._leak_env([2.0, 2.0], alpha=0.25)
        env.randomize_initial_state(torch.tensor([0]))
        env.fingertip_midpoint_linvel = torch.tensor([[0.0, 0.0, -0.02]])
        env._compute_intermediate_values(DT)
        self.assertAlmostEqual(env.noisy_force[0, 2].item(), 0.5, places=4)  # 0.25 * 2 N
        for _ in range(60):
            env._compute_intermediate_values(DT)
        self.assertAlmostEqual(env.noisy_force[0, 2].item(), 2.0, places=3)
        env.randomize_initial_state(torch.tensor([0]))
        self.assertEqual(env._leak_smooth.abs().max().item(), 0.0)

    def test_leak_size_is_drawn_per_axis_in_range(self):
        env = self._leak_env([1.0, 3.0])
        sizes = []
        for _ in range(200):
            env.randomize_initial_state(torch.tensor([0]))
            sizes.append(env._leak_N.clone())
        sizes = torch.cat(sizes)
        self.assertGreaterEqual(sizes.min().item(), 1.0)
        self.assertLessEqual(sizes.max().item(), 3.0)
        self.assertAlmostEqual(sizes.mean().item(), 2.0, delta=0.1)
        self.assertFalse(torch.equal(sizes[:, 0], sizes[:, 1]))  # independent axes

    def _pull_env(self, alpha, offset=(0.0, 0.0), ema=1.0):
        env = self._leak_env([0.0, 0.0], alpha=ema)
        env.cfg.obs_rand.ft_pull_leak_alpha = list(alpha)
        env.cfg.obs_rand.ft_pull_leak_offset_N = list(offset)
        env.task_prop_gains = torch.tensor([[565.0, 565.0, 565.0, 28.0, 28.0, 28.0]])
        env.fingertip_midpoint_pos = torch.zeros(1, 3)
        return env

    def _command(self, env, target):
        env.generate_ctrl_signals(
            ctrl_target_fingertip_midpoint_pos=torch.tensor([target]),
            ctrl_target_fingertip_midpoint_quat=torch.tensor([DOWN]),
            ctrl_target_gripper_dof_pos=0.0,
        )

    def test_pull_leak_is_off_by_default(self):
        env = self._pull_env([0.0, 0.0])
        env.randomize_initial_state(torch.tensor([0]))
        self._command(env, [0.01, 0.0, 0.0])
        env._compute_intermediate_values(DT)
        self.assertEqual(env.noisy_force.abs().max().item(), 0.0)
        self.assertEqual(env.ctrl_calls, 1)  # still reaches the upstream controller

    def test_pull_leak_opposes_the_pull(self):
        env = self._pull_env([0.4, 0.4])
        env.randomize_initial_state(torch.tensor([0]))
        self._command(env, [0.004, 0.0, -0.002])  # hand stuck at 0: pull 2.26 N in +x, -1.13 N in z
        env._compute_intermediate_values(DT)
        torch.testing.assert_close(env.controller_pull(), torch.tensor([[2.26, 0.0, -1.13]]), atol=1e-4, rtol=0)
        torch.testing.assert_close(env.noisy_force, torch.tensor([[-0.904, 0.0, 0.452]]), atol=1e-4, rtol=0)
        env.fingertip_midpoint_pos = torch.tensor([[0.004, 0.0, -0.002]])  # the hand catches up
        env._compute_intermediate_values(DT)
        self.assertLess(env.noisy_force.abs().max().item(), 1e-6)

    def test_pull_uses_each_episodes_gains(self):
        env = self._pull_env([0.5, 0.5])
        env.randomize_initial_state(torch.tensor([0]))
        env.task_prop_gains = torch.tensor([[300.0, 600.0, 900.0, 28.0, 28.0, 28.0]])
        self._command(env, [0.01, 0.01, 0.01])
        torch.testing.assert_close(env.controller_pull(), torch.tensor([[3.0, 6.0, 9.0]]), atol=1e-4, rtol=0)

    def test_reset_clears_the_pull(self):
        env = self._pull_env([0.4, 0.4])
        env.randomize_initial_state(torch.tensor([0]))
        self._command(env, [0.3, 0.0, 0.0])  # e.g. the reset's own motion
        env.randomize_initial_state(torch.tensor([0]))
        env._compute_intermediate_values(DT)
        self.assertEqual(env.controller_pull().abs().max().item(), 0.0)
        self.assertEqual(env.noisy_force.abs().max().item(), 0.0)

    def test_pull_leak_offset_and_smoothing(self):
        env = self._pull_env([0.0, 0.0], offset=(-1.0, -1.0), ema=0.25)
        env.randomize_initial_state(torch.tensor([0]))
        env._compute_intermediate_values(DT)
        torch.testing.assert_close(env.noisy_force, torch.full((1, 3), -0.25), atol=1e-6, rtol=0)
        for _ in range(60):
            env._compute_intermediate_values(DT)
        torch.testing.assert_close(env.noisy_force, torch.full((1, 3), -1.0), atol=1e-4, rtol=0)
        env.randomize_initial_state(torch.tensor([0]))
        self.assertEqual(env._leak_smooth.abs().max().item(), 0.0)

    def test_pull_leak_draws_in_range_per_axis(self):
        env = self._pull_env([0.2, 0.45], offset=(-1.5, 0.0))
        alphas, offsets = [], []
        for _ in range(200):
            env.randomize_initial_state(torch.tensor([0]))
            alphas.append(env._pull_alpha.clone())
            offsets.append(env._leak_offset.clone())
        alphas, offsets = torch.cat(alphas), torch.cat(offsets)
        self.assertGreaterEqual(alphas.min().item(), 0.2)
        self.assertLessEqual(alphas.max().item(), 0.45)
        self.assertGreaterEqual(offsets.min().item(), -1.5)
        self.assertLessEqual(offsets.max().item(), 0.0)
        self.assertFalse(torch.equal(alphas[:, 0], alphas[:, 1]))  # independent axes

    def test_speed_and_pull_leaks_add(self):
        env = self._pull_env([0.4, 0.4])
        env.cfg.obs_rand.ft_force_leak_N = [2.0, 2.0]
        env.randomize_initial_state(torch.tensor([0]))
        env.fingertip_midpoint_linvel = torch.tensor([[0.02, 0.0, 0.0]])  # speed leak ~ -2 N in x
        self._command(env, [0.004, 0.0, 0.0])  # pull leak -0.904 N in x
        env._compute_intermediate_values(DT)
        self.assertAlmostEqual(env.noisy_force[0, 0].item(), -2.904, places=3)


    def _friction_env(self, lo, hi):
        env = self.env
        env.num_envs = 1
        env.cfg.ctrl = types.SimpleNamespace(task_friction_lo_N=list(lo), task_friction_hi_N=list(hi))
        env.physics_dt = DT
        J = torch.zeros(1, 6, 7)
        J[0, 0, 0] = J[0, 1, 1] = J[0, 2, 2] = 1.0  # joint i moves the fingertip along axis i
        env.fingertip_midpoint_jacobian = J
        env.arm_mass_matrix = 2.0 * torch.eye(7).unsqueeze(0)
        env.fingertip_midpoint_linvel = torch.zeros(1, 3)
        env.fingertip_midpoint_pos = torch.zeros(1, 3)
        env.efforts = []
        env._robot = types.SimpleNamespace(set_joint_effort_target=lambda t: env.efforts.append(t.clone()))
        env._compute_intermediate_values(DT)
        return env

    def _ctrl(self, env, wrench):
        env.stub_wrench = torch.tensor([list(wrench) + [0.0, 0.0, 0.0]])
        env.efforts.clear()
        env.generate_ctrl_signals(torch.zeros(1, 3), torch.tensor([DOWN]), 0.0)
        return env.efforts[-1][0, 0:3]

    def test_friction_is_off_by_default(self):
        env = self._friction_env([0.0] * 6, [0.0] * 6)
        env.randomize_initial_state(torch.tensor([0]))
        self.assertIsNone(env._task_friction)
        torch.testing.assert_close(self._ctrl(env, [3.0, 0.0, 0.0]), torch.zeros(3))
        self.assertEqual(len(env.efforts), 1)

    def test_friction_holds_a_small_push_and_limits_a_large_one(self):
        f = [4.6, 1.9, 1.5, 2.0, 2.1, 1.0]
        env = self._friction_env(f, f)
        env.randomize_initial_state(torch.tensor([0]))
        # At rest: 3 N in +x is below x+ friction (held); 3 N in -y is above y- friction (2 N slides).
        torch.testing.assert_close(self._ctrl(env, [3.0, -3.0, 0.0]), torch.tensor([-3.0, 2.0, 0.0]))
        # Moving fast in -x: friction pushes +x with x- friction (1.9 N), whatever the push.
        env.fingertip_midpoint_linvel = torch.tensor([[-0.02, 0.0, 0.0]])
        torch.testing.assert_close(self._ctrl(env, [0.0, 0.0, 0.0]), torch.tensor([1.9, 0.0, 0.0]))

    def test_friction_is_not_applied_during_reset(self):
        f = [3.0] * 6
        env = self._friction_env(f, f)
        env.randomize_initial_state(torch.tensor([0]))  # allocates
        env.stub_wrench = torch.tensor([[2.0, 0.0, 0.0, 0.0, 0.0, 0.0]])
        env.efforts.clear()
        env.randomize_initial_state(torch.tensor([0]))  # the stub's reset calls the controller
        self.assertEqual(len(env.efforts), 1)  # only FactoryEnv's own call, no friction re-send
        torch.testing.assert_close(self._ctrl(env, [2.0, 0.0, 0.0])[0], torch.tensor(-2.0))  # after reset: on

    def test_friction_ramp_scales_the_limits_with_the_env_step(self):
        f = [4.0] * 6
        env = self._friction_env(f, f)
        env.cfg.ctrl.task_friction_ramp_steps = [100, 300]
        env.randomize_initial_state(torch.tensor([0]))
        env.fingertip_midpoint_linvel = torch.tensor([[0.02, 0.0, 0.0]])  # sliding in +x: friction at its limit
        for step, expected in [(0, 0.0), (100, 0.0), (200, -2.0), (300, -4.0), (5000, -4.0)]:
            env.common_step_counter = step
            self.assertAlmostEqual(self._ctrl(env, [0.0, 0.0, 0.0])[0].item(), expected, places=5, msg=step)

    def test_friction_ramp_off_by_default_means_full_friction(self):
        f = [4.0] * 6
        env = self._friction_env(f, f)
        env.randomize_initial_state(torch.tensor([0]))
        env.common_step_counter = 0
        env.fingertip_midpoint_linvel = torch.tensor([[0.02, 0.0, 0.0]])
        self.assertAlmostEqual(self._ctrl(env, [0.0, 0.0, 0.0])[0].item(), -4.0, places=5)

    def test_friction_draws_per_direction_in_range(self):
        env = self._friction_env([1.0, 0.0, 0.0, 0.0, 0.0, 0.0], [2.0, 0.5, 0.0, 0.0, 0.0, 3.0])
        draws = []
        for _ in range(200):
            env.randomize_initial_state(torch.tensor([0]))
            draws.append(env._task_friction.clone())
        d = torch.cat(draws)
        self.assertGreaterEqual(d[:, 0].min().item(), 1.0)
        self.assertLessEqual(d[:, 0].max().item(), 2.0)
        self.assertLessEqual(d[:, 1].max().item(), 0.5)
        self.assertEqual(d[:, 2:5].abs().max().item(), 0.0)
        self.assertGreater(d[:, 5].std().item(), 0.5)


@unittest.skipIf(torch is None, "needs torch")
class TestCoulombFriction(unittest.TestCase):
    def setUp(self):
        from forge_blocks.observation import coulomb_friction

        self.f = coulomb_friction
        self.mass = 2.0 * torch.eye(3).unsqueeze(0)
        self.plus = torch.tensor([[4.0, 4.0, 4.0]])
        self.minus = torch.tensor([[1.0, 1.0, 1.0]])

    def test_sticks_below_the_limit(self):
        out = self.f(torch.tensor([[3.0, -0.5, 0.0]]), torch.zeros(1, 3), self.mass, DT, self.plus, self.minus)
        torch.testing.assert_close(out, torch.tensor([[-3.0, 0.5, 0.0]]))

    def test_slides_at_the_directional_limit(self):
        out = self.f(torch.tensor([[6.0, -3.0, 0.0]]), torch.zeros(1, 3), self.mass, DT, self.plus, self.minus)
        torch.testing.assert_close(out, torch.tensor([[-4.0, 1.0, 0.0]]))

    def test_stops_a_slow_hand_without_reversing_it(self):
        # 0.1 mm/s with 2 kg: 0.024 N stops it within one 1/120 s step; friction gives exactly that.
        v = torch.tensor([[0.0001, 0.0, 0.0]])
        out = self.f(torch.zeros(1, 3), v, self.mass, DT, self.plus, self.minus)
        self.assertAlmostEqual(out[0, 0].item(), -2.0 * 0.0001 / DT, places=6)


@unittest.skipIf(torch is None, "needs torch")
class TestPullLeak(unittest.TestCase):
    def test_shape(self):
        from forge_blocks.observation import pull_leak

        out = pull_leak(torch.tensor([[2.0, -4.0, 0.0]]), torch.tensor([[0.25, 0.5, 0.4]]))
        torch.testing.assert_close(out, torch.tensor([[-0.5, 2.0, 0.0]]))


@unittest.skipIf(torch is None, "needs torch")
class TestSampleRadiusError(unittest.TestCase):
    def setUp(self):
        from forge_blocks.observation import sample_radius_error

        self.sample = sample_radius_error
        torch.manual_seed(0)

    def test_radius_stays_in_range(self):
        for mode in ("xy", "xyz"):
            radius = torch.linalg.norm(self.sample(5000, [0.001, 0.0025], mode), dim=1)
            self.assertGreaterEqual(radius.min().item(), 0.001 - 1e-7)
            self.assertLessEqual(radius.max().item(), 0.0025 + 1e-7)
            self.assertAlmostEqual(radius.mean().item(), 0.00175, delta=5e-5)

    def test_xy_has_no_z(self):
        self.assertEqual(self.sample(100, [0.0, 0.01], "xy")[:, 2].abs().max().item(), 0.0)

    def test_direction_is_uniform(self):
        for mode in ("xy", "xyz"):
            self.assertLess(self.sample(20000, [0.01, 0.01], mode).mean(dim=0).abs().max().item(), 3e-4)

    def test_unknown_mode_is_rejected(self):
        with self.assertRaises(ValueError):
            self.sample(1, [0.0, 0.01], "z")


@unittest.skipIf(torch is None, "needs torch")
class TestFrictionLeak(unittest.TestCase):
    def test_shape(self):
        from forge_blocks.observation import friction_leak

        v = torch.tensor([[0.002, -0.05, 0.0]])
        out = friction_leak(v, torch.tensor([[2.0, 2.0, 2.0]]), 0.002)
        torch.testing.assert_close(out, torch.tensor([[-2.0 * math.tanh(1.0), 2.0, 0.0]]), atol=1e-5, rtol=0)


@unittest.skipIf(torch is None, "needs torch")
class TestForceDeadband(unittest.TestCase):
    def test_axis(self):
        from forge_blocks.observation import force_deadband

        f = torch.tensor([[4.0, -7.0, 12.0]])
        torch.testing.assert_close(force_deadband(f, 5.0, "axis"), torch.tensor([[0.0, -2.0, 7.0]]))

    def test_norm_keeps_direction(self):
        from forge_blocks.observation import force_deadband

        f = torch.tensor([[6.0, 0.0, 8.0], [3.0, 0.0, 4.0]])  # |f| = 10 and 5
        torch.testing.assert_close(force_deadband(f, 5.0, "norm"), torch.tensor([[3.0, 0.0, 4.0], [0.0, 0.0, 0.0]]))


if __name__ == "__main__":
    unittest.main()
