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

        def randomize_initial_state(self, env_ids):
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
