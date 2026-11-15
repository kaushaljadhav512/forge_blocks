"""scripts/_isaaclab.py: the policy follows --device. Needs no Isaac Sim (task registration is stubbed)."""

import os
import sys
import types
import unittest
from unittest import mock

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))


class TestAgentDeviceOverrides(unittest.TestCase):
    def setUp(self):
        stub = {"forge_blocks.tasks": types.ModuleType("forge_blocks.tasks")}
        patcher = mock.patch.dict(sys.modules, stub)
        patcher.start()
        self.addCleanup(patcher.stop)
        sys.modules.pop("_isaaclab", None)
        self.addCleanup(sys.modules.pop, "_isaaclab", None)
        import _isaaclab

        self.overrides = _isaaclab.agent_device_overrides

    def test_device_moves_the_policy_too(self):
        want = ["agent.params.config.device=cuda:1", "agent.params.config.device_name=cuda:1"]
        self.assertEqual(self.overrides(["--task", "T", "--device", "cuda:1", "--headless"]), want)
        self.assertEqual(self.overrides(["--device=cuda:1"]), want)

    def test_no_device_changes_nothing(self):
        self.assertEqual(self.overrides(["--task", "T", "--headless", "agent.params.config.name=x"]), [])

    def test_explicit_overrides_win(self):
        argv = ["--device", "cuda:1", "agent.params.config.device=cuda:0"]
        self.assertEqual(self.overrides(argv), ["agent.params.config.device_name=cuda:1"])


if __name__ == "__main__":
    unittest.main()
