"""Robot-agnostic FORGE policy runner (no Isaac Sim / Isaac Lab imports).

Runs on the laptop or the robot PC with only numpy + torch. Reproduces, for one robot, what
ForgeEnv does around the policy (isaaclab_tasks/direct/forge/forge_env.py @ v2.3.2):

  observation  _get_observations: fingertip pose relative to the *estimated* slot opening,
               quat with w and z zeroed, finite-diff velocities (angular: z only), smoothed
               world-frame contact force, force threshold, previous (smoothed) actions.
               Compute the velocities over about 1/120 s on the robot, as forge_blocks does in
               sim, not over the 1/15 s policy step. Don't use with upstream checkpoints: their
               velocity inputs are noise (Isaac Lab issue #8874).
  action       _pre_physics_step: EMA of the clamped policy mean           -> update_actions()
               _apply_action:     targets relative to the estimated slot,   -> target()
                                  clipped to a step around the current
                                  fingertip pose, upright orientation

All quaternions are (w, x, y, z); all vectors are in the robot base frame, which plays the
role of the sim's world frame. "Fingertip" is Factory's fingertip midpoint (between the pads).

Units: m, rad, N. Call reset() at the start of every attempt, then step() at 15 Hz.

Two rates. ForgeEnv updates the actions once per policy step but recomputes the controller target
from the *current* fingertip pose on every physics step (120 Hz), so the 2 cm step clip follows the
hand as it moves. On the robot, call step() at 15 Hz and then target() with the latest fingertip
pose at 120 Hz or faster until the next step(); don't hold step()'s target for the whole 1/15 s.
"""

from __future__ import annotations

import json
import math
import os
from dataclasses import dataclass, field

import numpy as np

# Observation sizes, from factory_env_cfg.OBS_DIM_CFG + forge_env_cfg updates.
OBS_DIMS = {
    "fingertip_pos": 3,
    "fingertip_pos_rel_fixed": 3,
    "fingertip_quat": 4,
    "ee_linvel": 3,
    "ee_angvel": 3,
    "ft_force": 3,
    "force_threshold": 1,
}
ACTION_DIM = 7  # dx, dy, dz, (roll, pitch: ignored), yaw, success prediction


# --- Quaternion helpers, matching isaacsim.core.utils.torch.rotations ------------------------


def quat_mul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2,
            w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2,
        ]
    )


def quat_from_euler_xyz(roll, pitch, yaw):
    cy, sy = math.cos(yaw * 0.5), math.sin(yaw * 0.5)
    cr, sr = math.cos(roll * 0.5), math.sin(roll * 0.5)
    cp, sp = math.cos(pitch * 0.5), math.sin(pitch * 0.5)
    return np.array(
        [
            cy * cr * cp + sy * sr * sp,
            cy * sr * cp - sy * cr * sp,
            cy * cr * sp + sy * sr * cp,
            sy * cr * cp - cy * sr * sp,
        ]
    )


def get_euler_xyz(q):
    """Returns (roll, pitch, yaw), each wrapped to [0, 2*pi) like Isaac Sim's helper."""
    w, x, y, z = q
    roll = math.atan2(2.0 * (w * x + y * z), w * w - x * x - y * y + z * z)
    sinp = 2.0 * (w * y - z * x)
    pitch = math.copysign(math.pi / 2.0, sinp) if abs(sinp) >= 1 else math.asin(sinp)
    yaw = math.atan2(2.0 * (w * z + x * y), w * w + x * x - y * y - z * z)
    two_pi = 2 * math.pi
    return roll % two_pi, pitch % two_pi, yaw % two_pi


def wrap_yaw(angle):
    """factory_utils.wrap_yaw: keep yaw on a span uninterrupted by the joint-7 limit."""
    return angle - 2 * math.pi if angle > math.radians(235) else angle


# --- Force smoothing -------------------------------------------------------------------------


class ForceFilter:
    """FORGE's EMA on the contact force, re-expressed for any sample rate.

    In sim the EMA (alpha = ft_smoothing_factor) runs every physics step (1/120 s), i.e. 8 times
    per policy step. Feed this every robot force sample (e.g. 1 kHz from franka-interface) and it
    applies the alpha that gives the same time constant.
    """

    def __init__(self, alpha_per_sim_step: float, sim_dt: float):
        self.alpha_per_sim_step = alpha_per_sim_step
        self.sim_dt = sim_dt
        self.value = np.zeros(3)

    def reset(self):
        self.value[:] = 0.0  # ForgeEnv._reset_idx zeros the smoothed force

    def update(self, force: np.ndarray, dt: float) -> np.ndarray:
        alpha = 1.0 - (1.0 - self.alpha_per_sim_step) ** (dt / self.sim_dt)
        self.value = alpha * np.asarray(force, dtype=float) + (1.0 - alpha) * self.value
        return self.value


# --- Config ----------------------------------------------------------------------------------


@dataclass
class RunnerConfig:
    obs_order: list
    pos_action_bounds: list
    pos_action_threshold: list
    rot_action_threshold: list
    ema_factor: float  # sim randomises in ema_factor_range; we use its midpoint by default
    force_threshold: float  # sim randomises in contact_penalty_threshold_range
    ft_smoothing_factor: float
    sim_dt: float
    policy_dt: float
    rnn_layers: int
    rnn_units: int
    # Multiply the robot's force by this before it enters the observation. Sim reports the
    # joint reaction at the force_sensor link; the sign relative to libfranka's O_F_ext_hat_K
    # MUST be checked with tests/sim press test and the real press test before trusting it.
    force_sign: float = 1.0
    # Deploy only: aim roll/pitch at pointing straight down instead of ForgeEnv's current tilt + at most
    # rot_action_threshold. With the sim's rule the restoring torque is capped (28 Nm/rad x 0.097 rad = 2.7 Nm),
    # so a side load on the peg tilted the real arm to 37 deg (session 3, trial 2). The policy can't see tilt.
    hold_vertical: bool = False
    extra: dict = field(default_factory=dict)

    @classmethod
    def from_json(cls, path: str, **overrides) -> "RunnerConfig":
        with open(path) as f:
            d = json.load(f)
        d.update(overrides)
        known = {k: d.pop(k) for k in list(d) if k in cls.__dataclass_fields__ and k != "extra"}
        return cls(**known, extra=d)

    @property
    def obs_dim(self) -> int:
        return sum(OBS_DIMS[k] for k in self.obs_order) + ACTION_DIM


# --- Runner ----------------------------------------------------------------------------------


class ForgePolicyRunner:
    """Wraps an exported policy (scripts/export_policy.py) for closed-loop use.

    policy: callable (obs[1, N], h, c) -> (mu[1, 7], h, c) on torch tensors, or None to use
    only the observation/action maths (tests, sim comparisons).
    """

    def __init__(self, cfg: RunnerConfig, policy=None):
        self.cfg = cfg
        self.policy = policy
        self.actions = np.zeros(ACTION_DIM)  # ForgeEnv.actions: EMA-smoothed, in [-1, 1]
        self.slot_est = np.zeros(3)
        self._h = self._c = None

    @classmethod
    def from_export(cls, export_dir: str, device: str = "cpu", **cfg_overrides) -> "ForgePolicyRunner":
        import torch  # only needed when actually running a network

        cfg = RunnerConfig.from_json(os.path.join(export_dir, "runner_config.json"), **cfg_overrides)
        policy = torch.jit.load(os.path.join(export_dir, "policy.pt"), map_location=device).eval()
        return cls(cfg, policy)

    # .. episode ..................................................................................

    def reset(self, fingertip_pos, fingertip_quat, slot_tip_pos_est):
        """Start an attempt. slot_tip_pos_est: estimated centre of the slot *opening* (rim height)."""
        cfg = self.cfg
        self.slot_est = np.asarray(slot_tip_pos_est, dtype=float)
        self.actions = np.zeros(ACTION_DIM)
        # ForgeEnv._reset_idx: make the first EMA step start from the current pose.
        self.actions[0:3] = (np.asarray(fingertip_pos) - self.slot_est) / np.asarray(cfg.pos_action_bounds)
        self.actions[5] = self._yaw_to_action(fingertip_quat)
        self.actions[6] = -1.0
        if self.policy is not None:
            import torch

            device = next(iter(self.policy.parameters()), torch.zeros(0)).device
            self._h = torch.zeros(cfg.rnn_layers, 1, cfg.rnn_units, device=device)
            self._c = torch.zeros_like(self._h)

    def step(self, fingertip_pos, fingertip_quat, linvel, angvel, force_smoothed):
        """One 15 Hz step. Returns (target_pos, target_quat, success_prob, obs).

        The returned target is for the current pose only; keep calling target() with fresh poses
        until the next step() (see the module docstring).
        """
        obs = self.build_obs(fingertip_pos, fingertip_quat, linvel, angvel, force_smoothed)
        mu = self._run_policy(obs)
        target_pos, target_quat = self.apply_action(mu, fingertip_pos, fingertip_quat)
        return target_pos, target_quat, self.success_prob, obs

    @property
    def success_prob(self) -> float:
        return (self.actions[6] + 1.0) / 2.0

    # .. pieces (public so the sim bridge test can check each against ForgeEnv) ................

    def build_obs(self, fingertip_pos, fingertip_quat, linvel, angvel, force_smoothed) -> np.ndarray:
        cfg = self.cfg
        quat = np.array(fingertip_quat, dtype=float)
        quat[[0, 3]] = 0.0  # ForgeEnv: noisy_fingertip_quat[:, [0, 3]] = 0
        angvel_obs = np.array([0.0, 0.0, angvel[2]])  # ForgeEnv: ee_angvel_fd[:, 0:2] = 0
        prev_actions = self.actions.copy()
        prev_actions[3:5] = 0.0
        parts = {
            "fingertip_pos": np.asarray(fingertip_pos, dtype=float),
            "fingertip_pos_rel_fixed": np.asarray(fingertip_pos, dtype=float) - self.slot_est,
            "fingertip_quat": quat,
            "ee_linvel": np.asarray(linvel, dtype=float),
            "ee_angvel": angvel_obs,
            "ft_force": cfg.force_sign * np.asarray(force_smoothed, dtype=float)[:3],
            "force_threshold": np.array([cfg.force_threshold]),
        }
        obs = np.concatenate([parts[k] for k in cfg.obs_order] + [prev_actions])
        assert obs.shape == (cfg.obs_dim,), (obs.shape, cfg.obs_dim)
        return obs.astype(np.float32)

    def apply_action(self, mu, fingertip_pos, fingertip_quat):
        """update_actions(mu) then target(pose): one policy step's action, first physics step's target."""
        self.update_actions(mu)
        return self.target(fingertip_pos, fingertip_quat)

    def update_actions(self, mu) -> np.ndarray:
        """FactoryEnv._pre_physics_step: EMA of the clamped policy mean. Once per policy step."""
        cfg = self.cfg
        mu = np.clip(np.asarray(mu, dtype=float), -1.0, 1.0)  # rl_games player clamps to [-1, 1]
        self.actions = cfg.ema_factor * mu + (1.0 - cfg.ema_factor) * self.actions
        return self.actions

    def target(self, fingertip_pos, fingertip_quat):
        """ForgeEnv._apply_action for one env: pose target from the current actions and pose.

        Pure function of self.actions and the given pose; call it every control tick (>= 120 Hz).
        """
        cfg = self.cfg
        a = self.actions
        fingertip_pos = np.asarray(fingertip_pos, dtype=float)

        # Position: target relative to the estimated slot opening, then clipped around current pose.
        pre_pos = self.slot_est + a[0:3] * np.asarray(cfg.pos_action_bounds)
        thr = np.asarray(cfg.pos_action_threshold)
        target_pos = fingertip_pos + np.clip(pre_pos - fingertip_pos, -thr, thr)

        # Orientation: roll/pitch actions ignored; yaw action spans the joint-7 range.
        yaw = math.radians(-180.0) + math.radians(270.0) * (a[5] + 1.0) / 2.0
        pre_quat = quat_mul(quat_from_euler_xyz(math.pi, 0.0, 0.0), quat_from_euler_xyz(0.0, 0.0, yaw))

        curr_roll, curr_pitch, curr_yaw = get_euler_xyz(fingertip_quat)
        des_roll, des_pitch, des_yaw = get_euler_xyz(pre_quat)
        rthr = cfg.rot_action_threshold

        curr_yaw, des_yaw = wrap_yaw(curr_yaw), wrap_yaw(des_yaw)
        new_yaw = curr_yaw + np.clip(des_yaw - curr_yaw, -rthr[2], rthr[2])

        des_roll = des_roll + 2 * math.pi if des_roll < 0.0 else des_roll
        des_pitch = des_pitch + 2 * math.pi if des_pitch < 0.0 else des_pitch
        new_roll = curr_roll + np.clip(des_roll - curr_roll, -rthr[0], rthr[0])

        curr_pitch = curr_pitch - 2 * math.pi if curr_pitch > math.pi else curr_pitch
        des_pitch = des_pitch - 2 * math.pi if des_pitch > math.pi else des_pitch
        new_pitch = curr_pitch + np.clip(des_pitch - curr_pitch, -rthr[1], rthr[1])
        if cfg.hold_vertical:
            new_roll, new_pitch = des_roll, des_pitch

        return target_pos, quat_from_euler_xyz(new_roll, new_pitch, new_yaw)

    # .. internals ................................................................................

    def _yaw_to_action(self, fingertip_quat) -> float:
        rel = quat_mul(quat_from_euler_xyz(-math.pi, 0.0, 0.0), np.asarray(fingertip_quat, dtype=float))
        yaw = get_euler_xyz(rel)[2]
        if yaw > math.pi / 2:
            yaw -= 2 * math.pi
        if yaw < -math.pi:
            yaw += 2 * math.pi
        return (yaw + math.radians(180.0)) / math.radians(270.0) * 2.0 - 1.0

    def _run_policy(self, obs: np.ndarray) -> np.ndarray:
        if self.policy is None:
            raise RuntimeError("no policy loaded; use build_obs/apply_action directly")
        import torch

        with torch.inference_mode():
            x = torch.as_tensor(obs, device=self._h.device).unsqueeze(0)
            mu, self._h, self._c = self.policy(x, self._h, self._c)
        return mu[0].cpu().numpy()
