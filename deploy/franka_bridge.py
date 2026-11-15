"""Robot side of the peg policy: FrankaPy <-> ForgePolicyRunner. See deploy/README.md.

Runs on the lab computer that has FrankaPy/ROS. Only numpy is needed at import time;
FrankaPy, rospy and autolab_core are imported when a FrankaPyArm is created, so the loop logic can
be tested anywhere with a fake arm (tests/test_franka_bridge.py).

Frames. Everything is in the robot base frame, which plays the sim's world frame. FrankaPy reports its
tool point 0.1034 m along the hand's z axis; the policy's "fingertip" is the sim's
panda_fingertip_centered, 0.1121 m along the same axis with the same orientation
(scripts/measure_frames.py; the Lab2 code confirmed FrankaPy's frame on this robot). So:

    sim_tip = tool_pos + R[:, 2] * TIP_OFFSET          tool_pos = sim_tip - R[:, 2] * TIP_OFFSET

Rates. The policy runs at 15 Hz (runner.step). Between policy steps the controller target is
recomputed from the latest pose at the control rate (runner.target), as ForgeEnv does every 1/120 s.
"""

from __future__ import annotations

import json
import math
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np

from policy_runner import ForceFilter, ForgePolicyRunner

# --- Geometry (scripts/measure_frames.py, hardware/README.md) ----------------------------------

TIP_OFFSET = 0.1121 - 0.1034  # sim fingertip point beyond FrankaPy's tool point, along tool z [m]
GRASP_TIP_ABOVE_PEG_BOTTOM = 0.0321  # sim grasp: fingertip point (≈ finger ends) above the peg's bottom
HOLE_DEPTH = 0.025
SEATED_TIP_ABOVE_RIM = GRASP_TIP_ABOVE_PEG_BOTTOM - HOLE_DEPTH  # fingertip z above the rim when seated
START_ABOVE_RIM = 0.047  # Factory hand_init_pos z
START_NOISE = (0.02, 0.02, 0.01)  # Factory hand_init_pos_noise (uniform ±)
START_YAW_NOISE = 0.785  # Factory hand_init_orn_noise z (uniform ±, rad)
SUCCESS_XY = 0.0025  # Factory: peg within 2.5 mm of the hole axis ...
SUCCESS_Z = 0.04 * HOLE_DEPTH  # ... and within success_threshold * height (1 mm) of seated
DOWN = np.diag([1.0, -1.0, -1.0])  # gripper pointing down, yaw 0 (Lab2's TOP_R; sim roll = pi)
SIM_FORCE_UP_WHEN_PRESSING = 1.0  # measure_frames.py: pressing down on a surface reads +z in sim


# --- Rotations (quaternions are w, x, y, z like the runner) ---------------------------------------


def quat_to_mat(q) -> np.ndarray:
    w, x, y, z = np.asarray(q, dtype=float) / np.linalg.norm(q)
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
            [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
            [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)],
        ]
    )


def mat_to_quat(R) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z). Sign is arbitrary; see canonical_quat."""
    R = np.asarray(R, dtype=float)
    tr = np.trace(R)
    if tr > 0:
        s = 2.0 * math.sqrt(tr + 1.0)
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2])
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = 2.0 * math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2])
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1])
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    q = np.array(q)
    return q / np.linalg.norm(q)


def canonical_quat(q, ref=(0.0, 1.0, 0.0, 0.0)) -> np.ndarray:
    """Pick the sign of q closest to ref. The policy sees x and y only, which flip with the sign,
    so the robot must use the sim's convention. ref is set from the sim (bridge config)."""
    q = np.asarray(q, dtype=float)
    return -q if float(np.dot(q, ref)) < 0 else q


def rot_z(yaw: float) -> np.ndarray:
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]])


def down_with_yaw(yaw: float) -> np.ndarray:
    """Isaac's quat_from_euler_xyz(pi, 0, yaw) as a matrix: Rz(yaw) @ Rx(pi)."""
    return rot_z(yaw) @ DOWN


def tool_to_tip(pos, R, offset=TIP_OFFSET) -> np.ndarray:
    return np.asarray(pos, dtype=float) + np.asarray(R)[:, 2] * offset


def tip_to_tool(pos, R, offset=TIP_OFFSET) -> np.ndarray:
    return np.asarray(pos, dtype=float) - np.asarray(R)[:, 2] * offset


# --- Robot interface ------------------------------------------------------------------------------


class HardwareError(RuntimeError):
    pass


class SafetyStop(RuntimeError):
    pass


@dataclass
class RobotState:
    t: float  # monotonic clock [s]
    q: np.ndarray
    dq: np.ndarray
    tool_pos: np.ndarray
    tool_R: np.ndarray
    force: np.ndarray  # O_F_ext_hat_K[:3], base-frame axes, raw (no sign flip, no tare)
    J: np.ndarray | None = None  # 6x7 tool-point Jacobian [v; w], if available

    def check(self):
        parts = [self.q, self.dq, self.tool_pos, self.tool_R, self.force]
        if self.q.shape != (7,) or self.tool_pos.shape != (3,) or self.tool_R.shape != (3, 3):
            raise HardwareError("robot state has the wrong shape")
        if not all(np.isfinite(p).all() for p in parts):
            raise HardwareError("robot state is not finite")
        return self

    def tool_velocity(self, prev: "RobotState | None"):
        """(v, w) of the tool point: J @ dq when available, else a pose difference with prev."""
        if self.J is not None:
            twist = self.J @ self.dq
            return twist[:3], twist[3:]
        if prev is None or self.t <= prev.t:
            return np.zeros(3), np.zeros(3)
        dt = self.t - prev.t
        dR = self.tool_R @ prev.tool_R.T
        angle = math.acos(max(-1.0, min(1.0, (np.trace(dR) - 1.0) / 2.0)))
        if angle < 1e-9:
            w = np.zeros(3)
        else:
            axis = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / (2 * math.sin(angle))
            w = axis * angle / dt
        return (self.tool_pos - prev.tool_pos) / dt, w


class FrankaPyArm:
    """The only class that talks to FrankaPy. Calls follow FrankaPy's examples (run_dynamic_pose.py)
    and the Lab2 adapter that runs on this robot; check them against the installed FrankaPy."""

    def __init__(self, fa=None):
        import rospy  # noqa: F401  (FrankaArm starts the ROS node)
        from autolab_core import RigidTransform
        from franka_interface_msgs.msg import SensorDataGroup
        from frankapy import FrankaArm, FrankaConstants, SensorDataMessageType
        from frankapy.proto import PosePositionSensorMessage, ShouldTerminateSensorMessage
        from frankapy.proto_utils import make_sensor_group_msg, sensor_proto2ros_msg

        self._m = dict(
            rospy=rospy,
            RigidTransform=RigidTransform,
            SensorDataGroup=SensorDataGroup,
            FC=FrankaConstants,
            SDMT=SensorDataMessageType,
            PosePositionSensorMessage=PosePositionSensorMessage,
            ShouldTerminateSensorMessage=ShouldTerminateSensorMessage,
            make_sensor_group_msg=make_sensor_group_msg,
            sensor_proto2ros_msg=sensor_proto2ros_msg,
        )
        self.fa = fa if fa is not None else FrankaArm()
        self._pub = None
        self._id = 0
        self._t0 = 0.0
        self.has_jacobian = hasattr(self.fa, "get_jacobian")
        # One request for the whole state (~1.5 ms on the lab robot) instead of four (~6.5 ms), and all
        # fields come from the same instant. Older FrankaPy versions may lack it.
        self.has_robot_state = hasattr(self.fa, "get_robot_state")

    def _pose(self, pos, R):
        return self._m["RigidTransform"](
            rotation=np.asarray(R, dtype=float), translation=np.asarray(pos, dtype=float), from_frame="franka_tool", to_frame="world"
        )

    def read(self) -> RobotState:
        fa = self.fa
        if self.has_robot_state:
            s = fa.get_robot_state()
            pose, q, dq, ft = s["pose"], s["joints"], s["joint_velocities"], s["ee_force_torque"]
        else:
            pose, q, dq, ft = fa.get_pose(), fa.get_joints(), fa.get_joint_velocities(), fa.get_ee_force_torque()
        q = np.asarray(q, dtype=float)
        J = np.asarray(fa.get_jacobian(q), dtype=float) if self.has_jacobian else None  # local, no request
        return RobotState(
            t=time.monotonic(),
            q=q,
            dq=np.asarray(dq, dtype=float),
            tool_pos=np.asarray(pose.translation, dtype=float),
            tool_R=np.asarray(pose.rotation, dtype=float),
            force=np.asarray(ft, dtype=float)[:3],
            J=J if J is not None and J.shape == (6, 7) else None,
        ).check()

    def goto(self, pos, R, duration, stiffness=None):
        """Blocking Cartesian move (FrankaPy's impedance controller)."""
        kw = {"cartesian_impedances": list(stiffness)} if stiffness is not None else {}
        self.fa.goto_pose(self._pose(pos, R), duration=float(duration), use_impedance=True, ignore_virtual_walls=False, **kw)

    def start_stream(self, pos, R, stiffness, max_duration):
        """Start a dynamic pose skill; then call send_target at the control rate."""
        m = self._m
        self.fa.goto_pose(
            self._pose(pos, R), duration=float(max_duration), dynamic=True, buffer_time=10, cartesian_impedances=list(stiffness)
        )
        self._pub = m["rospy"].Publisher(m["FC"].DEFAULT_SENSOR_PUBLISHER_TOPIC, m["SensorDataGroup"], queue_size=10)
        self._id = 0
        self._t0 = m["rospy"].Time.now().to_time()

    def send_target(self, pos, R):
        m = self._m
        msg = m["PosePositionSensorMessage"](
            id=self._id,
            timestamp=m["rospy"].Time.now().to_time() - self._t0,
            position=[float(v) for v in pos],
            quaternion=[float(v) for v in mat_to_quat(R)],  # FrankaPy/autolab_core: w, x, y, z
        )
        sensor = m["sensor_proto2ros_msg"](msg, m["SDMT"].POSE_POSITION)
        self._pub.publish(m["make_sensor_group_msg"](trajectory_generator_sensor_msg=sensor))
        self._id += 1

    def stop_stream(self):
        m = self._m
        if self._pub is None:
            return
        try:
            msg = m["ShouldTerminateSensorMessage"](
                timestamp=m["rospy"].Time.now().to_time() - self._t0, should_terminate=True
            )
            sensor = m["sensor_proto2ros_msg"](msg, m["SDMT"].SHOULD_TERMINATE)
            self._pub.publish(m["make_sensor_group_msg"](termination_handler_sensor_msg=sensor))
            self.fa.wait_for_skill()
        except Exception:
            self.stop()
        finally:
            self._pub = None

    def grasp(self, width=0.008, force=40.0) -> float:
        """Close on the peg. 40 N so the peg doesn't slide under ~11 N of axial contact force."""
        self.fa.goto_gripper(
            width=width, grasp=True, speed=0.02, force=force, epsilon_inner=0.004, epsilon_outer=0.004, block=True,
            ignore_errors=False,
        )
        got = float(self.fa.get_gripper_width())
        if not 0.006 <= got <= 0.010:
            raise HardwareError(f"unexpected peg grasp width {got:.4f} m")
        return got

    def open(self) -> float:
        self.fa.open_gripper(block=True)
        return float(self.fa.get_gripper_width())

    def guide(self, duration, block=True):
        """Hand-guiding (zero-stiffness) mode."""
        self.fa.run_guide_mode(duration=float(duration), block=block)

    def stop(self):
        try:
            self.fa.stop_skill()
        except Exception:
            pass


# --- Hole registration ----------------------------------------------------------------------------


@dataclass
class Hole:
    """The registered (true) hole. seated_tip: sim fingertip when the peg is fully seated."""

    seated_tip: list
    samples: list = field(default_factory=list)

    @property
    def rim(self) -> np.ndarray:
        """Centre of the hole opening: the sim's fixed_pos_obs_frame."""
        return np.asarray(self.seated_tip) - np.array([0.0, 0.0, SEATED_TIP_ABOVE_RIM])

    @property
    def spread_m(self) -> float:
        if len(self.samples) < 2:
            return 0.0
        s = np.asarray(self.samples)
        return float(np.linalg.norm(s - s.mean(0), axis=1).max())

    def save(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        Path(path).write_text(json.dumps({**asdict(self), "rim": self.rim.tolist(), "spread_m": self.spread_m}, indent=2))

    @classmethod
    def load(cls, path) -> "Hole":
        d = json.loads(Path(path).read_text())
        return cls(seated_tip=d["seated_tip"], samples=d.get("samples", []))


def register_hole(arm, n=5, confirm=input, guide_s=15.0) -> Hole:
    """Hand-guide the grasped peg into the hole n times; average the seated fingertip pose."""
    samples = []
    for i in range(n):
        confirm(f"[{i + 1}/{n}] Guide mode for {guide_s:.0f} s: seat the peg fully, hold it still. Enter to start.")
        arm.guide(guide_s, block=True)
        st = arm.read()
        tip = tool_to_tip(st.tool_pos, st.tool_R)
        samples.append(tip.tolist())
        print(f"  seated fingertip {np.round(tip, 4).tolist()}")
        if i < n - 1:
            confirm("Lift the peg out (guide mode follows). Enter when clear.")
            arm.guide(5.0, block=True)
    hole = Hole(seated_tip=np.mean(samples, axis=0).tolist(), samples=samples)
    print(f"hole rim {np.round(hole.rim, 4).tolist()}, spread {hole.spread_m * 1000:.2f} mm (want < 0.5)")
    return hole


# --- Safety -----------------------------------------------------------------------------------------


@dataclass
class SafetyLimits:
    max_force_N: float = 25.0  # tared, smoothed force norm
    box_xy_m: float = 0.04  # around the registered hole and the estimate
    tip_floor_above_rim_m: float = SEATED_TIP_ABOVE_RIM - 0.003  # below seated = peg slipped or wrong hole
    box_top_above_est_m: float = 0.09
    max_speed_m_s: float = 0.3


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits, hole_rim, hole_est):
        self.lim = limits
        self.centres = [np.asarray(hole_rim, dtype=float)[:2], np.asarray(hole_est, dtype=float)[:2]]
        self.z_min = float(hole_rim[2]) + limits.tip_floor_above_rim_m
        self.z_max = max(float(hole_rim[2]), float(hole_est[2])) + limits.box_top_above_est_m

    def check(self, tip, force_norm, speed):
        lim = self.lim
        if force_norm > lim.max_force_N:
            raise SafetyStop(f"force {force_norm:.1f} N > {lim.max_force_N} N")
        if min(np.abs(tip[:2] - c).max() for c in self.centres) > lim.box_xy_m:
            raise SafetyStop(f"fingertip {np.round(tip, 4).tolist()} left the xy box")
        if not self.z_min <= tip[2] <= self.z_max:
            raise SafetyStop(f"fingertip z {tip[2]:.4f} outside [{self.z_min:.4f}, {self.z_max:.4f}]")
        if speed > lim.max_speed_m_s:
            raise SafetyStop(f"speed {speed:.2f} m/s > {lim.max_speed_m_s}")


# --- One attempt ------------------------------------------------------------------------------------


@dataclass
class BridgeConfig:
    control_hz: float = 200.0  # target updates; measure what FrankaPy sustains with `run_peg.py rate-test`
    stiffness: tuple = (565.0, 565.0, 565.0, 28.0, 28.0, 28.0)  # sim default_task_prop_gains
    timeout_s: float = 10.0  # sim episode length (150 policy steps)
    pred_threshold: float = 0.5
    tare_s: float = 0.5
    quat_ref: tuple = (0.0, 1.0, 0.0, 0.0)  # sign convention for observed quats; confirm against the sim
    tip_offset: float = TIP_OFFSET


def tare_force(arm, seconds, hz=200.0, clock=time.monotonic, sleep=time.sleep) -> np.ndarray:
    """Mean raw force over `seconds` in free space; subtract it for the rest of the attempt."""
    samples, end = [], clock() + seconds
    while clock() < end:
        samples.append(arm.read().force)
        sleep(1.0 / hz)
    return np.mean(samples, axis=0)


def tip_pose(st: RobotState, cfg: BridgeConfig):
    R = st.tool_R
    return tool_to_tip(st.tool_pos, R, cfg.tip_offset), canonical_quat(mat_to_quat(R), cfg.quat_ref), R


def tip_velocity(st: RobotState, prev, cfg: BridgeConfig):
    v, w = st.tool_velocity(prev)
    r = st.tool_R[:, 2] * cfg.tip_offset
    return v + np.cross(w, r), w


def run_attempt(arm, runner: ForgePolicyRunner, hole: Hole, hole_est, cfg: BridgeConfig, limits: SafetyLimits,
                log_steps: list | None = None, clock=time.monotonic, sleep=time.sleep) -> dict:
    """Run one insertion from the current pose. The arm must already be at the start pose, holding
    the peg, in free space. Returns a summary like eval_sweep.py's per-episode row."""
    hole_est = np.asarray(hole_est, dtype=float)
    rim, seated = hole.rim, np.asarray(hole.seated_tip)
    rc = runner.cfg
    safety = SafetyMonitor(limits, rim, hole_est)

    tare = tare_force(arm, cfg.tare_s, cfg.control_hz, clock, sleep)
    st = arm.read()
    tip, quat, R = tip_pose(st, cfg)
    runner.reset(tip, quat, hole_est)
    ff = ForceFilter(rc.ft_smoothing_factor, rc.sim_dt)
    ff.reset()

    result = {"outcome": None, "success": False, "time_to_success_s": None, "time_to_prediction_s": None,
              "peak_force_smooth_N": 0.0, "mean_force_smooth_N": 0.0, "policy_steps": 0, "error": None,
              "tare_N": tare.tolist(), "hole_est": hole_est.tolist(), "hole_rim": rim.tolist()}
    max_steps = int(round(cfg.timeout_s / rc.policy_dt))
    arm.start_stream(st.tool_pos, st.tool_R, cfg.stiffness, max_duration=cfg.timeout_s + 5.0)
    t0 = clock()
    next_policy, next_tick = t0, t0
    prev, force_sum, ticks = None, 0.0, 0
    try:
        while True:
            st = arm.read()
            now = clock()
            dt = (st.t - prev.t) if prev is not None and st.t > prev.t else 1.0 / cfg.control_hz
            fs = ff.update(st.force - tare, dt)  # runner.build_obs applies force_sign
            tip, quat, R = tip_pose(st, cfg)
            v, w = tip_velocity(st, prev, cfg)
            fnorm = float(np.linalg.norm(fs))
            safety.check(tip, fnorm, float(np.linalg.norm(v)))
            result["peak_force_smooth_N"] = max(result["peak_force_smooth_N"], fnorm)
            force_sum, ticks = force_sum + fnorm, ticks + 1

            seated_now = np.linalg.norm(tip[:2] - seated[:2]) < SUCCESS_XY and tip[2] < seated[2] + SUCCESS_Z
            if seated_now and result["time_to_success_s"] is None:
                result["time_to_success_s"] = now - t0

            if now >= next_policy:
                target_pos, target_quat, p_succ, obs = runner.step(tip, quat, v, w, fs)
                result["policy_steps"] += 1
                if log_steps is not None:
                    log_steps.append({"t": now - t0, "obs": obs.tolist(), "actions": runner.actions.tolist(),
                                      "target": target_pos.tolist(), "tip": tip.tolist(), "force_raw": st.force.tolist(),
                                      "force_smooth": fs.tolist(), "success_prob": p_succ})
                next_policy += rc.policy_dt
                if p_succ > cfg.pred_threshold:
                    result["outcome"], result["time_to_prediction_s"] = "predicted_success", now - t0
                    break
                if result["policy_steps"] >= max_steps:
                    result["outcome"] = "timeout"
                    break
            else:
                target_pos, target_quat = runner.target(tip, quat)

            target_R = quat_to_mat(target_quat)
            arm.send_target(tip_to_tool(target_pos, target_R, cfg.tip_offset), target_R)
            prev = st
            next_tick += 1.0 / cfg.control_hz
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
            else:
                next_tick = clock()  # fell behind; don't try to catch up with a burst
    except SafetyStop as exc:
        result["outcome"], result["error"] = "safety_stop", str(exc)
    finally:
        arm.stop_stream()

    st = arm.read()
    tip = tool_to_tip(st.tool_pos, st.tool_R, cfg.tip_offset)
    result["final_tip"] = tip.tolist()
    result["success"] = bool(np.linalg.norm(tip[:2] - seated[:2]) < SUCCESS_XY and tip[2] < seated[2] + SUCCESS_Z)
    result["mean_force_smooth_N"] = force_sum / max(ticks, 1)
    return result


# --- Start poses and errors (the sim's distributions) ---------------------------------------------------


def sample_error(r_min, r_max, rng) -> np.ndarray:
    """forge_blocks.observation.sample_radius_error, mode "xyz": uniform radius, direction on the sphere."""
    d = rng.normal(size=3)
    return (r_min + (r_max - r_min) * rng.random()) * d / max(np.linalg.norm(d), 1e-9)


def sample_start(hole: Hole, rng, cfg: BridgeConfig):
    """Factory randomize_initial_state: above the TRUE rim, uniform noise, yaw noise about yaw 0.
    Returns the FrankaPy tool pose (pos, R) to move to."""
    tip = hole.rim + np.array([0.0, 0.0, START_ABOVE_RIM]) + (2 * rng.random(3) - 1) * np.asarray(START_NOISE)
    R = down_with_yaw((2 * rng.random() - 1) * START_YAW_NOISE)
    return tip_to_tool(tip, R, cfg.tip_offset), R


# --- Calibration helpers --------------------------------------------------------------------------------


def press_test(arm, cfg: BridgeConfig, descend_m_s=0.005, max_descent=0.06, stop_N=6.0, hold_s=1.0, clock=time.monotonic,
               sleep=time.sleep):
    """Lower the held peg onto a free flat surface below until the force builds up, then report the
    raw force z. The sim reads +z for this press, so force_sign = sign(raw z). Starts at the current pose."""
    tare = tare_force(arm, cfg.tare_s, cfg.control_hz, clock, sleep)
    st = arm.read()
    start, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(start, R, cfg.stiffness, max_duration=max_descent / descend_m_s + 10)
    t0, pressed, contact = clock(), [], None
    try:
        while True:
            st = arm.read()
            f = st.force - tare
            t = clock() - t0
            if contact is None and np.linalg.norm(f) > stop_N:
                contact = t
            if contact is not None:
                pressed.append(f)
                if t - contact > hold_s:
                    break
                depth = descend_m_s * contact
            else:
                depth = descend_m_s * t
            if depth > max_descent:
                raise HardwareError("no contact within the descent limit; is there a surface below?")
            arm.send_target(start - np.array([0.0, 0.0, depth]), R)
            sleep(1.0 / cfg.control_hz)
    finally:
        arm.stop_stream()
        arm.goto(start, R, 3.0)
    fz = float(np.mean(pressed, axis=0)[2])
    sign = SIM_FORCE_UP_WHEN_PRESSING if fz > 0 else -SIM_FORCE_UP_WHEN_PRESSING
    return {"raw_mean_force": np.mean(pressed, axis=0).tolist(), "raw_fz": fz, "force_sign": sign}


def rate_test(arm, cfg: BridgeConfig, seconds=5.0, runner: ForgePolicyRunner | None = None, step_m=0.002,
              clock=time.monotonic, sleep=time.sleep):
    """Hold the current pose while streaming it back at cfg.control_hz, then step the target
    step_m straight UP. Reports the achieved send rate, how often the robot state actually changes,
    the delay from the step command to the first 0.3 mm of motion and to 63% of the step (the sim
    loses 10-15 points with one 67 ms policy step of delay), and the policy's inference time."""
    st = arm.read()
    hold_pos, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(hold_pos, R, cfg.stiffness, max_duration=seconds + 10)
    sends, changes, last_q, last_f = 0, 0, None, None
    t0 = clock()
    next_tick = t0
    out = {}
    try:
        while clock() - t0 < seconds:
            s = arm.read()
            if last_q is None or not np.array_equal(s.q, last_q) or not np.array_equal(s.force, last_f):
                changes += 1
            last_q, last_f = s.q, s.force
            arm.send_target(hold_pos, R)
            sends += 1
            # Sleep only what is left of the tick, like run_attempt, so the reported rate is the one the
            # policy loop can reach (a full-period sleep on top of the read under-reported it).
            next_tick += 1.0 / cfg.control_hz
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
            else:
                next_tick = clock()
        out = {"send_hz": sends / seconds, "state_update_hz": changes / seconds}

        z0 = arm.read().tool_pos[2]
        up = hold_pos + np.array([0.0, 0.0, step_m])
        t_step, first, rise = clock(), None, None
        while clock() - t_step < 2.0 and rise is None:
            arm.send_target(up, R)
            dz = arm.read().tool_pos[2] - z0
            if first is None and dz > 0.0003:
                first = clock() - t_step
            if dz > 0.63 * step_m:
                rise = clock() - t_step
            sleep(1.0 / cfg.control_hz)
        out["step_first_motion_ms"] = None if first is None else 1000 * first
        out["step_63pct_ms"] = None if rise is None else 1000 * rise
    finally:
        arm.stop_stream()
    if runner is not None and runner.policy is not None:
        tip, quat, _ = tip_pose(st, cfg)
        runner.reset(tip, quat, tip - np.array([0.0, 0.0, START_ABOVE_RIM]))
        times = []
        for _ in range(30):
            t = time.perf_counter()
            runner.step(tip, quat, np.zeros(3), np.zeros(3), np.zeros(3))
            times.append(time.perf_counter() - t)
        out["policy_ms_median"] = 1000 * float(np.median(times))
        out["policy_ms_max"] = 1000 * float(np.max(times))
    return out


def force_curve(arm, cfg: BridgeConfig, descend_m_s=0.004, max_descent=0.08, stall_m=0.003, max_cmd_N=15.0,
                bins_N=1.0, clock=time.monotonic, sleep=time.sleep):
    """Measure how the force estimate responds to small forces (the TA's "below ~5 N" effect).

    Lowers the held peg onto a free flat surface at constant speed. Contact is detected from
    position (the target keeps descending, the fingertip stops), not from the force reading, so a
    dead band can't hide it. After contact the push grows slowly; the commanded push is
    stiffness_z * (fingertip - target) (minus whatever joint friction eats). Records the estimate's
    z against it up to max_cmd_N, then lifts back to the start.

    Returns the raw samples and a table: commanded push bin -> mean |estimate z| (tared).
    A clean sensor follows the commanded line; a dead band reads ~0 up to its width.
    """
    k = cfg.stiffness[2]
    tare = tare_force(arm, cfg.tare_s, cfg.control_hz, clock, sleep)
    st = arm.read()
    start_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    start_tip = tool_to_tip(start_tool, R, cfg.tip_offset)
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=max_descent / descend_m_s + 10)
    t0, samples, contact_t = clock(), [], None
    try:
        while True:
            t = clock() - t0
            depth = descend_m_s * t
            if depth > max_descent:
                raise HardwareError("no contact within the descent limit; is there a surface below?")
            target_tip = start_tip - np.array([0.0, 0.0, depth])
            st = arm.read()
            tip = tool_to_tip(st.tool_pos, st.tool_R, cfg.tip_offset)
            lag = tip[2] - target_tip[2]
            cmd = k * max(lag, 0.0)
            measured = st.force - tare
            if contact_t is None and lag > stall_m:
                contact_t = t
            samples.append({"t": t, "lag_m": lag, "cmd_N": cmd, "force": measured.tolist(), "contact": contact_t is not None})
            if cmd > max_cmd_N:
                break
            arm.send_target(tip_to_tool(target_tip, R, cfg.tip_offset), R)
            sleep(1.0 / cfg.control_hz)
    finally:
        arm.stop_stream()
        arm.goto(start_tool, R, 3.0)
    table = []
    for lo in np.arange(0.0, max_cmd_N, bins_N):
        sel = [abs(s["force"][2]) for s in samples if s["contact"] and lo <= s["cmd_N"] < lo + bins_N]
        if sel:
            table.append({"cmd_N": float(lo + bins_N / 2), "mean_abs_fz_N": float(np.mean(sel)), "n": len(sel)})
    free = [s["force"] for s in samples if not s["contact"]]
    return {"samples": samples, "table": table, "free_space_std_N": np.std(free, axis=0).tolist() if free else None}


# --- Force-estimate checks with a known reference (scale, weights) and drift ------------------------
#
# force_curve compares the estimate with the *commanded* push, which joint friction can eat, so it
# can't tell "the estimate misses small forces" (sensing dead band, the TA's warning) from "small
# commands never reach the surface" (actuation dead zone, which the sim already randomizes).
# force_scale and force_weights use a reference that doesn't depend on the robot; force_drift checks
# how far the tared estimate wanders in free space, where every reading is error.

G = 9.80665


def _hold(arm, tool_pos, R, cfg: BridgeConfig, seconds, clock, sleep, task=None, tare=None, samples=None,
          max_force_N=25.0):
    """Keep streaming tool_pos (a stream must be running) for at least `seconds` and until `task`
    returns. `task` runs in a thread (it waits for the operator) so the target keeps flowing.
    Appends tared force samples to `samples`. Returns the task's result."""
    import threading

    tare = np.zeros(3) if tare is None else tare
    box = {}

    def run():
        try:
            box["result"] = task()
        except BaseException as e:  # re-raised in the control thread
            box["error"] = e

    thread = None
    if task is not None:
        thread = threading.Thread(target=run, daemon=True)
        thread.start()
    t0 = clock()
    while clock() - t0 < seconds or (thread is not None and thread.is_alive()):
        f = arm.read().force - tare
        if np.linalg.norm(f) > max_force_N:
            raise SafetyStop(f"force {np.linalg.norm(f):.1f} N > {max_force_N} N")
        if samples is not None:
            samples.append(f)
        arm.send_target(tool_pos, R)
        sleep(1.0 / cfg.control_hz)
    if "error" in box:
        raise box["error"]
    return box.get("result")


def _ask_float(ask, prompt, allow_empty=False):
    while True:
        s = str(ask(prompt)).strip()
        if allow_empty and s == "":
            return None
        try:
            return float(s)
        except ValueError:
            print(f"not a number: {s!r}")


def _deadband_fit(x, y, min_y=0.5):
    """Fit y = a*x - b on the points where y > min_y (above any dead band). Returns slope a and the
    x-intercept b/a: the width of the dead band in the units of x. None with fewer than 2 points."""
    pts = [(xi, yi) for xi, yi in zip(x, y) if yi > min_y]
    if len(pts) < 2:
        return {"slope": None, "deadband_N": None, "points": len(pts)}
    a, c = np.polyfit([p[0] for p in pts], [p[1] for p in pts], 1)
    return {"slope": float(a), "deadband_N": float(-c / a) if a > 0 else None, "points": len(pts)}


def force_scale(arm, cfg: BridgeConfig, levels_N=(1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 8.0, 10.0), read_scale=input,
                descend_m_s=0.004, max_descent=0.08, stall_m=0.003, settle_s=1.0, hold_s=2.0,
                clock=time.monotonic, sleep=time.sleep):
    """Press the held peg on a kitchen scale at a ladder of commanded pushes; the operator types the
    scale reading (grams) at each. Separates the two effects force_curve mixes up:
        commanded -> scale : actuation loss (joint friction; the sim's dead zone models this)
        scale -> estimate  : sensing error (the TA's "unreliable below ~5 N")
    Start with the peg 1-5 cm above the scale, scale zeroed."""
    k = cfg.stiffness[2]
    tare = tare_force(arm, cfg.tare_s, cfg.control_hz, clock, sleep)
    st = arm.read()
    start_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    start_tip = tool_to_tip(start_tool, R, cfg.tip_offset)
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=3600)
    rows = []
    try:
        # Down until the fingertip stalls on the scale (from position, so a sensing dead band can't hide it).
        t0 = clock()
        while True:
            depth = descend_m_s * (clock() - t0)
            if depth > max_descent:
                raise HardwareError("no contact within the descent limit; is the scale below the peg?")
            target_tip = start_tip - np.array([0.0, 0.0, depth])
            st = arm.read()
            tip = tool_to_tip(st.tool_pos, st.tool_R, cfg.tip_offset)
            if tip[2] - target_tip[2] > stall_m:
                surface_z = float(tip[2])
                break
            if np.linalg.norm(st.force - tare) > 25.0:
                raise SafetyStop("force above 25 N while descending")
            arm.send_target(tip_to_tool(target_tip, R, cfg.tip_offset), R)
            sleep(1.0 / cfg.control_hz)
        for F in levels_N:
            target_tip = np.array([start_tip[0], start_tip[1], surface_z - F / k])  # spring: F = k * depth
            tool = tip_to_tool(target_tip, R, cfg.tip_offset)
            _hold(arm, tool, R, cfg, settle_s, clock, sleep)
            samples = []
            grams = _hold(arm, tool, R, cfg, hold_s, clock, sleep, tare=tare, samples=samples,
                          task=lambda F=F: _ask_float(read_scale, f"commanded {F:4.1f} N - scale reading (g)? "))
            est = np.mean(samples, axis=0)
            rows.append({"cmd_N": float(F), "scale_N": grams * G / 1000.0, "scale_g": grams, "est_N": est.tolist(),
                         "est_abs_z_N": float(abs(est[2])), "n": len(samples)})
    finally:
        arm.stop_stream()
        arm.goto(start_tool, R, 3.0)
    return {
        "rows": rows,
        "tare_N": tare.tolist(),
        # x-intercepts: the force at which the scale / the estimate start to respond
        "actuation": _deadband_fit([r["cmd_N"] for r in rows], [r["scale_N"] for r in rows]),
        "sensing": _deadband_fit([r["scale_N"] for r in rows], [r["est_abs_z_N"] for r in rows]),
    }


def force_weights(arm, cfg: BridgeConfig, ask_mass=input, direction=(0.0, 0.0, -1.0), force_sign=None,
                  settle_s=1.5, hold_s=2.0, max_items=30, clock=time.monotonic, sleep=time.sleep):
    """Hold still in free space; the operator hangs known masses from the peg or gripper (string loop)
    and types each total mass in grams (0 = nothing hanging, empty = done). A hanging mass is a
    known, static force with no contact and no controller in the way: m*g along `direction` (the way
    the string pulls the hand; -z for a plain hanging weight, +-x/+-y over a pulley).

    With force_sign (from press-test), also checks the sign independently: in the sim's convention
    (force of the environment on the hand) the reading along `direction` must be +m*g."""
    d = np.asarray(direction, dtype=float)
    d = d / np.linalg.norm(d)
    st = arm.read()
    hold_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(hold_tool, R, cfg.stiffness, max_duration=3600)
    rows = []
    try:
        _hold(arm, hold_tool, R, cfg, settle_s, clock, sleep)
        tare_samples = []
        _hold(arm, hold_tool, R, cfg, hold_s, clock, sleep, samples=tare_samples)
        tare = np.mean(tare_samples, axis=0)  # nothing hanging
        for _ in range(max_items):
            grams = _hold(arm, hold_tool, R, cfg, 0.0, clock, sleep, task=lambda: _ask_float(
                ask_mass, "hang the next mass and type its TOTAL grams (0 = nothing, Enter = done): ", allow_empty=True))
            if grams is None:
                break
            _hold(arm, hold_tool, R, cfg, settle_s, clock, sleep)
            samples = []
            _hold(arm, hold_tool, R, cfg, hold_s, clock, sleep, tare=tare, samples=samples)
            est = np.mean(samples, axis=0)
            along = float(est @ d)
            row = {"mass_g": grams, "true_N": grams * G / 1000.0, "est_N": est.tolist(), "est_along_abs_N": abs(along),
                   "est_off_axis_N": float(np.linalg.norm(est - along * d)), "std_N": np.std(samples, axis=0).tolist()}
            if force_sign is not None:
                row["est_along_sim_N"] = force_sign * along  # should be +true_N
            rows.append(row)
    finally:
        arm.stop_stream()
    out = {"rows": rows, "direction": d.tolist(), "tare_N": tare.tolist(),
           "sensing": _deadband_fit([r["true_N"] for r in rows], [r["est_along_abs_N"] for r in rows])}
    if force_sign is not None:
        loaded = [r for r in rows if r["true_N"] > 1.0 and r["est_along_abs_N"] > 0.5]
        out["sign_ok"] = all(r["est_along_sim_N"] > 0 for r in loaded) if loaded else None
    return out


def force_drift(arm, cfg: BridgeConfig, hold_s=15.0, sweep_m=0.04, sweep_m_s=0.01, max_force_N=10.0,
                clock=time.monotonic, sleep=time.sleep):
    """Tare like an attempt does, then (1) hold still for hold_s and (2) move straight down sweep_m and
    back, all in free space. Nothing touches the peg, so every reading is error: (1) is drift over an
    attempt's duration, (2) is the pose-dependent error over an attempt's range of motion. The sim
    loses a few points at 1 N of offset and 10-20 at 2 N. Needs sweep_m + 1 cm of free space below."""
    tare = tare_force(arm, cfg.tare_s, cfg.control_hz, clock, sleep)
    st = arm.read()
    start_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=hold_s + 2 * sweep_m / sweep_m_s + 10)
    hold, sweep = [], []
    try:
        t0 = clock()
        while clock() - t0 < hold_s:
            st = arm.read()
            hold.append((clock() - t0, st.force - tare))
            if np.linalg.norm(hold[-1][1]) > max_force_N:
                raise SafetyStop("force in free space; is something touching the peg?")
            arm.send_target(start_tool, R)
            sleep(1.0 / cfg.control_hz)
        t1, T = clock(), sweep_m / sweep_m_s
        while clock() - t1 < 2 * T:
            s = clock() - t1
            depth = sweep_m_s * (s if s < T else 2 * T - s)
            st = arm.read()
            sweep.append((float(st.tool_pos[2] - start_tool[2]), st.force - tare))
            if np.linalg.norm(sweep[-1][1]) > max_force_N:
                raise SafetyStop("force in free space; is something below the peg?")
            arm.send_target(start_tool - np.array([0.0, 0.0, depth]), R)
            sleep(1.0 / cfg.control_hz)
    finally:
        arm.stop_stream()
        arm.goto(start_tool, R, 3.0)
    ht = np.array([t for t, _ in hold])
    hf = np.array([f for _, f in hold])
    out = {
        "hold_s": hold_s,
        "hold_max_abs_N": np.abs(hf).max(axis=0).tolist(),
        "hold_end_mean_N": hf[ht > hold_s - 1.0].mean(axis=0).tolist(),
        "hold_slope_N_per_s": [float(np.polyfit(ht, hf[:, i], 1)[0]) for i in range(3)],
        "hold_std_N": hf.std(axis=0).tolist(),
    }
    if sweep:
        sz = np.array([z for z, _ in sweep])
        sf = np.array([f for _, f in sweep])
        out["sweep_m"] = sweep_m
        out["sweep_max_abs_N"] = np.abs(sf).max(axis=0).tolist()
        out["sweep_slope_N_per_cm"] = [float(np.polyfit(sz, sf[:, i], 1)[0]) / 100.0 if np.ptp(sz) > 1e-4 else None
                                       for i in range(3)]
    out["worst_norm_N"] = float(max(np.linalg.norm(hf, axis=1).max(),
                                    np.linalg.norm(np.array([f for _, f in sweep]), axis=1).max() if sweep else 0.0))
    return out
