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
DEPTH_TOL_M = 0.0005  # register-hole warns when the measured bore depth is further than this from HOLE_DEPTH
DEPTH_TOUCH_FROM_AXIS_M = (0.0085, 0.013)  # peg-on-top touch: clear of the 9 mm bore, well inside the corner holes
# Hole mount (session 4 on): a cuboid with the 9 mm main hole in the middle and four 8.15 mm locating holes whose
# centres sit on a 60 x 60 mm square around it. The 7.98 mm peg has +-0.085 mm play in a locating hole and +-0.51 mm
# in the main one, so the corners pin the main hole's centre (their centroid) far tighter than seating it does.
CORNER_PITCH_M = 0.060
PEG_D_M, MAIN_HOLE_D_M, CORNER_HOLE_D_M = 0.00798, 0.0090, 0.00815
SQUARE_TOL_M = 0.0005  # corner sides/diagonals vs 60 / 84.85 mm: more = a corner wasn't seated or the print is off
# Main-hole check insertion vs the corners' centre: the peg's play in the main hole plus 0.1 mm for the fit and print.
CENTRE_TOL_M = (MAIN_HOLE_D_M - PEG_D_M) / 2 + 0.0001
SEATED_TIP_ABOVE_RIM = GRASP_TIP_ABOVE_PEG_BOTTOM - HOLE_DEPTH  # fingertip z above the rim when seated
START_ABOVE_RIM = 0.047  # Factory hand_init_pos z
START_NOISE = (0.02, 0.02, 0.01)  # Factory hand_init_pos_noise (uniform ±)
START_YAW_NOISE = 0.785  # Factory hand_init_orn_noise z (uniform ±, rad)
SUCCESS_XY = 0.0025  # Factory: peg within 2.5 mm of the hole axis ...
SUCCESS_Z = 0.04 * HOLE_DEPTH  # ... and within success_threshold * height (1 mm) of seated
DOWN = np.diag([1.0, -1.0, -1.0])  # gripper pointing down, yaw 0 (Lab2's TOP_R; sim roll = pi)
SIM_FORCE_UP_WHEN_PRESSING = 1.0  # measure_frames.py: pressing down on a surface reads +z in sim

# --- Robot state (FrankaPy robot_num 1) -----------------------------------------------------------

ROBOT_STATE_TOPIC = "/robot_state_publisher_node_1/robot_state"  # franka_interface_msgs/RobotState, 100 Hz in the lab
MAX_STATE_AGE_S = 0.1  # 10 missed messages: treat the state as lost and stop


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
    def __init__(self, msg, kind=None):
        super().__init__(msg)
        self.kind = kind  # "tilt"/"force" for those limits; run_attempt treats them as benign once the peg is seated


@dataclass
class RobotState:
    t: float  # monotonic clock [s]
    q: np.ndarray
    dq: np.ndarray
    tool_pos: np.ndarray
    tool_R: np.ndarray
    force: np.ndarray  # O_F_ext_hat_K[:3], base-frame axes, raw (no sign flip, no tare)
    J: np.ndarray | None = None  # 6x7 tool-point Jacobian [v; w], if available
    tau_J: np.ndarray | None = None  # measured joint torques [Nm] (libfranka tau_J), topic only
    tau_ext: np.ndarray | None = None  # estimated external joint torques [Nm] (tau_ext_hat_filtered), topic only

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


GRASP_MIN_M = 0.0035  # peg grasp widths FrankaPyArm.grasp accepts; libfranka's epsilon_inner matches the minimum
GRASP_MAX_M = 0.010
GRASP_FORCE_N = 60.0
GRASP_FORCE_MAX_N = 70.0  # Franka Hand continuous grasp force


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
        # FrankaPy's get_pose() is O_T_EE * tool_delta_pose (identity unless someone sets it); the raw
        # state (topic or get_robot_state) is O_T_EE only, so apply the same delta to match get_pose().
        delta = getattr(self.fa, "_tool_delta_pose", None)
        self._tool_delta = np.asarray(delta.matrix, dtype=float) if delta is not None else np.eye(4)
        # Fallback: one service request for the whole state (~1.5 ms) instead of four (~6.5 ms). On the
        # lab robot its answer only refreshes at ~30 Hz, while the topic below publishes at 100 Hz.
        self.has_robot_state = hasattr(self.fa, "get_robot_state")
        self.state_source = "get_robot_state" if self.has_robot_state else "separate calls"
        self._latest = None  # (RobotState msg, monotonic receive time), written by the ROS callback thread
        self._subscribe(rospy)

    def _subscribe(self, rospy, wait_s=1.0):
        """Listen to FrankaPy's robot-state topic; read() then costs no request at all."""
        try:
            from franka_interface_msgs.msg import RobotState as RobotStateMsg
        except ImportError:
            return
        sub = rospy.Subscriber(ROBOT_STATE_TOPIC, RobotStateMsg, self._on_state, queue_size=1, tcp_nodelay=True)
        end = time.monotonic() + wait_s
        while self._latest is None and time.monotonic() < end:
            time.sleep(0.01)
        if self._latest is None:
            sub.unregister()
            print(f"WARNING: no message on {ROBOT_STATE_TOPIC} within {wait_s:.0f} s; reading state with "
                  f"{self.state_source} (slower, ~30 Hz fresh data)")
        else:
            self.state_source = "topic"

    def _on_state(self, msg):
        self._latest = (msg, time.monotonic())  # one tuple assignment: read() never sees a half update

    def _pose(self, pos, R):
        return self._m["RigidTransform"](
            rotation=np.asarray(R, dtype=float), translation=np.asarray(pos, dtype=float), from_frame="franka_tool", to_frame="world"
        )

    def read(self) -> RobotState:
        """Latest robot state. `t` is when that state arrived (topic) or was fetched (fallbacks), so
        two reads of the same message have the same `t`."""
        fa = self.fa
        if self.state_source == "topic":
            msg, t = self._latest
            age = time.monotonic() - t
            if age > MAX_STATE_AGE_S:
                raise SafetyStop(f"robot state is {1000 * age:.0f} ms old; {ROBOT_STATE_TOPIC} stopped publishing?")
            T = np.asarray(msg.O_T_EE, dtype=float).reshape(4, 4).T @ self._tool_delta  # as FrankaPy's utils
            q, dq, ft = msg.q, msg.dq, msg.O_F_ext_hat_K
            tau_J, tau_ext = getattr(msg, "tau_J", None), getattr(msg, "tau_ext_hat_filtered", None)
        elif self.has_robot_state:
            t = time.monotonic()
            s = fa.get_robot_state()
            T = np.asarray(s["pose"].matrix, dtype=float) @ self._tool_delta
            q, dq, ft = s["joints"], s["joint_velocities"], s["ee_force_torque"]
            tau_J, tau_ext = None, None
        else:
            t = time.monotonic()
            T = np.asarray(fa.get_pose().matrix, dtype=float)  # get_pose() already includes the tool delta
            q, dq, ft = fa.get_joints(), fa.get_joint_velocities(), fa.get_ee_force_torque()
            tau_J, tau_ext = None, None
        q = np.asarray(q, dtype=float)
        J = np.asarray(fa.get_jacobian(q), dtype=float) if self.has_jacobian else None  # local, no request
        return RobotState(
            t=t,
            q=q,
            dq=np.asarray(dq, dtype=float),
            tool_pos=T[:3, 3].copy(),
            tool_R=T[:3, :3].copy(),
            force=np.asarray(ft, dtype=float)[:3],
            J=J if J is not None and J.shape == (6, 7) else None,
            tau_J=_joint_vector(tau_J),
            tau_ext=_joint_vector(tau_ext),
        ).check()

    def compare_with_service(self) -> dict:
        """Topic state vs FrankaPy's own get_pose()/get_joints(): they should agree to well under a mm."""
        st = self.read()
        pose, q = self.fa.get_pose(), np.asarray(self.fa.get_joints(), dtype=float)
        dR = st.tool_R @ np.asarray(pose.rotation, dtype=float).T
        # Small-angle form from the skew part: acos((trace - 1) / 2) is ill-conditioned near zero and turns
        # O_T_EE's ~2e-5 non-orthonormality into a fake ~0.36 deg even for identical matrices.
        skew = np.array([dR[2, 1] - dR[1, 2], dR[0, 2] - dR[2, 0], dR[1, 0] - dR[0, 1]]) / 2.0
        return {
            "source": self.state_source,
            "state_age_ms": 1000 * (time.monotonic() - st.t),
            "pos_diff_mm": 1000 * float(np.linalg.norm(st.tool_pos - np.asarray(pose.translation, dtype=float))),
            "rot_diff_deg": math.degrees(math.asin(min(1.0, float(np.linalg.norm(skew))))),
            "joint_diff_max_rad": float(np.max(np.abs(st.q - q))),
        }

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

    def grasp(self, width=0.008, force=GRASP_FORCE_N) -> float:
        """Close on the peg. 40 N let it slip when pushed into the hole by hand (session 3); 60 N by default,
        at most the Franka Hand's 70 N continuous. The lab gripper's
        fingers have divots that centre the peg, so they close past its 8 mm diameter (6.3-6.4 mm measured, down
        to ~4 mm possible); GRASP_MIN_M only catches a grasp that missed the peg (fingers closed to ~0)."""
        if not 0.0 < force <= GRASP_FORCE_MAX_N:
            raise ValueError(f"grasp force {force} N outside (0, {GRASP_FORCE_MAX_N}]")
        self.fa.goto_gripper(
            width=width, grasp=True, speed=0.02, force=force, epsilon_inner=width - GRASP_MIN_M,
            epsilon_outer=0.004, block=True, ignore_errors=False,
        )
        got = self.wait_gripper_settled()
        if not GRASP_MIN_M <= got <= GRASP_MAX_M:
            raise HardwareError(f"unexpected peg grasp width {got:.4f} m")
        return got

    def open(self) -> float:
        self.fa.open_gripper(block=True)
        return self.wait_gripper_settled()

    def wait_gripper_settled(self, timeout_s=10.0, still_s=0.5, tol_m=0.0002, clock=time.monotonic, sleep=time.sleep):
        """Gripper width once the fingers stop moving.

        FrankaPy's goto_gripper(block=True) doesn't wait for grasp or move goals: it never records them as
        the last gripper command, so wait_for_gripper() just sleeps 2 s. Closing from fully open to the
        peg at 2 cm/s takes over 3 s, so the width was read mid-motion (~45 mm in the lab)."""
        end = clock() + timeout_s
        last, still_since = None, None
        while clock() < end:
            w, now = float(self.fa.get_gripper_width()), clock()
            if last is not None and abs(w - last) < tol_m:
                still_since = now if still_since is None else still_since
                if now - still_since >= still_s:
                    return w
            else:
                still_since = None
            last = w
            sleep(0.1)
        raise HardwareError(f"gripper still moving after {timeout_s:.0f} s (width {last:.4f} m)")

    def guide(self, duration, block=True):
        """Hand-guiding (zero-stiffness) mode."""
        self.fa.run_guide_mode(duration=float(duration), block=block)

    def skill_done(self) -> bool:
        return bool(self.fa.is_skill_done())

    def wait_skill(self):
        self.fa.wait_for_skill()

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
    surface_tip: list | None = None  # fingertip with the peg bottom resting on the top face beside the hole
    corners: list | None = None  # seated fingertips in the four locating holes (corner registration)
    centre_sample: list | None = None  # seated fingertip in the main hole, the check insertion

    @property
    def square_error_m(self) -> float | None:
        """Worst deviation of the corners' 4 sides and 2 diagonals from the nominal square (order-independent)."""
        if self.corners is None:
            return None
        c = np.asarray(self.corners)[:, :2]
        d = sorted(float(np.linalg.norm(c[i] - c[j])) for i in range(4) for j in range(i + 1, 4))
        want = [CORNER_PITCH_M] * 4 + [CORNER_PITCH_M * math.sqrt(2.0)] * 2
        return float(max(abs(a - b) for a, b in zip(d, want)))

    @property
    def centre_offset_m(self) -> float | None:
        """xy distance of the main-hole check insertion from the centre the corners give."""
        if self.centre_sample is None:
            return None
        return float(np.linalg.norm(np.asarray(self.centre_sample)[:2] - np.asarray(self.seated_tip)[:2]))

    @property
    def depth_m(self) -> float | None:
        """Measured bore depth: the peg bottom's drop from the boss top to seated. The grip height cancels out,
        so this checks the one real dimension the policy relies on (the sim's HOLE_DEPTH)."""
        return None if self.surface_tip is None else float(self.surface_tip[2] - self.seated_tip[2])

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
        Path(path).write_text(json.dumps({**asdict(self), "rim": self.rim.tolist(), "spread_m": self.spread_m,
                                          "depth_m": self.depth_m, "square_error_m": self.square_error_m,
                                          "centre_offset_m": self.centre_offset_m}, indent=2))

    @classmethod
    def load(cls, path) -> "Hole":
        d = json.loads(Path(path).read_text())
        return cls(seated_tip=d["seated_tip"], samples=d.get("samples", []), surface_tip=d.get("surface_tip"),
                   corners=d.get("corners"), centre_sample=d.get("centre_sample"))


def guide_countdown(arm, seconds, what, out=print, clock=time.monotonic, sleep=time.sleep):
    """Guide mode for `seconds` with a live countdown on one line, then wait until the arm is stiff again.
    Uses the arm's own skill status when it has one (the guide skill starts a moment after the request)."""
    arm.guide(seconds, block=False)
    t0, shown = clock(), None
    done = getattr(arm, "skill_done", None)
    while True:
        left = seconds - (clock() - t0)
        if (done() if done is not None else left <= 0.0) or left < -5.0:
            break
        s = max(0, math.ceil(left))
        if s != shown:
            out(f"\r  ARM COMPLIANT: {s:2d} s left -- {what}   ", end="", flush=True)
            shown = s
        sleep(0.1)
    if hasattr(arm, "wait_skill"):
        arm.wait_skill()
    out("\a\r  ARM STIFF again (guide mode over)." + " " * 40, flush=True)


def _guided_seat(arm, title, hole_name, confirm, guide_s, lift_s, out, clock, sleep):
    """One hand-guided insertion: seat the peg, read the seated fingertip, lift it out again."""
    out(f"\n{title} Arm is STIFF. Put your hands on it, then:")
    confirm(f"  Press Enter -> compliant for {guide_s:.0f} s. Seat the peg fully in the {hole_name} (gently, stop "
            "when it bottoms out), then hold it still until the countdown ends.")
    guide_countdown(arm, guide_s, f"seat the peg in the {hole_name}, then hold still", out, clock, sleep)
    st = arm.read()
    tip = tool_to_tip(st.tool_pos, st.tool_R)
    out(f"  seated fingertip {np.round(tip, 4).tolist()}")
    confirm(f"  The arm holds the peg in the hole. Press Enter -> compliant for {lift_s:.0f} s to lift it out.")
    guide_countdown(arm, lift_s, "lift the peg out of the hole", out, clock, sleep)
    confirm("  Peg out and clear of the mount, not moved in the fingers (else Ctrl-C, regrasp, start over)? "
            "Press Enter to go on.")
    return tip


def register_hole(arm, confirm=input, guide_s=15.0, lift_s=10.0, out=print, clock=time.monotonic,
                  sleep=time.sleep, measure_depth=True, touch_s=10.0) -> Hole:
    """Register the main hole from the mount's four 8.15 mm locating holes.

    One hand-guided insertion into each corner hole (any order); the main hole's xy is their centroid (for four
    corners of a square that is also the least-squares fit, whatever its rotation). Then one insertion into the main
    hole: its seated height is the registration's z, and its xy is checked against the corners' centre. Last, the
    depth check: the peg bottom on the top face beside the main hole. Between insertions the arm is stiff; every
    prompt says what Enter does, and guide mode shows a countdown."""
    corners = []
    for k in range(4):
        corners.append(_guided_seat(arm, f"[corner {k + 1}/4] (the corner holes in any order, going around)",
                                    "corner hole", confirm, guide_s, lift_s, out, clock, sleep).tolist())
    centre_xy = np.mean(np.asarray(corners)[:, :2], axis=0)
    main = _guided_seat(arm, "[main hole] The 9 mm hole in the middle, once, as a check.", "main (middle) hole",
                        confirm, guide_s, lift_s, out, clock, sleep)
    hole = Hole(seated_tip=[float(centre_xy[0]), float(centre_xy[1]), float(main[2])], samples=corners + [main.tolist()],
                corners=corners, centre_sample=main.tolist())
    sq, off = hole.square_error_m, hole.centre_offset_m
    zs = np.asarray(corners)[:, 2]
    out(f"\nmain hole rim {np.round(hole.rim, 4).tolist()} (xy from the corners, z from the main hole)")
    out(f"  corner square: worst side/diagonal error {1000 * sq:.2f} mm (want < {1000 * SQUARE_TOL_M:.1f}); "
        f"corner seated heights span {1000 * (zs.max() - zs.min()):.2f} mm")
    out(f"  main-hole insertion {1000 * off:.2f} mm from the corners' centre "
        f"(want < {1000 * CENTRE_TOL_M:.2f}: the peg's play in the 9 mm hole)")
    if sq > SQUARE_TOL_M:
        out("  WARNING: the corners don't form the 60 mm square: a corner wasn't seated, the wrong hole was used, or "
            "the mount moved. Repeat the registration.")
    if off > CENTRE_TOL_M:
        out("  WARNING: the main hole isn't where the corners put it: the mount moved, or the main hole is printed off "
            "centre. Check the clamps and repeat.")
    if measure_depth:
        lo, hi = (1000 * r for r in DEPTH_TOUCH_FROM_AXIS_M)
        out("\nDepth check. Arm is STIFF. Put your hands on it, then:")
        confirm(f"  Press Enter -> compliant for {touch_s:.0f} s. Rest the peg bottom flat on the top face, "
                f"about 1 cm ({lo:.0f}-{hi:.0f} mm) from the main hole's centre, then hold it still.")
        guide_countdown(arm, touch_s, "peg flat on the top face beside the main hole, hold still", out, clock, sleep)
        st = arm.read()
        tip = tool_to_tip(st.tool_pos, st.tool_R)
        toff = float(np.linalg.norm(tip[:2] - np.asarray(hole.seated_tip)[:2]))
        hole.surface_tip = tip.tolist()
        out(f"  touch {1000 * toff:.1f} mm from the main hole's axis; bore depth {1000 * hole.depth_m:.1f} mm "
            f"(sim {1000 * HOLE_DEPTH:.1f})")
        if not DEPTH_TOUCH_FROM_AXIS_M[0] <= toff <= DEPTH_TOUCH_FROM_AXIS_M[1]:
            out(f"  WARNING: touch should be {lo:.0f}-{hi:.0f} mm from the main hole's axis (on the top face, clear of "
                "the bore); the depth may be wrong. Repeat the registration.")
        elif abs(hole.depth_m - HOLE_DEPTH) > DEPTH_TOL_M:
            out(f"  WARNING: depth is {1000 * (hole.depth_m - HOLE_DEPTH):+.1f} mm off the sim's: the policy will meet "
                "the top face that much lower (+) or higher (-) than in sim. Check the bore print and that the peg "
                "is fully home in its collar.")
        confirm(f"  The arm holds the peg on the mount. Press Enter -> compliant for {lift_s:.0f} s to lift it off.")
        guide_countdown(arm, lift_s, "lift the peg clear of the mount", out, clock, sleep)
    return hole


# --- Safety -----------------------------------------------------------------------------------------


@dataclass
class SafetyLimits:
    max_force_N: float = 25.0  # tared, smoothed force norm
    box_xy_m: float = 0.04  # around the registered hole and the estimate
    tip_floor_above_rim_m: float = SEATED_TIP_ABOVE_RIM - 0.003  # below seated = peg slipped or wrong hole
    box_top_above_est_m: float = 0.09
    max_speed_m_s: float = 0.3
    max_tilt_deg: float = 10.0  # gripper axis from straight down; trial 2 of session 3 ran away to 37 deg
    # Robot state (joints + force estimate) bit-identical this long = it stopped updating. Session 3 trial 2:
    # messages kept arriving (so read()'s age check passed) but their content froze for seconds.
    max_frozen_s: float = 0.1


def tilt_deg(R) -> float:
    """Angle between the gripper axis and straight down."""
    return math.degrees(math.acos(float(np.clip(-np.asarray(R)[2, 2], -1.0, 1.0))))


class FreezeCheck:
    """SafetyStop when the robot state's content stops changing (joint angles and force estimate are noisy,
    so real updates never repeat exactly)."""

    def __init__(self, max_frozen_s):
        self.max_frozen_s, self.last, self.since = max_frozen_s, None, None

    def update(self, st, now):
        key = (tuple(np.asarray(st.q, dtype=float)), tuple(np.asarray(st.force, dtype=float)))
        if key != self.last:
            self.last, self.since = key, now
        elif now - self.since > self.max_frozen_s:
            raise SafetyStop(f"robot state unchanged for {1000 * (now - self.since):.0f} ms: "
                             "the controller stopped updating (skill ended on the control PC?)")


class SafetyMonitor:
    def __init__(self, limits: SafetyLimits, hole_rim, hole_est):
        self.lim = limits
        self.centres = [np.asarray(hole_rim, dtype=float)[:2], np.asarray(hole_est, dtype=float)[:2]]
        self.z_min = float(hole_rim[2]) + limits.tip_floor_above_rim_m
        self.z_max = max(float(hole_rim[2]), float(hole_est[2])) + limits.box_top_above_est_m

    def check(self, tip, force_norm, speed, R=None):
        lim = self.lim
        if R is not None:
            tilt = tilt_deg(R)
            if tilt > lim.max_tilt_deg:
                raise SafetyStop(f"tilt {tilt:.1f} deg > {lim.max_tilt_deg} deg", kind="tilt")
        if force_norm > lim.max_force_N:
            raise SafetyStop(f"force {force_norm:.1f} N > {lim.max_force_N} N", kind="force")
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
    # Policy attempts only: rotation 80 Nm/rad instead of the sim's 28. At 28 the real wrist started every
    # session-3 attempt 2-3 deg off vertical and side loads on the peg tipped it 10-37 deg; the policy can't
    # see tilt, and in sim the peg stays upright. Translation stays at the sim's 565 N/m.
    attempt_stiffness: tuple = (565.0, 565.0, 565.0, 80.0, 80.0, 80.0)
    timeout_s: float = 10.0  # sim episode length (150 policy steps)
    # End the attempt when the policy's success prediction exceeds this; None: never (the sim doesn't either).
    # On the robot the prediction sits at 0.47-0.52 for the whole approach: in session 4 it crossed 0.5 at
    # 1.1-1.9 s with the peg still on the top face and ended all three 3.8 attempts.
    pred_threshold: float | None = None
    # End the attempt this long after the peg is first seated (outcome "seated"); None: run on. In session 3 the
    # policy shoved ~13 mm sideways into the bore wall right after seating and tilted the arm to 10-37 deg. The
    # success prediction came 0.14-0.25 s after seating when it came. Success is latched like the sim's.
    stop_after_seated_s: float | None = 0.3
    tare_s: float = 0.5
    quat_ref: tuple = (0.0, 1.0, 0.0, 0.0)  # sign convention for observed quats; confirm against the sim
    tip_offset: float = TIP_OFFSET


def tare_force(arm, seconds, hz=200.0, clock=time.monotonic, sleep=time.sleep) -> np.ndarray:
    """Mean raw force over `seconds` in free space; subtract it for the rest of the attempt.
    Only for when no stream is running; with one, use stream_tare."""
    samples, end = [], clock() + seconds
    while clock() < end:
        samples.append(arm.read().force)
        sleep(1.0 / hz)
    return np.mean(samples, axis=0)


STREAM_SETTLE_S = 0.5  # after start_stream, before taring


def stream_tare(arm, tool_pos, R, cfg: BridgeConfig, clock=time.monotonic, sleep=time.sleep) -> np.ndarray:
    """Tare with the streaming controller already running (call right after start_stream, holding
    tool_pos). Starting the impedance skill shifts the force estimate: ~1.7 N in z on the lab robot
    (force-drift, 2026-11-14), which a tare taken before start_stream leaves in every reading."""
    _hold(arm, tool_pos, R, cfg, STREAM_SETTLE_S, clock, sleep)
    samples = []
    _hold(arm, tool_pos, R, cfg, cfg.tare_s, clock, sleep, samples=samples)
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

    result = {"outcome": None, "success": False, "time_to_success_s": None, "time_to_prediction_s": None,
              "peak_force_smooth_N": 0.0, "mean_force_smooth_N": 0.0, "policy_steps": 0, "error": None,
              "tare_N": None, "hole_est": hole_est.tolist(), "hole_rim": rim.tolist()}
    max_steps = int(round(cfg.timeout_s / rc.policy_dt))
    prev, prev_now, force_sum, ticks = None, None, 0.0, 0
    st = arm.read()
    hold_tool, hold_R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(hold_tool, hold_R, cfg.attempt_stiffness, max_duration=cfg.timeout_s + STREAM_SETTLE_S + cfg.tare_s + 5.0)
    try:
        tare = stream_tare(arm, hold_tool, hold_R, cfg, clock, sleep)
        result["tare_N"] = tare.tolist()
        st = arm.read()
        tip, quat, R = tip_pose(st, cfg)
        runner.reset(tip, quat, hole_est)
        ff = ForceFilter(rc.ft_smoothing_factor, rc.sim_dt)
        ff.reset()
        t0 = clock()
        next_policy, next_tick = t0, t0
        frozen = FreezeCheck(limits.max_frozen_s)
        while True:
            st = arm.read()
            now = clock()
            frozen.update(st, now)
            # Filter over the time since the last tick, holding the latest sample (state arrives at 100 Hz,
            # ticks run at up to 200 Hz, so a sample can be read twice; using st.t would over-weight it).
            dt = now - prev_now if prev_now is not None else 1.0 / cfg.control_hz
            fs = ff.update(st.force - tare, dt)  # runner.build_obs applies force_sign
            tip, quat, R = tip_pose(st, cfg)
            v, w = tip_velocity(st, prev, cfg)
            fnorm = float(np.linalg.norm(fs))
            safety.check(tip, fnorm, float(np.linalg.norm(v)), R)
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
                                      "force_smooth": fs.tolist(), "success_prob": p_succ,
                                      "tool_R": R.tolist(), "tilt_deg": tilt_deg(R)})
                next_policy += rc.policy_dt
                if cfg.pred_threshold is not None and p_succ > cfg.pred_threshold:
                    result["outcome"], result["time_to_prediction_s"] = "predicted_success", now - t0
                    break
                if result["policy_steps"] >= max_steps:
                    result["outcome"] = "timeout"
                    break
            if (cfg.stop_after_seated_s is not None and result["time_to_success_s"] is not None
                    and now - t0 - result["time_to_success_s"] >= cfg.stop_after_seated_s):
                result["outcome"] = "seated"
                break
            else:
                target_pos, target_quat = runner.target(tip, quat)

            target_R = quat_to_mat(target_quat)
            arm.send_target(tip_to_tool(target_pos, target_R, cfg.tip_offset), target_R)
            prev, prev_now = st, now
            next_tick += 1.0 / cfg.control_hz
            delay = next_tick - clock()
            if delay > 0:
                sleep(delay)
            else:
                next_tick = clock()  # fell behind; don't try to catch up with a burst
    except SafetyStop as exc:
        if exc.kind in ("tilt", "force") and result["time_to_success_s"] is not None:
            # After seating, the policy's sideways push tips the gripper (session 4: 4-10 deg within the
            # stop_after_seated_s window) and presses the seated peg into the bore (26 N once). The attempt
            # still ends here, but as seated, not as a safety stop.
            result["outcome"], result["post_seat_stop"] = "seated", str(exc)
        else:
            result["outcome"], result["error"], result["stop_kind"] = "safety_stop", str(exc), exc.kind
    finally:
        arm.stop_stream()

    st = arm.read()
    tip = tool_to_tip(st.tool_pos, st.tool_R, cfg.tip_offset)
    result["final_tip"] = tip.tolist()
    # Like the sim (ep_succeeded latches): success = seated at any point. Also record the end pose's check.
    result["seated_at_end"] = bool(np.linalg.norm(tip[:2] - seated[:2]) < SUCCESS_XY and tip[2] < seated[2] + SUCCESS_Z)
    result["success"] = result["time_to_success_s"] is not None
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
    st = arm.read()
    start, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(start, R, cfg.stiffness, max_duration=max_descent / descend_m_s + 12)
    try:
        tare = stream_tare(arm, start, R, cfg, clock, sleep)
        t0, pressed, contact = clock(), [], None
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
    st = arm.read()
    start_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    start_tip = tool_to_tip(start_tool, R, cfg.tip_offset)
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=max_descent / descend_m_s + 12)
    samples, contact_t = [], None
    try:
        tare = stream_tare(arm, start_tool, R, cfg, clock, sleep)
        t0 = clock()
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
    st = arm.read()
    start_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    start_tip = tool_to_tip(start_tool, R, cfg.tip_offset)
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=3600)
    rows, tare = [], np.zeros(3)
    try:
        tare = stream_tare(arm, start_tool, R, cfg, clock, sleep)
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
    """Tare like an attempt does (stream running), then (1) hold still for hold_s and (2) move straight down sweep_m and
    back, all in free space. Nothing touches the peg, so every reading is error: (1) is drift over an
    attempt's duration, (2) is the pose-dependent error over an attempt's range of motion. The sim
    loses a few points at 1 N of offset and 10-20 at 2 N. Needs sweep_m + 1 cm of free space below."""
    pre_stream = tare_force(arm, cfg.tare_s, cfg.control_hz, clock, sleep)  # what the old code tared on
    st = arm.read()
    start_tool, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=hold_s + 2 * sweep_m / sweep_m_s + 12)
    hold, sweep, tare = [], [], np.zeros(3)
    try:
        tare = stream_tare(arm, start_tool, R, cfg, clock, sleep)
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
        # starting the controller moves the estimate by this much; the old tare-before-stream kept it
        "stream_start_shift_N": (tare - pre_stream).tolist(),
        "tare_N": tare.tolist(),
        "hold_series": [[round(float(t), 4)] + [round(float(x), 4) for x in f] for t, f in hold],  # t, fx, fy, fz
        "sweep_series": [[round(float(z), 5)] + [round(float(x), 4) for x in f] for z, f in sweep],  # dz, fx, fy, fz
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


# --- Joint friction (session 2) -------------------------------------------------------------------
#
# Session 1 (TEST_SUMMARY.md, test 6): a 2 mm target step (1.1 N of spring pull) didn't move the arm,
# ~2 N was needed to start moving, and moving made the force estimate read 1.5-2.4 N opposing the
# motion. The policy commands targets a few mm from the fingertip (1-5 N of pull) near the hole, and
# the sim has no friction. These three checks measure what a sim friction model (or a controller
# compensation) needs. All of them run in free space: nothing may touch the peg.
#
# The spring pull along an axis is stiffness * (target - hand). At rest it is what friction holds
# back; at a steady speed v it is Fc + b*v, where b also includes the controller's own damping.

AXES = {"x": np.array([1.0, 0.0, 0.0]), "y": np.array([0.0, 1.0, 0.0]), "z": np.array([0.0, 0.0, 1.0])}
FRICTION_MAX_FORCE_N = 10.0  # free space: a tared reading above this means something touches the peg
FRICTION_MAX_TRAVEL_M = 0.05  # the hand never goes further than this from where the check started


JOINT_LOG_COLUMNS = ["t"] + [f"{name}{j}" for name in ("q", "dq", "tau_J", "tau_ext") for j in range(1, 8)]


class JointLog:
    """Joint state alongside a friction check, so a per-joint friction model (e.g. Gaz et al. 2019) can be
    fitted later without another lab session. One row per new robot state (100 Hz on the topic; the
    control loop's repeated reads of the same message are skipped): t since the check's start, q [rad],
    dq [rad/s], tau_J and tau_ext [Nm] (None where the state source doesn't provide them)."""

    def __init__(self):
        self.rows = []
        self._last_t = None

    def add(self, t_rel, st):
        if st.t == self._last_t:
            return
        self._last_t = st.t
        row = [round(float(t_rel), 4)]
        for v, digits in ((st.q, 5), (st.dq, 5), (st.tau_J, 4), (st.tau_ext, 4)):
            row += [None] * 7 if v is None else [round(float(x), digits) for x in v]
        self.rows.append(row)


def _joint_vector(v):
    if v is None:
        return None
    v = np.asarray(v, dtype=float)
    return v if v.shape == (7,) and np.isfinite(v).all() else None


def _free_tick(arm, cfg: BridgeConfig, target_tool, R, tare, home, sleep, max_force_N=FRICTION_MAX_FORCE_N,
               max_travel_m=FRICTION_MAX_TRAVEL_M):
    """One control tick in free space: read, check, send the target, sleep. Returns (state, tared force)."""
    st = arm.read()
    f = st.force - tare
    if np.linalg.norm(f) > max_force_N:
        raise SafetyStop(f"force {np.linalg.norm(f):.1f} N in free space; is something touching the peg?")
    if np.linalg.norm(st.tool_pos - home) > max_travel_m:
        raise SafetyStop(f"hand moved more than {100 * max_travel_m:.0f} cm from the start")
    arm.send_target(target_tool, R)
    sleep(1.0 / cfg.control_hz)
    return st, f


def _hold_free(arm, cfg, target_tool, R, tare, home, seconds, clock, sleep):
    t0 = clock()
    st = None
    while clock() - t0 < seconds:
        st, _ = _free_tick(arm, cfg, target_tool, R, tare, home, sleep)
    return st


def _move_free(arm, cfg, a, b, R, tare, home, speed_m_s, clock, sleep, record=None):
    """Move the target from a to b at constant speed. Appends (t, target, state, force) to record."""
    a, b = np.asarray(a, float), np.asarray(b, float)
    T = max(np.linalg.norm(b - a) / speed_m_s, 1e-9)
    t0 = clock()
    while True:
        s = min((clock() - t0) / T, 1.0)
        target = a + s * (b - a)
        st, f = _free_tick(arm, cfg, target, R, tare, home, sleep)
        if record is not None:
            record.append((clock() - t0, target.copy(), st, f))
        if s >= 1.0:
            return


def friction_breakaway(arm, cfg: BridgeConfig, directions=("+x", "-x", "+y", "-y", "+z", "-z"), ramp_m_s=0.0005,
                       moved_m=0.0005, max_pull_N=8.0, settle_s=1.0, force_sign=1.0, clock=time.monotonic,
                       sleep=time.sleep):
    """Stiction. For each direction, move the target away from the hand slowly (0.5 mm/s: the spring
    pull grows by 0.28 N/s at 565 N/m) until the hand has moved `moved_m`. The breakaway force is the
    largest pull before that (just before it slips). Gives up at max_pull_N. Then the target returns.
    est_along_N / est_N are force_sign x the tared estimate: the force on the hand, as the sim and the policy
    see it (friction holding the hand back reads < 0 along the pull). The series keeps the raw estimate."""
    st = arm.read()
    home, R = st.tool_pos.copy(), st.tool_R.copy()
    per_dir = settle_s + 2 * (max_pull_N / min(cfg.stiffness[:3]) + moved_m) / ramp_m_s + 2.0
    arm.start_stream(home, R, cfg.stiffness, max_duration=len(directions) * per_dir + 15)
    rows = []
    try:
        tare = stream_tare(arm, home, R, cfg, clock, sleep)
        for name in directions:
            d = (-1.0 if name[0] == "-" else 1.0) * AXES[name[1]]
            k = float(cfg.stiffness["xyz".index(name[1])])
            p0 = _hold_free(arm, cfg, home, R, tare, home, settle_s, clock, sleep).tool_pos.copy()
            series, peak, peak_f, broke = [], 0.0, np.zeros(3), False
            joints = JointLog()
            t0 = clock()
            while True:
                s = ramp_m_s * (clock() - t0)
                target = p0 + d * s
                st, f = _free_tick(arm, cfg, target, R, tare, home, sleep)
                joints.add(clock() - t0, st)
                moved = float((st.tool_pos - p0) @ d)
                pull = k * (s - moved)
                series.append([round(clock() - t0, 4), round(s, 6), round(moved, 6), round(pull, 4)]
                              + [round(float(x), 4) for x in f])
                if pull > peak:
                    peak, peak_f = pull, f.copy()
                if moved > moved_m:
                    broke = True
                    break
                if pull > max_pull_N:
                    break
            _move_free(arm, cfg, target, home, R, tare, home, 0.005, clock, sleep)
            rows.append({
                "direction": name,
                "breakaway_N": float(peak) if broke else None,
                "max_pull_N": float(peak),
                "est_along_N": float(force_sign * peak_f @ d),  # force on the hand at the peak, along the pull
                "est_N": (force_sign * peak_f).tolist(),
                "series": series,  # t, target offset, moved, pull, fx, fy, fz
                "joints": joints.rows,  # columns: joint_columns
            })
        _hold_free(arm, cfg, home, R, tare, home, 0.5, clock, sleep)
    finally:
        arm.stop_stream()
        arm.goto(home, R, 3.0)
    return {"ramp_mm_s": 1000 * ramp_m_s, "moved_mm": 1000 * moved_m, "max_pull_N": max_pull_N,
            "stiffness": list(cfg.stiffness[:3]), "tare_N": tare.tolist(), "force_sign": force_sign,
            "joint_columns": JOINT_LOG_COLUMNS,
            "rows": rows}


def friction_sweep(arm, cfg: BridgeConfig, axis="x", speeds_m_s=(0.002, 0.005, 0.01, 0.02), sweep_m=0.03,
                   settle_s=0.5, steady=(0.35, 0.95), force_sign=1.0, clock=time.monotonic, sleep=time.sleep):
    """Kinetic friction and what it does to the force estimate. For each speed, the target moves
    sweep_m along +axis and back at that speed. On the steady part of each stroke (the `steady`
    fraction of the target's path), fit, for the pull and for each axis of the tared estimate:

        value = a + g * position + h * direction        (direction = +1 / -1)

    Friction flips with the direction of motion and a pose-dependent error doesn't, so h is the
    friction (in the pull: the friction force; in the estimate: the fake force it adds) and g is the
    pose-dependent error. Across speeds, friction = Fc + b * speed. b includes the controller's damping, so
    read Fc, not b, as joint friction. The steady window skips the start of each stroke, where the hand
    first sticks and then catches up. The fit uses force_sign x the tared estimate, the force on the hand as
    the sim and the policy see it, so a fake force opposing the motion is < 0 whatever the robot's raw sign
    (the lab robot's is -1). The series keeps the raw estimate."""
    e = AXES[axis]
    i = "xyz".index(axis)
    k = float(cfg.stiffness[i])
    st = arm.read()
    home, R = st.tool_pos.copy(), st.tool_R.copy()
    total = sum(2 * (sweep_m / v + settle_s) for v in speeds_m_s)
    arm.start_stream(home, R, cfg.stiffness, max_duration=total + 20)
    rows = []
    try:
        tare = stream_tare(arm, home, R, cfg, clock, sleep)
        for v in speeds_m_s:
            pts = []  # position along the axis, direction, pull, fx, fy, fz
            series = []
            joints = JointLog()  # t from the start of this speed's out stroke
            t_speed = clock()
            for sgn, a, b in ((1.0, home, home + e * sweep_m), (-1.0, home + e * sweep_m, home)):
                rec = []
                t_stroke = clock() - t_speed
                _move_free(arm, cfg, a, b, R, tare, home, v, clock, sleep, record=rec)
                _hold_free(arm, cfg, b, R, tare, home, settle_s, clock, sleep)
                for t, target, s, f in rec:
                    joints.add(t_stroke + t, s)
                    prog = float((target - a) @ e) * sgn / sweep_m
                    pos = float((s.tool_pos - home) @ e)
                    pull = k * float((target - s.tool_pos) @ e)
                    series.append([round(sgn), round(t, 4), round(prog, 4), round(pos, 6), round(pull, 4)]
                                  + [round(float(x), 4) for x in f])
                    if steady[0] <= prog <= steady[1]:
                        pts.append([pos, sgn, pull, *(force_sign * f)])
            P = np.array(pts)
            if len(P) < 4 or len(set(P[:, 1])) < 2:
                raise HardwareError("too few steady samples; lower the speed or raise --control_hz")
            A = np.column_stack([np.ones(len(P)), P[:, 0], P[:, 1]])
            coef = np.linalg.lstsq(A, P[:, 2:], rcond=None)[0]  # rows: a, g, h; columns: pull, fx, fy, fz
            fake = coef[2, 1:]
            rows.append({
                "speed_mm_s": 1000 * v,
                "friction_N": float(coef[2, 0]),  # pull needed to keep moving: Fc + b*v
                "fake_along_N": float(fake[i]),  # estimate along the motion (signed: <0 opposes it)
                "fake_across_N": float(np.max(np.abs(np.delete(fake, i)))),
                "fake_N": fake.tolist(),  # per axis, for motion in +axis
                "pose_error_N_per_cm": (coef[1, 1:] / 100.0).tolist(),
                "n": len(P),
                "series": series,  # direction, t, progress, position, pull, fx, fy, fz
                "joints": joints.rows,  # columns: joint_columns
            })
    finally:
        arm.stop_stream()
        arm.goto(home, R, 3.0)
    out = {"axis": axis, "sweep_mm": 1000 * sweep_m, "stiffness": list(cfg.stiffness[:3]), "tare_N": tare.tolist(),
           "force_sign": force_sign, "joint_columns": JOINT_LOG_COLUMNS, "rows": rows}
    sp = np.array([r["speed_mm_s"] / 1000 for r in rows])
    fr = np.array([r["friction_N"] for r in rows])
    if len(rows) >= 2 and np.ptp(sp) > 0:
        b, fc = np.polyfit(sp, fr, 1)
        out["coulomb_N"], out["viscous_N_per_m_s"] = float(fc), float(b)
    else:
        out["coulomb_N"], out["viscous_N_per_m_s"] = float(fr[0]), None
    out["worst_fake_along_N"] = float(max(abs(r["fake_along_N"]) for r in rows))
    out["worst_fake_across_N"] = float(max(r["fake_across_N"] for r in rows))
    out["pose_error_N_per_cm"] = np.mean([r["pose_error_N_per_cm"] for r in rows], axis=0).tolist()
    return out


def friction_steps(arm, cfg: BridgeConfig, steps_m=(0.0005, 0.001, 0.002, 0.004, 0.008), axes="xyz", hold_s=1.5,
                   clock=time.monotonic, sleep=time.sleep):
    """Policy-sized moves. For each axis, sign and step size, jump the target by the step, hold it
    for hold_s, record how far the hand got and when it reached 63% of the step, then jump back
    and hold again. With friction the hand stops where the pull drops below the kinetic friction,
    short of the target; below the breakaway force it doesn't move at all."""
    st = arm.read()
    home, R = st.tool_pos.copy(), st.tool_R.copy()
    arm.start_stream(home, R, cfg.stiffness, max_duration=len(axes) * 2 * len(steps_m) * 2 * hold_s + 20)
    rows = []
    try:
        tare = stream_tare(arm, home, R, cfg, clock, sleep)
        for ax in axes:
            for sgn in (1.0, -1.0):
                d = sgn * AXES[ax]
                for step in steps_m:
                    p0 = arm.read().tool_pos.copy()  # friction may have left it short of home
                    target = p0 + d * step
                    t0, t63, moved, series, joints = clock(), None, 0.0, [], JointLog()
                    while clock() - t0 < hold_s:
                        s, f = _free_tick(arm, cfg, target, R, tare, home, sleep)
                        joints.add(clock() - t0, s)
                        moved = float((s.tool_pos - p0) @ d)
                        series.append([round(clock() - t0, 4), round(moved, 6)])
                        if t63 is None and moved >= 0.63 * step:
                            t63 = clock() - t0
                    _hold_free(arm, cfg, home, R, tare, home, hold_s, clock, sleep)
                    rows.append({"axis": ax, "sign": int(sgn), "step_mm": 1000 * step, "moved_mm": 1000 * moved,
                                 "short_mm": 1000 * (step - moved), "t63_ms": None if t63 is None else 1000 * t63,
                                 "series": series[::4],  # t, moved
                                 "joints": joints.rows})  # columns: joint_columns
    finally:
        arm.stop_stream()
        arm.goto(home, R, 3.0)
    return {"stiffness": list(cfg.stiffness), "hold_s": hold_s, "tare_N": tare.tolist(),
            "joint_columns": JOINT_LOG_COLUMNS, "rows": rows}


# --- Friction random walk (session 2) -------------------------------------------------------------
#
# The checks above move along one axis at a time, at constant speeds, at one or two poses. The walk moves
# the target through the region the policy works in the way a policy does: random directions and speeds,
# stops, reversals and policy-sized jumps. In free space the true force is zero, so every reading is
# estimate error (friction leak + pose-dependent error). The trajectory is a pure function of WalkSpec
# (seeded), so the same walk can be repeated on the robot (cold / warm) and replayed in sim.
#
# Region: Lab2's MuJoCo scene (Robot_Autonomy_Lab/Lab2/franka_emika_panda/panda_torque_table_shelves.xml),
# robot base frame. Table top z ≈ 0, x 0.185-0.735, y ±0.504. Shelf bottom boards start at |y| = 0.279
# (x 0.095-0.60; posts at |y| 0.41-0.46 up to z 0.65). Hole spot (0.59, 0), boss top z ≈ 0.028. The box
# bounds the sim fingertip point (≈ finger ends; the peg bottom is GRASP_TIP_ABOVE_PEG_BOTTOM lower):
# - |y| <= 0.15: the hand is ~0.2 m wide along y (fingers open along y with DOWN), so ~3 cm from the boards.
# - z >= 0.12: peg bottom >= 0.088, 6 cm above the boss top; clamp handles may stick up, check them.
# - x 0.40-0.68: inside the table (edge 0.735) and in top-down reach (Lab2 picks at 0.59).
WALK_BOX_LO = (0.40, -0.15, 0.12)
WALK_BOX_HI = (0.68, 0.15, 0.30)
WALK_LIMIT_LO = (0.30, -0.20, 0.08)  # no --box may go beyond these
WALK_LIMIT_HI = (0.70, 0.20, 0.40)
WALK_CENTRE = (0.59, 0.0, 0.14)  # above the hole spot, low: where the policy works
WALK_STOP_MARGIN_M = 0.02  # the hand may lag outside the box by this much before the walk stops
# Where the walk may start from: the corridor in front of the robot above the table, up to 0.60 m. Nothing in
# the Lab2 scene is there (the shelf arches at |y| 0.44 are at z 0.65), and FrankaPy's home pose (fingertip
# ≈ (0.31, 0, 0.48)) is inside it. The box itself is inside it too.
WALK_APPROACH_LO = (0.25, -0.15)
WALK_APPROACH_HI = (0.70, 0.15)
WALK_APPROACH_Z_MAX = 0.60
WALK_APPROACH_M_S = 0.05
WALK_MAX_TILT_DEG = 10.0  # the walk runs gripper-down at yaw 0 (DOWN), like the sim replay
WALK_KINDS = {"hold": 0, "move": 1, "step": 2}


@dataclass
class WalkSpec:
    seed: int = 0
    duration_s: float = 180.0
    box_lo: tuple = WALK_BOX_LO
    box_hi: tuple = WALK_BOX_HI
    centre: tuple = WALK_CENTRE
    spread_m: tuple = (0.06, 0.06, 0.04)  # std of move waypoints around the centre (then clipped to the box)
    speed_mm_s: tuple = (1.0, 20.0)  # log-uniform per move
    max_move_s: float = 8.0  # a slow move toward a far waypoint stops after this long
    p_hold: float = 0.2
    p_step: float = 0.2
    hold_s: tuple = (0.3, 2.0)
    step_mm: tuple = (0.5, 8.0)  # log-uniform jump along one axis, like friction-steps
    step_hold_s: float = 1.0
    dt: float = 0.005

    def check(self):
        lo, hi, c = (np.array(v, dtype=float) for v in (self.box_lo, self.box_hi, self.centre))
        if (lo < np.array(WALK_LIMIT_LO) - 1e-9).any() or (hi > np.array(WALK_LIMIT_HI) + 1e-9).any() or (lo >= hi).any():
            raise ValueError(f"walk box must lie within {WALK_LIMIT_LO}..{WALK_LIMIT_HI} (Lab2 table, clear of the shelves)")
        if (c < lo).any() or (c > hi).any():
            raise ValueError("walk centre must be inside the box")
        if not 0.0 < self.duration_s <= 600.0 or not 0.0 < self.speed_mm_s[0] <= self.speed_mm_s[1] <= 30.0:
            raise ValueError("duration must be in (0, 600] s and speeds in (0, 30] mm/s")
        return self


def walk_trajectory(spec: WalkSpec) -> dict:
    """Fingertip targets for a seeded random walk: t [s], target (N, 3) [m], kind (N,) (WALK_KINDS).

    Segments, until duration_s: hold (the target stays put), move (straight line at a constant speed to a
    waypoint drawn around the centre, which pulls the walk back toward it), step (the target jumps 0.5-8 mm
    along one axis and holds). Starts at the centre; never leaves the box (it is convex, waypoints are clipped)."""
    spec.check()
    rng = np.random.default_rng(spec.seed)
    lo, hi, c = (np.array(v, dtype=float) for v in (spec.box_lo, spec.box_hi, spec.centre))
    p, t = c.copy(), 0.0
    segs = []  # (kind, t_start, duration, start, end)
    while t < spec.duration_s:
        u = rng.random()
        if u < spec.p_hold:
            dur, end, kind = rng.uniform(*spec.hold_s), p.copy(), "hold"
        elif u < spec.p_hold + spec.p_step:
            size = math.exp(rng.uniform(*np.log(spec.step_mm))) / 1000.0
            end = p.copy()
            ax = rng.integers(3)
            end[ax] += rng.choice([-1.0, 1.0]) * size
            end = np.clip(end, lo, hi)
            dur, kind = spec.step_hold_s, "step"
        else:
            w = np.clip(c + rng.normal(0.0, spec.spread_m), lo, hi)
            v = math.exp(rng.uniform(*np.log(spec.speed_mm_s))) / 1000.0
            d = np.linalg.norm(w - p)
            if d / v > spec.max_move_s:
                w = p + (w - p) * (v * spec.max_move_s / d)
            dur, end, kind = max(np.linalg.norm(w - p) / v, spec.dt), w, "move"
        segs.append((kind, t, dur, p, end))
        p, t = end, t + dur
    ts = np.arange(0.0, t, spec.dt)
    target, kinds = np.zeros((len(ts), 3)), np.zeros(len(ts), dtype=int)
    starts = np.array([s[1] for s in segs])
    idx = np.searchsorted(starts, ts, side="right") - 1
    for j, (kind, t0, dur, a, b) in enumerate(segs):
        sel = idx == j
        if kind == "move":
            s = np.clip((ts[sel] - t0) / dur, 0.0, 1.0)[:, None]
            target[sel] = a + s * (b - a)
        else:
            target[sel] = b  # hold: b == a; step: the jump happens at the segment's start
        kinds[sel] = WALK_KINDS[kind]
    return {"t": ts, "target": target, "kind": kinds}


def walk_approach(tip0, spec: WalkSpec):
    """Fingertip waypoints from tip0 to the walk's start: across at the current height to above the start,
    then straight down (or up). Only from inside the approach corridor (or the box); None from anywhere else."""
    lo = np.minimum(np.array(WALK_APPROACH_LO), np.array(spec.box_lo[:2]))
    hi = np.maximum(np.array(WALK_APPROACH_HI), np.array(spec.box_hi[:2]))
    tip0 = np.asarray(tip0, dtype=float)
    if (tip0[:2] < lo).any() or (tip0[:2] > hi).any() or not spec.box_lo[2] <= tip0[2] <= WALK_APPROACH_Z_MAX:
        return None
    start = np.array(spec.centre, dtype=float)
    across = np.array([start[0], start[1], tip0[2]])
    return [across, start] if np.linalg.norm(across - tip0) > 0.005 else [start]


def tilt_from_down_deg(R) -> float:
    return float(np.degrees(np.arccos(np.clip((np.trace(DOWN.T @ np.asarray(R)) - 1) / 2, -1.0, 1.0))))


WALK_COLUMNS = ["t", "target_x", "target_y", "target_z", "tip_x", "tip_y", "tip_z", "fx", "fy", "fz", "kind"]


def friction_walk(arm, cfg: BridgeConfig, spec: WalkSpec, force_sign=1.0, max_force_N=FRICTION_MAX_FORCE_N,
                  clock=time.monotonic, sleep=time.sleep) -> dict:
    """Run a seeded random walk (walk_trajectory) in free space and record, per new robot state, the target,
    the fingertip, the tared force estimate and the joint state. The hand may start anywhere in the approach
    corridor, e.g. at FrankaPy's home pose, gripper down within WALK_MAX_TILT_DEG: it moves (FrankaPy goto)
    across at its height to above the start, then down to it (walk_approach), and tares there. The walk ends
    early, keeping its data, if the force passes max_force_N (something touches the peg: the arm then stays
    where it is) or the hand leaves the box by WALK_STOP_MARGIN_M. The series keeps the raw (tared) estimate;
    fit_walk_leak's result, included, uses force_sign x raw, the sim's convention (force on the hand)."""
    traj = walk_trajectory(spec)
    lo, hi = np.array(spec.box_lo), np.array(spec.box_hi)
    st = arm.read()
    tilt = tilt_from_down_deg(st.tool_R)
    if tilt > WALK_MAX_TILT_DEG:
        raise HardwareError(f"gripper is {tilt:.0f} deg from pointing down at yaw 0; run `home` first")
    R = DOWN.copy()
    tip0 = tool_to_tip(st.tool_pos, st.tool_R, cfg.tip_offset)
    path = walk_approach(tip0, spec)
    if path is None:
        raise HardwareError(f"fingertip {np.round(tip0, 3).tolist()} is outside the area the walk can start from "
                            f"(x {WALK_APPROACH_LO[0]}-{WALK_APPROACH_HI[0]}, |y| <= {WALK_APPROACH_HI[1]}, "
                            f"z {spec.box_lo[2]}-{WALK_APPROACH_Z_MAX}); run `home` first")
    prev = tip0
    for wp in path:
        arm.goto(tip_to_tool(wp, R, cfg.tip_offset), R, max(3.0, float(np.linalg.norm(wp - prev)) / WALK_APPROACH_M_S))
        prev = wp
    start_tool = tip_to_tool(traj["target"][0], R, cfg.tip_offset)
    arm.start_stream(start_tool, R, cfg.stiffness, max_duration=spec.duration_s + 30.0)
    series, joints, stopped, last_t = [], JointLog(), None, None
    t_end = float(traj["t"][-1])
    try:
        tare = stream_tare(arm, start_tool, R, cfg, clock, sleep)
        t0 = clock()
        while (t := clock() - t0) < t_end:
            i = min(int(t / spec.dt), len(traj["t"]) - 1)
            j = min(i + 1, len(traj["t"]) - 1)
            a = (t - traj["t"][i]) / spec.dt if j > i else 0.0
            target = traj["target"][i] + min(max(a, 0.0), 1.0) * (traj["target"][j] - traj["target"][i])
            st = arm.read()
            f = st.force - tare
            tip = tool_to_tip(st.tool_pos, R, cfg.tip_offset)
            if st.t != last_t:
                last_t = st.t
                series.append([round(t, 4)] + [round(float(x), 6) for x in (*target, *tip)]
                              + [round(float(x), 4) for x in f] + [int(traj["kind"][i])])
                joints.add(t, st)
            if np.linalg.norm(f) > max_force_N:
                stopped = f"force {np.linalg.norm(f):.1f} N in free space at t = {t:.1f} s; is something touching the peg?"
                break
            if (tip < lo - WALK_STOP_MARGIN_M).any() or (tip > hi + WALK_STOP_MARGIN_M).any():
                stopped = f"fingertip {np.round(tip, 3).tolist()} left the walk box at t = {t:.1f} s"
                break
            arm.send_target(tip_to_tool(target, R, cfg.tip_offset), R)
            sleep(1.0 / cfg.control_hz)
    finally:
        arm.stop_stream()
    out = {"spec": asdict(spec), "stiffness": list(cfg.stiffness), "tare_N": tare.tolist(), "R": R.tolist(),
           "approach": [wp.tolist() for wp in path],
           "tip_offset": cfg.tip_offset, "force_sign": force_sign, "stopped": stopped, "seconds_done": series[-1][0] if series else 0.0,
           "planned_s": t_end, "columns": WALK_COLUMNS, "series": series,
           "joint_columns": JOINT_LOG_COLUMNS, "joints": joints.rows}
    if series:
        tips = np.array([r[4:7] for r in series])
        out["tip_min"], out["tip_max"] = tips.min(axis=0).tolist(), tips.max(axis=0).tolist()
    out["fit"] = fit_walk_leak(series, force_sign=force_sign) if len(series) >= 200 else None
    return out


def fit_walk_leak(series, force_sign=1.0, v_s_mm_s=(0.5, 1.0, 2.0, 5.0, 10.0), smooth_s=0.05) -> dict:
    """Fit the sim's friction-leak model to a walk (rows as WALK_COLUMNS). f = force_sign x the raw tared
    estimate, i.e. the force on the hand as the sim and the policy see it. Per axis i, by least squares:

        f_i = c_i + g_i . (tip - mean tip) - F+_i tanh(max(v_i, 0) / v_s) - F-_i tanh(min(v_i, 0) / v_s)

    v = fingertip velocity (central differences of the fingertip, smoothed over smooth_s). F+/F- > 0 is a
    force opposing the motion, as in the sim's friction_leak(-F tanh(v / v_s)); separate signs because
    session 1 read ~2 N moving down and ~1 N moving up. g is the pose-dependent error (N/m). v_s is shared
    by the three axes (the sim has one), picked from v_s_mm_s by the smallest total residual. Also reports
    the residual with no friction terms, so the gain from the friction model is visible.

    The model has no stiction. With friction that also holds a force at rest (the fake arm's Coulomb mode),
    3-min walks give the moving leak to about ±0.3 N, and a symmetric friction shows up to 0.5 N of +/-
    difference: compare repeats before reading an asymmetry as real (tests: TestFrictionWalk)."""
    S = np.array([r[:10] for r in series], dtype=float)
    t, tip, f = S[:, 0], S[:, 4:7], force_sign * S[:, 7:10]
    dt = float(np.median(np.diff(t)))
    k = max(1, int(round(smooth_s / dt)))
    sm = np.column_stack([np.convolve(tip[:, i], np.ones(k) / k, mode="same") for i in range(3)])
    v = np.column_stack([np.gradient(sm[:, i], t) for i in range(3)])
    n = len(t)
    keep = slice(k, n - k)  # the moving average is biased at the ends
    X = tip - tip.mean(axis=0)
    base = np.column_stack([np.ones(n), X])
    no_fric = [float(np.sqrt(np.mean((f[keep, i] - base[keep] @ np.linalg.lstsq(base[keep], f[keep, i], rcond=None)[0]) ** 2)))
               for i in range(3)]
    best = None
    per_vs = {}
    for vs in v_s_mm_s:
        coefs, rms = [], []
        for i in range(3):
            A = np.column_stack([base, np.tanh(np.maximum(v[:, i], 0) / (vs / 1000)), np.tanh(np.minimum(v[:, i], 0) / (vs / 1000))])
            c = np.linalg.lstsq(A[keep], f[keep, i], rcond=None)[0]
            coefs.append(c)
            rms.append(float(np.sqrt(np.mean((f[keep, i] - A[keep] @ c) ** 2))))
        per_vs[vs] = float(np.sqrt(np.mean(np.square(rms))))
        if best is None or per_vs[vs] < per_vs[best[0]]:
            best = (vs, coefs, rms)
    vs, coefs, rms = best
    axes = {}
    for i, name in enumerate("xyz"):
        c = coefs[i]
        axes[name] = {"leak_plus_N": float(-c[4]), "leak_minus_N": float(-c[5]), "offset_N": float(c[0]),
                      "pose_N_per_cm": (c[1:4] / 100.0).tolist(), "rms_N": rms[i], "rms_no_friction_N": no_fric[i]}
    return {"v_s_mm_s": vs, "rms_by_v_s": per_vs, "speed_max_mm_s": (1000 * np.abs(v[keep]).max(axis=0)).tolist(),
            "axes": axes}
