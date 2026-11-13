"""Measure, in sim, the frame offsets and force sign the real-robot bridge needs.

    ./isaaclab.sh -p scripts/measure_frames.py --task Isaac-ForgeBlocks-Peg8mm-Direct-v0 --headless

Prints:
  1. Body poses relative to panda_hand: the fingertip midpoint the policy observes and the
     force_sensor link. On the real robot panda_hand sits on the flange (panda_link8, rotated
     -45 deg about z), so this gives the flange -> fingertip transform for the bridge.
  2. The grasp: how far above the bottom of the held part the fingertip midpoint sits. Match this
     on the robot (grasping jig), or every position the policy sees is shifted.
  3. A press test: the hand pushes the held part down onto the hole's rim (6 mm off-centre, so it
     can't go in) and the script reports the sign of the force the policy observes. Compare it with
     the same press on the robot (O_F_ext_hat_K) to set force_sign in runner_config.json.
"""

import argparse
import json
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", default="Isaac-ForgeBlocks-Peg8mm-Direct-v0")
parser.add_argument("--num_envs", type=int, default=16)
parser.add_argument("--offset_mm", type=float, default=6.0, help="press point, off the hole centre in x")
parser.add_argument("--steps", type=int, default=90, help="policy steps of pressing (15 Hz)")
parser.add_argument("--out", default="results/measure_frames.json")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
simulation_app = AppLauncher(args_cli).app

import gymnasium as gym  # noqa: E402
import torch  # noqa: E402

import isaaclab.utils.math as math_utils  # noqa: E402
import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import forge_blocks.tasks  # noqa: E402,F401


def rel_pose(data, parent_idx, child_idx):
    """Pose of child in parent's frame, per env."""
    pp, pq = data.body_pos_w[:, parent_idx], data.body_quat_w[:, parent_idx]
    cp, cq = data.body_pos_w[:, child_idx], data.body_quat_w[:, child_idx]
    pos = math_utils.quat_apply(math_utils.quat_conjugate(pq), cp - pp)
    quat = math_utils.quat_mul(math_utils.quat_conjugate(pq), cq)
    return pos, quat


def stats(x):
    x = x.detach().cpu()
    return {"mean": x.mean(0).tolist(), "min": x.min(0).values.tolist(), "max": x.max(0).values.tolist()}


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=args_cli.num_envs)
    # Exact slot estimate, so the press lands where we aim.
    env_cfg.obs_rand.error_radius_mode = "xyz"
    env_cfg.obs_rand.error_radius = [0.0, 0.0]
    env = gym.make(args_cli.task, cfg=env_cfg)
    u = env.unwrapped
    env.reset()
    robot = u._robot
    names = robot.body_names
    print("[measure] robot bodies:", names)
    out = {"task": args_cli.task, "bodies": names}

    # 1. Frames relative to panda_hand (identical in every env; report the spread anyway).
    hand = names.index("panda_hand")
    for body in ["panda_fingertip_centered", "force_sensor", "panda_leftfinger", "panda_rightfinger"]:
        if body in names:
            pos, quat = rel_pose(robot.data, hand, names.index(body))
            out[f"{body}_in_hand"] = {"pos_m": stats(pos), "quat_wxyz": stats(quat)}
            print(f"[measure] {body} in panda_hand frame: pos {pos[0].tolist()} quat {quat[0].tolist()}")

    # 2. Grasp height: fingertip midpoint above the held part's origin (its bottom face for the peg).
    tip_above_held = u.fingertip_midpoint_pos - u.held_pos
    out["fingertip_minus_held_origin_m"] = stats(tip_above_held)
    out["held_asset_height_m"] = u.cfg_task.held_asset_cfg.height
    print(f"[measure] fingertip - held origin (world): {stats(tip_above_held)}")

    # 3. Press test. Actions are absolute targets around the slot estimate (x 5 cm per unit).
    n = u.num_envs
    mu = torch.zeros((n, 7), device=u.device)
    mu[:, 0] = args_cli.offset_mm * 1e-3 / u.cfg.ctrl.pos_action_bounds[0]
    mu[:, 2] = -1.0  # 5 cm below the rim: the 2 cm step clip keeps pushing down
    mu[:, 5] = u.actions[:, 5]  # hold yaw
    mu[:, 6] = -1.0
    log = []
    for t in range(args_cli.steps):
        env.step(mu)
        log.append(
            {
                "t_s": (t + 1) * u.step_dt,
                "tip_z_above_rim_m": (u.fingertip_midpoint_pos[:, 2] - u.fixed_pos_obs_frame[:, 2]).mean().item(),
                "force_raw_world": u.force_sensor_world[:, :3].mean(0).tolist(),
                "force_obs_smooth": u.force_sensor_smooth[:, :3].mean(0).tolist(),
            }
        )
    free, pressed = log[0], log[-1]
    out["press"] = {"offset_mm": args_cli.offset_mm, "first_step": free, "last_step": pressed, "trace": log}
    fz = pressed["force_obs_smooth"][2]
    sign = "+z (up: the force the environment exerts on the hand)" if fz > 0 else "-z (down: the force the hand exerts)"
    out["press"]["observed_fz_sign_when_pressing_down"] = 1 if fz > 0 else -1
    print(f"[measure] free space (step 1): force obs {free['force_obs_smooth']}, tip {free['tip_z_above_rim_m']:.4f} m above rim")
    print(f"[measure] pressing (step {args_cli.steps}): force obs {pressed['force_obs_smooth']}, tip {pressed['tip_z_above_rim_m']:.4f} m above rim")
    print(f"[measure] pressing down reads as {sign}")

    os.makedirs(os.path.dirname(os.path.abspath(args_cli.out)), exist_ok=True)
    with open(args_cli.out, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[measure] wrote {args_cli.out}")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
