"""Record one video per position-error level, using the same policy setup and bins as eval_sweep.py.

    ./isaaclab.sh -p scripts/record_video.py --task Isaac-ForgeBlocks-Peg8mm-Direct-v0 --headless \
        --checkpoint logs/rl_games/peg_forge/seed0/nn/<checkpoint>.pth --out videos/peg_forge_seed0

Writes <out>/error_<min>-<max>mm.mp4, playing back at real time. Each frame shows the episode, this
episode's error, the smoothed contact force and whether the peg has been inserted.
"""

import argparse
import math
import os
import sys

from isaaclab.app import AppLauncher

parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
parser.add_argument("--task", required=True)
parser.add_argument("--checkpoint", required=True)
parser.add_argument("--out", required=True, help="output folder")
parser.add_argument("--episodes", type=int, default=6, help="per error level")
parser.add_argument("--seed", type=int, default=0)
parser.add_argument("--force_threshold", type=float, default=7.5, help="N; the paper deploys at 7.5")
AppLauncher.add_app_launcher_args(parser)
args_cli = parser.parse_args()
args_cli.enable_cameras = True
simulation_app = AppLauncher(args_cli).app

import cv2  # noqa: E402
import gymnasium as gym  # noqa: E402
import imageio  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from rl_games.common import env_configurations, vecenv  # noqa: E402
from rl_games.torch_runner import Runner  # noqa: E402

from isaaclab_rl.rl_games import RlGamesGpuEnv, RlGamesVecEnvWrapper  # noqa: E402

import isaaclab_tasks  # noqa: E402,F401
from isaaclab_tasks.utils import load_cfg_from_registry, parse_env_cfg  # noqa: E402

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
import forge_blocks.tasks  # noqa: E402,F401

LEVELS = [(0.0, 1.0), (1.0, 2.5), (2.5, 5.0)]  # mm, the paper's bands
CAMERA_OFFSET = (0.28, 0.22, 0.14)  # m from the slot, on the side away from the robot
WARMUP_RENDERS = 20  # the first frames from a new render product are blank


def make_player(env, agent_cfg):
    """Same steps as eval_sweep.py."""
    cfg = agent_cfg["params"]["config"]
    env_params = agent_cfg["params"]["env"]
    wrapped = RlGamesVecEnvWrapper(
        env,
        cfg["device"],
        env_params.get("clip_observations", math.inf),
        env_params.get("clip_actions", math.inf),
        env_params.get("obs_groups"),
        env_params.get("concate_obs_groups", True),
    )
    vecenv.register("IsaacRlgWrapper", lambda config_name, num_actors, **kw: RlGamesGpuEnv(config_name, num_actors, **kw))
    env_configurations.register("rlgpu", {"vecenv_type": "IsaacRlgWrapper", "env_creator": lambda **kw: wrapped})
    agent_cfg["params"]["load_checkpoint"] = True
    agent_cfg["params"]["load_path"] = args_cli.checkpoint
    cfg["num_actors"] = env.unwrapped.num_envs
    runner = Runner()
    runner.load(agent_cfg)
    player = runner.create_player()
    player.restore(args_cli.checkpoint)
    player.reset()
    player.is_deterministic = True
    return wrapped, player


def aim_camera(u):
    slot = u.fixed_pos[0].cpu().numpy()  # relative to the env origin
    u.viewport_camera_controller.update_view_location(
        eye=tuple(float(s + o) for s, o in zip(slot, CAMERA_OFFSET)), lookat=tuple(float(s) for s in slot)
    )


def annotate(frame, lines, color):
    frame = np.array(frame[:, :, :3], dtype=np.uint8)
    for i, text in enumerate(lines):
        y = 40 + 36 * i
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5, cv2.LINE_AA)
        cv2.putText(frame, text, (24, y), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color if i == len(lines) - 1 else (255, 255, 255), 2, cv2.LINE_AA)
    return frame


def run_episode(env, wrapped, player, obs, writer=None, label=""):
    """Run until the episode ends, writing a frame per policy step. Returns (obs, success, error_mm)."""
    u = env.unwrapped
    error_mm = torch.linalg.norm(u.init_fixed_pos_obs_noise[0]).item() * 1e3
    if player.is_rnn:
        player.init_rnn()
    if writer is not None:
        aim_camera(u)
        for _ in range(WARMUP_RENDERS):
            env.render()
    t, inserted_at = 0, None
    while True:
        with torch.inference_mode():
            actions = player.get_action(player.obs_to_torch(obs), is_deterministic=True)
            obs, _, dones, _ = wrapped.step(actions)
        t += 1
        if dones.any():
            # ep_succeeded is only cleared on the next _pre_physics_step.
            return obs, bool(u.ep_succeeded[0]), error_mm
        if inserted_at is None and u.ep_succeeded[0]:
            inserted_at = t * u.step_dt
        if writer is not None:
            force = torch.linalg.norm(u.force_sensor_smooth[0, :3]).item()
            status = f"inserted at {inserted_at:.1f} s" if inserted_at is not None else "not inserted"
            color = (80, 255, 80) if inserted_at is not None else (255, 200, 80)
            writer.append_data(
                annotate(
                    env.render(),
                    [label, f"position error {error_mm:.1f} mm", f"t {t * u.step_dt:.1f} s   force {force:.1f} N", status],
                    color,
                )
            )


def main():
    env_cfg = parse_env_cfg(args_cli.task, device=args_cli.device or "cuda:0", num_envs=1)
    env_cfg.seed = args_cli.seed
    env_cfg.sim.render_interval = env_cfg.decimation
    env_cfg.viewer.origin_type = "env"
    env_cfg.viewer.env_index = 0
    agent_cfg = load_cfg_from_registry(args_cli.task, "rl_games_cfg_entry_point")

    env = gym.make(args_cli.task, cfg=env_cfg, render_mode="rgb_array")
    u = env.unwrapped
    wrapped, player = make_player(env, agent_cfg)
    os.makedirs(args_cli.out, exist_ok=True)

    u.cfg.obs_rand.error_radius_mode = "xyz"
    u.cfg.task.contact_penalty_threshold_range = [args_cli.force_threshold, args_cli.force_threshold]
    obs = None
    for err_min, err_max in LEVELS:
        u.cfg.obs_rand.error_radius = [err_min * 1e-3, err_max * 1e-3]
        if obs is None:
            obs = wrapped.reset()
        else:
            # The episode in flight was reset with the previous level's error; run and discard it.
            obs, _, _ = run_episode(env, wrapped, player, obs)
        if isinstance(obs, dict):
            obs = obs["obs"]
        _ = player.get_batch_size(obs, 1)

        path = os.path.join(args_cli.out, f"error_{err_min:g}-{err_max:g}mm.mp4")
        writer = imageio.get_writer(path, fps=round(1.0 / u.step_dt), codec="libx264", quality=7, pixelformat="yuv420p")
        wins = 0
        for ep in range(args_cli.episodes):
            label = f"{err_min:g}-{err_max:g} mm band, episode {ep + 1}/{args_cli.episodes}"
            obs, success, error_mm = run_episode(env, wrapped, player, obs, writer, label)
            wins += success
            print(f"[video] {err_min:g}-{err_max:g} mm episode {ep + 1}: error {error_mm:.2f} mm, success {success}")
        writer.close()
        print(f"[video] wrote {path}: {wins}/{args_cli.episodes} succeeded")
    env.close()


if __name__ == "__main__":
    main()
    simulation_app.close()
