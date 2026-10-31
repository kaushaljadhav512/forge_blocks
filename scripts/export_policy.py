"""Export an rl_games checkpoint to a self-contained folder for deploy/policy_runner.py.

    python scripts/export_policy.py logs/rl_games/<run>/<timestamp>/nn/<run>.pth --out exports/<name>

Needs torch + rl_games (+ pyyaml), not Isaac Sim: everything about the env is read from the
params/env.yaml and params/agent.yaml that Isaac Lab's train.py saves next to the checkpoint.

Writes:
  policy.pt            TorchScript: (obs[1,N], h, c) -> (mu[1,7], h, c). Includes the observation
                       normaliser from the checkpoint, so it takes raw observations.
  runner_config.json   everything ForgePolicyRunner needs (obs order, action scaling, filters ...)
  env.yaml, agent.yaml copies, for the record
"""

import argparse
import json
import os
import shutil
import sys

import torch
import yaml

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "deploy"))
from policy_runner import ACTION_DIM, OBS_DIMS  # noqa: E402


def _load_yaml(path):
    # Our own files written by Isaac Lab's dump_yaml, which can contain python tuples.
    with open(path) as f:
        return yaml.unsafe_load(f)


class Actor(torch.nn.Module):
    """Deterministic actor path of rl_games' ModelA2CContinuousLogStd, without sampling."""

    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, obs, h, c):
        obs = self.model.norm_obs(obs)
        mu, _logstd, _value, states = self.model.a2c_network(
            {"obs": obs, "rnn_states": (h, c), "seq_length": 1, "is_train": False}
        )
        return mu, states[0], states[1]


def build_model(agent_cfg, obs_dim, checkpoint):
    from rl_games.algos_torch import model_builder

    params = agent_cfg["params"]
    network = model_builder.ModelBuilder().load(params)
    model = network.build(
        {
            "actions_num": ACTION_DIM,
            "input_shape": (obs_dim,),
            "num_seqs": 1,
            "value_size": 1,
            "normalize_value": params["config"].get("normalize_value", False),
            "normalize_input": params["config"].get("normalize_input", False),
        }
    )
    model.load_state_dict(checkpoint["model"])
    return model.eval()  # eval: the running mean/std must not update


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("checkpoint")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    run_dir = os.path.dirname(os.path.dirname(os.path.abspath(args.checkpoint)))
    env_yaml = os.path.join(run_dir, "params", "env.yaml")
    agent_yaml = os.path.join(run_dir, "params", "agent.yaml")
    env_cfg, agent_cfg = _load_yaml(env_yaml), _load_yaml(agent_yaml)

    obs_order = list(env_cfg["obs_order"])
    obs_dim = sum(OBS_DIMS[k] for k in obs_order) + ACTION_DIM

    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    model = build_model(agent_cfg, obs_dim, checkpoint)
    running_mean = checkpoint["model"].get("running_mean_std.running_mean")
    if running_mean is not None and running_mean.shape[0] != obs_dim:
        sys.exit(f"checkpoint obs dim {running_mean.shape[0]} != {obs_dim} from obs_order {obs_order}")

    rnn = agent_cfg["params"]["network"]["rnn"]
    actor = Actor(model)
    example = (
        torch.zeros(1, obs_dim),
        torch.zeros(rnn["layers"], 1, rnn["units"]),
        torch.zeros(rnn["layers"], 1, rnn["units"]),
    )
    with torch.no_grad():
        traced = torch.jit.trace(actor, example, check_trace=False)
        # The trace must match the eager model, including the LSTM state carry-over.
        h, c = example[1], example[2]
        h_t, c_t = h, c
        for _ in range(5):
            obs = torch.randn(1, obs_dim)
            mu, h, c = actor(obs, h, c)
            mu_t, h_t, c_t = traced(obs, h_t, c_t)
            assert torch.allclose(mu, mu_t, atol=1e-5) and torch.allclose(h, h_t, atol=1e-5), "trace mismatch"

    ctrl, task = env_cfg["ctrl"], env_cfg["task"]
    sim_dt = env_cfg["sim"]["dt"]
    runner_cfg = {
        "obs_order": obs_order,
        "pos_action_bounds": list(ctrl["pos_action_bounds"]),
        "pos_action_threshold": list(ctrl["pos_action_threshold"]),
        "rot_action_threshold": list(ctrl["rot_action_threshold"]),
        "ema_factor": sum(ctrl["ema_factor_range"]) / 2,
        "force_threshold": sum(task["contact_penalty_threshold_range"]) / 2,
        "ft_smoothing_factor": env_cfg["ft_smoothing_factor"],
        "sim_dt": sim_dt,
        "policy_dt": sim_dt * env_cfg["decimation"],
        "rnn_layers": rnn["layers"],
        "rnn_units": rnn["units"],
        # Not used by the runner; recorded so the robot side can match the sim.
        "task_prop_gains": list(ctrl["default_task_prop_gains"]),
        "ema_factor_range": list(ctrl["ema_factor_range"]),
        "contact_penalty_threshold_range": list(task["contact_penalty_threshold_range"]),
        "obs_rand": env_cfg["obs_rand"],
        "episode_length_s": env_cfg["episode_length_s"],
        "checkpoint": os.path.abspath(args.checkpoint),
        "epoch": checkpoint.get("epoch"),
    }

    os.makedirs(args.out, exist_ok=True)
    traced.save(os.path.join(args.out, "policy.pt"))
    with open(os.path.join(args.out, "runner_config.json"), "w") as f:
        json.dump(runner_cfg, f, indent=2, default=list)
    shutil.copy(env_yaml, os.path.join(args.out, "env.yaml"))
    shutil.copy(agent_yaml, os.path.join(args.out, "agent.yaml"))
    print(f"exported epoch {runner_cfg['epoch']} to {args.out} (obs_dim={obs_dim})")


if __name__ == "__main__":
    main()
