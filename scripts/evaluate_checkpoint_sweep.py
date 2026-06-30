from __future__ import annotations

import argparse
import csv
from pathlib import Path

from carracing_rl.agent import DQNAgent
from carracing_rl.config import load_config
from carracing_rl.envs import make_env
from carracing_rl.evaluate import evaluate_policy
from carracing_rl.utils import configure_torch_runtime, resolve_device, set_seed


FIELDNAMES = [
    "seed_offset",
    "checkpoint_step",
    "mean_reward",
    "std_reward",
    "min_reward",
    "max_reward",
    "mean_length",
    "episodes",
]


def evaluate_once(config: dict, checkpoint: Path, seed_offset: int, episodes: int, epsilon: float) -> dict:
    seed = int(config.get("seed", 42))
    env = make_env(config, seed=seed + seed_offset)
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, config, device=device)
    payload = agent.load(checkpoint, load_optimizer=False)
    max_episode_steps = int(config.get("train", {}).get("max_episode_steps", 1_000))
    metrics = evaluate_policy(agent, env, episodes=episodes, max_episode_steps=max_episode_steps, epsilon=epsilon)
    env.close()
    return {
        "seed_offset": seed_offset,
        "checkpoint_step": int(payload.get("step", -1)),
        **metrics,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML config used to build the agent.")
    parser.add_argument("--checkpoint", required=True, help="Path to the checkpoint to evaluate.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--seed-offsets", nargs="+", type=int, required=True)
    parser.add_argument("--output", required=True, help="CSV output path.")
    parser.add_argument("--epsilon", type=float, default=0.0)
    args = parser.parse_args()

    config = load_config(args.config)
    set_seed(int(config.get("seed", 42)))
    configure_torch_runtime(config)
    checkpoint = Path(args.checkpoint)
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)

    with output.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=FIELDNAMES)
        writer.writeheader()
        for seed_offset in args.seed_offsets:
            row = evaluate_once(config, checkpoint, seed_offset, args.episodes, args.epsilon)
            writer.writerow(row)
            print(
                f"seed_offset={seed_offset} step={row['checkpoint_step']} "
                f"mean={row['mean_reward']:.1f} std={row['std_reward']:.1f}"
            )


if __name__ == "__main__":
    main()
