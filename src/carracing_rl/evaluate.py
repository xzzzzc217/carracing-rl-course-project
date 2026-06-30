from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from .agent import DQNAgent
from .config import load_config
from .envs import make_env
from .utils import configure_torch_runtime, resolve_device, set_seed


def evaluate_policy(agent: DQNAgent, env, episodes: int, max_episode_steps: int, epsilon: float = 0.0) -> dict:
    rewards: list[float] = []
    lengths: list[int] = []
    for _ in range(episodes):
        observation, _ = env.reset()
        episode_reward = 0.0
        episode_length = 0
        for _step in range(max_episode_steps):
            action = agent.select_action(observation, epsilon=epsilon)
            observation, reward, terminated, truncated, _ = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            if terminated or truncated:
                break
        rewards.append(episode_reward)
        lengths.append(episode_length)
    reward_array = np.asarray(rewards, dtype=np.float32)
    length_array = np.asarray(lengths, dtype=np.float32)
    return {
        "mean_reward": float(reward_array.mean()),
        "std_reward": float(reward_array.std(ddof=0)),
        "min_reward": float(reward_array.min()),
        "max_reward": float(reward_array.max()),
        "mean_length": float(length_array.mean()),
        "episodes": int(episodes),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to the YAML config used to build the agent.")
    parser.add_argument("--checkpoint", required=True, help="Path to a .pt checkpoint.")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--epsilon", type=float, default=0.0)
    parser.add_argument("--render", action="store_true")
    args = parser.parse_args()

    config = load_config(args.config)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    configure_torch_runtime(config)
    render_mode = "human" if args.render else None
    env = make_env(config, seed=seed + 20_000, render_mode=render_mode)
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    agent = DQNAgent(env.observation_space.shape, env.action_space.n, config, device=device)
    payload = agent.load(args.checkpoint, load_optimizer=False)
    max_episode_steps = int(config.get("train", {}).get("max_episode_steps", 1_000))
    metrics = evaluate_policy(agent, env, episodes=args.episodes, max_episode_steps=max_episode_steps, epsilon=args.epsilon)
    metrics["checkpoint_step"] = int(payload.get("step", -1))
    env.close()
    print(json.dumps(metrics, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
