from __future__ import annotations

import argparse
import csv
from pathlib import Path
import shutil
import time

from tqdm import tqdm

from .agent import DQNAgent
from .config import load_config, save_config
from .envs import make_env
from .evaluate import evaluate_policy
from .replay import PrioritizedReplayBuffer, UniformReplayBuffer
from .utils import CsvLogger, configure_torch_runtime, ensure_dir, resolve_device, set_seed


def build_replay(config: dict, observation_shape: tuple[int, ...], seed: int):
    train_config = config.get("train", {})
    agent_config = config.get("agent", {})
    capacity = int(train_config.get("replay_capacity", 20_000))
    if bool(agent_config.get("prioritized_replay", True)):
        return PrioritizedReplayBuffer(
            capacity=capacity,
            observation_shape=observation_shape,
            alpha=float(agent_config.get("per_alpha", 0.6)),
            beta_start=float(agent_config.get("per_beta_start", 0.4)),
            beta_frames=int(agent_config.get("per_beta_frames", train_config.get("total_steps", 200_000))),
            epsilon=float(agent_config.get("per_epsilon", 1e-6)),
            seed=seed,
        )
    return UniformReplayBuffer(capacity=capacity, observation_shape=observation_shape, seed=seed)


def _prepare_run_dir(run_dir: Path, overwrite: bool = False) -> Path:
    if overwrite and run_dir.exists():
        results_root = run_dir.parent.resolve()
        resolved = run_dir.resolve()
        if results_root not in resolved.parents:
            raise ValueError(f"Refusing to remove a run directory outside {results_root}: {resolved}")
        shutil.rmtree(resolved)
    return ensure_dir(run_dir)


def _load_local_best_eval(eval_path: Path) -> tuple[float, int] | None:
    if not eval_path.exists():
        return None

    best_reward = float("-inf")
    best_step = 0
    with eval_path.open("r", encoding="utf-8", newline="") as handle:
        for row in csv.DictReader(handle):
            try:
                reward = float(row["mean_reward"])
                step = int(row["step"])
            except (KeyError, TypeError, ValueError):
                continue
            if reward > best_reward:
                best_reward = reward
                best_step = step

    if best_step == 0:
        return None
    return best_reward, best_step


def train(
    config: dict,
    results_dir: str | Path = "results",
    resume_checkpoint: str | Path | None = None,
    resume_model_only: bool = False,
    overwrite: bool = False,
    limit_steps: int | None = None,
    show_progress: bool = True,
) -> Path:
    seed = int(config.get("seed", 42))
    set_seed(seed)
    configure_torch_runtime(config)
    run_name = config.get("logging", {}).get("run_name", f"run_seed{seed}")
    run_dir = _prepare_run_dir(Path(results_dir) / run_name, overwrite=overwrite)
    if resume_checkpoint is None and not overwrite and any((run_dir / name).exists() for name in ("train.csv", "eval.csv")):
        raise FileExistsError(
            f"{run_dir} already contains logs. Use --resume to continue or --overwrite to restart."
        )
    checkpoint_dir = ensure_dir(run_dir / "checkpoints")
    save_config(config, run_dir / "config.yaml")

    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    env = make_env(config, seed=seed)
    eval_env = make_env(config, seed=seed + 10_000)
    observation_shape = env.observation_space.shape
    num_actions = env.action_space.n

    agent = DQNAgent(observation_shape, num_actions, config, device=device)
    resume_step = 0
    best_eval_reward = float("-inf")
    local_best_eval_step: int | None = None
    if resume_checkpoint is not None:
        payload = agent.load(resume_checkpoint, load_optimizer=not resume_model_only)
        resume_step = int(payload.get("step", 0))
        best_eval_reward = float(payload.get("extra", {}).get("best_eval_reward", float("-inf")))
        local_best_eval = _load_local_best_eval(run_dir / "eval.csv")
        if local_best_eval is not None:
            best_eval_reward, local_best_eval_step = local_best_eval
        elif Path(resume_checkpoint).name == "best.pt" and best_eval_reward > float("-inf"):
            agent.save(
                checkpoint_dir / "best.pt",
                step=resume_step,
                config=config,
                extra={"best_eval_reward": best_eval_reward, "elapsed_seconds": 0.0},
            )
        if local_best_eval_step == resume_step and not (checkpoint_dir / "best.pt").exists():
            agent.save(
                checkpoint_dir / "best.pt",
                step=resume_step,
                config=config,
                extra={"best_eval_reward": best_eval_reward, "elapsed_seconds": 0.0},
            )

    replay_buffer = build_replay(config, observation_shape, seed=seed)

    train_config = config.get("train", {})
    total_steps = int(train_config.get("total_steps", 200_000))
    end_step = total_steps if limit_steps is None else min(total_steps, resume_step + int(limit_steps))
    learning_starts = int(train_config.get("learning_starts", 5_000))
    train_frequency = int(train_config.get("train_frequency", 4))
    target_update_interval = int(train_config.get("target_update_interval", 2_000))
    eval_interval_steps = int(train_config.get("eval_interval_steps", 5_000))
    save_interval_steps = int(train_config.get("save_interval_steps", 5_000))
    eval_episodes = int(train_config.get("eval_episodes", 10))
    max_episode_steps = int(train_config.get("max_episode_steps", 1_000))
    early_stop_reward = train_config.get("early_stop_reward")
    early_stop_reward = None if early_stop_reward is None else float(early_stop_reward)
    restore_best_on_early_stop = bool(train_config.get("restore_best_on_early_stop", False))

    train_logger = CsvLogger(
        run_dir / "train.csv",
        [
            "step",
            "episode",
            "episode_reward",
            "episode_length",
            "epsilon",
            "loss",
            "mean_q",
            "mean_target",
            "mean_td_abs",
            "beta",
        ],
    )
    eval_logger = CsvLogger(
        run_dir / "eval.csv",
        ["step", "mean_reward", "std_reward", "min_reward", "max_reward", "mean_length", "episodes"],
    )

    observation, _ = env.reset(seed=seed)
    episode_reward = 0.0
    episode_length = 0
    episode_index = 0
    latest_metrics: dict[str, float] = {}
    start_time = time.time()
    stopped_early = False

    progress = tqdm(range(resume_step + 1, end_step + 1), desc=run_name, unit="step", disable=not show_progress)
    for step in progress:
        epsilon = agent.epsilon_by_step(step, config)
        if step < learning_starts:
            action = agent.sample_random_action()
        else:
            action = agent.select_action(observation, epsilon=epsilon)

        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay_buffer.add(observation, action, reward, next_observation, done)
        observation = next_observation
        episode_reward += float(reward)
        episode_length += 1

        if step >= learning_starts and step % train_frequency == 0 and len(replay_buffer) >= agent.batch_size:
            latest_metrics = agent.train_step(replay_buffer, global_step=step)

        if step % target_update_interval == 0:
            agent.update_target()

        if done:
            train_logger.write(
                {
                    "step": step,
                    "episode": episode_index,
                    "episode_reward": episode_reward,
                    "episode_length": episode_length,
                    "epsilon": epsilon,
                    **latest_metrics,
                }
            )
            progress.set_postfix(
                reward=f"{episode_reward:.1f}",
                eps=f"{epsilon:.2f}",
                loss=f"{latest_metrics.get('loss', 0.0):.3f}",
            )
            episode_index += 1
            observation, _ = env.reset()
            episode_reward = 0.0
            episode_length = 0

        if step % eval_interval_steps == 0:
            metrics = evaluate_policy(
                agent,
                eval_env,
                episodes=eval_episodes,
                max_episode_steps=max_episode_steps,
                epsilon=0.0,
            )
            eval_logger.write({"step": step, **metrics})
            if metrics["mean_reward"] > best_eval_reward:
                best_eval_reward = metrics["mean_reward"]
                agent.save(
                    checkpoint_dir / "best.pt",
                    step=step,
                    config=config,
                    extra={"best_eval_reward": best_eval_reward, "elapsed_seconds": time.time() - start_time},
                )
            if early_stop_reward is not None and metrics["mean_reward"] >= early_stop_reward:
                if restore_best_on_early_stop and (checkpoint_dir / "best.pt").exists():
                    shutil.copy2(checkpoint_dir / "best.pt", checkpoint_dir / "latest.pt")
                else:
                    agent.save(
                        checkpoint_dir / "latest.pt",
                        step=step,
                        config=config,
                        extra={
                            "best_eval_reward": best_eval_reward,
                            "elapsed_seconds": time.time() - start_time,
                            "stopped_early": True,
                            "early_stop_reward": early_stop_reward,
                        },
                    )
                stopped_early = True
                break

        if step % save_interval_steps == 0:
            agent.save(
                checkpoint_dir / "latest.pt",
                step=step,
                config=config,
                extra={"best_eval_reward": best_eval_reward, "elapsed_seconds": time.time() - start_time},
            )

    if not stopped_early:
        agent.save(
            checkpoint_dir / "latest.pt",
            step=end_step,
            config=config,
            extra={"best_eval_reward": best_eval_reward, "elapsed_seconds": time.time() - start_time},
        )
    env.close()
    eval_env.close()
    return run_dir


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to a YAML training config.")
    parser.add_argument("--results-dir", default="results", help="Directory used for run outputs.")
    parser.add_argument("--resume", help="Path to a checkpoint to resume model and optimizer state.")
    parser.add_argument("--resume-model-only", action="store_true", help="Load model weights from --resume but start with a fresh optimizer.")
    parser.add_argument("--overwrite", action="store_true", help="Delete this run directory before starting.")
    parser.add_argument("--limit-steps", type=int, help="Run at most this many additional environment steps.")
    parser.add_argument("--no-progress", action="store_true", help="Disable tqdm progress output.")
    args = parser.parse_args()
    config = load_config(args.config)
    train(
        config,
        results_dir=args.results_dir,
        resume_checkpoint=args.resume,
        resume_model_only=args.resume_model_only,
        overwrite=args.overwrite,
        limit_steps=args.limit_steps,
        show_progress=not args.no_progress,
    )


if __name__ == "__main__":
    main()
