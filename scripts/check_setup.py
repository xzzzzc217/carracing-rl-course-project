from __future__ import annotations

import argparse
import importlib.util
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


REQUIRED_MODULES = [
    "gymnasium",
    "Box2D",
    "numpy",
    "PIL",
    "torch",
    "matplotlib",
    "tqdm",
    "yaml",
]


def check_imports() -> bool:
    ok = True
    for module in REQUIRED_MODULES:
        found = importlib.util.find_spec(module) is not None
        print(f"{module:12s}: {'ok' if found else 'missing'}")
        ok = ok and found
    return ok


def run_runtime_smoke(config_path: Path) -> None:
    import numpy as np
    import torch

    from carracing_rl.agent import DQNAgent
    from carracing_rl.config import load_config
    from carracing_rl.envs import make_env
    from carracing_rl.train import build_replay
    from carracing_rl.utils import configure_torch_runtime, resolve_device, set_seed

    config = load_config(config_path)
    seed = int(config.get("seed", 42))
    set_seed(seed)
    configure_torch_runtime(config)
    device = resolve_device(config.get("runtime", {}).get("device", "auto"))
    env = make_env(config, seed=seed)

    observation, _ = env.reset(seed=seed)
    assert observation.shape == env.observation_space.shape
    assert observation.dtype == np.float32
    print(f"env observation: shape={observation.shape}, dtype={observation.dtype}")
    print(f"env actions: n={env.action_space.n}")

    agent = DQNAgent(env.observation_space.shape, env.action_space.n, config, device=device)
    with torch.no_grad():
        q_values = agent.online(torch.as_tensor(observation[None, ...], dtype=torch.float32, device=device))
    assert q_values.shape == (1, env.action_space.n)
    print(f"network q-values: shape={tuple(q_values.shape)}, device={device}")

    replay = build_replay(config, env.observation_space.shape, seed=seed)
    for _ in range(max(agent.batch_size, 4)):
        action = env.action_space.sample()
        next_observation, reward, terminated, truncated, _ = env.step(action)
        done = bool(terminated or truncated)
        replay.add(observation, action, reward, next_observation, done)
        observation = next_observation
        if done:
            observation, _ = env.reset()

    metrics = agent.train_step(replay, global_step=1)
    print(
        "train step: "
        f"loss={metrics['loss']:.4f}, mean_q={metrics['mean_q']:.4f}, "
        f"mean_td_abs={metrics['mean_td_abs']:.4f}"
    )
    env.close()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--runtime-smoke", action="store_true", help="Also instantiate CarRacing and run one DQN update.")
    parser.add_argument("--config", default="configs/smoke.yaml", help="Config used for runtime smoke checks.")
    args = parser.parse_args()

    imports_ok = check_imports()
    if not imports_ok:
        raise SystemExit("Missing dependencies. Install them with `python -m pip install -e .`.")
    if args.runtime_smoke:
        run_runtime_smoke(ROOT / args.config)


if __name__ == "__main__":
    main()
