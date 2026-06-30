from __future__ import annotations

from collections import deque
import warnings

import numpy as np

from .preprocessing import preprocess_frame

try:
    import gymnasium as gym
    from gymnasium import spaces
except ImportError:  # pragma: no cover - exercised only when dependency is missing.
    gym = None
    spaces = None


if gym is not None:

    class CarRacingPreprocessWrapper(gym.ObservationWrapper):
        """Preprocess raw 96x96 RGB frames for pixel-based value learning."""

        def __init__(
            self,
            env: gym.Env,
            image_size: int = 64,
            crop_bottom: int = 12,
            grayscale: bool = True,
        ) -> None:
            super().__init__(env)
            self.image_size = image_size
            self.crop_bottom = crop_bottom
            self.grayscale = grayscale
            channels = 1 if grayscale else 3
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(channels, image_size, image_size),
                dtype=np.float32,
            )

        def observation(self, observation: np.ndarray) -> np.ndarray:
            return preprocess_frame(
                observation,
                image_size=self.image_size,
                crop_bottom=self.crop_bottom,
                grayscale=self.grayscale,
            )


    class FrameStackWrapper(gym.Wrapper):
        """Stack the latest k channel-first observations along the channel axis."""

        def __init__(self, env: gym.Env, num_stack: int = 4) -> None:
            if num_stack < 1:
                raise ValueError("num_stack must be at least 1.")
            super().__init__(env)
            self.num_stack = num_stack
            self.frames: deque[np.ndarray] = deque(maxlen=num_stack)
            low = np.repeat(env.observation_space.low, num_stack, axis=0)
            high = np.repeat(env.observation_space.high, num_stack, axis=0)
            self.observation_space = spaces.Box(
                low=low,
                high=high,
                dtype=env.observation_space.dtype,
            )

        def reset(self, **kwargs):
            observation, info = self.env.reset(**kwargs)
            self.frames.clear()
            for _ in range(self.num_stack):
                self.frames.append(observation)
            return self._get_observation(), info

        def step(self, action):
            observation, reward, terminated, truncated, info = self.env.step(action)
            self.frames.append(observation)
            return self._get_observation(), reward, terminated, truncated, info

        def _get_observation(self) -> np.ndarray:
            return np.concatenate(list(self.frames), axis=0).astype(np.float32, copy=False)


    class ActionRepeatWrapper(gym.Wrapper):
        """Repeat each chosen action for a small number of environment steps."""

        def __init__(self, env: gym.Env, repeat: int = 1) -> None:
            if repeat < 1:
                raise ValueError("repeat must be at least 1.")
            super().__init__(env)
            self.repeat = repeat

        def step(self, action):
            total_reward = 0.0
            last_observation = None
            last_info = {}
            terminated = False
            truncated = False
            for _ in range(self.repeat):
                last_observation, reward, terminated, truncated, last_info = self.env.step(action)
                total_reward += float(reward)
                if terminated or truncated:
                    break
            return last_observation, total_reward, terminated, truncated, last_info


    class RewardTransformWrapper(gym.RewardWrapper):
        """Apply optional reward scaling and clipping for DQN stability."""

        def __init__(
            self,
            env: gym.Env,
            reward_scale: float = 1.0,
            reward_clip: float | None = None,
        ) -> None:
            super().__init__(env)
            self.reward_scale = reward_scale
            self.reward_clip = reward_clip

        def reward(self, reward):
            value = float(reward) * self.reward_scale
            if self.reward_clip is not None:
                value = float(np.clip(value, -self.reward_clip, self.reward_clip))
            return value

else:
    CarRacingPreprocessWrapper = None
    FrameStackWrapper = None
    ActionRepeatWrapper = None
    RewardTransformWrapper = None


def _make_base_env(env_id: str, seed: int | None, render_mode: str | None, max_episode_steps: int | None):
    if gym is None:
        raise ImportError(
            "gymnasium is required for CarRacing. Install dependencies with "
            "`python -m pip install -e .`."
        )

    kwargs = {"continuous": False}
    if render_mode is not None:
        kwargs["render_mode"] = render_mode
    if max_episode_steps is not None:
        kwargs["max_episode_steps"] = max_episode_steps

    try:
        env = gym.make(env_id, **kwargs)
    except Exception as exc:
        if env_id == "CarRacing-v2":
            warnings.warn(
                "CarRacing-v2 was not found; trying CarRacing-v3 for newer Gymnasium installations.",
                RuntimeWarning,
            )
            env = gym.make("CarRacing-v3", **kwargs)
        else:
            raise exc

    if seed is not None:
        env.reset(seed=seed)
        env.action_space.seed(seed)
        env.observation_space.seed(seed)
    return env


def make_env(config: dict, seed: int | None = None, render_mode: str | None = None):
    """Create the configured CarRacing environment with preprocessing wrappers."""
    env_config = config.get("env", {})
    env = _make_base_env(
        env_id=env_config.get("id", "CarRacing-v2"),
        seed=seed,
        render_mode=render_mode,
        max_episode_steps=env_config.get("max_episode_steps", 1000),
    )
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = ActionRepeatWrapper(env, repeat=int(env_config.get("action_repeat", 1)))
    env = CarRacingPreprocessWrapper(
        env,
        image_size=int(env_config.get("image_size", 64)),
        crop_bottom=int(env_config.get("crop_bottom", 12)),
        grayscale=bool(env_config.get("grayscale", True)),
    )
    env = FrameStackWrapper(env, num_stack=int(env_config.get("frame_stack", 4)))
    env = RewardTransformWrapper(
        env,
        reward_scale=float(env_config.get("reward_scale", 1.0)),
        reward_clip=env_config.get("reward_clip"),
    )
    return env
