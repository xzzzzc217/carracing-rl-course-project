from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .preprocessing import decode_observation, encode_observation


@dataclass(slots=True)
class TransitionBatch:
    states: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_states: np.ndarray
    dones: np.ndarray
    weights: np.ndarray
    indices: np.ndarray


class UniformReplayBuffer:
    """Fixed-size replay buffer storing observations as uint8 to reduce memory."""

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        seed: int = 0,
    ) -> None:
        if capacity <= 0:
            raise ValueError("capacity must be positive.")
        self.capacity = int(capacity)
        self.observation_shape = tuple(observation_shape)
        self.rng = np.random.default_rng(seed)
        self.states = np.empty((capacity, *self.observation_shape), dtype=np.uint8)
        self.next_states = np.empty((capacity, *self.observation_shape), dtype=np.uint8)
        self.actions = np.empty((capacity,), dtype=np.int64)
        self.rewards = np.empty((capacity,), dtype=np.float32)
        self.dones = np.empty((capacity,), dtype=np.float32)
        self.position = 0
        self.size = 0

    def __len__(self) -> int:
        return self.size

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        self.states[self.position] = encode_observation(state)
        self.next_states[self.position] = encode_observation(next_state)
        self.actions[self.position] = int(action)
        self.rewards[self.position] = float(reward)
        self.dones[self.position] = float(done)
        self.position = (self.position + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int, beta: float | None = None) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        indices = self.rng.integers(0, self.size, size=batch_size)
        weights = np.ones((batch_size,), dtype=np.float32)
        return self._make_batch(indices, weights)

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        return None

    def beta_by_step(self, step: int) -> float:
        return 1.0

    def _make_batch(self, indices: np.ndarray, weights: np.ndarray) -> TransitionBatch:
        return TransitionBatch(
            states=decode_observation(self.states[indices]),
            actions=self.actions[indices].copy(),
            rewards=self.rewards[indices].copy(),
            next_states=decode_observation(self.next_states[indices]),
            dones=self.dones[indices].copy(),
            weights=weights.astype(np.float32, copy=False),
            indices=indices.astype(np.int64, copy=False),
        )


class PrioritizedReplayBuffer(UniformReplayBuffer):
    """Proportional prioritized replay.

    A direct probability vector is used instead of a segment tree because the
    course-scale buffer is small enough and the implementation is easier to audit.
    """

    def __init__(
        self,
        capacity: int,
        observation_shape: tuple[int, ...],
        alpha: float = 0.6,
        beta_start: float = 0.4,
        beta_frames: int = 200_000,
        epsilon: float = 1e-6,
        seed: int = 0,
    ) -> None:
        super().__init__(capacity, observation_shape, seed)
        self.alpha = float(alpha)
        self.beta_start = float(beta_start)
        self.beta_frames = int(beta_frames)
        self.epsilon = float(epsilon)
        self.priorities = np.zeros((capacity,), dtype=np.float32)
        self.max_priority = 1.0

    def add(
        self,
        state: np.ndarray,
        action: int,
        reward: float,
        next_state: np.ndarray,
        done: bool,
    ) -> None:
        index = self.position
        super().add(state, action, reward, next_state, done)
        self.priorities[index] = self.max_priority

    def sample(self, batch_size: int, beta: float | None = None) -> TransitionBatch:
        if self.size == 0:
            raise ValueError("Cannot sample from an empty replay buffer.")
        beta_value = self.beta_start if beta is None else float(beta)
        scaled = np.power(self.priorities[: self.size] + self.epsilon, self.alpha)
        probabilities = scaled / scaled.sum()
        indices = self.rng.choice(self.size, size=batch_size, replace=True, p=probabilities)
        weights = np.power(self.size * probabilities[indices], -beta_value)
        weights /= weights.max()
        return self._make_batch(indices, weights.astype(np.float32))

    def update_priorities(self, indices: np.ndarray, priorities: np.ndarray) -> None:
        values = np.asarray(priorities, dtype=np.float32)
        values = np.maximum(values, self.epsilon)
        self.priorities[indices] = values
        self.max_priority = max(self.max_priority, float(values.max(initial=self.max_priority)))

    def beta_by_step(self, step: int) -> float:
        fraction = min(1.0, max(0.0, step / max(1, self.beta_frames)))
        return self.beta_start + fraction * (1.0 - self.beta_start)
