from __future__ import annotations

import numpy as np

from carracing_rl.replay import PrioritizedReplayBuffer, UniformReplayBuffer


def _transition(index: int):
    state = np.full((4, 8, 8), fill_value=index / 10.0, dtype=np.float32)
    next_state = np.full((4, 8, 8), fill_value=(index + 1) / 10.0, dtype=np.float32)
    return state, index % 5, float(index), next_state, index % 2 == 0


def test_uniform_replay_samples_batch() -> None:
    buffer = UniformReplayBuffer(capacity=8, observation_shape=(4, 8, 8), seed=0)
    for index in range(6):
        buffer.add(*_transition(index))
    batch = buffer.sample(batch_size=4)
    assert batch.states.shape == (4, 4, 8, 8)
    assert batch.next_states.shape == (4, 4, 8, 8)
    assert batch.actions.shape == (4,)
    assert batch.weights.shape == (4,)


def test_prioritized_replay_updates_priorities() -> None:
    buffer = PrioritizedReplayBuffer(capacity=8, observation_shape=(4, 8, 8), seed=1)
    for index in range(6):
        buffer.add(*_transition(index))
    batch = buffer.sample(batch_size=4, beta=0.4)
    priorities = np.linspace(1.0, 2.0, num=4, dtype=np.float32)
    buffer.update_priorities(batch.indices, priorities)
    assert buffer.max_priority >= 2.0
    assert 0.4 <= buffer.beta_by_step(0) <= buffer.beta_by_step(200_000) <= 1.0
