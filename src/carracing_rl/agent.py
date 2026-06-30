from __future__ import annotations

from pathlib import Path

import numpy as np
import torch
from torch import nn

from .models import DQNNetwork


class DQNAgent:
    def __init__(
        self,
        observation_shape: tuple[int, int, int],
        num_actions: int,
        config: dict,
        device: torch.device,
    ) -> None:
        train_config = config.get("train", {})
        agent_config = config.get("agent", {})
        self.num_actions = num_actions
        self.device = device
        self.gamma = float(train_config.get("gamma", 0.99))
        self.batch_size = int(train_config.get("batch_size", 32))
        self.double_dqn = bool(agent_config.get("double_dqn", True))
        self.grad_clip_norm = float(train_config.get("grad_clip_norm", 10.0))
        self.per_epsilon = float(agent_config.get("per_epsilon", 1e-6))
        self.lr = float(train_config.get("lr", 1e-4))
        probabilities = agent_config.get("random_action_probabilities")
        if probabilities is None:
            self.random_action_probabilities = None
        else:
            values = np.asarray(probabilities, dtype=np.float64)
            if values.shape != (num_actions,):
                raise ValueError(
                    f"random_action_probabilities must have shape ({num_actions},), got {values.shape}."
                )
            if np.any(values < 0) or values.sum() <= 0:
                raise ValueError("random_action_probabilities must be non-negative and have positive sum.")
            self.random_action_probabilities = values / values.sum()

        hidden_dim = int(agent_config.get("hidden_dim", 512))
        dueling = bool(agent_config.get("dueling", True))
        self.online = DQNNetwork(
            observation_shape,
            num_actions,
            hidden_dim=hidden_dim,
            dueling=dueling,
        ).to(device)
        self.target = DQNNetwork(
            observation_shape,
            num_actions,
            hidden_dim=hidden_dim,
            dueling=dueling,
        ).to(device)
        self.target.load_state_dict(self.online.state_dict())
        self.target.eval()
        self.optimizer = torch.optim.Adam(self.online.parameters(), lr=self.lr)
        self.loss_fn = nn.SmoothL1Loss(reduction="none")

    def epsilon_by_step(self, step: int, config: dict) -> float:
        train_config = config.get("train", {})
        start = float(train_config.get("epsilon_start", 1.0))
        final = float(train_config.get("epsilon_final", 0.05))
        decay_steps = max(1, int(train_config.get("epsilon_decay_steps", 100_000)))
        fraction = min(1.0, step / decay_steps)
        return start + fraction * (final - start)

    @torch.no_grad()
    def select_action(self, observation: np.ndarray, epsilon: float = 0.0) -> int:
        if np.random.random() < epsilon:
            return self.sample_random_action()
        state = torch.as_tensor(observation[None, ...], dtype=torch.float32, device=self.device)
        q_values = self.online(state)
        return int(q_values.argmax(dim=1).item())

    def sample_random_action(self) -> int:
        if self.random_action_probabilities is None:
            return int(np.random.randint(self.num_actions))
        return int(np.random.choice(self.num_actions, p=self.random_action_probabilities))

    def train_step(self, replay_buffer, global_step: int) -> dict[str, float]:
        beta = replay_buffer.beta_by_step(global_step)
        batch = replay_buffer.sample(self.batch_size, beta=beta)
        states = torch.as_tensor(batch.states, dtype=torch.float32, device=self.device)
        actions = torch.as_tensor(batch.actions, dtype=torch.int64, device=self.device).unsqueeze(1)
        rewards = torch.as_tensor(batch.rewards, dtype=torch.float32, device=self.device).unsqueeze(1)
        next_states = torch.as_tensor(batch.next_states, dtype=torch.float32, device=self.device)
        dones = torch.as_tensor(batch.dones, dtype=torch.float32, device=self.device).unsqueeze(1)
        weights = torch.as_tensor(batch.weights, dtype=torch.float32, device=self.device).unsqueeze(1)

        q_values = self.online(states).gather(1, actions)
        with torch.no_grad():
            if self.double_dqn:
                next_actions = self.online(next_states).argmax(dim=1, keepdim=True)
                next_q = self.target(next_states).gather(1, next_actions)
            else:
                next_q = self.target(next_states).max(dim=1, keepdim=True).values
            targets = rewards + self.gamma * (1.0 - dones) * next_q

        td_errors = targets - q_values
        losses = self.loss_fn(q_values, targets)
        loss = (losses * weights).mean()

        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(self.online.parameters(), self.grad_clip_norm)
        self.optimizer.step()

        priorities = np.abs(td_errors.detach().cpu().numpy().squeeze(1)) + self.per_epsilon
        replay_buffer.update_priorities(batch.indices, priorities)

        return {
            "loss": float(loss.item()),
            "mean_q": float(q_values.detach().mean().item()),
            "mean_target": float(targets.detach().mean().item()),
            "mean_td_abs": float(np.mean(priorities)),
            "beta": float(beta),
        }

    def update_target(self) -> None:
        self.target.load_state_dict(self.online.state_dict())

    def save(self, path: str | Path, step: int, config: dict, extra: dict | None = None) -> None:
        payload = {
            "step": step,
            "config": config,
            "online_state_dict": self.online.state_dict(),
            "target_state_dict": self.target.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "extra": extra or {},
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        torch.save(payload, path)

    def load(self, path: str | Path, load_optimizer: bool = True) -> dict:
        payload = torch.load(path, map_location=self.device, weights_only=False)
        self.online.load_state_dict(payload["online_state_dict"])
        self.target.load_state_dict(payload.get("target_state_dict", payload["online_state_dict"]))
        if load_optimizer and "optimizer_state_dict" in payload:
            self.optimizer.load_state_dict(payload["optimizer_state_dict"])
            for group in self.optimizer.param_groups:
                group["lr"] = self.lr
        return payload
