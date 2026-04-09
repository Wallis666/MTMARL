"""
连续动作 SAC 用的 numpy 经验回放缓冲区。

简单环形缓冲，存 ``(obs, action, reward, next_obs, done)``，``sample`` 返回
torch tensor，便于直接喂给 SAC 更新函数。
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch


@dataclass
class Batch:
    obs: torch.Tensor
    actions: torch.Tensor
    rewards: torch.Tensor
    next_obs: torch.Tensor
    dones: torch.Tensor


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        obs_shape: tuple[int, ...],
        action_dim: int,
        device: str | torch.device = "cpu",
    ) -> None:
        self.capacity = int(capacity)
        self.device = torch.device(device)
        self.ptr = 0
        self.size = 0

        self.obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.next_obs = np.zeros((self.capacity, *obs_shape), dtype=np.float32)
        self.actions = np.zeros((self.capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((self.capacity, 1), dtype=np.float32)
        self.dones = np.zeros((self.capacity, 1), dtype=np.float32)

    def add(
        self,
        obs: np.ndarray,
        action: np.ndarray,
        reward: float,
        next_obs: np.ndarray,
        done: bool,
    ) -> None:
        i = self.ptr
        self.obs[i] = obs
        self.next_obs[i] = next_obs
        self.actions[i] = action
        self.rewards[i, 0] = reward
        self.dones[i, 0] = float(done)
        self.ptr = (self.ptr + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def sample(self, batch_size: int) -> Batch:
        idx = np.random.randint(0, self.size, size=batch_size)
        to_t = lambda a: torch.from_numpy(a[idx]).to(self.device)  # noqa: E731
        return Batch(
            obs=to_t(self.obs),
            actions=to_t(self.actions),
            rewards=to_t(self.rewards),
            next_obs=to_t(self.next_obs),
            dones=to_t(self.dones),
        )

    def __len__(self) -> int:
        return self.size
