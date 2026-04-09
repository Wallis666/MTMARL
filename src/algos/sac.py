"""
CleanRL 风格的 Soft Actor-Critic（连续动作版）。

包含:
  * ``Actor``    : tanh-squashed 对角高斯策略；
  * ``QNetwork`` : 双 Q 网络结构（这里写一份，实例化两个）；
  * ``SAC``      : 封装 actor / 双 Q / target Q / 自适应温度 α 的更新逻辑。

使用方式（伪代码）::

    agent = SAC(obs_dim, action_dim, action_low, action_high)
    a = agent.select_action(obs)
    ...
    info = agent.update(batch)
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam

from src.buffers.replay import Batch

LOG_STD_MIN = -5.0
LOG_STD_MAX = 2.0


def _mlp(in_dim: int, out_dim: int, hidden: int = 256) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.ReLU(),
        nn.Linear(hidden, hidden),
        nn.ReLU(),
        nn.Linear(hidden, out_dim),
    )


class Actor(nn.Module):
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        hidden: int = 256,
    ) -> None:
        super().__init__()
        self.trunk = nn.Sequential(
            nn.Linear(obs_dim, hidden), nn.ReLU(),
            nn.Linear(hidden, hidden), nn.ReLU(),
        )
        self.mu = nn.Linear(hidden, action_dim)
        self.log_std = nn.Linear(hidden, action_dim)

        # 把 [low, high] 映射到 tanh 输出 [-1, 1]
        self.register_buffer(
            "act_scale",
            torch.tensor((action_high - action_low) / 2.0, dtype=torch.float32),
        )
        self.register_buffer(
            "act_bias",
            torch.tensor((action_high + action_low) / 2.0, dtype=torch.float32),
        )

    def forward(self, obs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.trunk(obs)
        mu = self.mu(h)
        log_std = self.log_std(h).clamp(LOG_STD_MIN, LOG_STD_MAX)
        return mu, log_std

    def sample(
        self,
        obs: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """返回 (action, log_prob, deterministic_action)。"""
        mu, log_std = self.forward(obs)
        std = log_std.exp()
        normal = torch.distributions.Normal(mu, std)
        x = normal.rsample()
        y = torch.tanh(x)
        action = y * self.act_scale + self.act_bias
        # tanh 修正
        log_prob = normal.log_prob(x) - torch.log(
            self.act_scale * (1 - y.pow(2)) + 1e-6
        )
        log_prob = log_prob.sum(-1, keepdim=True)
        det_action = torch.tanh(mu) * self.act_scale + self.act_bias
        return action, log_prob, det_action


class QNetwork(nn.Module):
    def __init__(self, obs_dim: int, action_dim: int, hidden: int = 256) -> None:
        super().__init__()
        self.net = _mlp(obs_dim + action_dim, 1, hidden)

    def forward(self, obs: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([obs, action], dim=-1))


@dataclass
class SACConfig:
    gamma: float = 0.99
    tau: float = 0.005
    actor_lr: float = 3e-4
    critic_lr: float = 3e-4
    alpha_lr: float = 3e-4
    target_entropy: float | None = None  # None → -action_dim
    hidden: int = 256


class SAC:
    def __init__(
        self,
        obs_dim: int,
        action_dim: int,
        action_low: np.ndarray,
        action_high: np.ndarray,
        device: str | torch.device = "cpu",
        config: SACConfig | None = None,
    ) -> None:
        self.cfg = config or SACConfig()
        self.device = torch.device(device)
        self.action_dim = action_dim

        self.actor = Actor(
            obs_dim, action_dim, action_low, action_high, self.cfg.hidden,
        ).to(self.device)
        self.q1 = QNetwork(obs_dim, action_dim, self.cfg.hidden).to(self.device)
        self.q2 = QNetwork(obs_dim, action_dim, self.cfg.hidden).to(self.device)
        self.q1_target = QNetwork(obs_dim, action_dim, self.cfg.hidden).to(self.device)
        self.q2_target = QNetwork(obs_dim, action_dim, self.cfg.hidden).to(self.device)
        self.q1_target.load_state_dict(self.q1.state_dict())
        self.q2_target.load_state_dict(self.q2.state_dict())

        self.actor_opt = Adam(self.actor.parameters(), lr=self.cfg.actor_lr)
        self.q_opt = Adam(
            list(self.q1.parameters()) + list(self.q2.parameters()),
            lr=self.cfg.critic_lr,
        )

        # 自适应温度
        self.target_entropy = (
            self.cfg.target_entropy
            if self.cfg.target_entropy is not None
            else -float(action_dim)
        )
        self.log_alpha = torch.zeros(1, requires_grad=True, device=self.device)
        self.alpha_opt = Adam([self.log_alpha], lr=self.cfg.alpha_lr)

    @property
    def alpha(self) -> torch.Tensor:
        return self.log_alpha.exp()

    @torch.no_grad()
    def select_action(
        self,
        obs: np.ndarray,
        deterministic: bool = False,
    ) -> np.ndarray:
        obs_t = torch.as_tensor(obs, dtype=torch.float32, device=self.device)
        if obs_t.ndim == 1:
            obs_t = obs_t.unsqueeze(0)
        action, _, det = self.actor.sample(obs_t)
        out = det if deterministic else action
        return out.squeeze(0).cpu().numpy()

    def update(self, batch: Batch) -> dict[str, float]:
        # ----- critic -----
        with torch.no_grad():
            next_a, next_logp, _ = self.actor.sample(batch.next_obs)
            target_q = torch.min(
                self.q1_target(batch.next_obs, next_a),
                self.q2_target(batch.next_obs, next_a),
            ) - self.alpha * next_logp
            target = batch.rewards + (1.0 - batch.dones) * self.cfg.gamma * target_q

        q1_pred = self.q1(batch.obs, batch.actions)
        q2_pred = self.q2(batch.obs, batch.actions)
        q_loss = F.mse_loss(q1_pred, target) + F.mse_loss(q2_pred, target)

        self.q_opt.zero_grad()
        q_loss.backward()
        self.q_opt.step()

        # ----- actor -----
        a, logp, _ = self.actor.sample(batch.obs)
        q_min = torch.min(self.q1(batch.obs, a), self.q2(batch.obs, a))
        actor_loss = (self.alpha.detach() * logp - q_min).mean()

        self.actor_opt.zero_grad()
        actor_loss.backward()
        self.actor_opt.step()

        # ----- alpha -----
        alpha_loss = -(self.log_alpha * (logp.detach() + self.target_entropy)).mean()
        self.alpha_opt.zero_grad()
        alpha_loss.backward()
        self.alpha_opt.step()

        # ----- target soft update -----
        with torch.no_grad():
            for p, tp in zip(self.q1.parameters(), self.q1_target.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)
            for p, tp in zip(self.q2.parameters(), self.q2_target.parameters()):
                tp.data.mul_(1 - self.cfg.tau).add_(self.cfg.tau * p.data)

        return {
            "q_loss": float(q_loss.item()),
            "actor_loss": float(actor_loss.item()),
            "alpha_loss": float(alpha_loss.item()),
            "alpha": float(self.alpha.item()),
            "entropy": float(-logp.mean().item()),
        }

    def state_dict(self) -> dict:
        return {
            "actor": self.actor.state_dict(),
            "q1": self.q1.state_dict(),
            "q2": self.q2.state_dict(),
            "q1_target": self.q1_target.state_dict(),
            "q2_target": self.q2_target.state_dict(),
            "log_alpha": self.log_alpha.detach().cpu(),
        }

    def load_state_dict(self, sd: dict) -> None:
        self.actor.load_state_dict(sd["actor"])
        self.q1.load_state_dict(sd["q1"])
        self.q2.load_state_dict(sd["q2"])
        self.q1_target.load_state_dict(sd["q1_target"])
        self.q2_target.load_state_dict(sd["q2_target"])
        with torch.no_grad():
            self.log_alpha.copy_(sd["log_alpha"].to(self.device))
