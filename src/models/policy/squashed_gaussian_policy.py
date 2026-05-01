"""压缩高斯策略网络模块，用于 HASAC 算法。"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.base.mlp import mlp
from src.utils.env import get_shape_from_obs_space

LOG_STD_MAX = 2
LOG_STD_MIN = -10


class SquashedGaussianPolicy(nn.Module):
    """用于 HASAC 的压缩高斯策略网络。"""

    def __init__(
        self,
        args,
        obs_space,
        action_space,
        device=torch.device("cpu"),
    ):
        """初始化压缩高斯策略模型。

        Args:
            args: 包含模型相关信息的字典。
            obs_space: 观测空间。
            action_space: 动作空间。
            device: 指定运行设备（cpu/gpu）。
        """
        super().__init__()
        self.tpdv = dict(dtype=torch.float32, device=device)
        self.log_std_min = torch.tensor(
            args.get("log_std_min", LOG_STD_MIN),
        ).to(**self.tpdv)
        self.log_std_max = torch.tensor(
            args.get("log_std_max", LOG_STD_MAX),
        ).to(**self.tpdv)

        hidden_sizes = args["hidden_sizes"]
        obs_shape = get_shape_from_obs_space(obs_space)
        act_dim = action_space.shape[0]

        self.net = mlp(
            in_dim=obs_shape[0],
            mlp_dims=hidden_sizes[:-1],
            out_dim=hidden_sizes[-1],
        )
        self.mu_layer = nn.Linear(
            hidden_sizes[-1],
            act_dim,
        )
        self.log_std_layer = nn.Linear(
            hidden_sizes[-1],
            act_dim,
        )
        self.act_limit = torch.tensor(
            action_space.high[0],
        ).to(**self.tpdv)
        self.to(device)

    def forward(
        self,
        obs,
        stochastic=True,
        with_logprob=False,
    ):
        """前向传播，返回缩放到动作空间范围内的输出。"""
        net_out = self.net(obs)
        mu = self.mu_layer(net_out)
        log_std = self.log_std_layer(net_out)
        # tanh 软截断，始终保持梯度流
        log_std = self._soft_clamp(
            log_std, self.log_std_min, self.log_std_max,
        )

        if stochastic:
            eps = torch.randn_like(mu).to(**self.tpdv)
            pi = mu + eps * log_std.exp()
        else:
            eps = torch.zeros_like(mu).to(**self.tpdv)
            pi = mu

        if with_logprob:
            log_pi = self._gaussian_logprob(eps, log_std)
        else:
            log_pi = None

        mu, pi, log_pi = self._squash(mu, pi, log_pi)

        return pi, log_pi

    def _gaussian_logprob(
        self,
        eps: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """计算高斯分布的对数概率。"""
        residual = self._gaussian_residual(
            eps, log_std,
        ).sum(-1, keepdim=True)
        size = eps.size(-1)
        return self._gaussian_logprob_from_residual(
            residual,
        ) * size

    def _squash(self, mu, pi, log_pi):
        """应用 tanh 压缩并修正对数概率。"""
        mu = torch.tanh(mu)
        pi = torch.tanh(pi)

        if log_pi is not None:
            log_pi -= self._squash_correction(pi).sum(
                -1, keepdim=True,
            )

        mu = mu * self.act_limit
        pi = pi * self.act_limit
        return mu, pi, log_pi

    @staticmethod
    @torch.jit.script
    def _soft_clamp(
        x: torch.Tensor,
        low: torch.Tensor,
        high: torch.Tensor,
    ) -> torch.Tensor:
        """tanh 软截断，映射到 [low, high] 且始终有梯度。"""
        return low + 0.5 * (high - low) * (torch.tanh(x) + 1)

    @staticmethod
    @torch.jit.script
    def _gaussian_residual(
        eps: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """高斯分布对数概率的残差项。"""
        return -0.5 * eps.pow(2) - log_std

    @staticmethod
    @torch.jit.script
    def _gaussian_logprob_from_residual(
        residual: torch.Tensor,
    ) -> torch.Tensor:
        """从残差计算完整的对数概率。"""
        log_two_pi = torch.log(torch.tensor(
            2.0 * torch.pi,
            dtype=residual.dtype,
            device=residual.device,
        ))
        return residual - 0.5 * log_two_pi

    @staticmethod
    @torch.jit.script
    def _squash_correction(
        pi: torch.Tensor,
    ) -> torch.Tensor:
        """tanh 压缩的对数概率修正项。"""
        return torch.log(F.relu(1 - pi.pow(2)) + 1e-6)
