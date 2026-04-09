"""
多智能体多任务策略网络（actor）。

参考 m3w-marl ``WorldModelPolicy`` 的设计，采用 SAC 风格的
**tanh-squashed Gaussian** 策略：

* 主干为一个由 ``NormedLinear`` 堆叠的 MLP，输入是
  ``ObsEncoder`` 得到的潜在 ``z``；
* 末端两个独立线性头分别预测高斯分布的均值 ``mu`` 与
  ``log_std``，``log_std`` 用 ``tanh`` 重参数化到
  ``[log_std_min, log_std_max]`` 之间；
* 采样后用 ``tanh`` 把动作压到 ``[-1, 1]``，再按动作上限缩放到
  环境实际范围；同步给出 log-prob 的 tanh 修正项以便 SAC 训练。

针对 MTMARL 的关键约定:

* 输入张量形状为 ``(..., n_agents, latent_dim)``，所有智能体共享
  同一套策略权重；任务 one-hot 已通过 ``ObsEncoder`` 进入 ``z``，
  无需在 actor 内显式处理。
* 输出动作维度统一为 ``MultiTaskMaMuJoCo.max_action_dim``，未使用
  的尾部维度由上层 ``_crop_actions`` 自行裁剪。
* 输出动作范围被缩放到 ``[-action_limit, +action_limit]``，与
  MaMuJoCo / Gymnasium-Robotics 中 ``Box`` 动作空间一致。
"""

from __future__ import annotations

import math
from typing import Sequence

import torch
from torch import nn

from src.models.layers import build_normed_mlp


_LOG_STD_MIN_DEFAULT: float = -10.0
_LOG_STD_MAX_DEFAULT: float = 2.0


class SquashedGaussianActor(nn.Module):
    """
    SAC 风格的 tanh-squashed Gaussian 策略。

    所有智能体共享同一套权重；前向接受任意前缀维度，末维必须等于
    ``latent_dim``，输出末维等于 ``action_dim``。
    """

    def __init__(
        self,
        latent_dim: int,
        action_dim: int,
        hidden_dims: Sequence[int] = (512, 512),
        action_limit: float = 1.0,
        log_std_min: float = _LOG_STD_MIN_DEFAULT,
        log_std_max: float = _LOG_STD_MAX_DEFAULT,
    ) -> None:
        """
        参数:
            latent_dim: 单个智能体的潜在维度，与 ``ObsEncoder.latent_dim`` 对齐。
            action_dim: 单个智能体的动作维度，对应 ``max_action_dim``。
            hidden_dims: 主干 MLP 的隐藏层宽度序列。
            action_limit: 动作上限，``tanh`` 输出会被乘以该值。
            log_std_min: ``log_std`` 重参数化的下界。
            log_std_max: ``log_std`` 重参数化的上界。

        异常:
            ValueError: 当 ``hidden_dims`` 为空，或 ``log_std_min`` 不小于
                ``log_std_max`` 时抛出。
        """
        super().__init__()
        if len(hidden_dims) == 0:
            raise ValueError("hidden_dims 至少需要一层。")
        if log_std_min >= log_std_max:
            raise ValueError(
                f"log_std_min ({log_std_min}) 必须小于 log_std_max "
                f"({log_std_max})。"
            )

        self.latent_dim = latent_dim
        self.action_dim = action_dim
        self.action_limit = action_limit

        self.register_buffer(
            "log_std_min",
            torch.tensor(log_std_min, dtype=torch.float32),
        )
        self.register_buffer(
            "log_std_max",
            torch.tensor(log_std_max, dtype=torch.float32),
        )

        # 主干 MLP：末层不加激活，紧跟 mu / log_std 两个线性头
        self.trunk = build_normed_mlp(
            in_dim=latent_dim,
            hidden_dims=hidden_dims[:-1],
            out_dim=hidden_dims[-1],
            output_activation=nn.Mish(),
        )
        self.mu_head = nn.Linear(hidden_dims[-1], action_dim)
        self.log_std_head = nn.Linear(hidden_dims[-1], action_dim)

    # ------------------------------------------------------------------
    # 主接口
    # ------------------------------------------------------------------

    def forward(
        self,
        latents: torch.Tensor,
        stochastic: bool = True,
        with_logprob: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        采样动作并可选地返回对应的 log-prob。

        参数:
            latents: 形状 ``(..., latent_dim)`` 的潜在张量，典型为
                ``(num_envs, n_agents, latent_dim)``。
            stochastic: 为 ``True`` 时按高斯采样，否则直接取均值。
            with_logprob: 是否同时返回经 tanh 修正后的 log-prob。

        返回:
            ``(actions, log_probs)``，其中 ``actions`` 形状 ``(..., action_dim)``，
            落在 ``[-action_limit, +action_limit]`` 范围内；``log_probs``
            形状 ``(..., 1)``，``with_logprob=False`` 时为 ``None``。

        异常:
            ValueError: 当输入末维与 ``latent_dim`` 不一致时抛出。
        """
        if latents.shape[-1] != self.latent_dim:
            raise ValueError(
                f"latents 末维 {latents.shape[-1]} 与 latent_dim "
                f"{self.latent_dim} 不一致。"
            )

        features = self.trunk(latents)
        mu = self.mu_head(features)
        raw_log_std = self.log_std_head(features)
        log_std = self._rescale_log_std(raw_log_std)

        if stochastic:
            noise = torch.randn_like(mu)
            sampled = mu + noise * log_std.exp()
        else:
            noise = torch.zeros_like(mu)
            sampled = mu

        log_probs: torch.Tensor | None = None
        if with_logprob:
            log_probs = self._gaussian_logprob(noise, log_std)

        squashed_actions, log_probs = self._squash(sampled, log_probs)
        return squashed_actions, log_probs

    def deterministic(
        self,
        latents: torch.Tensor,
    ) -> torch.Tensor:
        """
        返回当前策略的确定性动作（即 ``tanh(mu) * action_limit``）。

        参数:
            latents: 形状 ``(..., latent_dim)`` 的潜在张量。

        返回:
            形状 ``(..., action_dim)`` 的动作张量。
        """
        actions, _ = self.forward(
            latents,
            stochastic=False,
            with_logprob=False,
        )
        return actions

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _rescale_log_std(
        self,
        raw_log_std: torch.Tensor,
    ) -> torch.Tensor:
        """把原始 log_std 通过 tanh 映射到 ``[log_std_min, log_std_max]``。"""
        return self.log_std_min + 0.5 * (
            self.log_std_max - self.log_std_min
        ) * (torch.tanh(raw_log_std) + 1.0)

    @staticmethod
    def _gaussian_logprob(
        noise: torch.Tensor,
        log_std: torch.Tensor,
    ) -> torch.Tensor:
        """
        计算重参数化采样的高斯 log-prob，沿动作维求和。

        参数:
            noise: 形状 ``(..., action_dim)`` 的标准正态噪声。
            log_std: 与 ``noise`` 同形状的 log 标准差。

        返回:
            形状 ``(..., 1)`` 的 log-prob。
        """
        residual = -0.5 * noise.pow(2) - log_std
        log_two_pi = math.log(2.0 * math.pi)
        per_dim = residual - 0.5 * log_two_pi
        return per_dim.sum(dim=-1, keepdim=True)

    def _squash(
        self,
        sampled: torch.Tensor,
        log_probs: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """
        对采样动作做 ``tanh`` 压缩并修正 log-prob，最后乘以动作上限。
        """
        squashed = torch.tanh(sampled)
        if log_probs is not None:
            correction = torch.log(
                torch.nn.functional.relu(1.0 - squashed.pow(2)) + 1e-6,
            ).sum(dim=-1, keepdim=True)
            log_probs = log_probs - correction
        return squashed * self.action_limit, log_probs
