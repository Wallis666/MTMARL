"""World Model Actor 算法模块。

封装 World Model 策略网络，提供动作采样、对数概率
计算、梯度控制和模型持久化等功能。适用于 model-based
多智能体强化学习中 imagination 阶段的策略优化。
"""

from __future__ import annotations

import numpy as np
import torch
from gymnasium import spaces

from src.models.policy.squashed_gaussian_policy import (
    SquashedGaussianPolicy,
)
from src.utils.env import check
from src.utils.model import update_linear_schedule


class WorldModelActor:
    """World Model Actor 算法。

    管理策略网络的训练生命周期，包括梯度开关控制
    （用于 world model imagination 中选择性冻结策略）、
    学习率衰减和按智能体编号的模型存储与加载。
    """

    def __init__(
        self,
        args: dict,
        obs_space: spaces.Space,
        act_space: spaces.Box,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        初始化 World Model Actor。

        参数:
            args: 算法参数字典，需包含:
                - lr: 学习率。
                - hidden_sizes: 隐藏层维度列表。
            obs_space: 观测空间。
            act_space: 动作空间，仅支持连续动作空间。
            device: 用于张量运算的设备。

        异常:
            AssertionError: 当动作空间不是 Box 类型时
                抛出。
        """
        assert act_space.__class__.__name__ == "Box", (
            "WorldModelActor 仅支持连续动作空间"
        )

        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        self.lr = args["lr"]
        self.device = device

        self.actor = SquashedGaussianPolicy(
            args, obs_space, act_space, device,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr,
        )
        self.turn_off_grad()

    def get_actions(
        self,
        obs: np.ndarray | torch.Tensor,
        available_actions: None = None,
        stochastic: bool = True,
    ) -> torch.Tensor:
        """
        根据观测采样动作。

        参数:
            obs: 观测值。
            available_actions: 可用动作掩码，连续空间
                下必须为 None。
            stochastic: 是否使用随机策略。

        返回:
            采样得到的动作张量。
        """
        obs = check(obs).to(**self.tpdv)
        actions, _ = self.actor(
            obs,
            stochastic=stochastic,
            with_logprob=False,
        )
        return actions

    def get_actions_with_logprobs(
        self,
        obs: np.ndarray | torch.Tensor,
        available_actions: None = None,
        stochastic: bool = True,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        采样动作并返回对应的对数概率。

        参数:
            obs: 观测值。
            available_actions: 可用动作掩码，连续空间
                下必须为 None。
            stochastic: 是否使用随机策略。

        返回:
            (动作, 对数概率) 二元组。
        """
        obs = check(obs).to(**self.tpdv)
        actions, logp_actions = self.actor(
            obs,
            stochastic=stochastic,
            with_logprob=True,
        )
        return actions, logp_actions

    def lr_decay(
        self,
        step: int,
        steps: int,
    ) -> None:
        """
        线性衰减学习率。

        参数:
            step: 当前训练步数。
            steps: 总训练步数。
        """
        update_linear_schedule(
            self.actor_optimizer, step, steps, self.lr,
        )

    def save(
        self,
        save_dir: str,
        agent_id: int,
    ) -> None:
        """
        保存策略网络参数。

        参数:
            save_dir: 保存目录路径。
            agent_id: 智能体编号。
        """
        torch.save(
            self.actor.state_dict(),
            str(save_dir)
            + "/actor_agent"
            + str(agent_id)
            + ".pt",
        )

    def restore(
        self,
        model_dir: str,
        agent_id: int,
    ) -> None:
        """
        加载策略网络参数。

        参数:
            model_dir: 模型目录路径。
            agent_id: 智能体编号。
        """
        actor_state_dict = torch.load(
            str(model_dir)
            + "/actor_agent"
            + str(agent_id)
            + ".pt",
            map_location=self.device,
        )
        self.actor.load_state_dict(actor_state_dict)

    def turn_on_grad(self) -> None:
        """开启策略网络参数的梯度计算。"""
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self) -> None:
        """关闭策略网络参数的梯度计算。"""
        for p in self.actor.parameters():
            p.requires_grad = False
