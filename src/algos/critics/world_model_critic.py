"""World Model Critic 算法模块。

封装 Twin Q 网络及其目标网络，提供分布式 Q 值计算、
软更新、自动温度调节和模型持久化等功能。适用于
model-based 多智能体强化学习中 imagination 阶段
的价值评估。
"""

from __future__ import annotations

import itertools
from copy import deepcopy

import numpy as np
import torch
from gymnasium import spaces

from src.models.value.world_model_value import (
    ContinuousQNet,
)
from src.utils.env import check
from src.utils.math import TwoHotProcessor
from src.utils.model import update_linear_schedule
from src.utils.scale import RunningScale


class WorldModelCritic:
    """World Model Critic 算法。

    采用 Twin Q 网络（双 Q 网络）抑制过估计，通过
    TwoHotProcessor 实现分布式 Q 值回归，并使用
    RunningScale 对 Q 值做在线百分位归一化。支持
    SAC 风格的自动温度系数调节。
    """

    def __init__(
        self,
        args: dict,
        share_obs_space: spaces.Space,
        act_space: list[spaces.Space],
        num_agents: int,
        state_type: str,
        device: torch.device = torch.device("cpu"),
    ) -> None:
        """
        初始化 World Model Critic。

        参数:
            args: 算法参数字典，需包含:
                - critic_lr: Critic 学习率。
                - gamma: 折扣因子。
                - polyak: 目标网络软更新系数。
                - auto_alpha: 是否自动调节温度。
                - alpha: 固定温度系数
                    （auto_alpha 为 False 时使用）。
                - num_bins: Two-Hot 编码的 bin 数量。
                - reward_min: symlog 空间最小值。
                - reward_max: symlog 空间最大值。
                - scale_tau: RunningScale 的 EMA 系数。
            share_obs_space: 共享观测空间。
            act_space: 各智能体的动作空间列表。
            num_agents: 智能体数量。
            state_type: 状态类型（"EP" 或 "FP"）。
            device: 用于张量运算的设备。
        """
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        self.device = device
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = (
            act_space[0].__class__.__name__
        )

        # Twin Q 网络
        self.critic = ContinuousQNet(
            args, share_obs_space, act_space, device,
        )
        self.critic2 = ContinuousQNet(
            args, share_obs_space, act_space, device,
        )

        # 零初始化最后一层，使初始 Q 值接近零
        self.critic.mlp[-1].weight.data.fill_(0)
        self.critic2.mlp[-1].weight.data.fill_(0)

        # 目标网络
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False

        # 超参数
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]

        # 联合优化器
        critic_params = itertools.chain(
            self.critic.parameters(),
            self.critic2.parameters(),
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params, lr=self.critic_lr,
        )

        # 分布式 Q 值处理器
        self.processor = TwoHotProcessor(
            num_bins=args.get("num_bins", 101),
            vmin=args.get("reward_min", -10.0),
            vmax=args.get("reward_max", 10.0),
            device=device,
        )

        # Q 值在线缩放
        self.scale = RunningScale(
            tau=args.get("scale_tau", 0.01),
            device=device,
        )

        # 自动温度调节
        self.auto_alpha = args.get("auto_alpha", True)
        if self.auto_alpha:
            self.log_alpha = torch.zeros(
                1, requires_grad=True, device=device,
            )
            self.alpha_optimizer = torch.optim.Adam(
                [self.log_alpha],
                lr=args.get("alpha_lr", 3e-4),
            )
            self.alpha = torch.exp(
                self.log_alpha.detach(),
            )
        else:
            self.alpha = args["alpha"]

        self.turn_off_grad()

    def get_values(
        self,
        share_obs: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
        mode: str = "mean",
    ) -> torch.Tensor:
        """
        计算 Q 值。

        参数:
            share_obs: 共享观测。
            actions: 动作。
            mode: 聚合方式，"min" 取两个 Q 网络的
                最小值，"mean" 取均值。

        返回:
            解码后的标量 Q 值张量。

        异常:
            ValueError: 当 mode 不是 "min" 或 "mean"
                时抛出。
        """
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        q_logits = self.critic(share_obs, actions)
        q_logits2 = self.critic2(share_obs, actions)
        q_value = self.processor.decode(q_logits)
        q_value2 = self.processor.decode(q_logits2)
        if mode == "min":
            return torch.min(q_value, q_value2)
        if mode == "mean":
            return (q_value + q_value2) / 2
        raise ValueError(
            f"不支持的聚合方式: {mode!r}，"
            "可选 'min' 或 'mean'"
        )

    @torch.no_grad()
    def get_target_values(
        self,
        share_obs: np.ndarray | torch.Tensor,
        actions: np.ndarray | torch.Tensor,
    ) -> torch.Tensor:
        """
        计算目标网络的 Q 值（无梯度）。

        参数:
            share_obs: 共享观测。
            actions: 动作。

        返回:
            两个目标 Q 网络的最小值。
        """
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        q_logits = self.target_critic(
            share_obs, actions,
        )
        q_logits2 = self.target_critic2(
            share_obs, actions,
        )
        q_value = self.processor.decode(q_logits)
        q_value2 = self.processor.decode(q_logits2)
        return torch.min(q_value, q_value2)

    def update_alpha(
        self,
        logp_actions: list[torch.Tensor],
        target_entropy: float,
    ) -> None:
        """
        自动调节温度系数 alpha。

        参数:
            logp_actions: 各智能体动作对数概率列表。
            target_entropy: 目标熵值。
        """
        log_prob = (
            torch.sum(
                torch.cat(logp_actions, dim=-1),
                dim=-1,
                keepdim=True,
            )
            .detach()
            .to(**self.tpdv)
            + target_entropy
        )
        alpha_loss = -(self.log_alpha * log_prob).mean()
        self.alpha_optimizer.zero_grad()
        alpha_loss.backward()
        self.alpha_optimizer.step()
        self.alpha = torch.exp(
            self.log_alpha.detach(),
        )

    def soft_update(self) -> None:
        """按 polyak 系数软更新目标网络。"""
        for param_target, param in zip(
            self.target_critic.parameters(),
            self.critic.parameters(),
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak)
                + param.data * self.polyak
            )
        for param_target, param in zip(
            self.target_critic2.parameters(),
            self.critic2.parameters(),
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak)
                + param.data * self.polyak
            )

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
            self.critic_optimizer,
            step,
            steps,
            self.critic_lr,
        )

    def save(
        self,
        save_dir: str,
    ) -> None:
        """
        保存所有网络参数和缩放状态。

        参数:
            save_dir: 保存目录路径。
        """
        path = str(save_dir)
        torch.save(
            self.critic.state_dict(),
            path + "/critic_agent.pt",
        )
        torch.save(
            self.target_critic.state_dict(),
            path + "/target_critic_agent.pt",
        )
        torch.save(
            self.critic2.state_dict(),
            path + "/critic_agent2.pt",
        )
        torch.save(
            self.target_critic2.state_dict(),
            path + "/target_critic_agent2.pt",
        )
        torch.save(
            self.scale.state_dict(),
            path + "/q_scale.pt",
        )

    def restore(
        self,
        model_dir: str,
    ) -> None:
        """
        加载所有网络参数和缩放状态。

        参数:
            model_dir: 模型目录路径。
        """
        path = str(model_dir)
        self.critic.load_state_dict(torch.load(
            path + "/critic_agent.pt",
            map_location=self.device,
        ))
        self.target_critic.load_state_dict(torch.load(
            path + "/target_critic_agent.pt",
            map_location=self.device,
        ))
        self.critic2.load_state_dict(torch.load(
            path + "/critic_agent2.pt",
            map_location=self.device,
        ))
        self.target_critic2.load_state_dict(torch.load(
            path + "/target_critic_agent2.pt",
            map_location=self.device,
        ))
        self.scale.load_state_dict(torch.load(
            path + "/q_scale.pt",
            map_location=self.device,
        ))

    def turn_on_grad(self) -> None:
        """开启 Critic 参数的梯度计算。"""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self) -> None:
        """关闭 Critic 参数的梯度计算。"""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
