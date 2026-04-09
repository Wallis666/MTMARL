"""
多智能体多任务世界模型训练器。

把项目的五大模型组件 —— ``ObsEncoder`` / ``SoftMoEDynamics`` /
``SparseMoERewardModel`` / ``SquashedGaussianActor`` / ``TwinQCritic``
—— 串成一次完整的更新流程，融合 TD-MPC2 风格的世界模型联合训练
与 SAC 风格的 actor-critic 更新，针对 ``MultiTaskMaMuJoCo`` 的
**多智能体共享潜空间 + 团队标量奖励** 场景做了一致化适配。

整体训练目标:

* **World model 联合损失**（在 :meth:`update_world_model` 中）

  - 一致性 (consistency)::

        L_z = Σ_{t=1..H} ρ^t · MSE(z_pred_t, sg(encoder(obs_t)))

    采用 stop-gradient 的目标编码，避免编码器追踪自身预测；
  - 奖励 (reward)::

        L_r = Σ_{t=0..H-1} ρ^t · CE(reward_logits_t, r_t)

    通过 :class:`TwoHotProcessor` 把团队标量奖励变成 two-hot 软目标；
  - MoE 负载均衡 (来自 ``SparseMoERewardModel`` 的辅助损失)::

        L_balance = Σ_{t} ρ^t · aux["load_balancing_loss"]

* **Critic 更新**（在 :meth:`update_critic` 中）：标准 SAC 双 Q
  + n-step 折扣 bootstrap，target 由目标网络在 next 潜空间评估，
  Q 头通过 :class:`TwoHotProcessor` 走 two-hot 离散回归。

* **Actor 更新**（在 :meth:`update_actor` 中）：SAC 风格的 reparam
  目标 ``α · logπ - min(Q1, Q2)``，潜在表示从编码器 detach。

* **Polyak 软更新**：每次 critic 更新后调用 :meth:`soft_update_target`。

约定:

* 所有 batch 字典来自 :class:`ReplayBuffer`，张量形状与
  ``MultiTaskMaMuJoCo`` 输出对齐：
  ``obs / next_obs ∈ (B, n_agents, obs_dim)``，
  ``actions ∈ (B, n_agents, action_dim)``，
  ``rewards / dones / gammas ∈ (B, 1)``；
* :meth:`update_world_model` 接收 ``sample_horizon`` 的输出，每项
  形状最外层为 ``(H, B, ...)``。
"""

from __future__ import annotations

from copy import deepcopy
from typing import Any

import numpy as np
import torch
from torch import nn

from src.models.actor import SquashedGaussianActor
from src.models.critic import TwinQCritic
from src.models.dynamics import SoftMoEDynamics
from src.models.obs_encoder import ObsEncoder
from src.models.reward import SparseMoERewardModel
from src.utils.two_hot import TwoHotProcessor


class WorldModelTrainer:
    """
    TD-MPC2 + SAC 风格的多智能体多任务世界模型训练器。

    本类不持有环境与缓冲区，仅负责"给一批数据 → 跑一次梯度更新"。
    外层 runner 负责采样 / 调用 / 日志。
    """

    def __init__(
        self,
        encoder: ObsEncoder,
        dynamics: SoftMoEDynamics,
        reward_model: SparseMoERewardModel,
        actor: SquashedGaussianActor,
        critic: TwinQCritic,
        two_hot: TwoHotProcessor,
        device: torch.device | str = "cpu",
        world_model_lr: float = 3e-4,
        actor_lr: float = 3e-4,
        critic_lr: float = 3e-4,
        gamma: float = 0.99,
        polyak: float = 0.005,
        alpha: float = 0.2,
        consistency_coef: float = 1.0,
        reward_coef: float = 1.0,
        balance_coef: float = 0.01,
        rho: float = 0.5,
        grad_clip_norm: float = 10.0,
    ) -> None:
        """
        参数:
            encoder: 观测编码器。
            dynamics: 潜在动力学模型。
            reward_model: 团队奖励模型。
            actor: 策略网络。
            critic: 双 Q critic 网络。
            two_hot: 用于 reward / Q 目标编码与解码的 ``TwoHotProcessor``。
            device: 训练设备。
            world_model_lr: encoder + dynamics + reward 共享 optimizer 的学习率。
            actor_lr: actor 学习率。
            critic_lr: critic 学习率。
            gamma: 折扣因子（critic 的 TD 目标使用）。
            polyak: 目标网络软更新系数 ``τ``。
            alpha: SAC 熵正则系数（固定，不做自动调节）。
            consistency_coef: 世界模型一致性损失权重。
            reward_coef: 世界模型奖励损失权重。
            balance_coef: MoE 负载均衡损失权重。
            rho: 时序衰减因子，对 ``t`` 步贡献加权 ``rho^t``。
            grad_clip_norm: 梯度裁剪的最大范数。
        """
        self.device = torch.device(device)
        self.encoder = encoder.to(self.device)
        self.dynamics = dynamics.to(self.device)
        self.reward_model = reward_model.to(self.device)
        self.actor = actor.to(self.device)
        self.critic = critic.to(self.device)
        self.two_hot = two_hot.to(self.device)

        # 目标 critic：深拷贝 + 冻结梯度，由 polyak 软更新维护
        self.target_critic = deepcopy(self.critic).to(self.device)
        for parameter in self.target_critic.parameters():
            parameter.requires_grad = False

        # 三套 optimizer：world_model / actor / critic
        world_model_parameters = list(self.encoder.parameters())
        world_model_parameters += list(self.dynamics.parameters())
        world_model_parameters += list(self.reward_model.parameters())
        self.world_model_optimizer = torch.optim.Adam(
            world_model_parameters,
            lr=world_model_lr,
        )
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(),
            lr=actor_lr,
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=critic_lr,
        )

        self.gamma = gamma
        self.polyak = polyak
        self.alpha = alpha
        self.consistency_coef = consistency_coef
        self.reward_coef = reward_coef
        self.balance_coef = balance_coef
        self.rho = rho
        self.grad_clip_norm = grad_clip_norm

    # ------------------------------------------------------------------
    # World model 更新
    # ------------------------------------------------------------------

    def update_world_model(
        self,
        horizon_batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        基于 ``ReplayBuffer.sample_horizon`` 的输出做一次世界模型更新。

        参数:
            horizon_batch: 字典，键含义见 ``ReplayBuffer.sample_horizon``，
                每项形状最外层为 ``(H, B, ...)``。

        返回:
            标量损失字典，方便上层日志记录。
        """
        obs = self._to_tensor(horizon_batch["obs"])  # (H, B, N_a, obs_dim)
        actions = self._to_tensor(horizon_batch["actions"])
        rewards = self._to_tensor(horizon_batch["rewards"])  # (H, B, 1)
        next_obs = self._to_tensor(horizon_batch["next_obs"])

        horizon = obs.shape[0]
        batch_size = obs.shape[1]

        # 把 (H, B, N_a, D) flatten 成 (H*B, N_a, D) 供 encoder 一次性编码
        with torch.no_grad():
            target_latents_flat = self.encoder(
                next_obs.reshape(horizon * batch_size, *next_obs.shape[2:]),
            )
        target_latents = target_latents_flat.reshape(
            horizon,
            batch_size,
            *target_latents_flat.shape[1:],
        )

        # 起点潜在 z_0：来自第一步的 obs
        current_latents = self.encoder(obs[0])  # (B, N_a, latent_dim)

        consistency_loss = torch.zeros((), device=self.device)
        reward_loss = torch.zeros((), device=self.device)
        balance_loss = torch.zeros((), device=self.device)

        rho_weight = 1.0
        for step_index in range(horizon):
            action_t = actions[step_index]
            reward_t = rewards[step_index]

            # 奖励预测使用当前潜在 + 当前动作
            reward_logits, reward_aux = self.reward_model(
                current_latents,
                action_t,
            )
            reward_loss = reward_loss + rho_weight * self.two_hot.cross_entropy_loss(
                reward_logits,
                reward_t,
            ).mean()
            balance_loss = balance_loss + rho_weight * reward_aux[
                "load_balancing_loss"
            ]

            # 推进一步动力学
            current_latents = self.dynamics(current_latents, action_t)

            # 一致性损失：与 sg(encoder(next_obs_t)) 对齐
            consistency_loss = consistency_loss + rho_weight * (
                (current_latents - target_latents[step_index]).pow(2).mean()
            )

            rho_weight *= self.rho

        total_loss = (
            self.consistency_coef * consistency_loss
            + self.reward_coef * reward_loss
            + self.balance_coef * balance_loss
        )

        self.world_model_optimizer.zero_grad(set_to_none=True)
        total_loss.backward()
        nn.utils.clip_grad_norm_(
            list(self.encoder.parameters())
            + list(self.dynamics.parameters())
            + list(self.reward_model.parameters()),
            max_norm=self.grad_clip_norm,
        )
        self.world_model_optimizer.step()

        return {
            "wm/total": float(total_loss.item()),
            "wm/consistency": float(consistency_loss.item()),
            "wm/reward": float(reward_loss.item()),
            "wm/balance": float(balance_loss.item()),
        }

    # ------------------------------------------------------------------
    # Critic 更新
    # ------------------------------------------------------------------

    def update_critic(
        self,
        batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        基于 ``ReplayBuffer.sample`` 的输出做一次 critic 更新。

        参数:
            batch: 字典，键含义见 ``ReplayBuffer.sample``。

        返回:
            标量损失字典。
        """
        obs = self._to_tensor(batch["obs"])
        actions = self._to_tensor(batch["actions"])
        rewards = self._to_tensor(batch["rewards"])
        gammas = self._to_tensor(batch["gammas"])
        dones = self._to_tensor(batch["dones"].astype(np.float32))
        next_obs = self._to_tensor(batch["next_obs"])

        # 当前潜在用于 Q(z, a)
        latents = self.encoder(obs).detach()
        q1_logits, q2_logits = self.critic(latents, actions)

        # 目标值：在 next 潜空间用当前 actor 采样并经过目标 critic
        with torch.no_grad():
            next_latents = self.encoder(next_obs)
            next_actions, next_log_probs = self.actor(
                next_latents,
                stochastic=True,
                with_logprob=True,
            )
            target_q1_logits, target_q2_logits = self.target_critic(
                next_latents,
                next_actions,
            )
            target_q1 = self.two_hot.decode(target_q1_logits)
            target_q2 = self.two_hot.decode(target_q2_logits)
            target_q = torch.min(target_q1, target_q2)

            # SAC 熵项：把每个智能体的 logπ 求和后取平均（团队标量）
            entropy_term = self.alpha * next_log_probs.mean(dim=1)
            target_value = target_q - entropy_term

            # n-step 折扣 bootstrap
            target_scalar = rewards + gammas * (1.0 - dones) * target_value

        critic_loss_q1 = self.two_hot.cross_entropy_loss(
            q1_logits,
            target_scalar,
        ).mean()
        critic_loss_q2 = self.two_hot.cross_entropy_loss(
            q2_logits,
            target_scalar,
        ).mean()
        critic_loss = critic_loss_q1 + critic_loss_q2

        self.critic_optimizer.zero_grad(set_to_none=True)
        critic_loss.backward()
        nn.utils.clip_grad_norm_(
            self.critic.parameters(),
            max_norm=self.grad_clip_norm,
        )
        self.critic_optimizer.step()

        return {
            "critic/total": float(critic_loss.item()),
            "critic/q1": float(critic_loss_q1.item()),
            "critic/q2": float(critic_loss_q2.item()),
            "critic/target_mean": float(target_scalar.mean().item()),
        }

    # ------------------------------------------------------------------
    # Actor 更新
    # ------------------------------------------------------------------

    def update_actor(
        self,
        batch: dict[str, np.ndarray],
    ) -> dict[str, float]:
        """
        基于 ``ReplayBuffer.sample`` 的输出做一次 actor 更新。

        参数:
            batch: 字典，键含义见 ``ReplayBuffer.sample``。

        返回:
            标量损失字典。
        """
        obs = self._to_tensor(batch["obs"])
        latents = self.encoder(obs).detach()

        sampled_actions, log_probs = self.actor(
            latents,
            stochastic=True,
            with_logprob=True,
        )
        q1_logits, q2_logits = self.critic(latents, sampled_actions)
        q1_value = self.two_hot.decode(q1_logits)
        q2_value = self.two_hot.decode(q2_logits)
        q_value = torch.min(q1_value, q2_value)

        # log_probs 形状 (B, n_agents, 1)，对智能体维度求和后再取均值，
        # 得到团队级别的策略熵
        entropy_term = log_probs.mean(dim=1)
        actor_loss = (self.alpha * entropy_term - q_value).mean()

        self.actor_optimizer.zero_grad(set_to_none=True)
        actor_loss.backward()
        nn.utils.clip_grad_norm_(
            self.actor.parameters(),
            max_norm=self.grad_clip_norm,
        )
        self.actor_optimizer.step()

        return {
            "actor/loss": float(actor_loss.item()),
            "actor/entropy": float(-entropy_term.mean().item()),
            "actor/q_value": float(q_value.mean().item()),
        }

    # ------------------------------------------------------------------
    # 目标网络软更新
    # ------------------------------------------------------------------

    def soft_update_target(self) -> None:
        """对目标 critic 做一次 Polyak 软更新。"""
        with torch.no_grad():
            for target_param, online_param in zip(
                self.target_critic.parameters(),
                self.critic.parameters(),
                strict=True,
            ):
                target_param.data.lerp_(online_param.data, self.polyak)

    # ------------------------------------------------------------------
    # 推理 / 评估接口
    # ------------------------------------------------------------------

    @torch.no_grad()
    def select_actions(
        self,
        obs: np.ndarray,
        stochastic: bool = True,
    ) -> np.ndarray:
        """
        给定一批环境观测，返回 actor 采样出的动作。

        参数:
            obs: 形状 ``(num_envs, n_agents, obs_dim)`` 的 numpy 观测。
            stochastic: 为 ``True`` 时按高斯采样，否则取均值。

        返回:
            形状 ``(num_envs, n_agents, action_dim)`` 的 numpy 动作。
        """
        obs_tensor = self._to_tensor(obs)
        latents = self.encoder(obs_tensor)
        actions, _ = self.actor(
            latents,
            stochastic=stochastic,
            with_logprob=False,
        )
        return actions.detach().cpu().numpy()

    # ------------------------------------------------------------------
    # 内部辅助
    # ------------------------------------------------------------------

    def _to_tensor(
        self,
        array: np.ndarray,
    ) -> torch.Tensor:
        """把 numpy 数组迁移到训练设备的 ``float32`` 张量。"""
        return torch.as_tensor(array, dtype=torch.float32, device=self.device)

    # ------------------------------------------------------------------
    # 模式切换
    # ------------------------------------------------------------------

    def train(self) -> None:
        """把所有可训练模块切到训练模式。"""
        self.encoder.train()
        self.dynamics.train()
        self.reward_model.train()
        self.actor.train()
        self.critic.train()

    def eval(self) -> None:
        """把所有可训练模块切到评估模式。"""
        self.encoder.eval()
        self.dynamics.eval()
        self.reward_model.eval()
        self.actor.eval()
        self.critic.eval()

    def state_dict(self) -> dict[str, Any]:
        """返回所有模型与 optimizer 的 state_dict，便于检查点保存。"""
        return {
            "encoder": self.encoder.state_dict(),
            "dynamics": self.dynamics.state_dict(),
            "reward_model": self.reward_model.state_dict(),
            "actor": self.actor.state_dict(),
            "critic": self.critic.state_dict(),
            "target_critic": self.target_critic.state_dict(),
            "world_model_optimizer": self.world_model_optimizer.state_dict(),
            "actor_optimizer": self.actor_optimizer.state_dict(),
            "critic_optimizer": self.critic_optimizer.state_dict(),
        }

    def load_state_dict(
        self,
        state: dict[str, Any],
    ) -> None:
        """从 :meth:`state_dict` 的输出恢复所有模型与 optimizer。"""
        self.encoder.load_state_dict(state["encoder"])
        self.dynamics.load_state_dict(state["dynamics"])
        self.reward_model.load_state_dict(state["reward_model"])
        self.actor.load_state_dict(state["actor"])
        self.critic.load_state_dict(state["critic"])
        self.target_critic.load_state_dict(state["target_critic"])
        self.world_model_optimizer.load_state_dict(
            state["world_model_optimizer"],
        )
        self.actor_optimizer.load_state_dict(state["actor_optimizer"])
        self.critic_optimizer.load_state_dict(state["critic_optimizer"])
