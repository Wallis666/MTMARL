"""双 Q 网络连续动作空间 Critic 模块。"""

import itertools
from copy import deepcopy

import torch

from baselines.models.value.continuous_q_net import (
    ContinuousQNet,
)
from baselines.utils.env import check
from baselines.utils.model import update_linear_schedule


class TwinContinuousQCritic:
    """学习两个 Q 函数的 Critic，适用于连续动作空间。"""

    def __init__(
        self,
        args,
        share_obs_space,
        act_space,
        num_agents,
        state_type,
        device=torch.device("cpu"),
    ):
        """
        初始化 TwinContinuousQCritic。

        参数:
            args: 算法参数字典。
            share_obs_space: 共享观测空间。
            act_space: 动作空间。
            num_agents: 智能体数量。
            state_type: 状态类型。
            device: 用于张量运算的设备。
        """
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.action_type = (
            act_space[0].__class__.__name__
        )
        self.critic = ContinuousQNet(
            args, share_obs_space, act_space, device,
        )
        self.critic2 = ContinuousQNet(
            args, share_obs_space, act_space, device,
        )
        self.target_critic = deepcopy(self.critic)
        self.target_critic2 = deepcopy(self.critic2)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        for param in self.target_critic2.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = (
            args["use_proper_time_limits"]
        )
        critic_params = itertools.chain(
            self.critic.parameters(),
            self.critic2.parameters(),
        )
        self.critic_optimizer = torch.optim.Adam(
            critic_params, lr=self.critic_lr,
        )
        self.turn_off_grad()

    def lr_decay(
        self,
        step,
        steps,
    ):
        """
        衰减学习率。

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

    def soft_update(self):
        """软更新目标网络。"""
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

    def get_values(
        self,
        share_obs,
        actions,
    ):
        """获取 Q 值。"""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        return self.critic(share_obs, actions)

    def train(
        self,
        share_obs,
        actions,
        reward,
        done,
        term,
        next_share_obs,
        next_actions,
        gamma,
    ):
        """
        训练 Critic。

        参数:
            share_obs: 共享观测，形状 (batch_size, dim)。
            actions: 动作，形状 (n_agents, batch_size, dim)。
            reward: 奖励，形状 (batch_size, 1)。
            done: 终止标志，形状 (batch_size, 1)。
            term: 截断标志，形状 (batch_size, 1)。
            next_share_obs: 下一步共享观测。
            next_actions: 下一步动作。
            gamma: 折扣因子，形状 (batch_size, 1)。
        """
        assert (
            share_obs.__class__.__name__ == "ndarray"
        )
        assert actions.__class__.__name__ == "ndarray"
        assert reward.__class__.__name__ == "ndarray"
        assert done.__class__.__name__ == "ndarray"
        assert term.__class__.__name__ == "ndarray"
        assert (
            next_share_obs.__class__.__name__
            == "ndarray"
        )
        assert gamma.__class__.__name__ == "ndarray"
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv)
        actions = torch.cat(
            [actions[i] for i in range(actions.shape[0])],
            dim=-1,
        )
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(
            **self.tpdv,
        )
        next_actions = torch.cat(
            next_actions, dim=-1,
        ).to(**self.tpdv)
        next_q_values1 = self.target_critic(
            next_share_obs, next_actions,
        )
        next_q_values2 = self.target_critic2(
            next_share_obs, next_actions,
        )
        next_q_values = torch.min(
            next_q_values1, next_q_values2,
        )
        if self.use_proper_time_limits:
            q_targets = (
                reward
                + gamma * next_q_values * (1 - term)
            )
        else:
            q_targets = (
                reward
                + gamma * next_q_values * (1 - done)
            )
        critic_loss1 = torch.mean(
            torch.nn.functional.mse_loss(
                self.critic(share_obs, actions),
                q_targets,
            )
        )
        critic_loss2 = torch.mean(
            torch.nn.functional.mse_loss(
                self.critic2(share_obs, actions),
                q_targets,
            )
        )
        critic_loss = critic_loss1 + critic_loss2
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def save(
        self,
        save_dir,
    ):
        """保存模型参数。"""
        torch.save(
            self.critic.state_dict(),
            str(save_dir) + "/critic_agent.pt",
        )
        torch.save(
            self.target_critic.state_dict(),
            str(save_dir) + "/target_critic_agent.pt",
        )
        torch.save(
            self.critic2.state_dict(),
            str(save_dir) + "/critic_agent2.pt",
        )
        torch.save(
            self.target_critic2.state_dict(),
            str(save_dir)
            + "/target_critic_agent2.pt",
        )

    def restore(
        self,
        model_dir,
    ):
        """恢复模型参数。"""
        critic_state_dict = torch.load(
            str(model_dir) + "/critic_agent.pt",
            map_location=self.device,
        )
        self.critic.load_state_dict(critic_state_dict)
        target_critic_state_dict = torch.load(
            str(model_dir) + "/target_critic_agent.pt",
            map_location=self.device,
        )
        self.target_critic.load_state_dict(
            target_critic_state_dict,
        )
        critic_state_dict2 = torch.load(
            str(model_dir) + "/critic_agent2.pt",
            map_location=self.device,
        )
        self.critic2.load_state_dict(critic_state_dict2)
        target_critic_state_dict2 = torch.load(
            str(model_dir)
            + "/target_critic_agent2.pt",
            map_location=self.device,
        )
        self.target_critic2.load_state_dict(
            target_critic_state_dict2,
        )

    def turn_on_grad(self):
        """开启 Critic 参数的梯度计算。"""
        for param in self.critic.parameters():
            param.requires_grad = True
        for param in self.critic2.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """关闭 Critic 参数的梯度计算。"""
        for param in self.critic.parameters():
            param.requires_grad = False
        for param in self.critic2.parameters():
            param.requires_grad = False
