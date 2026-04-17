"""离散动作空间 Q Critic 模块。"""

from copy import deepcopy

import torch

from baselines.models.value.dueling_q_net import (
    DuelingQNet,
)
from baselines.utils.env import check
from baselines.utils.model import update_linear_schedule


class DiscreteQCritic:
    """学习 Q 函数的 Critic，适用于离散动作空间。"""

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
        初始化 DiscreteQCritic。

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
        self.tpdv_a = dict(
            dtype=torch.int64, device=device,
        )
        self.act_space = act_space
        self.num_agents = num_agents
        self.state_type = state_type
        self.process_action_spaces(act_space)
        self.critic = DuelingQNet(
            args,
            share_obs_space,
            self.joint_action_dim,
            device,
        )
        self.target_critic = deepcopy(self.critic)
        for param in self.target_critic.parameters():
            param.requires_grad = False
        self.gamma = args["gamma"]
        self.critic_lr = args["critic_lr"]
        self.polyak = args["polyak"]
        self.use_proper_time_limits = (
            args["use_proper_time_limits"]
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(),
            lr=self.critic_lr,
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

    def get_values(
        self,
        share_obs,
        actions,
    ):
        """获取给定观测和动作的 Q 值。"""
        share_obs = check(share_obs).to(**self.tpdv)
        actions = check(actions).to(**self.tpdv_a)
        joint_action = self.indiv_to_joint(actions)
        return torch.gather(
            self.critic(share_obs), 1, joint_action,
        )

    def train_values(
        self,
        share_obs,
        actions,
    ):
        """
        训练 Critic 的辅助函数。

        参数:
            share_obs: 共享观测，形状 (batch_size, dim)。
            actions: 动作，形状 (n_agents, batch_size, dim)。
        """
        share_obs = check(share_obs).to(**self.tpdv)
        all_values = self.critic(share_obs)
        actions = deepcopy(actions)

        def update_actions(agent_id):
            joint_idx = self.get_joint_idx(
                actions, agent_id,
            )
            values = torch.gather(
                all_values, 1, joint_idx,
            )
            action = torch.argmax(
                values, dim=-1, keepdim=True,
            )
            actions[agent_id] = action

        def get_values():
            joint_action = self.indiv_to_joint(actions)
            return torch.gather(
                all_values, 1, joint_action,
            )

        return update_actions, get_values

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
        actions = check(actions).to(**self.tpdv_a)
        action = self.indiv_to_joint(actions).to(
            **self.tpdv_a,
        )
        reward = check(reward).to(**self.tpdv)
        done = check(done).to(**self.tpdv)
        term = check(term).to(**self.tpdv)
        gamma = check(gamma).to(**self.tpdv)
        next_share_obs = check(next_share_obs).to(
            **self.tpdv,
        )
        next_action = self.indiv_to_joint(
            next_actions,
        ).to(**self.tpdv_a)
        next_q_values = torch.gather(
            self.target_critic(next_share_obs),
            1,
            next_action,
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
        critic_loss = torch.mean(
            torch.nn.functional.mse_loss(
                torch.gather(
                    self.critic(share_obs), 1, action,
                ),
                q_targets,
            )
        )
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

    def process_action_spaces(
        self,
        action_spaces,
    ):
        """处理动作空间，计算联合动作维度。"""
        self.action_dims = []
        self.joint_action_dim = 1
        for space in action_spaces:
            self.action_dims.append(space.n)
            self.joint_action_dim *= space.n

    def joint_to_indiv(
        self,
        orig_action,
    ):
        """
        将联合动作转换为各智能体的独立动作。

        参数:
            orig_action: 联合动作。

        返回:
            actions: 各智能体的独立动作列表。
        """
        action = deepcopy(orig_action)
        actions = []
        for dim in self.action_dims:
            actions.append(action % dim)
            action = torch.div(
                action, dim, rounding_mode="floor",
            )
        return actions

    def indiv_to_joint(
        self,
        orig_actions,
    ):
        """
        将各智能体的独立动作转换为联合动作。

        参数:
            orig_actions: 各智能体的独立动作列表。

        返回:
            action: 联合动作。
        """
        actions = deepcopy(orig_actions)
        action = torch.zeros_like(actions[0])
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            action += accum_dim * actions[i]
            accum_dim *= dim
        return action

    def get_joint_idx(
        self,
        actions,
        agent_id,
    ):
        """
        获取某个智能体可选动作对应的联合动作索引。

        参数:
            actions: 各智能体的独立动作列表。
            agent_id: 智能体编号。

        返回:
            joint_idx: 联合动作索引张量。
        """
        batch_size = actions[0].shape[0]
        joint_idx = torch.zeros(
            (
                batch_size,
                self.action_dims[agent_id],
            ),
        ).to(**self.tpdv_a)
        accum_dim = 1
        for i, dim in enumerate(self.action_dims):
            if i == agent_id:
                for j in range(
                    self.action_dims[agent_id],
                ):
                    joint_idx[:, j] += accum_dim * j
            else:
                joint_idx += accum_dim * actions[i]
            accum_dim *= dim
        return joint_idx

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

    def turn_on_grad(self):
        """开启 Critic 参数的梯度计算。"""
        for param in self.critic.parameters():
            param.requires_grad = True

    def turn_off_grad(self):
        """关闭 Critic 参数的梯度计算。"""
        for param in self.critic.parameters():
            param.requires_grad = False
