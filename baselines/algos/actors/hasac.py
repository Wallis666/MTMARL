"""HASAC 算法模块。"""

import torch

from baselines.algos.actors.off_policy_base import OffPolicyBase
from baselines.models.policy.squashed_gaussian_policy import (
    SquashedGaussianPolicy,
)
from baselines.models.policy.stochastic_mlp_policy import (
    StochasticMlpPolicy,
)
from baselines.utils.discrete import gumbel_softmax
from baselines.utils.env import check


class HASAC(OffPolicyBase):
    """HASAC 算法。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 HASAC 算法。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间。
            device: 用于张量运算的设备。
        """
        self.tpdv = dict(
            dtype=torch.float32, device=device,
        )
        self.polyak = args["polyak"]
        self.lr = args["lr"]
        self.device = device
        self.action_type = act_space.__class__.__name__

        if act_space.__class__.__name__ == "Box":
            self.actor = SquashedGaussianPolicy(
                args, obs_space, act_space, device,
            )
        else:
            self.actor = StochasticMlpPolicy(
                args, obs_space, act_space, device,
            )

        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.lr,
        )
        self.turn_off_grad()

    def get_actions(
        self,
        obs,
        available_actions=None,
        stochastic=True,
    ):
        """
        根据观测获取动作。

        参数:
            obs: Actor 的观测值。
            available_actions: 可用动作掩码。
            stochastic: 是否使用随机策略。

        返回:
            actions: Actor 采取的动作。
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions, _ = self.actor(
                obs,
                stochastic=stochastic,
                with_logprob=False,
            )
        else:
            actions = self.actor(
                obs, available_actions, stochastic,
            )
        return actions

    def get_actions_with_logprobs(
        self,
        obs,
        available_actions=None,
        stochastic=True,
    ):
        """
        获取动作及其对数概率。

        参数:
            obs: Actor 的观测值。
            available_actions: 可用动作掩码。
            stochastic: 是否使用随机策略。

        返回:
            actions: Actor 采取的动作。
            logp_actions: 动作的对数概率。
        """
        obs = check(obs).to(**self.tpdv)
        if self.action_type == "Box":
            actions, logp_actions = self.actor(
                obs,
                stochastic=stochastic,
                with_logprob=True,
            )
        elif self.action_type == "Discrete":
            logits = self.actor.get_logits(
                obs, available_actions,
            )
            # onehot 动作
            actions = gumbel_softmax(
                logits, hard=True, device=self.device,
            )
            logp_actions = torch.sum(
                actions * logits, dim=-1, keepdim=True,
            )
        elif self.action_type == "MultiDiscrete":
            logits = self.actor.get_logits(
                obs, available_actions,
            )
            actions = []
            logp_actions = []
            for logit in logits:
                # onehot 动作
                action = gumbel_softmax(
                    logit,
                    hard=True,
                    device=self.device,
                )
                logp_action = torch.sum(
                    action * logit,
                    dim=-1,
                    keepdim=True,
                )
                actions.append(action)
                logp_actions.append(logp_action)
            actions = torch.cat(actions, dim=-1)
            logp_actions = torch.cat(
                logp_actions, dim=-1,
            )
        return actions, logp_actions

    def save(
        self,
        save_dir,
        id,
    ):
        """保存 Actor。"""
        torch.save(
            self.actor.state_dict(),
            str(save_dir)
            + "/actor_agent"
            + str(id)
            + ".pt",
        )

    def restore(
        self,
        model_dir,
        id,
    ):
        """恢复 Actor。"""
        actor_state_dict = torch.load(
            str(model_dir)
            + "/actor_agent"
            + str(id)
            + ".pt",
            map_location=self.device,
        )
        self.actor.load_state_dict(actor_state_dict)
