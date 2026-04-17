"""Off-Policy 算法基类模块。"""

import torch

from baselines.utils.model import update_linear_schedule


class OffPolicyBase:
    """Off-Policy 算法基类。"""

    def __init__(
        self,
        args,
        obs_space,
        act_space,
        device=torch.device("cpu"),
    ):
        """
        初始化 Off-Policy 算法基类。

        参数:
            args: 算法参数字典。
            obs_space: 观测空间。
            act_space: 动作空间。
            device: 用于张量运算的设备。
        """
        pass

    def lr_decay(
        self,
        step,
        steps,
    ):
        """
        衰减 Actor 和 Critic 的学习率。

        参数:
            step: 当前训练步数。
            steps: 总训练步数。
        """
        update_linear_schedule(
            self.actor_optimizer, step, steps, self.lr,
        )

    def get_actions(
        self,
        obs,
        randomness,
    ):
        """获取动作。"""
        pass

    def get_target_actions(
        self,
        obs,
    ):
        """获取目标 Actor 的动作。"""
        pass

    def soft_update(self):
        """软更新目标 Actor。"""
        for param_target, param in zip(
            self.target_actor.parameters(),
            self.actor.parameters(),
        ):
            param_target.data.copy_(
                param_target.data * (1.0 - self.polyak)
                + param.data * self.polyak
            )

    def save(
        self,
        save_dir,
        id,
    ):
        """保存 Actor 和目标 Actor。"""
        torch.save(
            self.actor.state_dict(),
            str(save_dir)
            + "/actor_agent"
            + str(id)
            + ".pt",
        )
        torch.save(
            self.target_actor.state_dict(),
            str(save_dir)
            + "/target_actor_agent"
            + str(id)
            + ".pt",
        )

    def restore(
        self,
        model_dir,
        id,
    ):
        """恢复 Actor 和目标 Actor。"""
        actor_state_dict = torch.load(
            str(model_dir)
            + "/actor_agent"
            + str(id)
            + ".pt",
            map_location=self.device,
        )
        self.actor.load_state_dict(actor_state_dict)
        target_actor_state_dict = torch.load(
            str(model_dir)
            + "/target_actor_agent"
            + str(id)
            + ".pt",
            map_location=self.device,
        )
        self.target_actor.load_state_dict(
            target_actor_state_dict,
        )

    def turn_on_grad(self):
        """开启 Actor 参数的梯度计算。"""
        for p in self.actor.parameters():
            p.requires_grad = True

    def turn_off_grad(self):
        """关闭 Actor 参数的梯度计算。"""
        for p in self.actor.parameters():
            p.requires_grad = False
