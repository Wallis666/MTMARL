"""连续动作空间 Q 网络模块。"""

import torch
import torch.nn as nn

from src.models.base.mlp import mlp
from src.utils.env import get_shape_from_obs_space


def get_combined_dim(
    cent_obs_feature_dim,
    act_spaces,
):
    """
    获取中心化观测和各智能体动作的合并维度。

    参数:
        cent_obs_feature_dim: 中心化观测特征维度。
        act_spaces: 各智能体的动作空间列表。

    返回:
        combined_dim: 合并后的维度。
    """
    combined_dim = cent_obs_feature_dim
    for space in act_spaces:
        if space.__class__.__name__ == "Box":
            combined_dim += space.shape[0]
        elif space.__class__.__name__ == "Discrete":
            combined_dim += space.n
        else:
            action_dims = space.nvec
            for action_dim in action_dims:
                combined_dim += action_dim
    return combined_dim


class ContinuousQNet(nn.Module):
    """Q 网络，适用于连续和离散动作空间。"""

    def __init__(
        self,
        args,
        cent_obs_space,
        act_spaces,
        device=torch.device("cpu"),
    ):
        """
        初始化 ContinuousQNet 模型。

        参数:
            args: 包含模型相关信息的参数字典。
            cent_obs_space: 中心化观测空间。
            act_spaces: 各智能体的动作空间列表。
            device: 运行设备。
        """
        super().__init__()
        hidden_sizes = args["hidden_sizes"]
        cent_obs_shape = get_shape_from_obs_space(
            cent_obs_space,
        )
        num_bins = args.get("num_bins", 101)
        dropout = args.get("dropout", 0.01)
        self.mlp = mlp(
            in_dim=get_combined_dim(
                cent_obs_shape[0], act_spaces,
            ),
            mlp_dims=hidden_sizes,
            out_dim=num_bins,
            dropout=dropout,
        )
        self.to(device)

    def forward(
        self,
        cent_obs,
        actions,
    ):
        """
        前向传播，计算 Q 值。

        参数:
            cent_obs: 中心化观测。
            actions: 动作。

        返回:
            q_values: Q 值预测。
        """
        concat_x = torch.cat(
            [cent_obs, actions], dim=-1,
        )
        q_values = self.mlp(concat_x)
        return q_values
