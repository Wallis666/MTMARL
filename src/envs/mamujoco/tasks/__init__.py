"""多任务环境注册表模块。"""

from src.envs.mamujoco.tasks.cheetah import HalfCheetahMultiTask
from src.envs.mamujoco.tasks.hopper import HopperMultiTask
# from src.envs.mamujoco.tasks.humanoid import HumanoidMultiTask
from src.envs.mamujoco.tasks.humanoid_standup import HumanoidStandupMultiTask
from src.envs.mamujoco.tasks.reacher import ReacherMultiTask
# from src.envs.mamujoco.tasks.swimmer import SwimmerMultiTask
from src.envs.mamujoco.tasks.walker import Walker2dMultiTask


# 以 scenario 名称为键，映射到对应的多任务环境类。
# 新增环境时只需在此注册即可。
ENV_REGISTRY = {
    "HalfCheetah": HalfCheetahMultiTask,
    "Hopper": HopperMultiTask,
    # "Humanoid": HumanoidMultiTask,
    "HumanoidStandup": HumanoidStandupMultiTask,
    "Walker2d": Walker2dMultiTask,
    # "Swimmer": SwimmerMultiTask,
    "Reacher": ReacherMultiTask,
}

# 预设智能体分割配置的默认参数。
# 键为可读的配置名称，值包含构建环境所需的完整参数。
ARGS_REGISTRY = {
    "HalfCheetah": {
        "agent_conf": "2x3",
        "agent_obsk": 1,
    },
    "Hopper": {
        "agent_conf": "3x1",
        "agent_obsk": 1,
    },
    "Humanoid": {
        "agent_conf": "9|8",
        "agent_obsk": 1,
    },
    "HumanoidStandup": {
        "agent_conf": "9|8",
        "agent_obsk": 1,
    },
    "Walker2d": {
        "agent_conf": "2x3",
        "agent_obsk": 1,
    },
    "Swimmer": {
        "agent_conf": "2x1",
        "agent_obsk": 1,
    },
    "Reacher": {
        "agent_conf": "2x1",
        "agent_obsk": 1,
    },
}
