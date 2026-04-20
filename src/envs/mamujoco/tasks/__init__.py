"""多任务环境注册表模块。"""

from src.envs.mamujoco.tasks.cheetah import HalfCheetahMultiTask
from src.envs.mamujoco.tasks.hopper import HopperMultiTask
from src.envs.mamujoco.tasks.humanoid import HumanoidMultiTask


# 以 scenario 名称为键，映射到对应的多任务环境类。
# 新增环境时只需在此注册即可。
ENV_REGISTRY = {
    "HalfCheetah": HalfCheetahMultiTask,
    "Hopper": HopperMultiTask,
    "Humanoid": HumanoidMultiTask,
}

# 预设智能体分割配置的默认参数。
# 键为可读的配置名称，值包含构建环境所需的完整参数。
ARGS_REGISTRY = {
    "HalfCheetah": {
        "agent_conf": "2x3",
        "agent_obsk": 0,
    },
    "Hopper": {
        "agent_conf": "3x1",
        "agent_obsk": 0,
    },
    "Humanoid": {
        "agent_conf": "9|8",
        "agent_obsk": 0,
    },
}
