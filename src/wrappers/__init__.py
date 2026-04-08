"""
向量化环境包装器子包。

提供面向多智能体多任务环境（如 ``MultiTaskMaMuJoCo``）的并行采样
接口，包含：

* :class:`ShareVecEnv`        ：共享观测向量化环境抽象基类。
* :class:`ShareDummyVecEnv`   ：单进程顺序版本，便于调试。
* :class:`ShareSubprocVecEnv` ：多进程并行版本，用于真正的训练加速。
* :class:`CloudpickleWrapper` ：跨进程序列化辅助类。
"""

from src.wrappers.base import ShareVecEnv
from src.wrappers.cloudpickle_wrapper import CloudpickleWrapper
from src.wrappers.dummy_vec_env import ShareDummyVecEnv
from src.wrappers.subproc_vec_env import ShareSubprocVecEnv

__all__ = [
    "CloudpickleWrapper",
    "ShareDummyVecEnv",
    "ShareSubprocVecEnv",
    "ShareVecEnv",
]
