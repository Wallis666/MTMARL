"""
跨进程序列化工具。

Python 内置 ``multiprocessing`` 默认使用 ``pickle`` 序列化要传给子进程
的对象，但 ``pickle`` 无法序列化 lambda、闭包以及部分动态创建的类。
``cloudpickle`` 可以处理这些情况；本模块提供的 :class:`CloudpickleWrapper`
通过重写 ``__getstate__`` / ``__setstate__``，让被包装对象在跨进程
传输时自动改用 ``cloudpickle`` 序列化。
"""

from __future__ import annotations

import pickle
from typing import Any

import cloudpickle


class CloudpickleWrapper:
    """
    将任意 Python 对象包装为可被 ``multiprocessing`` 安全序列化的形式。

    用法::

        wrapper = CloudpickleWrapper(my_env_factory)
        Process(target=worker, args=(pipe, wrapper)).start()
    """

    def __init__(
        self,
        wrapped_object: Any,
    ) -> None:
        """
        参数:
            wrapped_object: 任意需要被传入子进程的 Python 对象，常见
                场景是无参可调用的环境工厂函数。
        """
        self.wrapped_object = wrapped_object

    def __getstate__(self) -> bytes:
        """以 ``cloudpickle`` 序列化被包装对象。"""
        return cloudpickle.dumps(self.wrapped_object)

    def __setstate__(
        self,
        serialized: bytes,
    ) -> None:
        """从字节串反序列化恢复被包装对象。"""
        self.wrapped_object = pickle.loads(serialized)
