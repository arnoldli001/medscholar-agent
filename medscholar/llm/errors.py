"""LLM 相关异常。

单独一个模块是为了断开环：``client`` 需要抛 :class:`LLMError`，
``json_parsing`` 解析失败时也要抛同一个异常。如果异常定义在 ``client`` 里，
``json_parsing`` 就得反向 import ``client``，两个模块立刻互相依赖。
把异常下沉到最底层的独立模块后，依赖方向变成单向：
``errors`` ← ``json_parsing`` ← ``client``。
"""

from __future__ import annotations

__all__ = ["LLMError"]


class LLMError(RuntimeError):
    """LLM 调用失败。

    约定：面向用户的消息必须说清"哪一步失败、下一步该做什么"。
    早期版本把云端返回的英文 400 原样抛给用户，界面只有一行
    ``HTTP 400``，用户完全不知道是模型没拉取、Key 没配、还是上下文超了。
    """
