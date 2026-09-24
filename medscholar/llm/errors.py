"""LLM 异常。独立成模块以断开依赖环：errors ← json_parsing ← client 单向依赖。"""

from __future__ import annotations

__all__ = ["LLMError"]


class LLMError(RuntimeError):
    """LLM 调用失败；面向用户的消息必须说清哪一步失败、下一步怎么做。"""
