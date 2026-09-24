"""模型输出的 JSON 容错解析（纯函数、无 IO，密集测试在 tests/test_llm_json.py）。

小模型不保证纯 JSON：Markdown 围栏/前置解释文字、空白循环导致截断、
字符串内未转义引号、尾随逗号，均需容错。形状感知（expect="object"/"array"）必须保留：
残缺对象里第一个配平的 ``[...]`` 可能只是 pico.outcomes，不限定形状会把指标数组
当成整份计划，topic_zh/queries/outline 静默丢失。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import LLMError

__all__ = ["extract_json"]

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, *, expect: str = "any") -> Any:
    """尽力提取 JSON：直接解析 → 剥 Markdown 围栏 → 截取首个平衡块
    → 补全截断对象 → 修复尾随逗号/中文引号。expect 限定顶层形状（object/array）。"""
    if not text:
        raise LLMError("模型返回为空，无法解析 JSON")
    text = text.strip()

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(text):
        parsed: Any = None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            fixed = _repair(candidate)
            if fixed != candidate:
                try:
                    parsed = json.loads(fixed)
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if _matches(parsed, expect):
            return parsed

    want = {"object": "JSON 对象", "array": "JSON 数组"}.get(expect, "JSON")
    detail = f"（{last_error}）" if last_error else ""
    raise LLMError(f"无法从模型输出中解析出{want}{detail}：{text[:400]}")


def _matches(value: Any, expect: str) -> bool:
    """顶层形状是否符合调用方的期望。"""
    if expect == "object":
        return isinstance(value, dict)
    if expect == "array":
        return isinstance(value, list)
    return True


def _leading_opener(text: str) -> str | None:
    """文本自己声明的顶层形状：第一个非空白字符是 ``{`` 还是 ``[``。"""
    for ch in text:
        if ch in "{[":
            return ch
        if not ch.isspace():
            return None
    return None


def _close_truncated(fragment: str) -> str | None:
    """补全被截断的 JSON：退到最后一个完整值再补未闭合括号。
    小模型陷入空白循环烧完 token 时，前面已生成的字段仍然完好，值得捞回。"""
    stack: list[str] = []
    in_string = False
    escaped = False
    cut = -1  # 最后一个完整值的结束位置
    for index, ch in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                cut = index + 1
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            if not stack:
                return None  # 顶层已经闭合，说明不是截断
            cut = index + 1
        elif ch == "," and stack:
            cut = index
    if in_string or not stack or cut <= 0:
        return None
    # 回退到最后一个完整值，去掉悬空的逗号与半截成员
    repaired = fragment[:cut].rstrip().rstrip(",")
    return repaired + "".join(reversed(stack))


def _json_candidates(text: str) -> list[str]:
    bases = [text]
    fenced = _FENCE_RE.search(text)
    if fenced:
        bases.append(fenced.group(1).strip())

    # 只按文本自己声明的形状找块，避免残缺对象里第一个配平的 [...] 被误当答案（见模块说明）。
    lead = _leading_opener(text)
    if lead == "{":
        pairs = [("{", "}")]
    elif lead == "[":
        pairs = [("[", "]")]
    else:
        pairs = [("{", "}"), ("[", "]")]

    candidates: list[str] = list(bases)
    for base in bases:
        for opener, closer in pairs:
            block = _balanced_block(base, opener, closer)
            if block:
                candidates.append(block)
                continue  # 已配平，无需再尝试补全
            start = base.find(opener)
            if start >= 0:
                closed = _close_truncated(base[start:])
                if closed:
                    candidates.append(closed)

    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _balanced_block(text: str, opener: str, closer: str) -> str | None:
    """截取第一个括号配平的块（跳过字符串字面量内的括号）。"""
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair(text: str) -> str:
    """修复常见 JSON 瑕疵：尾随逗号、中文引号。"""
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    repaired = repaired.replace("“", '"').replace("”", '"')
    return repaired
