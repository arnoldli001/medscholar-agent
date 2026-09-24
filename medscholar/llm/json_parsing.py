"""模型输出的 JSON 容错解析（纯函数，无 IO、无状态）。

从 ``llm/client.py`` 拆出来的原因：这部分是纯算法（剥离围栏、截取平衡括号、
补全截断、修复尾随逗号），与"怎么发请求、怎么记账"无关，
却是整个 LLM 层里最需要密集测试的一段。拆开后可以被单独测
（``tests/test_llm_json.py``），``client.py`` 也回到 500 行以内。

为什么不能直接 ``json.loads``：本地小模型几乎不会严格输出纯 JSON。
实测 qwen3:8b 至少四种坏法：

1. 外面包一层 ```` ```json ```` 围栏，或前面带一句"好的，这是结果："；
2. 生成到一半陷入空白循环，把 token 预算烧完，JSON 被截断；
3. 字符串里出现未转义的引号（检索式里最常见）；
4. 尾部多一个逗号。

形状感知为什么关键：``expect="object"`` 会只接受顶层对象。这不是洁癖：
实测残缺对象里第一个配平的 ``[...]`` 恰好是 ``pico.outcomes``；
不限定形状时，就会把"结局指标数组"当成整份计划返回，
``topic_zh`` / ``queries`` / ``outline`` 全部静默丢失，
表现为"规划莫名其妙只出来一个结局列表"。
"""

from __future__ import annotations

import json
import re
from typing import Any

from .errors import LLMError

__all__ = ["extract_json"]

_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, *, expect: str = "any") -> Any:
    """从模型输出中尽力提取 JSON。

    依次尝试：直接解析 → 剥离 Markdown 围栏 → 截取首个平衡的 ``{...}`` 或 ``[...]``
    → 补全被截断的对象 → 修复尾随逗号 / 中文引号后重试。

    ``expect`` 为 ``"object"`` 或 ``"array"`` 时只接受该形状的顶层结果。
    这一点很关键：实测 qwen3:8b 生成 ``queries`` 时写坏过 JSON，而残缺对象里
    第一个配平的 ``[...]`` 恰好是 ``pico.outcomes``；不限定形状就会把
    结局指标数组当成整份计划返回，``topic_zh`` / ``queries`` / ``outline`` 全部丢失。
    """
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
    """补全被截断的 JSON：退到最后一个完整的值，再补上未闭合的括号。

    本地小模型偶尔会在生成到一半时陷入空白循环，把 token 预算烧完（实测
    qwen3:8b 在 ``"queries"`` 里输出 ``"query": "("`` 之后就只剩换行）。
    这时整个对象虽然不合法，但前面已经生成好的 ``topic_zh`` / ``pico``
    都是完好的，值得捞回来。
    """
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

    # 只按文本自己声明的形状找块：以 `{` 开头就只认对象。否则残缺对象里第一个
    # 配平的 `[...]` 会被当成答案（见 extract_json 的说明）。
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
