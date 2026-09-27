"""检索式解析与跨数据源翻译：先解析成结构化查询，再按数据源能力翻译。

布尔能力分两档：pubmed/europepmc/arxiv 支持完整布尔表达式（arXiv 用 ANDNOT）；
semantic_scholar/openalex/crossref 只做相关度检索，拿全部核心词（同义词全带，排序更准）。
语法：空格/逗号=AND，| 或 OR=任选，- 或 NOT=排除，"引号"=精确短语；分号=OR（中文并列习惯）。
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Sequence

__all__ = [
    "ParsedQuery",
    "parse_query",
    "describe_query",
    "for_source",
    "SUPPORTED_SYNTAX_HELP",
    "BOOLEAN_SOURCES",
]

#: 支持完整布尔语法的数据源
BOOLEAN_SOURCES = frozenset({"pubmed", "europepmc", "arxiv"})

SUPPORTED_SYNTAX_HELP = (
    "空格或逗号 = AND（同时包含）｜ | 或 OR = 任选其一｜ - 或 NOT = 排除｜ \"引号\" = 精确短语"
)

# 用于切分：先按引号切，保留引号内容
_TOKEN_RE = re.compile(r'"[^"]*"|\S+')
_OR_WORDS = {"or", "或者", "｜"}
# AND 是显式连接符，必须识别为"分隔符"而非词：之前漏识会让 for_source 拼出
# "(rTMS AND AND ...)" 的双重 AND，Europe PMC 对此返回 hitCount=0（HTTP 200，
# 静默空结果；见回归测试 tests/test_query_alignment.py）。
_AND_WORDS = {"and", "且", "&"}
_NOT_WORDS = {"not", "-", "排除", "非"}
_SEPARATORS = re.compile(r"[,，]")


@dataclass(slots=True)
class ParsedQuery:
    """结构化的检索式。"""

    raw: str = ""
    #: 必须包含的词（AND）
    must: list[str] = field(default_factory=list)
    #: OR 组：每组内部"任选其一"，组与组之间是 AND
    any_groups: list[list[str]] = field(default_factory=list)
    #: 排除的词（NOT）
    exclude: list[str] = field(default_factory=list)
    #: 精确短语（视为 must，但检索时加引号）
    phrases: list[str] = field(default_factory=list)

    @property
    def is_simple(self) -> bool:
        """只有 AND 词、没有 OR/NOT/短语 —— 这种情况无需特殊处理。"""
        return not self.any_groups and not self.exclude and not self.phrases

    @property
    def is_empty(self) -> bool:
        return not (self.must or self.any_groups or self.phrases)

    def core_terms(self) -> list[str]:
        """给不支持布尔的数据源用的检索词。

        OR 组会把全部候选词都带上，而不是只取第一个：OpenAlex / Crossref
        这类是相关度检索，多给同义词能让排序更准；只取一个反而丢召回。
        """
        terms = [*self.phrases, *self.must]
        for group in self.any_groups:
            terms.extend(group)
        return _dedupe(terms)

    def core_text(self) -> str:
        return " ".join(self.core_terms())

    def to_dict(self) -> dict[str, Any]:
        return {
            "raw": self.raw,
            "must": list(self.must),
            "any_groups": [list(g) for g in self.any_groups],
            "exclude": list(self.exclude),
            "phrases": list(self.phrases),
            "is_simple": self.is_simple,
            "core_text": self.core_text(),
            "description": describe_query(self),
        }


def _strip_quotes(token: str) -> str:
    return token.strip().strip('"').strip()


def _tokenize(raw: str) -> list[str]:
    """切分词元：空格/逗号=AND 相邻，分号插入 | 标记（OR），引号整体保留不切。"""
    out: list[str] = []
    for chunk in _TOKEN_RE.findall(raw):
        if chunk.startswith('"'):
            out.append(chunk)
            continue
        parts = [p for p in re.split(r"[;；]", chunk) if p.strip()]
        for index, part in enumerate(parts):
            if index:
                out.append("|")
            out.extend(p for p in re.split(r"[,，]", part) if p.strip())
    return out


def parse_query(text: str) -> ParsedQuery:
    """把用户输入解析成 :class:`ParsedQuery`。

    >>> q = parse_query('rTMS 卒中后抑郁')
    >>> q.must
    ['rTMS', '卒中后抑郁']
    >>> q = parse_query('rTMS | 经颅磁刺激 -动物')
    >>> q.any_groups, q.exclude
    ([['rTMS', '经颅磁刺激']], ['动物'])
    >>> q = parse_query('"post-stroke depression" rTMS')
    >>> q.phrases, q.must
    (['post-stroke depression'], ['rTMS'])
    >>> q = parse_query('a | b | c')
    >>> q.any_groups
    [['a', 'b', 'c']]
    >>> q = parse_query('rTMS NOT 动物实验')
    >>> q.exclude
    ['动物实验']
    """
    raw = (text or "").strip()
    parsed = ParsedQuery(raw=raw)
    if not raw:
        return parsed

    segments = _tokenize(raw)

    pending_or = False
    pending_not = False
    # 记录"最近加入的词"及其所在容器，OR 需要把前一个词从容器里提出来合并
    last_term: str | None = None
    last_kind: str | None = None  # must | phrase | any

    for token in segments:
        token = token.strip()
        if not token:
            continue

        lowered = token.lower()
        if lowered in _OR_WORDS or token == "|":
            pending_or = True
            continue
        if lowered in _AND_WORDS:
            # 显式 AND：分隔符，不入词，也不与前词拼成新组；
            # 同时把 pending_or/pending_not 清零，避免与前一个 OR/NOT 标记复合。
            pending_or = pending_not = False
            continue
        if lowered in _NOT_WORDS:
            # 单独的 NOT 关键字：作用于下一个词
            pending_not = True
            continue
        if token.startswith("-") and len(token) > 1:
            value = _strip_quotes(token[1:])
            if value:
                parsed.exclude.append(value)
            pending_or = pending_not = False
            continue

        is_phrase = token.startswith('"') and token.endswith('"') and len(token) > 2
        value = _strip_quotes(token)
        if not value:
            continue

        if pending_not:
            parsed.exclude.append(value)
            pending_not = pending_or = False
            last_term, last_kind = None, None
            continue

        if pending_or:
            pending_or = False
            if last_kind == "any" and parsed.any_groups:
                parsed.any_groups[-1].append(value)
            elif last_kind in {"must", "phrase"} and last_term:
                # 把上一个词从原容器取出，与当前词组成 OR 组
                container = parsed.must if last_kind == "must" else parsed.phrases
                if last_term in container:
                    container.remove(last_term)
                if parsed.any_groups and last_term in parsed.any_groups[-1]:
                    parsed.any_groups[-1].append(value)
                else:
                    parsed.any_groups.append([last_term, value])
            else:
                parsed.any_groups.append([value])
            last_term, last_kind = value, "any"
            continue

        if is_phrase:
            parsed.phrases.append(value)
            last_term, last_kind = value, "phrase"
        else:
            parsed.must.append(value)
            last_term, last_kind = value, "must"

    # 去重保序
    parsed.must = _dedupe(parsed.must)
    parsed.phrases = _dedupe(parsed.phrases)
    parsed.exclude = _dedupe(parsed.exclude)
    parsed.any_groups = [_dedupe(g) for g in parsed.any_groups if g]
    # OR 组里只剩一项 → 退化为普通必须词，避免生成无意义的 (...)
    collapsed: list[list[str]] = []
    for group in parsed.any_groups:
        if len(group) == 1:
            parsed.must.append(group[0])
        else:
            collapsed.append(group)
    parsed.any_groups = collapsed
    parsed.must = _dedupe(parsed.must)
    return parsed


def _dedupe(items: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for item in items:
        key = item.lower()
        if key and key not in seen:
            seen.add(key)
            out.append(item)
    return out


def describe_query(parsed: ParsedQuery) -> str:
    """人类可读的解析结果，用于前端"将检索：…"回显。"""
    if parsed.is_empty:
        return "（空检索式）"
    parts: list[str] = []
    parts.extend(f'"{p}"' for p in parsed.phrases)
    parts.extend(parsed.must)
    for group in parsed.any_groups:
        parts.append("(" + " 或 ".join(group) + ")")
    if parsed.exclude:
        parts.append("排除 " + "、".join(parsed.exclude))
    return " 且 ".join(parts)


def _quote_if_needed(term: str) -> str:
    """包含空格或连字符的词在布尔表达式里要加引号。"""
    if re.search(r"[\s\-]", term):
        return f'"{term}"'
    return term


def for_source(query: str | ParsedQuery, source: str) -> str:
    """按数据源能力翻译检索式。

    >>> for_source("rTMS 卒中后抑郁", "pubmed")
    '(rTMS AND 卒中后抑郁)'
    >>> for_source("rTMS | 经颅磁刺激", "pubmed")
    '((rTMS OR 经颅磁刺激))'
    >>> for_source("rTMS | 经颅磁刺激 -动物", "openalex")
    'rTMS 经颅磁刺激'
    """
    parsed = query if isinstance(query, ParsedQuery) else parse_query(query)
    if parsed.is_empty:
        return ""

    if source not in BOOLEAN_SOURCES:
        # 相关度检索数据源：给全部核心词（OR 组同义词全带上，多给词排序更准、不丢召回）。
        # exclude 不在本地 FTS 里参与（_build_filters 无排除项），但 remote relevance
        # 检索也容易把 -xxx 当普通词参与匹配——先在文本里剔除负向词，剩下的再 core_text()。
        if parsed.exclude:
            bad = {x.lower() for x in parsed.exclude}
            terms = [t for t in parsed.core_terms() if t.lower() not in bad]
            return " ".join(terms) if terms else parsed.core_text()
        return parsed.core_text()

    units: list[str] = []
    for phrase in parsed.phrases:
        units.append(f'"{phrase}"' if source == "arxiv" else f'"{phrase}"')
    for term in parsed.must:
        units.append(_quote_if_needed(term))
    for group in parsed.any_groups:
        joined = " OR ".join(_quote_if_needed(t) for t in group)
        units.append(f"({joined})")

    expression = " AND ".join(units)
    if len(units) > 1:
        expression = f"({expression})"
    if parsed.exclude:
        negated = " OR ".join(_quote_if_needed(t) for t in parsed.exclude)
        operator = "ANDNOT" if source == "arxiv" else "NOT"
        expression = f"{expression} {operator} ({negated})" if len(parsed.exclude) > 1 else (
            f"{expression} {operator} {_quote_if_needed(parsed.exclude[0])}"
        )
    # 输出合法性自检：AND/OR/NOT 旁边紧邻同一种运算符 = 翻译器 bug。
    # 历史上的 bug：parse_query 不识别 AND → for_source 拼出 "(... AND AND ...)"，
    # Europe PMC 返回 hitCount=0（HTTP 200，静默空结果）。守在这里，
    # 一旦再次触发，立即报错而不是把坏表达式送出去。
    if expression:
        for double in ("AND AND", "OR OR", "NOT NOT", "(AND ", "(OR ", "(NOT ", " AND)"):
            if double in f" {expression} ":
                raise AssertionError(
                    f"翻译后的检索式含畸形序列 {double!r}：{expression!r}"
                    f"——parse_query 或 for_source 出错，请检查 tokens 解析。"
                )
        # 括号必须配平（每开括号必有对应闭括号）。
        if expression.count("(") != expression.count(")"):
            raise AssertionError(
                f"翻译后的检索式括号不配平：{expression!r}"
            )
    return expression
