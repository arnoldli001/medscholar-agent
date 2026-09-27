"""Agent 提示词的兼容外壳：提示词正文已带版本号迁至 platform.prompt_library，

此处用 ``X as X`` 显式重导出常量（同一对象，不存在两份真相），并保留组装函数
（取哪些片段、拼接顺序是流程不是文案）与三个 JSON 示例结构（保证渲染缩进/转义一致）。
迁移完整性由 tests/test_prompts.py 锁定；新代码请直接用 platform.prompts.prompt_text。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from medscholar.platform.prompt_library import (
    ASK_SYSTEM as ASK_SYSTEM,
    CHAT_SYSTEM as CHAT_SYSTEM,
    CRITIQUE_SYSTEM as CRITIQUE_SYSTEM,
    OUTLINE_SYSTEM as OUTLINE_SYSTEM,
    PLAN_SYSTEM as PLAN_SYSTEM,
    REFLECT_SYSTEM as REFLECT_SYSTEM,
    SECTION_SYSTEM as SECTION_SYSTEM,
    SUMMARY_SYSTEM as SUMMARY_SYSTEM,
    TRANSLATE_SYSTEM as TRANSLATE_SYSTEM,
)
from medscholar.platform.prompt_library import HARD_RULES as _HARD_RULES  # noqa: F401
from medscholar.platform.prompt_library import ROLE as _ROLE  # noqa: F401
from medscholar.platform.prompts import prompt_text

__all__ = [
    "PLAN_SYSTEM",
    "plan_user",
    "TRANSLATE_SYSTEM",
    "translate_user",
    "OUTLINE_SYSTEM",
    "outline_user",
    "SECTION_SYSTEM",
    "section_user",
    "CRITIQUE_SYSTEM",
    "critique_user",
    "REFLECT_SYSTEM",
    "reflect_user",
    "SUMMARY_SYSTEM",
    "summary_user",
    "CHAT_SYSTEM",
    "chat_user",
    "digest_papers",
]

# _ROLE/_HARD_RULES 不在 __all__ 但被外部（agent/writer.py）直接 import，必须显式重导出。

_PLAN_SCHEMA = {
    "topic_zh": "课题的中文规范表述",
    "topic_en": "课题的英文规范表述（用于 PubMed 等英文库）",
    "pico": {
        "population": "研究对象",
        "intervention": "干预措施",
        "comparator": "对照",
        "outcomes": ["结局指标"],
    },
    "queries": [
        {
            "query": "检索式（英文库用英文，中文库用中文）",
            "sources": ["pubmed", "europepmc", "openalex", "crossref"],
            "rationale": "为什么这样检索",
        }
    ],
    "mesh_terms": ["相关 MeSH 主题词"],
    "year_from": 2015,
    "key_questions": ["需要回答的关键问题"],
    "outline": [{"title": "章节标题", "points": ["该章节要写的要点"]}],
}


def plan_user(topic: str, *, extra: str = "", offline: bool = False) -> str:
    """检索规划的用户提示词（正文在注册表 ``plan.user`` 及其片段 key 里）。"""
    note = prompt_text("plan.user.note_offline" if offline else "plan.user.note_online")
    extra_note = prompt_text("plan.user.extra_note", extra=extra) if extra else ""
    return prompt_text(
        "plan.user",
        topic=topic,
        extra_note=extra_note,
        note=note,
        schema=json.dumps(_PLAN_SCHEMA, ensure_ascii=False, indent=2),
    )


def translate_user(text: str) -> str:
    """术语翻译的用户提示词。"""
    return prompt_text("translate.user", text=text)


def outline_user(topic: str, digest: str, *, section_count: int = 5) -> str:
    """大纲规划的用户提示词。"""
    return prompt_text("outline.user", topic=topic, digest=digest, section_count=section_count)


def section_user(
    topic: str,
    section_title: str,
    points: Sequence[str],
    digest: str,
    *,
    max_chars: int = 1200,
    min_chars: int = 0,
    style: str = "综述正文",
) -> str:
    """单章节撰写提示词。字数给区间而非单一目标值，并明确写满下限。

    P2-8 修复：把 ``digest``（5 章节共用）放到 user.content 最前面，把
    ``topic/section_title/points`` 放到 digest 之后。这样在云端（DeepSeek / OpenAI）
    自动前缀缓存下，digest 作为稳定前缀被命中，后 4 个章节的输入 token 成本降 90%。

    重要：digest 必须是**字节级稳定**的（不可在前缀里塞时间戳 / 随机内容）。
    build_context_digest 已确保这一点（按论文引用编号升序 + max_abstract 固定）。
    """
    if points:
        bullet = "\n".join(f"- {p}" for p in points)
    else:
        bullet = prompt_text("writer.section.no_points")
    if min_chars and min_chars > 0:
        length_clause = prompt_text(
            "writer.section.length_with_min",
            style=style,
            min_chars=min_chars,
            max_chars=max_chars,
        )
    else:
        length_clause = prompt_text(
            "writer.section.length_plain", style=style, max_chars=max_chars
        )
    # 注意：原 writer.section 模板用占位符拼装。为了缓存命中，**不要**再用模板的
    # 散落占位符——直接拼装稳定前缀 (digest) + 章节级变化部分。
    parts: list[str] = []
    if digest:
        parts.append(digest)
    parts.append(f"研究课题：{topic}")
    parts.append(f"章节：{section_title}")
    if bullet:
        parts.append(f"本章要点：\n{bullet}")
    parts.append(length_clause)
    parts.append(
        "请基于上述材料撰写本章节正文：直接输出 Markdown，"
        "引用使用 `[n]` 形式，不输出 JSON、解释或标题层级之外的元信息。"
    )
    return "\n\n".join(parts)


_CRITIQUE_SCHEMA = {
    "assessments": [
        {
            "id": "材料中的文献编号（整数）",
            "relevance": "0~10 的相关性评分",
            "quality": "0~10 的方法学质量评分",
            "evidence_level": "RCT / Meta分析 / 队列 / 病例对照 / 综述 / 动物实验 / 其他",
            "key_finding": "一句话核心发现（不超过 40 字）",
            "limitation": "主要局限（不超过 30 字）",
            "use_in_review": "是否值得写入综述（是/否）",
        }
    ],
    "overall": {
        "evidence_quality": "整体证据质量（高/中/低/极低）",
        "gaps": ["研究空白"],
        "suggestions": ["对综述写作的建议"],
    },
}


def critique_user(topic: str, digest: str, count: int) -> str:
    """文献批判的用户提示词。"""
    return prompt_text(
        "critique.user",
        topic=topic,
        digest=digest,
        count=count,
        schema=json.dumps(_CRITIQUE_SCHEMA, ensure_ascii=False, indent=2),
    )


_REFLECT_SCHEMA = {
    "verdict": "pass / revise",
    "score": "0~10 的整体质量分",
    "issues": [
        {
            "severity": "high / medium / low",
            "type": "越界引用 / 无据数据 / 逻辑问题 / 遗漏文献 / 表述问题",
            "detail": "问题描述",
            "suggestion": "修改建议",
        }
    ],
    "strengths": ["做得好的地方"],
}


def reflect_user(topic: str, draft: str, valid_ids: Sequence[int], digest: str) -> str:
    """自我审查的用户提示词。"""
    ids = ", ".join(str(i) for i in valid_ids) or prompt_text("reflect.user.no_ids")
    return prompt_text(
        "reflect.user",
        topic=topic,
        ids=ids,
        draft=draft,
        digest=digest,
        schema=json.dumps(_REFLECT_SCHEMA, ensure_ascii=False, indent=2),
    )


def summary_user(paper_text: str, *, topic: str = "", focus: str = "") -> str:
    """单篇速读用户提示词；块间空行分隔（与迁移前 "\\n\\n".join 一致），本函数只决定块与顺序。"""
    parts = []
    if topic:
        parts.append(prompt_text("summary.user.topic", topic=topic))
    if focus:
        parts.append(prompt_text("summary.user.focus", focus=focus))
    parts.append(prompt_text("summary.user.body", paper_text=paper_text))
    parts.append(prompt_text("summary.user.tail"))
    return "\n\n".join(parts)


def chat_user(
    question: str,
    *,
    context: str = "",
    local_hits: str = "",
) -> str:
    """对话的用户提示词（上下文与本地检索命中都是可选的）。"""
    blocks = [prompt_text("chat.user.question", question=question)]
    if context:
        blocks.append(prompt_text("chat.user.context", context=context))
    if local_hits:
        blocks.append(prompt_text("chat.user.local_hits", local_hits=local_hits))
    blocks.append(prompt_text("chat.user.tail"))
    return "\n\n".join(blocks)


def ask_user(question: str, digest: str, *, paper_count: int = 0) -> str:
    """知识库问答的提示词。"""
    if digest:
        materials = prompt_text(
            "ask.materials.digest", paper_count=paper_count, digest=digest
        )
    else:
        materials = prompt_text("ask.materials.empty")
    return prompt_text("ask.user", question=question, materials=materials)


def _numbered_index(paper: Mapping[str, Any], fallback: int) -> int:
    """优先取数据携带的显式编号 __index__（正文 [n] 的含义由调用方指定，
    筛选子集可能有编号空洞如 1/4/7/12，重编号会把模型点评挂错文献）；
    缺失/非整数（含 bool）回退顺序编号而非抛异常，写作主路径不能被脏字段阻断。"""
    raw = paper.get("__index__")
    if isinstance(raw, bool) or not isinstance(raw, int):
        return fallback
    return raw


def digest_papers(
    papers: Sequence[Mapping[str, Any]],
    *,
    start_index: int = 1,
    max_abstract: int = 900,
    max_papers: int = 25,
) -> str:
    """把文献列表压成提示词用的材料摘要。编号即正文引用序号：优先用显式 __index__
    （筛选子集有编号空洞，重编号会让材料 [n] 与正文引用错位），否则按 start_index
    顺序递增，脏字段安全回退。本函数产出运行时数据块，故不放进提示词注册表。

    P0-3 修复：若 ``paper['__fulltext_excerpt__']`` 存在，会在摘要后追加一段
    "全文片段（仅前 N 字）"。这段内容**是高危注入面**——_guard_materials 已把全文
    纳入 ``detect_injection`` 扫描。
    """
    blocks: list[str] = []
    for offset, paper in enumerate(papers[:max_papers]):
        index = _numbered_index(paper, start_index + offset)
        title = str(paper.get("title") or "").strip()
        abstract = str(paper.get("abstract") or "").strip()
        if len(abstract) > max_abstract:
            abstract = abstract[:max_abstract] + "…"
        year = paper.get("pub_year") or "n.d."
        journal = str(paper.get("journal") or "").strip()
        authors = paper.get("authors") or []
        if isinstance(authors, str):
            authors = [authors]
        first_author = authors[0] if authors else "佚名"
        cited = paper.get("cited_by_count") or 0
        oa = "开放获取" if paper.get("is_open_access") else "非开放获取"
        ptype = str(paper.get("publication_type") or "").strip()

        head = f"[{index}] {title}"
        meta = f"    {first_author} 等 | {journal} | {year} | 被引 {cited} | {oa}"
        if ptype:
            meta += f" | {ptype}"
        body_lines: list[str] = []
        if abstract:
            body_lines.append(f"    摘要：{abstract}")
        else:
            body_lines.append("    摘要：（无）")
        # P0-3：拼接全文片段（由 build_context_digest 在 __fulltext_excerpt__ 注入）
        ft_excerpt = paper.get("__fulltext_excerpt__")
        if isinstance(ft_excerpt, str) and ft_excerpt.strip():
            body_lines.append(f"    全文片段：{ft_excerpt.strip()}")
        blocks.append(f"{head}\n{meta}\n" + "\n".join(body_lines))
    return "\n\n".join(blocks)
