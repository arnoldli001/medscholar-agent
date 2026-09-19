"""Agent 提示词集中管理（**兼容外壳**）。

提示词正文已带着版本号迁至 :mod:`medscholar.platform.prompt_library`，
本模块保留为**兼容外壳**：原有的常量名、函数名与 ``__all__`` 一个都不少，
值与迁移前**逐字相同**——这一点由 ``tests/test_prompts.py`` 的迁移完整性用例锁定。

## 为什么保留外壳而不是删掉

``medscholar.llm.prompts`` 是**公开契约**：``agent/graph.py``、``agent/critic.py``、
``agent/reader.py``、``agent/writer.py``、``retrieval.py``、``server/routes/search.py``
以及测试都在 import 它。一次性改掉所有调用点，会把"低风险的文本搬迁"变成
"高风险的大改造"，而只做重导出的外壳成本几乎为零。

## 为什么常量是重导出（``X as X``）而不是在这里再写一份

两份文本就是两份真相，早晚会有一份变旧，而且变旧的那份**不会有任何提示**。
重导出拿到的是同一个对象，不存在"哪份才是真的"的问题；
``X as X`` 这个写法本身也是在告诉静态检查与读者："这是有意重导出，不是没用到的 import"。

## 为什么函数留在这里

它们包含的是**流程**——取哪些片段、按什么顺序拼、什么时候整段留空；
所有成文的提示词句子都在注册表里。把流程也塞进注册表，等于在注册表里写代码，
那正是本项目拒绝引入模板引擎的原因（见 :mod:`medscholar.platform.prompts`）。
留在原处的还有三个 JSON 示例结构（``_PLAN_SCHEMA`` 等）：它们是**数据**，
由 ``json.dumps(..., ensure_ascii=False, indent=2)`` 渲染，放在这里才能保证
连缩进与转义都与迁移前一致。

新代码请直接使用 :mod:`medscholar.platform.prompts`（``prompt_text("writer.section", ...)``）——
那样才能拿到版本坐标，也才便于做 A/B。
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

# 以下两个名字**不在** ``__all__`` 里，但确实被外部引用：
# ``agent/writer.py`` 的 write_abstract 直接 import 了 ``_ROLE`` 来复用统一人设。
# 外壳漏掉私有名会让 import 直接失败（本仓库的 test_compat_shims.py 记录过同类事故），
# 所以这两个名字同样显式重导出，不能靠 import * 顺带带出来。

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
    """单章节撰写提示词。

    ``min_chars`` / ``max_chars`` 是**本节**的目标字数区间。以前只给一个
    "约 900 字"，模型就真的只写 900 字；给出区间并明确"写满下限"才能得到
    用户期望的篇幅。
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
    return prompt_text(
        "writer.section",
        topic=topic,
        section_title=section_title,
        bullet=bullet,
        digest=digest,
        length_clause=length_clause,
    )


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
    """单篇速读的用户提示词。

    块之间用空行分隔，与迁移前的 ``"\\n\\n".join(parts)`` 完全一致；
    每块文本本身来自注册表，这里只决定"放哪几块、什么顺序"。
    """
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
    """取数据里携带的**显式编号**（``__index__``），不可用时回退到顺序编号。

    为什么材料编号要允许"外部指定"：调用方给出的编号才是正文里 ``[n]`` 的含义。
    ``retrieval.build_context_digest`` 会把**筛过的子集**交给这里（Critic 按启发式
    打分挑出最相关的若干篇，真编号可能是 1/4/7/12），并在拿回模型点评后按**真编号**
    回查文献。如果这里按 1/2/3/4 重新编号，模型说的"第 2 篇"就会被挂到真编号 2 的
    那篇上——而那篇可能根本不在材料里：轻则点评丢失，重则**把差评挂到好文献头上**。

    回退而不是抛异常是刻意的：本函数在写作主路径上，一个编号字段脏了不该让整篇
    综述生成失败。``bool`` 也回退——它是 ``int`` 的子类，但 ``True`` 显然不是编号。
    """
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
    """把文献列表压成适合放进提示词的材料摘要。

    编号即综述正文里使用的引用序号，取值规则：

    1. 数据里带 ``__index__``（且是整数）时**优先用它**——调用方传进来的显式编号
       才是正文引用的含义。筛过的子集会有编号空洞（1/4/7/12），
       若在这里按顺序重新编号，材料里的 ``[n]`` 就与正文引用对不上了；
    2. 否则按 ``start_index + 顺序`` 递增编号（普通列表的默认行为，向后兼容）；
    3. ``__index__`` 缺失或不是整数时**安全回退**到顺序编号而不是抛异常——
       这个函数在写作主路径上，脏字段不该让整篇综述写不出来。

    这个函数**不**搬进注册表：它产出的是**每次运行都不同的材料数据块**，
    不是提示词文本。把数据渲染混进提示词库，会让"这段话有没有被改过"重新变得难判断。
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
        body = f"    摘要：{abstract}" if abstract else "    摘要：（无）"
        blocks.append(f"{head}\n{meta}\n{body}")
    return "\n\n".join(blocks)
