"""Writer Agent：综述与摘要生成（需求 3.1）。

基于评估后的文献材料流式生成带引用的综述正文、章节、摘要与结论。

引用约束：正文里的 ``[n]`` 只能用
:class:`~medscholar.agent.state.AgentState` 的 ``citation_map`` 里存在的
编号；成稿后由 :class:`~medscholar.agent.formatter.FormatterAgent`
统一校验并剔除越界引用，避免幻觉编号留在成稿里。
"""

from __future__ import annotations

import logging
import re
from typing import Any, Sequence

from ..config import AppConfig, get_config
from ..constants import (
    BODY_TRUNCATE_SUMMARY,
    CHARS_PER_TOKEN,
    CITATION_RANGE_MAX_SPAN,
    DEFAULT_SECTION_COUNT,
    DIGEST_MAX_ABSTRACT_OUTLINE,
    LLM_MAX_TOKENS_ABSTRACT,
    LLM_MAX_TOKENS_OUTLINE,
    LLM_MAX_TOKENS_SUMMARY,
    LLM_TEMPERATURE_ABSTRACT,
    LLM_TEMPERATURE_OUTLINE,
    LLM_TEMPERATURE_SECTION,
    LLM_TEMPERATURE_SUMMARY,
    MAX_TOKEN_CAP,
    MIN_BUDGET,
    MIN_HEADROOM,
    MIN_TOKEN_CAP,
    SECTION_CHARS_GAP,
    SECTION_MIN_CHARS,
    TOKEN_RESERVE,
    TOKEN_SAFETY_MARGIN,
)
from ..llm.client import LLMError, get_llm
from ..llm.prompts import (
    OUTLINE_SYSTEM,
    SECTION_SYSTEM,
    outline_user,
    section_user,
)
from ..models import Paper
from ..retrieval import build_context_digest
from ..textutil import estimate_tokens
from .scout import Emitter, emit_event
from .state import PlanSection, ResearchPlan

logger = logging.getLogger(__name__)

__all__ = ["WriterAgent", "DEFAULT_OUTLINE"]

#: 材料块按"论文数上限 / 单篇摘要字数"逐档收缩，直到给正文留出足够上下文。
#: num_ctx 是提示词与输出的共享预算（本地 8 GB 显存下只能开到 8192），
#: 材料塞满就会把正文挤没。第一档的极大值表示不限篇数，先试最丰富的材料。
_DIGEST_LADDER: tuple[tuple[int, int], ...] = (
    (1_000_000, 800),
    (1_000_000, 500),
    (18, 400),
    (12, 300),
    (8, 200),
)


def _fit_digest(
    entries: Sequence[tuple[int, Paper]],
    *,
    num_ctx: int,
    want_output_tokens: int,
    system_prompt: str,
) -> str:
    """在上下文预算内构造材料块：优先保正文篇幅，其次保材料丰富度。"""
    budget = num_ctx - want_output_tokens - estimate_tokens(system_prompt) - TOKEN_RESERVE
    budget = max(MIN_BUDGET, budget)
    digest = ""
    for papers_cap, abstract in _DIGEST_LADDER:
        subset = list(entries[:papers_cap])
        digest = build_context_digest(subset, max_abstract=abstract)
        if estimate_tokens(digest) <= budget:
            break
    return digest

#: Plan 未给出大纲时的兜底结构
DEFAULT_OUTLINE: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("1 引言", ("研究背景与疾病负担", "本综述要回答的问题")),
    ("2 方法", ("检索策略与数据来源", "纳入与排除标准")),
    ("3 主要研究进展", ("干预方案与参数", "疗效证据", "作用机制")),
    ("4 讨论", ("证据一致性", "现有研究的局限")),
    ("5 结论与展望", ("主要结论", "后续研究方向")),
)


class WriterAgent:
    """写作智能体。"""

    def __init__(self, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    # ---------------------------------------------------------------- 大纲
    async def refine_outline(
        self,
        topic: str,
        entries: Sequence[tuple[int, Paper]],
        *,
        fallback: Sequence[PlanSection] = (),
        emit: Emitter | None = None,
    ) -> list[PlanSection]:
        """基于实际检索到的材料细化大纲；失败时沿用 Plan 的大纲或默认结构。"""
        if not entries:
            return list(fallback) or _default_outline()

        digest = build_context_digest(
            entries[:20], max_abstract=DIGEST_MAX_ABSTRACT_OUTLINE
        )
        try:
            client = get_llm(self.config)
            await client.start()
            payload = await client.chat_json(
                [
                    {
                        "role": "user",
                        "content": outline_user(
                            topic, digest, section_count=len(fallback) or DEFAULT_SECTION_COUNT
                        ),
                    }
                ],
                system=OUTLINE_SYSTEM,
                temperature=LLM_TEMPERATURE_OUTLINE,
                max_tokens=LLM_MAX_TOKENS_OUTLINE,
            )
            sections = _parse_outline(payload)
            if sections:
                await emit_event(emit, "status", message=f"大纲已细化：{len(sections)} 个章节")
                return sections
        except LLMError as exc:
            logger.info("大纲细化失败，沿用 Plan 大纲：%s", exc)
        except Exception as exc:  # pragma: no cover
            logger.warning("大纲细化异常：%s", exc)

        return list(fallback) or _default_outline()

    # ---------------------------------------------------------------- 综述
    async def write_review(
        self,
        topic: str,
        plan: ResearchPlan | None,
        entries: Sequence[tuple[int, Paper]],
        outline: Sequence[PlanSection],
        *,
        emit: Emitter | None = None,
        on_token: Any = None,
        style: str = "综述正文",
        max_chars: int = 900,
        min_chars: int = 0,
        total_min_chars: int = 0,
        total_max_chars: int = 0,
    ) -> str:
        """逐章节撰写综述，返回完整 Markdown 草稿。

        Args:
            on_token: ``async def on_token(text: str)`` 流式回调。
            min_chars / max_chars: 单节目标字数（未给 total_* 时生效）。
            total_min_chars / total_max_chars: 整篇正文的目标字数区间，
                给出后按章节数均分到每一节。
        """
        if not entries:
            return (
                f"# {topic}\n\n"
                "> 未检索到可用文献，无法生成综述。请调整检索式或数据源后重试。\n"
            )

        section_count = max(1, len(outline))
        if total_max_chars and total_max_chars > 0:
            total_min = total_min_chars if total_min_chars > 0 else int(total_max_chars * 0.6)
            total_min = min(total_min, total_max_chars)
            min_chars = max(SECTION_MIN_CHARS, total_min // section_count)
            max_chars = max(min_chars + SECTION_CHARS_GAP, total_max_chars // section_count)

        # 单节输出需要的 token 数：实测 qwen3 中文约 1.7 字/token，留出余量
        want_tokens = int(max_chars / CHARS_PER_TOKEN) + TOKEN_SAFETY_MARGIN
        digest = _fit_digest(
            entries,
            num_ctx=self.config.llm.num_ctx,
            want_output_tokens=want_tokens,
            system_prompt=SECTION_SYSTEM,
        )
        prompt_tokens = estimate_tokens(digest) + estimate_tokens(SECTION_SYSTEM)
        # 剩余可生成量：不能让提示词把输出挤到 0（早期"只写了几百字"就有这个原因）
        headroom = max(MIN_HEADROOM, self.config.llm.num_ctx - prompt_tokens - TOKEN_RESERVE)
        token_cap = max(MIN_TOKEN_CAP, min(want_tokens, headroom, MAX_TOKEN_CAP))
        logger.info(
            "写作预算：正文目标 %d~%d 字（%d 节，每节 %d~%d 字）| "
            "材料约 %d token | 单节生成上限 %d token",
            total_min_chars or min_chars * section_count,
            total_max_chars or max_chars * section_count,
            section_count,
            min_chars,
            max_chars,
            prompt_tokens,
            token_cap,
        )

        valid_ids = [index for index, _ in entries]
        client = get_llm(self.config)
        await client.start()
        title = _review_title(topic, plan)
        parts: list[str] = [f"# {title}", ""]
        if plan and plan.key_questions:
            parts.append("**本综述聚焦的问题**：" + "；".join(plan.key_questions))
            parts.append("")

        for number, section in enumerate(outline, start=1):
            await emit_event(
                emit,
                "status",
                message=f"正在撰写 {number}/{len(outline)}：{section.title}",
                section=section.title,
            )
            parts.append(f"## {section.title}")
            parts.append("")

            prompt = section_user(
                topic,
                section.title,
                section.points,
                digest,
                max_chars=max_chars,
                min_chars=min_chars,
                style=style,
            )
            body = ""
            try:
                async for chunk in client.stream(
                    [{"role": "user", "content": prompt}],
                    system=SECTION_SYSTEM,
                    temperature=LLM_TEMPERATURE_SECTION,
                    max_tokens=token_cap,
                ):
                    body += chunk
                    if on_token is not None:
                        await on_token(chunk)
            except LLMError as exc:
                message = f"章节「{section.title}」生成失败：{exc}"
                logger.warning(message)
                await emit_event(emit, "error", message=message)
                body = f"（本节生成失败：{exc}）"

            body = _sanitize(body, valid_ids)
            parts.append(body.strip())
            parts.append("")
            if on_token is not None:
                await on_token("\n\n")

        return "\n".join(parts).strip() + "\n"

    # ---------------------------------------------------------- 摘要 / 结论
    async def write_abstract(
        self,
        topic: str,
        draft: str,
        *,
        emit: Emitter | None = None,
    ) -> str:
        """为已生成的综述写一段中文摘要。"""
        from ..llm.prompts import _ROLE  # noqa: PLC2701 - 复用统一角色设定

        client = get_llm(self.config)
        await client.start()
        system = (
            f"{_ROLE}\n\n当前处于**摘要生成**阶段。"
            "请为给定的综述草稿撰写一段 200~300 字的结构式中文摘要，"
            "包含目的、方法、主要结果与结论四部分。只输出摘要正文。"
        )
        try:
            return (
                await client.chat(
                    [
                        {
                            "role": "user",
                            "content": (
                                f"综述课题：{topic}\n\n草稿内容：\n{draft[:6000]}\n\n"
                                "请输出 200~300 字的结构式摘要。"
                            ),
                        }
                    ],
                    system=system,
                    temperature=LLM_TEMPERATURE_ABSTRACT,
                    max_tokens=LLM_MAX_TOKENS_ABSTRACT,
                )
            ).strip()
        except LLMError as exc:
            await emit_event(emit, "error", message=f"摘要生成失败：{exc}")
            return ""

    # ------------------------------------------------------------ 单篇速读
    async def summarize_paper(self, paper: Paper, *, topic: str = "", full_text: str = "") -> str:
        from ..llm.prompts import SUMMARY_SYSTEM, summary_user

        body = full_text or paper.abstract or ""
        if not body.strip():
            return "该文献没有可用的摘要或全文。"
        client = get_llm(self.config)
        await client.start()
        return (
            await client.chat(
                [{"role": "user", "content": summary_user(body[:BODY_TRUNCATE_SUMMARY], topic=topic)}],
                system=SUMMARY_SYSTEM,
                temperature=LLM_TEMPERATURE_SUMMARY,
                max_tokens=LLM_MAX_TOKENS_SUMMARY,
            )
        ).strip()


# ------------------------------------------------------------------ 工具函数
_CITATION_RE = re.compile(r"[\[【]\s*(\d{1,3}(?:\s*[,，\-–]\s*\d{1,3})*)\s*[\]】]")


def extract_citations(text: str) -> list[int]:
    """抽取正文中出现的全部引用编号（支持 ``[1]`` / ``[1,2]`` / ``[1-3]`` / ``【1】``）。"""
    found: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        for token in re.split(r"[,，]", match.group(1)):
            token = token.strip()
            if not token:
                continue
            range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", token)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                if 0 < start <= end <= start + CITATION_RANGE_MAX_SPAN:
                    found.extend(range(start, end + 1))
                continue
            if token.isdigit():
                found.append(int(token))
    return found


def _sanitize(text: str, valid_ids: Sequence[int]) -> str:
    """剔除越界与自造的引用编号（保留合法编号，去掉 [] 包裹的非法数字）。"""
    valid = set(valid_ids)

    def replace(match: re.Match[str]) -> str:
        kept: list[str] = []
        for token in re.split(r"[,，]", match.group(1)):
            token = token.strip()
            if not token:
                continue
            range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", token)
            if range_match:
                start, end = int(range_match.group(1)), int(range_match.group(2))
                inside = [n for n in range(start, end + 1) if n in valid]
                if inside:
                    kept.extend(str(n) for n in inside)
                continue
            if token.isdigit() and int(token) in valid:
                kept.append(str(int(token)))
        if not kept:
            return ""  # 整组都是越界引用 → 删除标记
        return "[" + ",".join(kept) + "]"

    cleaned = _CITATION_RE.sub(replace, text or "")
    # 清理因删除引用留下的空白与空标点
    cleaned = re.sub(r"\s+([，。；、）])", r"\1", cleaned)
    cleaned = re.sub(r"\(\s*\)", "", cleaned)
    return cleaned.strip()


def _parse_outline(payload: Any) -> list[PlanSection]:
    if not isinstance(payload, dict):
        return []
    raw = payload.get("outline")
    if not isinstance(raw, list):
        return []
    sections: list[PlanSection] = []
    for item in raw:
        if isinstance(item, dict) and str(item.get("title") or "").strip():
            sections.append(PlanSection.from_dict(item))
    return sections


def _default_outline() -> list[PlanSection]:
    return [PlanSection(title=title, points=list(points)) for title, points in DEFAULT_OUTLINE]


def _review_title(topic: str, plan: ResearchPlan | None) -> str:
    base = (plan.topic_zh if plan and plan.topic_zh else topic).strip().rstrip("。.")
    if base.endswith(("综述", "研究进展", "系统评价")):
        return base
    return f"{base}：研究进展综述"
