"""Critic Agent —— 批判性评估。

职责（需求 3.1）：评估文献的**方法学质量**、**证据等级**与**与课题的相关性**，
决定哪些文献值得写进综述。

两条互补的评估路径：

1. **启发式**（默认先跑）：从出版类型、摘要中的研究设计关键词、样本量、
   被引次数、时效性等可验证信号打分 —— 完全离线、可解释、不会幻觉；
2. **LLM 评估**：在启发式基础上让模型逐篇点评核心发现与局限。

LLM 不可用或输出不合法时**自动回退**到启发式结果，因此本节点永不失败。
"""

from __future__ import annotations

import logging
import re
from typing import Sequence

from ..config import AppConfig, get_config
from ..llm.client import LLMError, get_llm
from ..llm.prompts import CRITIQUE_SYSTEM, critique_user
from ..models import Paper
from ..retrieval import build_context_digest
from .scout import Emitter, emit_event
from .state import CritiqueResult, PaperAssessment

logger = logging.getLogger(__name__)

__all__ = ["CriticAgent", "heuristic_assessment", "detect_evidence_level", "extract_sample_size"]

#: 研究设计 → (证据等级名, 质量基准分)
_DESIGN_RULES: tuple[tuple[str, tuple[str, ...], float], ...] = (
    ("Meta分析/系统评价", ("meta-analysis", "meta analysis", "systematic review", "荟萃分析", "系统评价", "meta分析"), 8.5),
    ("指南/共识", ("guideline", "consensus", "recommendation", "指南", "共识", "专家共识"), 7.5),
    ("RCT", ("randomized controlled", "randomised controlled", "randomly assigned",
             "randomized trial", "rct", "随机对照", "随机分组"), 7.5),
    ("队列研究", ("cohort", "prospective study", "longitudinal", "队列", "前瞻性"), 5.5),
    ("病例对照", ("case-control", "case control", "病例对照"), 4.5),
    ("横断面", ("cross-sectional", "cross sectional", "survey", "横断面", "现况调查"), 4.0),
    ("综述", ("review", "overview", "narrative review", "综述", "研究进展"), 4.5),
    ("动物实验", ("rats", "rat ", "mice", "mouse", "murine", "rabbit", "zebrafish",
                  "大鼠", "小鼠", "动物模型", "家兔"), 3.0),
    ("病例报告", ("case report", "case series", "个案", "病例报告", "病例系列"), 2.5),
)

_SAMPLE_PATTERNS = (
    re.compile(r"n\s*=\s*(\d{1,6})", re.IGNORECASE),
    re.compile(r"(\d{1,6})\s*(?:patients|participants|subjects|participant|adults|children|cases)\b", re.IGNORECASE),
    re.compile(r"(\d{1,6})\s*(?:例|名|位)(?:患者|病人|受试者|对象)?"),
    re.compile(r"(?:纳入|共|选取|收集)\s*(\d{1,6})\s*(?:例|名|位)"),
)

_WORD_RE = re.compile(r"[a-z][a-z\-]{2,}|[\u4e00-\u9fff]{2,}")
_STOP = {
    "the", "and", "for", "with", "from", "into", "that", "this", "are", "was",
    "were", "its", "their", "study", "trial", "effect", "effects", "analysis",
    "patients", "results", "conclusion", "objective", "methods", "background",
    "研究", "方法", "结果", "结论", "目的", "疗效", "观察", "分析", "治疗",
}


def detect_evidence_level(paper: Paper) -> str:
    """从出版类型与题目/摘要推断证据等级。"""
    haystack = " ".join(
        [
            (paper.publication_type or "").lower(),
            (paper.title or "").lower(),
            (paper.abstract or "")[:1500].lower(),
        ]
    )
    # 动物实验优先级高于设计类型：动物 RCT 依然是动物实验
    if any(k in haystack for k in ("大鼠", "小鼠", "动物模型")) or re.search(
        r"\b(rats?|mice|mouse|murine|rabbits?)\b", haystack
    ):
        return "动物实验"
    for label, keywords, _score in _DESIGN_RULES:
        if any(k in haystack for k in keywords):
            return label
    return "其他"


def extract_sample_size(paper: Paper) -> int | None:
    """从摘要中提取样本量（取各模式中的最大值，避免误取年份等小数字）。"""
    text = " ".join([paper.title or "", paper.abstract or ""])[:4000]
    candidates: list[int] = []
    for pattern in _SAMPLE_PATTERNS:
        for match in pattern.finditer(text):
            try:
                value = int(match.group(1))
            except (TypeError, ValueError):
                continue
            if 5 <= value <= 500_000:
                candidates.append(value)
    return max(candidates) if candidates else None


def _design_score(level: str) -> float:
    for label, _keywords, score in _DESIGN_RULES:
        if label == level:
            return score
    return 4.0


def _relevance_score(paper: Paper, topic_terms: set[str]) -> float:
    """相关性 = 课题实词在标题/摘要中的覆盖率（标题权重更高）。"""
    if not topic_terms:
        return 6.0
    title = (paper.title or "").lower()
    abstract = (paper.abstract or "").lower()
    keywords = " ".join(paper.keywords + paper.mesh_terms).lower()

    title_hits = sum(1 for t in topic_terms if t in title)
    abstract_hits = sum(1 for t in topic_terms if t in abstract or t in keywords)

    coverage = min(1.0, (title_hits * 2.0 + abstract_hits * 1.0) / (len(topic_terms) * 2.0))
    score = 3.0 + coverage * 7.0
    # 主题词命中是强信号
    if title_hits and coverage >= 0.5:
        score = min(10.0, score + 0.8)
    return round(max(0.0, min(10.0, score)), 1)


def heuristic_assessment(
    paper: Paper, *, topic: str, index: int, current_year: int | None = None
) -> PaperAssessment:
    """纯规则评估（离线、可解释、无幻觉）。"""
    import datetime

    current_year = current_year or datetime.date.today().year
    level = detect_evidence_level(paper)
    quality = _design_score(level)

    # --- 样本量：大样本加分，过小减分
    sample = extract_sample_size(paper)
    if sample is not None:
        if sample >= 1000:
            quality += 1.2
        elif sample >= 300:
            quality += 0.8
        elif sample >= 100:
            quality += 0.4
        elif sample < 30:
            quality -= 0.8

    # --- 被引：领域影响力的粗略代理
    cited = paper.cited_by_count or 0
    if cited >= 500:
        quality += 1.2
    elif cited >= 100:
        quality += 0.8
    elif cited >= 30:
        quality += 0.4
    elif cited == 0:
        quality -= 0.3

    # --- 时效性
    if paper.pub_year:
        age = current_year - paper.pub_year
        if age <= 3:
            quality += 0.5
        elif age >= 12:
            quality -= 0.6

    # --- 完整度：没有摘要的文献难以评估
    if not (paper.abstract or "").strip():
        quality -= 1.2

    quality = round(max(0.0, min(10.0, quality)), 1)

    topic_terms = {t for t in _WORD_RE.findall((topic or "").lower()) if t not in _STOP}
    relevance = _relevance_score(paper, topic_terms)

    # 摘要缺失时相关性不可靠，向中性收敛
    if not (paper.abstract or "").strip():
        relevance = round(relevance * 0.7 + 6.0 * 0.3, 1)

    limitation = ""
    if sample is not None and sample < 30:
        limitation = f"样本量偏小（n={sample}）"
    elif level == "动物实验":
        limitation = "动物实验，外推至临床需谨慎"
    elif level in {"病例报告", "横断面"}:
        limitation = "设计证据等级较低，难以推断因果"
    elif not (paper.abstract or "").strip():
        limitation = "缺少摘要，无法评估方法学细节"
    elif cited == 0 and (paper.pub_year or current_year) <= current_year - 1:
        limitation = "尚无被引记录"

    key_finding = _first_sentence(paper.abstract)

    return PaperAssessment(
        index=index,
        paper_id=paper.paper_id,
        relevance=relevance,
        quality=quality,
        evidence_level=level,
        key_finding=key_finding,
        limitation=limitation,
        use_in_review=relevance >= 4.0,
        source="heuristic",
    )


def _first_sentence(abstract: str, *, limit: int = 80) -> str:
    """从摘要中截取一句结论性文字（优先取「结果」「结论」段落）。"""
    if not abstract:
        return ""
    text = abstract.replace("\n", " ")
    for marker in ("结论：", "结论:", "CONCLUSION:", "Conclusion:", "结果：", "结果:"):
        position = text.find(marker)
        if position >= 0:
            snippet = text[position + len(marker) :].strip()
            sentence = re.split(r"(?<=[。.!?])\s*", snippet)[0]
            if sentence:
                return sentence[:limit].rstrip("。. ") + "。"
    sentence = re.split(r"(?<=[。.!?])\s*", text.strip())[0]
    return sentence[:limit].rstrip("。. ") + ("。" if sentence else "")


class CriticAgent:
    """评估智能体。"""

    def __init__(self, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()

    async def assess(
        self,
        topic: str,
        entries: Sequence[tuple[int, Paper]],
        *,
        emit: Emitter | None = None,
        use_llm: bool = True,
    ) -> CritiqueResult:
        """评估全部候选文献。

        Args:
            entries: ``[(引用编号, Paper), ...]``
        """
        index_to_paper = {index: paper for index, paper in entries}
        result = CritiqueResult()

        # ---- 1) 启发式基线（总是先算，作为回退结果）
        heuristic = {
            index: heuristic_assessment(paper, topic=topic, index=index)
            for index, paper in entries
        }

        # ---- 2) LLM 精评
        if use_llm and entries and not self.config.offline:
            try:
                llm_assessments = await self._llm_assess(topic, entries, index_to_paper)
            except LLMError as exc:
                logger.info("LLM 评估不可用，使用启发式结果：%s", exc)
                await emit_event(
                    emit,
                    "status",
                    message=f"LLM 评估不可用（{str(exc)[:60]}），已改用规则评估",
                )
                llm_assessments = {}
            except Exception as exc:  # pragma: no cover
                logger.warning("LLM 评估异常：%s", exc)
                llm_assessments = {}

            if llm_assessments:
                result.used_llm = True
                # LLM 给相关性/结论，启发式守住质量分（避免小模型乱给高分）
                for index, base in heuristic.items():
                    llm_item = llm_assessments.get(index)
                    if llm_item is None:
                        result.assessments.append(base)
                        continue
                    merged = PaperAssessment(
                        index=index,
                        paper_id=base.paper_id,
                        relevance=round((llm_item.relevance + base.relevance) / 2, 1),
                        quality=round((llm_item.quality + base.quality) / 2, 1),
                        evidence_level=llm_item.evidence_level or base.evidence_level,
                        key_finding=llm_item.key_finding or base.key_finding,
                        limitation=llm_item.limitation or base.limitation,
                        use_in_review=llm_item.use_in_review and base.use_in_review,
                        source="llm+heuristic",
                    )
                    result.assessments.append(merged)

        if not result.assessments:
            result.assessments = [heuristic[i] for i in sorted(heuristic)]

        if not result.assessments:
            return result

        # ---- 3) 整体证据质量
        usable = [a for a in result.assessments if a.use_in_review]
        avg_quality = sum(a.quality for a in usable) / len(usable) if usable else 0.0
        rct_like = sum(
            1 for a in usable if a.evidence_level in {"RCT", "Meta分析/系统评价", "指南/共识"}
        )
        if avg_quality >= 7.5 and rct_like >= 3:
            result.evidence_quality = "高"
        elif avg_quality >= 6.0:
            result.evidence_quality = "中"
        elif avg_quality >= 4.0:
            result.evidence_quality = "低"
        else:
            result.evidence_quality = "极低"

        result.suggestions = self._build_suggestions(result, index_to_paper)
        result.gaps = self._build_gaps(result)
        return result

    # ------------------------------------------------------------------ LLM
    async def _llm_assess(
        self,
        topic: str,
        entries: Sequence[tuple[int, Paper]],
        index_to_paper: dict[int, Paper],
    ) -> dict[int, PaperAssessment]:
        """让 LLM 逐篇点评，但**限制篇数**。

        本地 8B 模型在纯 CPU 上约 7 tokens/s，逐篇点评 25 篇需要生成上千 token、
        耗时数分钟。因此只把最相关的若干篇交给 LLM（按启发式得分排序），
        其余文献仍由离线启发式给出完整评估 —— 两者在 :meth:`assess` 中等权融合。
        """
        limit = min(len(entries), max(4, self.config.agent.critique_max_papers))
        if len(entries) > limit:
            scored = sorted(
                entries,
                key=lambda pair: -heuristic_assessment(
                    pair[1], topic=topic, index=pair[0]
                ).combined,
            )
            subset = scored[:limit]
            subset.sort(key=lambda pair: pair[0])  # 保持引用编号顺序，便于模型对上号
        else:
            subset = list(entries)

        digest = build_context_digest(subset, max_abstract=600)
        logger.info("LLM 逐篇点评 %d/%d 篇文献", len(subset), len(entries))

        client = get_llm(self.config)
        await client.start()
        payload = await client.chat_json(
            [{"role": "user", "content": critique_user(topic, digest, len(subset))}],
            system=CRITIQUE_SYSTEM,
            temperature=0.1,
            max_tokens=min(3000, 320 * len(subset) + 400),
            retries=2,
        )
        raw_items = payload.get("assessments") if isinstance(payload, dict) else None
        if not isinstance(raw_items, list):
            return {}

        out: dict[int, PaperAssessment] = {}
        for item in raw_items:
            if not isinstance(item, dict):
                continue
            try:
                index = int(item.get("id"))
            except (TypeError, ValueError):
                continue
            if index not in index_to_paper or index in out:
                continue
            assessment = PaperAssessment.from_dict(item, index=index)
            assessment.paper_id = index_to_paper[index].paper_id
            out[index] = assessment
        return out

    # --------------------------------------------------------------- 归纳
    @staticmethod
    def _build_suggestions(
        result: CritiqueResult, index_to_paper: dict[int, Paper]
    ) -> list[str]:
        usable = [a for a in result.assessments if a.use_in_review]
        suggestions: list[str] = []
        if not usable:
            suggestions.append("没有文献达到纳入标准，建议放宽检索条件或补充检索式。")
            return suggestions
        top = sorted(usable, key=lambda a: -a.combined)[:3]
        names = []
        for assessment in top:
            paper = index_to_paper.get(assessment.index)
            if paper:
                names.append(f"[{assessment.index}] {paper.title[:36]}")
        if names:
            suggestions.append("优先引用质量最高的文献：" + "；".join(names))
        rct_count = sum(1 for a in usable if a.evidence_level in {"RCT", "Meta分析/系统评价"})
        if rct_count == 0:
            suggestions.append("纳入文献中缺少 RCT 或 Meta 分析，结论应避免使用因果性表述。")
        elif rct_count < 3:
            suggestions.append(f"仅 {rct_count} 篇高等级证据，建议补充检索以增强论证强度。")
        if len(usable) < 5:
            suggestions.append("可用文献偏少，建议扩大检索年限或增加同义词检索式。")
        return suggestions

    @staticmethod
    def _build_gaps(result: CritiqueResult) -> list[str]:
        gaps: list[str] = []
        usable = [a for a in result.assessments if a.use_in_review]
        if usable and all(a.evidence_level == "动物实验" for a in usable):
            gaps.append("现有材料均为动物实验，缺少临床证据。")
        small = [a for a in usable if "样本量偏小" in (a.limitation or "")]
        if small and len(small) >= max(2, len(usable) // 2):
            gaps.append("多数研究样本量偏小，需要大样本验证。")
        old = sum(1 for a in usable if a.evidence_level == "其他")
        if old and old >= len(usable) // 2:
            gaps.append("相当一部分文献研究设计不明确，方法学信息不足。")
        return gaps
