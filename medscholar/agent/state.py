"""Agent 状态与事件模型。

工作流的四节点（Plan → 审批 → Execute → Reflect → Synthesize）共享
:class:`AgentState`；每个节点通过产出 :class:`AgentEvent` 把进度推给 UI，
因此同一套图逻辑既能驱动 SSE 前端，也能驱动 CLI。
"""

from __future__ import annotations

import time
import uuid
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Mapping, Sequence

from ..models import Paper

__all__ = [
    "Phase",
    "AgentEvent",
    "PlanQuery",
    "PlanSection",
    "ResearchPlan",
    "PaperAssessment",
    "CritiqueResult",
    "ReviewResult",
    "AgentState",
    "new_run_id",
]


class Phase(str, Enum):
    """工作流阶段。"""

    PENDING = "pending"
    PLAN = "plan"
    AWAIT_APPROVAL = "await_approval"
    EXECUTE = "execute"
    REFLECT = "reflect"
    SYNTHESIZE = "synthesize"
    REVIEW = "review"
    DONE = "done"
    ERROR = "error"
    CANCELLED = "cancelled"

    @property
    def label(self) -> str:
        return {
            "pending": "待开始",
            "plan": "规划检索策略",
            "await_approval": "等待确认",
            "execute": "执行检索",
            "reflect": "批判性评估",
            "synthesize": "撰写综述",
            "review": "自我审查",
            "done": "已完成",
            "error": "出错",
            "cancelled": "已取消",
        }.get(self.value, self.value)


@dataclass(slots=True)
class AgentEvent:
    """推送给 UI / CLI 的一条事件。"""

    type: str
    data: dict[str, Any] = field(default_factory=dict)
    ts: float = field(default_factory=time.time)

    def to_dict(self) -> dict[str, Any]:
        return {"type": self.type, "data": self.data, "ts": self.ts}


def new_run_id() -> str:
    return uuid.uuid4().hex[:12]


# ------------------------------------------------------------------- 检索计划
@dataclass(slots=True)
class PlanQuery:
    """一条检索式。"""

    query: str
    sources: list[str] = field(default_factory=list)
    rationale: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {"query": self.query, "sources": list(self.sources), "rationale": self.rationale}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanQuery":
        sources = data.get("sources") or []
        if isinstance(sources, str):
            sources = [s.strip() for s in sources.split(",") if s.strip()]
        return cls(
            query=str(data.get("query") or "").strip(),
            sources=[str(s) for s in sources],
            rationale=str(data.get("rationale") or ""),
        )


@dataclass(slots=True)
class PlanSection:
    """综述大纲的一个章节。"""

    title: str
    points: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {"title": self.title, "points": list(self.points)}

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "PlanSection":
        points = data.get("points") or []
        if isinstance(points, str):
            points = [p.strip() for p in points.split("；") if p.strip()]
        return cls(
            title=str(data.get("title") or "").strip(),
            points=[str(p) for p in points if str(p).strip()],
        )


#: 计划 JSON 的顶层字段名，用于识别「模型把对象拆成了数组元素」这种情况。
_PLAN_KEYS = frozenset(
    {
        "topic_zh",
        "topic_en",
        "pico",
        "queries",
        "mesh_terms",
        "year_from",
        "year_to",
        "key_questions",
        "outline",
    }
)

#: 检索式特征：出现这些记号就认为字符串是一条检索式，而不是章节标题。
_SEARCH_MARKERS = (" AND ", " OR ", " NOT ", "[", "]", '"', ":", "*", "(", ")")


def coerce_plan_payload(payload: Any) -> dict[str, Any]:
    """把模型返回的 JSON 归一化成 :class:`ResearchPlan` 需要的字典。

    模型并不总是听话。实测 qwen3:8b 在规划阶段返回过**顶层数组**，直接交给
    ``ResearchPlan.from_dict`` 会抛 ``'list' object has no attribute 'get'``，
    整次规划降级成模板大纲。这里把常见的几种走样都救回来：

    - ``{...}``                      → 原样返回；
    - ``[{...}]``                    → 取出唯一的对象；
    - ``[{"topic_zh":...}, {"queries":[...]}]`` → 合并成一个对象（对象被拆散了）；
    - ``[{"query":...}, ...]``       → ``{"queries": [...]}``；
    - ``[{"title":...}, ...]``       → ``{"outline": [...]}``；
    - ``["...", ...]``               → 像检索式的进 queries，否则当 key_questions；
    - 其它（字符串/数字/null）        → ``{}``，交给调用方的兜底逻辑。

    实测 qwen3:8b 在规划阶段返回过 ``["抑郁症状改善（如HAMD评分）", "治疗反应率", ...]``
    —— 那其实是 ``pico.outcomes`` 数组。这类裸字符串列表含义不明：既不是章节标题，
    也不一定是检索式。所以只在含检索语法时当成 queries，其余一律放进 key_questions
    （只用于展示），让 outline / queries 留空，好让 graph 里既有的兜底补上
    标准综述大纲与「用课题本身检索」，而不是把结局指标当成章节标题。

    >>> coerce_plan_payload([{"topic_zh": "rTMS"}, {"outline": []}])["topic_zh"]
    'rTMS'
    >>> coerce_plan_payload("抱歉，我无法回答") == {}
    True
    >>> coerce_plan_payload(["疗效", "安全性"])["key_questions"]
    ['疗效', '安全性']
    """
    if isinstance(payload, Mapping):
        return dict(payload)
    if not isinstance(payload, list):
        return {}

    items = [item for item in payload if item not in (None, "", [], {})]
    if not items:
        return {}

    dicts = [item for item in items if isinstance(item, Mapping)]
    strings = [item for item in items if isinstance(item, str) and item.strip()]

    if dicts and len(dicts) == len(items):
        keys = [set(item) for item in dicts]
        if all("query" in key for key in keys):
            return {"queries": [dict(item) for item in dicts]}
        if all("title" in key for key in keys):
            return {"outline": [dict(item) for item in dicts]}
        if len(dicts) == 1:
            only = dict(dicts[0])
            # 单元素数组只是外面多包了一层壳
            return only
        # 多个对象各带一部分字段 → 合并，尽量把整份计划拼回来
        merged: dict[str, Any] = {}
        for item in dicts:
            for key, value in item.items():
                if key not in merged or not merged[key]:
                    merged[key] = value
        return merged

    if strings and len(strings) == len(items):
        looks_like_query = [
            any(marker in text for marker in _SEARCH_MARKERS) for text in strings
        ]
        if sum(looks_like_query) * 2 >= len(strings):
            return {"queries": [{"query": text.strip()} for text in strings]}
        # 含义不明 → 只当展示用的关键问题，outline/queries 留空交给兜底
        return {"key_questions": [text.strip() for text in strings]}

    # 混合类型：留下能用的部分
    if dicts:
        return coerce_plan_payload(dicts)
    return {}


def _usable_query(text: str) -> bool:
    """检索式必须含真正的词/字，不能只是标点。

    实测模型在被截断时留下过 ``"query": "("``。若把它当成有效检索式，
    Scout 就会拿一个左括号去检索，返回一堆无关结果；
    过滤掉之后 ``queries`` 为空，graph 会退回「用课题本身检索」。
    """
    stripped = text.strip()
    if not stripped:
        return False
    return any(ch.isalnum() for ch in stripped)


@dataclass(slots=True)
class ResearchPlan:
    """Plan 节点的产出。"""

    topic_zh: str = ""
    topic_en: str = ""
    pico: dict[str, Any] = field(default_factory=dict)
    queries: list[PlanQuery] = field(default_factory=list)
    mesh_terms: list[str] = field(default_factory=list)
    year_from: int | None = None
    year_to: int | None = None
    key_questions: list[str] = field(default_factory=list)
    outline: list[PlanSection] = field(default_factory=list)
    #: 用户在审批时给出的修改意见
    feedback: str = ""
    #: 是否为兜底方案（LLM 不可用 / 离线 / 解析失败），前端据此提示用户方案未定制
    degraded: bool = False
    degraded_reason: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "topic_zh": self.topic_zh,
            "topic_en": self.topic_en,
            "pico": self.pico,
            "queries": [q.to_dict() for q in self.queries],
            "mesh_terms": list(self.mesh_terms),
            "year_from": self.year_from,
            "year_to": self.year_to,
            "key_questions": list(self.key_questions),
            "outline": [s.to_dict() for s in self.outline],
            "degraded": self.degraded,
            "degraded_reason": self.degraded_reason,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ResearchPlan":
        def _int(value: Any) -> int | None:
            try:
                return int(value) if value not in (None, "", "null") else None
            except (TypeError, ValueError):
                return None

        # 防线：即使调用方忘了 coerce，也不能因为顶层不是对象就整次规划失败
        if not isinstance(data, Mapping):
            data = coerce_plan_payload(data)

        pico = data.get("pico")
        if not isinstance(pico, dict):
            pico = {}

        return cls(
            topic_zh=str(data.get("topic_zh") or "").strip(),
            topic_en=str(data.get("topic_en") or "").strip(),
            pico=dict(pico),
            queries=[
                PlanQuery.from_dict(q)
                for q in (data.get("queries") or [])
                if isinstance(q, Mapping) and _usable_query(str(q.get("query") or ""))
            ],
            mesh_terms=[str(m) for m in (data.get("mesh_terms") or []) if str(m).strip()],
            year_from=_int(data.get("year_from")),
            year_to=_int(data.get("year_to")),
            key_questions=[
                str(q) for q in (data.get("key_questions") or []) if str(q).strip()
            ],
            outline=[
                PlanSection.from_dict(s)
                for s in (data.get("outline") or [])
                if isinstance(s, Mapping) and str(s.get("title") or "").strip()
            ],
        )

    def english_query(self) -> str:
        """给英文库用的检索词。"""
        return self.topic_en or self.topic_zh

    def chinese_query(self) -> str:
        return self.topic_zh or self.topic_en


# ------------------------------------------------------------------- 评估结果
@dataclass(slots=True)
class PaperAssessment:
    """Critic 对单篇文献的评估。"""

    index: int
    paper_id: int | None = None
    relevance: float = 5.0
    quality: float = 5.0
    evidence_level: str = "其他"
    key_finding: str = ""
    limitation: str = ""
    use_in_review: bool = True
    source: str = "heuristic"  # heuristic | llm

    @property
    def combined(self) -> float:
        """综合分：相关性权重更高（写综述时首先要"对题"）。"""
        return round(self.relevance * 0.6 + self.quality * 0.4, 2)

    def to_dict(self) -> dict[str, Any]:
        return {
            "index": self.index,
            "paper_id": self.paper_id,
            "relevance": self.relevance,
            "quality": self.quality,
            "evidence_level": self.evidence_level,
            "key_finding": self.key_finding,
            "limitation": self.limitation,
            "use_in_review": self.use_in_review,
            "combined": self.combined,
            "source": self.source,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any], *, index: int) -> "PaperAssessment":
        def _score(value: Any, default: float) -> float:
            try:
                return max(0.0, min(10.0, float(value)))
            except (TypeError, ValueError):
                return default

        use = data.get("use_in_review", True)
        if isinstance(use, str):
            use = use.strip() in {"是", "yes", "true", "y", "1"}

        return cls(
            index=index,
            paper_id=data.get("paper_id"),
            relevance=_score(data.get("relevance"), 5.0),
            quality=_score(data.get("quality"), 5.0),
            evidence_level=str(data.get("evidence_level") or "其他"),
            key_finding=str(data.get("key_finding") or ""),
            limitation=str(data.get("limitation") or ""),
            use_in_review=bool(use),
            source="llm",
        )


@dataclass(slots=True)
class CritiqueResult:
    """Reflect 节点的产出。"""

    assessments: list[PaperAssessment] = field(default_factory=list)
    evidence_quality: str = "中"
    gaps: list[str] = field(default_factory=list)
    suggestions: list[str] = field(default_factory=list)
    used_llm: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "assessments": [a.to_dict() for a in self.assessments],
            "overall": {
                "evidence_quality": self.evidence_quality,
                "gaps": list(self.gaps),
                "suggestions": list(self.suggestions),
            },
            "used_llm": self.used_llm,
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "CritiqueResult":
        """从阶段快照还原（断点续跑用）。"""
        if not isinstance(data, Mapping):
            return cls()
        overall = data.get("overall")
        if not isinstance(overall, Mapping):
            overall = {}
        assessments: list[PaperAssessment] = []
        for index, item in enumerate(data.get("assessments") or [], start=1):
            if not isinstance(item, Mapping):
                continue
            try:
                idx = int(item.get("id") or item.get("index") or index)
            except (TypeError, ValueError):
                idx = index
            assessments.append(PaperAssessment.from_dict(item, index=idx))
        return cls(
            assessments=assessments,
            evidence_quality=str(overall.get("evidence_quality") or "中"),
            gaps=[str(g) for g in (overall.get("gaps") or []) if str(g).strip()],
            suggestions=[str(s) for s in (overall.get("suggestions") or []) if str(s).strip()],
            used_llm=bool(data.get("used_llm")),
        )


@dataclass(slots=True)
class ReviewResult:
    """综述草稿的自我审查结果。"""

    verdict: str = "pass"
    score: float = 0.0
    issues: list[dict[str, Any]] = field(default_factory=list)
    strengths: list[str] = field(default_factory=list)
    invalid_citations: list[int] = field(default_factory=list)

    @property
    def passed(self) -> bool:
        return self.verdict.lower().startswith("pass") and not any(
            i.get("severity") == "high" for i in self.issues
        )

    def to_dict(self) -> dict[str, Any]:
        return {
            "verdict": self.verdict,
            "score": self.score,
            "issues": list(self.issues),
            "strengths": list(self.strengths),
            "invalid_citations": list(self.invalid_citations),
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, Any]) -> "ReviewResult":
        """从阶段快照还原（断点续跑用）。"""
        if not isinstance(data, Mapping):
            return cls()
        try:
            score = float(data.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0
        return cls(
            verdict=str(data.get("verdict") or "pass").lower(),
            score=score,
            issues=[dict(i) for i in (data.get("issues") or []) if isinstance(i, Mapping)],
            strengths=[str(s) for s in (data.get("strengths") or []) if str(s).strip()],
            invalid_citations=[
                int(i) for i in (data.get("invalid_citations") or []) if str(i).lstrip("-").isdigit()
            ],
        )


# ------------------------------------------------------------------- 运行状态
@dataclass(slots=True)
class AgentState:
    """一次 Agent 运行的完整状态。"""

    topic: str
    run_id: str = field(default_factory=new_run_id)
    session_id: int | None = None
    project_id: int | None = None
    sources: list[str] = field(default_factory=list)
    offline: bool = False
    citation_style: str = "gb7714"

    phase: Phase = Phase.PENDING
    plan: ResearchPlan | None = None

    #: 检索到的文献（去重合并后，顺序即引用顺序基准）
    papers: list[Paper] = field(default_factory=list)
    #: 写入数据库后的 paper_id 列表（与 papers 一一对应）
    paper_ids: list[int] = field(default_factory=list)
    #: 引用编号 → Paper（正文里 [n] 中的 n）
    citation_map: dict[int, Paper] = field(default_factory=dict)

    critique: CritiqueResult | None = None
    outline: list[PlanSection] = field(default_factory=list)
    draft: str = ""
    review: ReviewResult | None = None

    artifact_id: int | None = None
    saved: dict[str, int] = field(default_factory=dict)
    embed_report: dict[str, Any] = field(default_factory=dict)
    search_stats: list[dict[str, Any]] = field(default_factory=list)

    errors: list[str] = field(default_factory=list)
    #: 本次运行的综述正文目标字数（0 表示沿用配置里的默认值）。
    #: 放在状态里，是为了让"每次运行一个字数范围"和断点续跑都能带上它。
    review_min_chars: int = 0
    review_max_chars: int = 0
    #: 断点续跑：上次已完成的阶段名（plan/execute/reflect/synthesize/review）。
    #: 为空表示这是一次全新运行。
    resumed_from: str = ""
    started_at: float = field(default_factory=time.time)
    finished_at: float | None = None

    # ------------------------------------------------------------- 便捷方法
    @property
    def elapsed_ms(self) -> int:
        end = self.finished_at or time.time()
        return int((end - self.started_at) * 1000)

    @property
    def valid_citation_ids(self) -> list[int]:
        return sorted(self.citation_map)

    def select_papers(
        self, *, max_papers: int = 25, use_critique: bool = True
    ) -> list[tuple[int, Paper]]:
        """挑选进入写作上下文的文献，返回 ``[(引用编号, Paper), ...]``。

        编号在**筛选之后**重新连续分配，确保正文里的 ``[n]`` 与参考文献表
        严格一一对应（这是引用准确性最容易出错的地方）。
        """
        if not self.citation_map:
            return []

        entries = sorted(self.citation_map.items())
        if use_critique and self.critique and self.critique.assessments:
            score_by_id = {
                a.paper_id: a for a in self.critique.assessments if a.paper_id is not None
            }
            by_index = {a.index: a for a in self.critique.assessments}
            scored: list[tuple[float, int, Paper]] = []
            for index, paper in entries:
                assessment = score_by_id.get(paper.paper_id) or by_index.get(index)
                if assessment is not None and not assessment.use_in_review:
                    continue
                score = assessment.combined if assessment else 5.0
                scored.append((score, index, paper))
            scored.sort(key=lambda item: (-item[0], item[1]))
            entries = [(index, paper) for _, index, paper in scored[:max_papers]]
            # 保持引用编号稳定：按原编号排序，避免正文编号与排序错位
            entries.sort(key=lambda item: item[0])
        else:
            entries = entries[:max_papers]

        return entries

    def renumber(self, entries: Sequence[tuple[int, Paper]]) -> dict[int, Paper]:
        """把选中的文献重新编号为 1..n，并更新 ``citation_map``。"""
        mapping: dict[int, Paper] = {}
        for new_index, (_old_index, paper) in enumerate(entries, start=1):
            mapping[new_index] = paper
        self.citation_map = mapping
        return mapping

    def add_error(self, message: str) -> None:
        if message and message not in self.errors:
            self.errors.append(message)

    def summary(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topic": self.topic,
            "phase": self.phase.value,
            "papers": len(self.papers),
            "citations": len(self.citation_map),
            "saved": dict(self.saved),
            "elapsed_ms": self.elapsed_ms,
            "errors": list(self.errors),
            "artifact_id": self.artifact_id,
        }
