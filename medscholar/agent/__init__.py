"""Agent 层：Scout/Reader/Critic/Writer/Formatter 五智能体 + graph 编排的四节点工作流。

运行时（后台执行 + SSE 事件流 + 人工审批）见 :mod:`~medscholar.agent.runtime`。
"""

from __future__ import annotations

from .critic import CriticAgent, heuristic_assessment
from .formatter import CitationReport, FormatterAgent
from .graph import ApprovalCallback, ResearchGraph
from .reader import PDF_AVAILABLE, PDF_NOTE, ReaderAgent
from .runtime import AgentRuntime, RunHandle, get_runtime
from .scout import Emitter, ScoutAgent, ScoutResult, emit_event
from .state import (
    AgentEvent,
    AgentState,
    CritiqueResult,
    PaperAssessment,
    Phase,
    PlanQuery,
    PlanSection,
    ResearchPlan,
    ReviewResult,
    new_run_id,
)
from .writer import DEFAULT_OUTLINE, WriterAgent, extract_citations

__all__ = [
    # 编排
    "ResearchGraph",
    "ApprovalCallback",
    "AgentRuntime",
    "RunHandle",
    "get_runtime",
    # 六个智能体
    "ScoutAgent",
    "ReaderAgent",
    "CriticAgent",
    "WriterAgent",
    "FormatterAgent",
    "ScoutResult",
    "CitationReport",
    "heuristic_assessment",
    "extract_citations",
    "DEFAULT_OUTLINE",
    "PDF_AVAILABLE",
    "PDF_NOTE",
    # 状态
    "AgentState",
    "AgentEvent",
    "Phase",
    "ResearchPlan",
    "PlanQuery",
    "PlanSection",
    "PaperAssessment",
    "CritiqueResult",
    "ReviewResult",
    "new_run_id",
    "Emitter",
    "emit_event",
]
