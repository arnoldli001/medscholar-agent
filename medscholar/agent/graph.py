"""LangGraph 风格的四节点工作流（需求 3.2）。

::

    [Plan] → [用户审批] → [Execute] → [Reflect] → [Synthesize] → [Review] → 成稿

实现说明：本项目不需要 LangGraph 的持久化图状态机（单机、单用户、
运行期状态全在内存里），因此用等价的 **async 编排 + 事件流** 实现同一语义，
既保持了四节点的清晰边界，又避免了为一个线性流程引入重依赖。
每个节点都是独立方法，可单独测试与替换。
"""

from __future__ import annotations

import asyncio
import logging
import time
from typing import Any, Awaitable, Callable, Sequence

from ..cite import detect_style
from ..config import AppConfig, get_config
from ..constants import (
    DIGEST_MAX_ABSTRACT_REVIEW,
    DIGEST_MAX_ABSTRACT_REVISE,
    DRAFT_TRUNCATE_REVIEW,
    DRAFT_TRUNCATE_REVISE,
    FEEDBACK_TRUNCATE,
    LLM_MAX_TOKENS_PLAN,
    LLM_MAX_TOKENS_REVIEW,
    LLM_TEMPERATURE_PLAN,
    LLM_TEMPERATURE_REVIEW,
    LLM_TEMPERATURE_REVISE,
    MAX_ERRORS_SNAPSHOT,
    MAX_ISSUES_REVISE,
    REVISE_MIN_LENGTH_RATIO,
)
from ..db.connect import Database, get_db
from ..db.repo import add_message, save_artifact
from ..llm.client import LLMError, get_llm
from ..llm.prompts import PLAN_SYSTEM, REFLECT_SYSTEM, plan_user, reflect_user
from ..models import Paper
from ..retrieval import build_context_digest
from .critic import CriticAgent
from .formatter import FormatterAgent
from .reader import ReaderAgent
from .scout import Emitter, ScoutAgent, emit_event
from .state import (
    AgentState,
    CritiqueResult,
    Phase,
    PlanQuery,
    ResearchPlan,
    ReviewResult,
    coerce_plan_payload,
)
from .writer import WriterAgent, extract_citations

logger = logging.getLogger(__name__)

__all__ = ["ResearchGraph", "ApprovalCallback"]

#: 审批回调：返回 ``(decision, feedback)``，decision ∈ approve / revise / cancel
ApprovalCallback = Callable[[], Awaitable[tuple[str, str]]]


class ResearchGraph:
    """把六个 Agent 串成一条可观测、可中断的流水线。"""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        db: Database | None = None,
        registry: Any = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db or get_db()
        self.scout = ScoutAgent(config=self.config, registry=registry, db=self.db)
        self.reader = ReaderAgent(config=self.config, db=self.db, registry=registry)
        self.critic = CriticAgent(config=self.config)
        self.writer = WriterAgent(config=self.config)
        self.formatter = FormatterAgent(config=self.config)

    async def close(self) -> None:
        await self.reader.close()

    # ============================================================ 主流程
    async def run(
        self,
        state: AgentState,
        *,
        emit: Emitter | None = None,
        approval: ApprovalCallback | None = None,
    ) -> AgentState:
        """执行完整工作流。任何异常都会被记录进 ``state.errors`` 并优雅收尾。

        若 ``state.resumed_from`` 指明某个阶段已经完成（断点续跑），
        该阶段及其之前的阶段会直接复用已有成果，不再重跑。
        """
        done_at = _phase_index(state.resumed_from)
        resumed = done_at >= 0
        if resumed:
            await emit_event(
                emit,
                "status",
                message=f"从「{state.resumed_from}」阶段继续，已完成的阶段不再重跑",
            )
        try:
            # ---------------------------------------------------- 1. Plan
            state.phase = Phase.PLAN
            await emit_event(emit, "phase", phase=Phase.PLAN.value, label=Phase.PLAN.label)
            if done_at >= _phase_index("plan") and state.plan is not None:
                await emit_event(
                    emit, "status", message="复用已完成的研究方案（不重新规划）"
                )
            else:
                state.plan = await self.plan(state, emit=emit)
            await emit_event(emit, "plan", plan=state.plan.to_dict())
            await self._checkpoint(state, "plan", emit=emit)

            # ------------------------------------------- 2. 用户审批（人机协同）
            # 只有"停在规划之后"的运行才需要重新审批；已经进过执行阶段的说明早就批过了
            need_approval = approval is not None and self.config.agent.require_approval
            if need_approval and done_at > _phase_index("plan"):
                await emit_event(
                    emit, "status", message="上次已批准过，跳过审批直接继续"
                )
                need_approval = False
            if need_approval:
                state.phase = Phase.AWAIT_APPROVAL
                await emit_event(
                    emit,
                    "phase",
                    phase=Phase.AWAIT_APPROVAL.value,
                    label=Phase.AWAIT_APPROVAL.label,
                )
                await emit_event(
                    emit,
                    "awaiting_approval",
                    run_id=state.run_id,
                    plan=state.plan.to_dict(),
                )
                decision, feedback = await approval()
                logger.info("用户审批结果：%s", decision)

                if decision == "cancel":
                    state.phase = Phase.CANCELLED
                    await emit_event(emit, "status", message="用户已取消本次任务")
                    return state

                if decision == "revise" and feedback.strip():
                    state.plan.feedback = feedback.strip()
                    await emit_event(
                        emit, "status", message=f"按用户意见调整检索策略：{feedback[:FEEDBACK_TRUNCATE]}"
                    )
                    state.plan = await self.plan(state, emit=emit, feedback=feedback.strip())
                    await emit_event(emit, "plan", plan=state.plan.to_dict(), revised=True)

            # ------------------------------------------------- 3. Execute
            state.phase = Phase.EXECUTE
            await emit_event(
                emit, "phase", phase=Phase.EXECUTE.value, label=Phase.EXECUTE.label
            )
            if done_at >= _phase_index("execute") and state.papers:
                await emit_event(
                    emit,
                    "status",
                    message=f"复用已入库的 {len(state.papers)} 篇文献（不重新检索）",
                )
            else:
                await self.execute(state, emit=emit)
            await self._checkpoint(state, "execute", emit=emit)
            if not state.papers:
                state.phase = Phase.DONE
                hint = (
                    "本次用的是兜底方案（大模型未参与规划），检索式就是课题原文，"
                    "命中率通常很低——建议先确认 Ollama 可用后重新发起。"
                    if state.plan is not None and state.plan.degraded
                    else "请检查网络、检索式，或稍后重试"
                    "（Semantic Scholar 等在无 API Key 时限流较严）。"
                )
                await emit_event(
                    emit,
                    "error",
                    message=f"所有数据源均未返回结果。{hint}",
                )
                return state

            # ------------------------------------------------- 4. Reflect
            state.phase = Phase.REFLECT
            await emit_event(
                emit, "phase", phase=Phase.REFLECT.value, label=Phase.REFLECT.label
            )
            if done_at >= _phase_index("reflect") and state.critique is not None:
                await emit_event(emit, "status", message="复用已完成的文献评估")
            else:
                await self.reflect(state, emit=emit)
            await self._checkpoint(state, "reflect", emit=emit)

            # ---------------------------------------------- 5. Synthesize
            state.phase = Phase.SYNTHESIZE
            await emit_event(
                emit, "phase", phase=Phase.SYNTHESIZE.value, label=Phase.SYNTHESIZE.label
            )
            if done_at >= _phase_index("synthesize") and state.draft.strip():
                await emit_event(
                    emit, "status", message="复用已写好的草稿（不重新撰写）"
                )
            else:
                await self.synthesize(state, emit=emit)
            await self._checkpoint(state, "synthesize", emit=emit)

            # -------------------------------------------------- 6. Review
            state.phase = Phase.REVIEW
            await emit_event(emit, "phase", phase=Phase.REVIEW.value, label=Phase.REVIEW.label)
            if done_at >= _phase_index("review") and state.review is not None:
                await emit_event(emit, "status", message="复用已完成的自我审查")
            else:
                await self.review(state, emit=emit)
            await self._checkpoint(state, "review", emit=emit)

            # ------------------------------------------------ 7. 成稿落库
            await self.finalize(state, emit=emit)

        except asyncio.CancelledError:
            state.phase = Phase.CANCELLED
            await emit_event(emit, "status", message="任务已取消")
            raise
        except Exception as exc:  # 兜底：任何未预期异常都不应吞掉进度
            logger.exception("工作流异常终止")
            state.phase = Phase.ERROR
            state.add_error(f"{type(exc).__name__}: {exc}")
            await emit_event(emit, "error", message=f"工作流异常：{exc}")
        finally:
            state.finished_at = time.time()
            if state.phase not in {Phase.ERROR, Phase.CANCELLED}:
                state.phase = Phase.DONE
            await emit_event(
                emit,
                "done",
                run_id=state.run_id,
                phase=state.phase.value,
                elapsed_ms=state.elapsed_ms,
                summary=state.summary(),
            )
        return state

    # ============================================================ 节点实现
    async def _checkpoint(self, state: AgentState, phase: str, *, emit: Emitter | None) -> None:
        """阶段结束后打一个快照，供断点续跑与页面刷新后恢复。

        走 ``step`` 事件交给 Runtime 落库；事件本身不推给浏览器
        （快照可能包含上万字的草稿，没必要占用事件流）。
        """
        snapshot: dict[str, Any] = {}
        if state.plan is not None:
            snapshot["plan"] = state.plan.to_dict()
        if state.paper_ids:
            snapshot["paper_ids"] = list(state.paper_ids)
        elif state.papers:
            snapshot["paper_ids"] = [p.paper_id for p in state.papers if p.paper_id]
        if state.critique is not None and phase in {"reflect", "synthesize", "review"}:
            snapshot["critique"] = state.critique.to_dict()
        if state.draft and phase in {"synthesize", "review"}:
            snapshot["draft"] = state.draft
        if state.review is not None and phase == "review":
            snapshot["review"] = state.review.to_dict()
        if state.artifact_id:
            snapshot["artifact_id"] = state.artifact_id
        if state.errors:
            snapshot["errors"] = list(state.errors[:MAX_ERRORS_SNAPSHOT])
        await emit_event(emit, "step", phase=phase, snapshot=snapshot)

    async def plan(
        self, state: AgentState, *, emit: Emitter | None = None, feedback: str = ""
    ) -> ResearchPlan:
        """Plan 节点：LLM 生成检索策略与写作大纲。"""
        if state.offline or self.config.offline:
            plan = _offline_plan(state.topic)
            plan.degraded = True
            plan.degraded_reason = "离线模式：未调用大模型，使用模板大纲 + 课题关键词检索"
            await emit_event(emit, "status", message="离线模式：跳过 LLM 规划，使用模板大纲")
            return plan

        client = get_llm(self.config)
        try:
            await client.start()
            payload = await client.chat_json(
                [
                    {
                        "role": "user",
                        "content": plan_user(state.topic, extra=feedback, offline=False),
                    }
                ],
                system=PLAN_SYSTEM,
                temperature=LLM_TEMPERATURE_PLAN,
                max_tokens=LLM_MAX_TOKENS_PLAN,
                retries=2,
            )
            plan = ResearchPlan.from_dict(coerce_plan_payload(payload))
            if not isinstance(payload, dict):
                logger.warning(
                    "规划返回的 JSON 顶层是 %s 而不是对象，已尝试归一化",
                    type(payload).__name__,
                )
            if not plan.topic_zh:
                plan.topic_zh = state.topic
            if not plan.queries:
                plan.queries = [PlanQuery(query=plan.english_query() or state.topic)]
            if not plan.outline:
                plan = _with_default_outline(plan)
            plan.feedback = feedback
            await emit_event(
                emit,
                "status",
                message=(
                    f"规划完成：{len(plan.queries)} 条检索式，"
                    f"{len(plan.outline)} 个章节"
                ),
            )
            return plan
        except LLMError as exc:
            message = f"LLM 规划失败，已降级为关键词检索：{exc}"
            logger.warning(message)
            state.add_error(message)
            await emit_event(emit, "error", message=message)
            degraded_reason = f"大模型未能生成检索策略（{exc}），已改用课题关键词 + 模板大纲"
        except Exception as exc:  # pragma: no cover
            # 必须带 traceback：曾经只打了 %s，导致定位不到真实抛点
            logger.warning("规划异常：%s", exc, exc_info=True)
            state.add_error(f"规划异常：{exc}")
            degraded_reason = f"规划出错（{type(exc).__name__}: {exc}），已改用课题关键词 + 模板大纲"

        fallback = _offline_plan(state.topic)
        fallback.degraded = True
        fallback.degraded_reason = degraded_reason
        return fallback

    async def execute(self, state: AgentState, *, emit: Emitter | None = None) -> None:
        """Execute 节点：Scout 并发检索 + 入库 + Reader 预取全文。"""
        plan = state.plan or _offline_plan(state.topic)
        state.plan = plan

        result = await self.scout.search_plan(
            plan,
            sources=state.sources,
            limit=max(self.config.agent.writer_max_papers * 2, 40),
            per_source_limit=self.config.agent.max_papers_per_source,
            emit=emit,
            persist=not state.offline,
            embed=not state.offline,
        )

        state.papers = result.papers
        state.saved = result.saved
        state.embed_report = result.embedded
        state.search_stats = result.stats
        # 初始引用编号：1..n，与检索排序一致
        state.citation_map = {i: p for i, p in enumerate(result.papers, start=1)}

        await emit_event(
            emit,
            "papers",
            count=len(state.papers),
            items=[p.to_dict() for p in state.papers[:40]],
        )

        # 为最相关的开放获取文献预取全文（有全文的文献写作质量明显更高）
        if state.papers and not state.offline and self.config.agent.warm_fulltext:
            entries = state.select_papers(max_papers=self.config.agent.fulltext_top_n, use_critique=False)
            subset = [paper for _index, paper in entries]
            fetched = await self.reader.warm_fulltext(
                subset, limit=self.config.agent.fulltext_top_n, emit=emit
            )
            if fetched:
                await emit_event(emit, "status", message=f"已预取 {fetched} 篇开放获取全文")

    async def reflect(self, state: AgentState, *, emit: Emitter | None = None) -> CritiqueResult:
        """Reflect 节点：Critic 评估证据质量，并据结果重排引用编号。"""
        entries = sorted(state.citation_map.items())
        result = await self.critic.assess(state.topic, entries, emit=emit)
        state.critique = result

        usable = [a for a in result.assessments if a.use_in_review]
        message = (
            f"评估完成：{len(result.assessments)} 篇中 {len(usable)} 篇纳入写作"
            f"（整体证据质量：{result.evidence_quality}）"
        )
        await emit_event(emit, "status", message=message)
        await emit_event(emit, "critique", **result.to_dict())

        # 重新挑选并**重编号**：保证正文 [n] 与参考文献表严格一一对应
        selected = state.select_papers(
            max_papers=self.config.agent.writer_max_papers, use_critique=True
        )
        if not selected:
            selected = sorted(state.citation_map.items())[: self.config.agent.writer_max_papers]
            await emit_event(
                emit,
                "status",
                message="没有文献达到纳入标准，已放宽为按相关性取前若干篇",
            )
        state.renumber(selected)
        if state.critique:
            # 同步评估结果的编号，便于前端按新编号高亮
            by_paper = {p.paper_id: i for i, p in state.citation_map.items()}
            for assessment in state.critique.assessments:
                if assessment.paper_id in by_paper:
                    assessment.index = by_paper[assessment.paper_id]
        return result

    async def synthesize(self, state: AgentState, *, emit: Emitter | None = None) -> str:
        """Synthesize 节点：Writer 流式撰写，Formatter 统一引用格式。"""
        entries = sorted(state.citation_map.items())
        if not entries:
            state.draft = f"# {state.topic}\n\n> 没有可用文献，无法生成综述。\n"
            return state.draft

        # 大纲：优先用 LLM 结合真实材料细化
        fallback = state.plan.outline if state.plan else []
        state.outline = await self.writer.refine_outline(
            state.topic, entries, fallback=fallback, emit=emit
        )

        async def on_token(chunk: str) -> None:
            await emit_event(emit, "token", text=chunk)

        draft = await self.writer.write_review(
            state.topic,
            state.plan,
            entries,
            state.outline,
            emit=emit,
            on_token=on_token,
            total_min_chars=state.review_min_chars or self.config.agent.review_min_chars,
            total_max_chars=state.review_max_chars or self.config.agent.review_max_chars,
        )

        style = detect_style(state.citation_style)
        draft = self.formatter.restyle_inline(draft, entries, style)
        state.draft = draft
        return draft

    async def review(self, state: AgentState, *, emit: Emitter | None = None) -> ReviewResult:
        """Review 节点：规则体检 + LLM 自我批判，必要时做一轮自动修订。"""
        entries = sorted(state.citation_map.items())
        result = ReviewResult()

        # ---- 规则体检（不依赖 LLM，永远可执行）
        check = self.formatter.selfcheck(state.draft, entries)
        result.invalid_citations = list(check["citations"]["missing_from_list"])
        result.issues = list(check["issues"])
        result.verdict = check["verdict"]

        # ---- LLM 自我批判
        if not state.offline and not self.config.offline and state.draft.strip():
            try:
                llm_review = await self._llm_review(state, entries)
            except Exception as exc:
                logger.info("LLM 自我审查不可用：%s", exc)
                llm_review = None
            if llm_review is not None:
                result.verdict = llm_review.verdict or result.verdict
                result.score = llm_review.score
                result.strengths = llm_review.strengths
                known = {(i.get("type"), i.get("detail")) for i in result.issues}
                for issue in llm_review.issues:
                    key = (issue.get("type"), issue.get("detail"))
                    if key not in known:
                        result.issues.append(issue)

        result.score = result.score or (9.0 if result.verdict == "pass" else 6.0)
        state.review = result

        await emit_event(
            emit,
            "status",
            message=(
                f"自我审查：{result.verdict}（{result.score:.1f}/10），"
                f"发现 {len(result.issues)} 个问题"
            ),
        )
        await emit_event(emit, "review", **result.to_dict())

        # ---- 自动修订（默认一轮）
        if (
            self.config.agent.auto_revise
            and not result.passed
            and result.issues
            and not state.offline
            and state.draft.strip()
        ):
            await self._auto_revise(state, result, emit=emit)

        return result

    async def finalize(self, state: AgentState, *, emit: Emitter | None = None) -> None:
        """成稿：生成参考文献表并保存产物。"""
        entries = sorted(state.citation_map.items())
        style = detect_style(state.citation_style)
        references = self.formatter.build_references(entries, style, only_cited=state.draft)

        content = state.draft.rstrip()
        cited = set(extract_citations(state.draft))
        used = [(i, p) for i, p in entries if i in cited] or entries

        if references.strip():
            content += "\n\n## 参考文献\n\n" + references
        if state.review and state.review.issues:
            high = [i for i in state.review.issues if i.get("severity") == "high"]
            if high:
                content += "\n\n---\n\n> **自动审查提示**：" + "；".join(
                    str(i.get("detail")) for i in high[:3]
                )

        artifact_id = None
        if state.session_id is not None:
            try:
                artifact_id = save_artifact(
                    title=_artifact_title(state.topic),
                    content=content,
                    kind="review",
                    fmt="markdown",
                    session_id=state.session_id,
                    meta={
                        "topic": state.topic,
                        "run_id": state.run_id,
                        "paper_count": len(used),
                        "style": style,
                        "evidence_quality": state.critique.evidence_quality
                        if state.critique
                        else "",
                        "references": [
                            {"index": i, "paper_id": p.paper_id, "title": p.title}
                            for i, p in used
                        ],
                    },
                    db=self.db,
                )
                add_message(
                    state.session_id,
                    "assistant",
                    content,
                    meta={"run_id": state.run_id, "artifact_id": artifact_id, "kind": "review"},
                    db=self.db,
                )
            except Exception as exc:  # pragma: no cover
                logger.warning("产物保存失败：%s", exc)
                state.add_error(f"产物保存失败：{exc}")

        state.artifact_id = artifact_id
        await emit_event(
            emit,
            "artifact",
            artifact_id=artifact_id,
            title=_artifact_title(state.topic),
            content=content,
            fmt="markdown",
            reference_entries=self.formatter.reference_entries(entries, style),
            style=style,
        )

    # ------------------------------------------------------------ 内部方法
    async def _llm_review(
        self, state: AgentState, entries: Sequence[tuple[int, Paper]]
    ) -> ReviewResult | None:
        client = get_llm(self.config)
        await client.start()
        digest = build_context_digest(entries, max_abstract=DIGEST_MAX_ABSTRACT_REVIEW)
        payload = await client.chat_json(
            [
                {
                    "role": "user",
                    "content": reflect_user(
                        state.topic,
                        state.draft[:DRAFT_TRUNCATE_REVIEW],
                        [index for index, _ in entries],
                        digest,
                    ),
                }
            ],
            system=REFLECT_SYSTEM,
            temperature=LLM_TEMPERATURE_REVIEW,
            max_tokens=LLM_MAX_TOKENS_REVIEW,
            retries=1,
        )
        if not isinstance(payload, dict):
            return None

        issues = [i for i in (payload.get("issues") or []) if isinstance(i, dict)]
        for issue in issues:
            issue.setdefault("severity", "medium")
            issue.setdefault("type", "其他")
        try:
            score = float(payload.get("score") or 0)
        except (TypeError, ValueError):
            score = 0.0

        return ReviewResult(
            verdict=str(payload.get("verdict") or "pass").lower(),
            score=score,
            issues=issues,
            strengths=[str(s) for s in (payload.get("strengths") or []) if s],
        )

    async def _auto_revise(
        self, state: AgentState, review: ReviewResult, *, emit: Emitter | None
    ) -> None:
        """按审查意见做一轮自动修订（有界，避免无限循环）。"""
        await emit_event(emit, "status", message="正在按审查意见自动修订草稿…")
        entries = sorted(state.citation_map.items())
        digest = build_context_digest(entries, max_abstract=DIGEST_MAX_ABSTRACT_REVISE)
        issues_text = "\n".join(
            f"- [{i.get('severity')}] {i.get('detail')} → {i.get('suggestion')}"
            for i in review.issues[:MAX_ISSUES_REVISE]
        )
        prompt = (
            f"研究课题：{state.topic}\n\n"
            f"以下是综述草稿，以及审稿意见。请**只做必要修改**，保持原有结构与引用编号不变，"
            f"输出修订后的完整草稿。\n\n"
            f"审稿意见：\n{issues_text}\n\n"
            f"可引用的文献材料（编号必须与草稿一致）：\n{digest}\n\n"
            f"草稿：\n{state.draft[:DRAFT_TRUNCATE_REVISE]}\n\n"
            "请直接输出修订后的完整 Markdown 草稿，不要解释修改内容。"
        )
        try:
            client = get_llm(self.config)
            await client.start()
            revised = await client.chat(
                [{"role": "user", "content": prompt}],
                system="你是医学综述编辑，只做事实性修正与结构优化，绝不新增材料中不存在的文献或数据。",
                temperature=LLM_TEMPERATURE_REVISE,
                max_tokens=min(4096, self.config.agent.context_char_budget // 6),
            )
        except LLMError as exc:
            await emit_event(emit, "error", message=f"自动修订失败：{exc}")
            return

        revised = revised.strip()
        if not revised or len(revised) < len(state.draft) * REVISE_MIN_LENGTH_RATIO:
            await emit_event(emit, "status", message="自动修订结果不完整，已保留原草稿")
            return

        from .writer import _sanitize  # noqa: PLC2701 - 复用引用清洗逻辑

        state.draft = _sanitize(revised, [index for index, _ in entries])
        if state.review is not None:
            check = self.formatter.selfcheck(state.draft, entries)
            state.review.verdict = check["verdict"]
            state.review.issues = list(check["issues"])
            state.review.invalid_citations = list(check["citations"]["missing_from_list"])
        await emit_event(emit, "status", message="自动修订完成，已更新草稿")


# ------------------------------------------------------------------ 辅助
#: 阶段先后顺序，断点续跑时用它判断"哪些阶段已经做完"。
_PHASE_SEQUENCE = ("plan", "execute", "reflect", "synthesize", "review")


def _phase_index(phase: str) -> int:
    """阶段在流水线中的序号；空字符串（全新运行）返回 -1。"""
    name = (phase or "").strip()
    if not name:
        return -1
    try:
        return _PHASE_SEQUENCE.index(name)
    except ValueError:
        return -1


def _with_default_outline(plan: ResearchPlan) -> ResearchPlan:
    from .writer import _default_outline

    plan.outline = _default_outline()
    return plan


def _offline_plan(topic: str) -> ResearchPlan:
    """离线 / LLM 不可用时的兜底计划：直接用课题本身检索 + 模板大纲。"""
    from .writer import _default_outline

    return ResearchPlan(
        topic_zh=topic,
        topic_en=topic,
        queries=[PlanQuery(query=topic, sources=[], rationale="LLM 不可用，直接使用课题关键词检索")],
        outline=_default_outline(),
    )


def _artifact_title(topic: str) -> str:
    base = (topic or "综述").strip().rstrip("。.")
    return f"{base}：研究进展综述"
