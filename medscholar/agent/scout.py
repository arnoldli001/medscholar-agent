"""Scout Agent —— 检索。

职责（需求 3.1）：

* 按 Plan 给出的检索式**并发**调用各学术数据源；
* 跨库去重合并，按相关性整理出候选文献池；
* 落库（含增量向量嵌入），为后续 Critic / Writer 提供本地可检索的知识库；
* 对开放获取文献可选抓取全文，供 Reader 深入解析。

单个数据源失败会被隔离并上报，绝不中断整条流水线。
"""

from __future__ import annotations

import asyncio
import logging
from dataclasses import dataclass, field
from typing import Any, Awaitable, Callable, Sequence

from ..api import SearchFilters, SourceRegistry, get_registry
from ..config import AppConfig, get_config
from ..db.connect import Database, get_db
from ..db.repo import insert_paper
from ..dedupe import merge_papers
from ..models import Paper, SearchLogEntry
from ..db.repo import log_search
from .state import PlanQuery, ResearchPlan

logger = logging.getLogger(__name__)

__all__ = ["ScoutResult", "ScoutAgent", "Emitter", "emit_event"]

Emitter = Callable[[str, dict[str, Any]], Awaitable[None]]

#: 同时执行的检索式数量。每个检索式内部还会并发查 5~6 个数据源，
#: 所以这个值不能太大，否则容易触发上游限流（Semantic Scholar 尤其严格）。
_MAX_CONCURRENT_QUERIES = 3


async def emit_event(emit: Emitter | None, event_type: str, **data: Any) -> None:
    """安全地推送事件（emit 为空或抛错都不应影响主流程）。"""
    if emit is None:
        return
    try:
        await emit(event_type, data)
    except Exception as exc:  # pragma: no cover - UI 断开连接等
        logger.debug("事件推送失败（%s）：%s", event_type, exc)


@dataclass(slots=True)
class ScoutResult:
    """Scout 一轮执行的产出。"""

    papers: list[Paper] = field(default_factory=list)
    paper_ids: list[int] = field(default_factory=list)
    stats: list[dict[str, Any]] = field(default_factory=list)
    saved: dict[str, int] = field(default_factory=dict)
    embedded: dict[str, Any] = field(default_factory=dict)
    fulltext_fetched: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "count": len(self.papers),
            "saved": dict(self.saved),
            "embedded": dict(self.embedded),
            "sources": list(self.stats),
            "fulltext_fetched": self.fulltext_fetched,
        }


class ScoutAgent:
    """检索智能体。"""

    def __init__(
        self,
        *,
        config: AppConfig | None = None,
        registry: SourceRegistry | None = None,
        db: Database | None = None,
    ) -> None:
        self.config = config or get_config()
        self.db = db or get_db()
        self._registry = registry

    @property
    def registry(self) -> SourceRegistry:
        if self._registry is None:
            self._registry = get_registry(self.config)
        return self._registry

    # ------------------------------------------------------------------ 主流程
    async def search_plan(
        self,
        plan: ResearchPlan,
        *,
        sources: Sequence[str] | None = None,
        per_source_limit: int | None = None,
        limit: int = 60,
        emit: Emitter | None = None,
        persist: bool = True,
        embed: bool = True,
    ) -> ScoutResult:
        """执行 Plan 中的全部检索式。"""
        queries = self._queries_to_run(plan, sources)
        result = ScoutResult()
        if not queries:
            await emit_event(emit, "status", message="没有可执行的检索式")
            return result

        cfg = self.config
        per_source = per_source_limit or cfg.agent.max_papers_per_source
        collected: list[Paper] = []

        # 多条检索式**并发**执行。这是纯粹的等待网络，不占 GPU，所以并发是净收益：
        # 4 条检索式 × 每个数据源 1~3 秒网络往返，串行要等一整轮，并发只等最慢的那条。
        # 单个检索式内部的多数据源并发由 SourceRegistry 负责。
        await emit_event(
            emit,
            "status",
            message=f"并发检索 {len(queries)} 条检索式（{', '.join(q.query[:32] for q in queries[:3])}"
            + ("…" if len(queries) > 3 else "")
            + "）",
        )

        filters = SearchFilters(
            year_from=plan.year_from,
            year_to=plan.year_to,
            sort="relevance",
        )

        async def run_query(plan_query: PlanQuery) -> Any:
            async with search_semaphore:
                return await self.registry.search(
                    plan_query.query,
                    sources=plan_query.sources or list(sources or []),
                    limit=limit,
                    per_source_limit=per_source,
                    filters=filters,
                    # 注册表是跨运行共享的单例，离线开关必须逐次传入
                    offline=self.config.offline,
                )

        search_semaphore = asyncio.Semaphore(_MAX_CONCURRENT_QUERIES)
        outcomes = await asyncio.gather(
            *(run_query(q) for q in queries), return_exceptions=True
        )

        # 结果按检索式原顺序整理：日志、统计、界面事件都保持确定性顺序
        for index, (plan_query, outcome) in enumerate(zip(queries, outcomes), start=1):
            label = plan_query.query
            if isinstance(outcome, BaseException):
                message = f"检索式「{label[:40]}」执行失败：{outcome}"
                logger.warning(message, exc_info=outcome)
                await emit_event(emit, "error", message=message)
                result.stats.append({"query": label, "error": str(outcome)})
                continue
            collected.extend(outcome.papers)

            for status in outcome.statuses:
                entry = {
                    "query": label,
                    "source": status.name,
                    "label": status.label,
                    "ok": status.ok,
                    "count": status.count,
                    "error": status.error,
                    "duration_ms": status.duration_ms,
                    "skipped": status.skipped,
                }
                result.stats.append(entry)
                log_search(
                    SearchLogEntry(
                        query=label,
                        source=status.name,
                        result_count=status.count,
                        duration_ms=status.duration_ms,
                        error=status.error or ("skipped" if status.skipped else ""),
                    ),
                    db=self.db,
                )

            await emit_event(
                emit,
                "search_result",
                query=label,
                rationale=plan_query.rationale,
                count=len(outcome.papers),
                raw_count=outcome.raw_count,
                duration_ms=outcome.duration_ms,
                sources=[s.to_dict() for s in outcome.statuses],
                items=[p.to_dict() for p in outcome.papers[:12]],
            )
            if outcome.errors:
                for name, message in outcome.errors.items():
                    result.stats.append({"query": label, "source": name, "error": message})

        # ---- 跨检索式去重合并（同一篇文献可能被多条检索式命中）
        merged = merge_papers(collected)
        await emit_event(
            emit,
            "status",
            message=f"检索完成：原始 {len(collected)} 条 → 去重后 {len(merged)} 篇",
        )

        if persist and merged:
            merged, result.saved, result.embedded = await self._persist(
                merged, emit=emit, embed=embed
            )

        result.papers = merged[: limit * 3]
        result.paper_ids = [p.paper_id or 0 for p in result.papers]
        return result

    # ------------------------------------------------------------------ 内部
    def _queries_to_run(
        self, plan: ResearchPlan, sources: Sequence[str] | None
    ) -> list[PlanQuery]:
        """整理检索式；Plan 为空时用课题本身兜底。"""
        queries = [q for q in plan.queries if q.query.strip()]
        if not queries:
            fallback = plan.english_query() or plan.chinese_query()
            if fallback:
                queries = [PlanQuery(query=fallback, sources=list(sources or []), rationale="兜底检索")]
        return queries

    async def _persist(
        self, papers: list[Paper], *, emit: Emitter | None, embed: bool
    ) -> tuple[list[Paper], dict[str, int], dict[str, Any]]:
        """落库并回填 ``paper_id``。"""
        # 注意：这里必须用 **异步** 版本。run_embedding_pipeline 是同步封装，
        # await 它只会得到 "object EmbeddingReport can't be used in 'await' expression"。
        from ..embedding.pipeline import run_embedding_pipeline_async

        new_count = 0
        updated_count = 0
        errors: list[str] = []
        new_ids: list[int] = []

        for paper in papers:
            try:
                paper_id, created = insert_paper(paper, db=self.db)
            except Exception as exc:
                errors.append(f"{paper.title[:50]}: {exc}")
                continue
            paper.paper_id = paper_id
            if created:
                new_count += 1
                new_ids.append(paper_id)
            else:
                updated_count += 1

        await emit_event(
            emit,
            "status",
            message=f"已入库：新增 {new_count} 篇，更新 {updated_count} 篇",
        )

        embedded: dict[str, Any] = {}
        if embed and new_ids and self.config.embedding.auto_embed:
            await emit_event(
                emit, "status", message=f"正在为 {len(new_ids)} 篇新文献生成向量…"
            )
            try:
                report = await run_embedding_pipeline_async(
                    ids=new_ids, db=self.db, config=self.config
                )
                embedded = report.to_dict()
                await emit_event(emit, "status", message=report.summary())
            except Exception as exc:
                logger.warning("向量生成失败：%s", exc)
                embedded = {"failed": len(new_ids), "errors": [str(exc)]}
                await emit_event(
                    emit,
                    "error",
                    message=f"向量生成失败（不影响关键词检索）：{exc}",
                )

        return papers, {"new": new_count, "updated": updated_count, "errors_seen": len(errors)}, embedded
