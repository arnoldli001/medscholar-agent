"""多数据源统一调度：并发调用、隔离单源失败（记入 SourceStatus）、合并去重。"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import AppConfig, get_config
from ..dedupe import merge_papers
from ..models import Paper, SOURCE_LABELS
from .arxiv_client import ArxivClient
from .base import BaseClient, SearchFilters, SourceError
from .cnki_client import CnkiClient
from .core_client import CoreClient
from .crossref_client import CrossrefClient
from .doaj_client import DoajClient
from .europepmc_client import EuropePMCClient
from .openalex_client import OpenAlexClient
from .pubmed_client import PubMedClient
from .semantic_scholar_client import SemanticScholarClient

logger = logging.getLogger(__name__)

__all__ = [
    "SourceStatus",
    "SearchOutcome",
    "SourceRegistry",
    "get_registry",
    "close_registry",
    "search_all",
    "SOURCE_ALIASES",
    "DEFAULT_SOURCES",
    "ALL_SOURCES",
    "CHINESE_SOURCES",
    "normalize_source_name",
]

_CLIENT_TYPES: dict[str, type[BaseClient]] = {
    "pubmed": PubMedClient,
    "europepmc": EuropePMCClient,
    "semantic_scholar": SemanticScholarClient,
    "openalex": OpenAlexClient,
    "crossref": CrossrefClient,
    "arxiv": ArxivClient,
    "cnki": CnkiClient,
    "doaj": DoajClient,
    "core": CoreClient,
}

#: 用户/前端可能写出的各种别名
SOURCE_ALIASES: dict[str, str] = {
    "pubmed": "pubmed",
    "pm": "pubmed",
    "ncbi": "pubmed",
    "europepmc": "europepmc",
    "europe_pmc": "europepmc",
    "epmc": "europepmc",
    "pmc": "europepmc",
    "s2": "semantic_scholar",
    "semanticscholar": "semantic_scholar",
    "semantic_scholar": "semantic_scholar",
    "openalex": "openalex",
    "oa": "openalex",
    "crossref": "crossref",
    "cr": "crossref",
    "arxiv": "arxiv",
    "cnki": "cnki",
    "知网": "cnki",
    "doaj": "doaj",
    "开放获取期刊": "doaj",
    "core": "core",
    "机构库": "core",
}

#: 默认检索的数据源（覆盖面 + 稳定性最佳的组合）
DEFAULT_SOURCES: tuple[str, ...] = (
    "pubmed",
    "europepmc",
    "openalex",
    "crossref",
    "semantic_scholar",
)

#: 全部可用数据源
ALL_SOURCES: tuple[str, ...] = tuple(_CLIENT_TYPES)

#: 中文课题建议追加的数据源（CNKI 默认关闭，见 api/cnki_client.py）
CHINESE_SOURCES: tuple[str, ...] = ("cnki", "openalex")


def normalize_source_name(name: str) -> str:
    """把别名统一成内部客户端名；未知名称原样返回小写形式。"""
    key = str(name).strip().lower()
    return SOURCE_ALIASES.get(key, key)


@dataclass(slots=True)
class SourceStatus:
    """单个数据源一次检索的执行结果。"""

    name: str
    label: str
    ok: bool
    count: int = 0
    error: str = ""
    duration_ms: int = 0
    skipped: bool = False

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "ok": self.ok,
            "count": self.count,
            "error": self.error,
            "duration_ms": self.duration_ms,
            "skipped": self.skipped,
        }


@dataclass(slots=True)
class SearchOutcome:
    """一次跨库检索的完整结果。"""

    query: str
    papers: list[Paper] = field(default_factory=list)
    statuses: list[SourceStatus] = field(default_factory=list)
    duration_ms: int = 0
    raw_count: int = 0

    @property
    def errors(self) -> dict[str, str]:
        return {s.name: s.error for s in self.statuses if s.error}

    @property
    def ok_sources(self) -> list[str]:
        return [s.name for s in self.statuses if s.ok]

    def summary(self) -> str:
        parts = [f"「{self.query}」跨库检索：{len(self.papers)} 篇（原始 {self.raw_count} 条）"]
        for status in self.statuses:
            if status.skipped:
                parts.append(f"  · {status.label}：跳过")
            elif status.ok:
                parts.append(f"  · {status.label}：{status.count} 篇（{status.duration_ms} ms）")
            else:
                parts.append(f"  · {status.label}：失败 —— {status.error}")
        return "\n".join(parts)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "count": len(self.papers),
            "raw_count": self.raw_count,
            "duration_ms": self.duration_ms,
            "sources": [s.to_dict() for s in self.statuses],
        }


class SourceRegistry:
    """数据源客户端注册表（懒创建 + 复用 HTTP 连接池）。"""

    def __init__(self, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self._clients: dict[str, BaseClient] = {}
        self._lock = asyncio.Lock()
        # P1-11：每个数据源一个 Bulkhead（concurrency=1 即"该源同时只能一个请求"）。
        # 防止某源被慢响应占满 asyncio 任务预算；批量限流且天然隔离。
        # 这里的 limit=1 是合理默认（已有 TokenBucket 限速）；后续可按源配置。
        self._bulkheads: dict[str, Any] = {}
        from ..platform.resilience import Bulkhead

        for name in ALL_SOURCES:
            self._bulkheads[name] = Bulkhead(name=f"source:{name}", limit=1)

    # ------------------------------------------------------------ 客户端获取
    def _make_client(self, name: str) -> BaseClient:
        client_type = _CLIENT_TYPES.get(name)
        if client_type is None:
            raise SourceError(name, f"未知数据源：{name}（可选：{', '.join(ALL_SOURCES)}）")
        return client_type(config=self.config)

    async def client(self, name: str) -> BaseClient:
        key = normalize_source_name(name)
        async with self._lock:
            client = self._clients.get(key)
            if client is None:
                client = self._make_client(key)
                await client.start()
                self._clients[key] = client
            return client

    def known(self, name: str) -> bool:
        return normalize_source_name(name) in _CLIENT_TYPES

    def describe(self) -> list[dict[str, Any]]:
        out = []
        for name in ALL_SOURCES:
            settings = self.config.sources.get(name)
            out.append(
                {
                    "name": name,
                    "label": SOURCE_LABELS.get(name, name),
                    "enabled": settings.enabled,
                    "has_api_key": bool(settings.api_key),
                    "rps": settings.rps,
                }
            )
        return out

    # ---------------------------------------------------------------- 检索
    def select_sources(self, sources: Sequence[str] | None = None) -> list[str]:
        """确定本次要用的数据源：显式指定优先，否则用默认组合。"""
        if sources:
            chosen = []
            for item in sources:
                key = normalize_source_name(item)
                if key in _CLIENT_TYPES and key not in chosen:
                    chosen.append(key)
            return chosen
        return [
            name
            for name in DEFAULT_SOURCES
            if self.config.sources.get(name).enabled
        ] or list(DEFAULT_SOURCES)

    async def search(
        self,
        query: str,
        *,
        sources: Sequence[str] | None = None,
        limit: int = 20,
        per_source_limit: int | None = None,
        filters: SearchFilters | None = None,
        offline: bool | None = None,
    ) -> SearchOutcome:
        """并发检索多个数据源，去重合并后返回。

        offline 必须由调用方传入：注册表是跨会话单例，而每次运行的离线设置不同，
        否则"离线运行"仍会联网。
        """
        started = time.perf_counter()
        names = self.select_sources(sources)
        is_offline = self.config.offline if offline is None else offline

        if is_offline:
            return SearchOutcome(
                query=query,
                statuses=[
                    SourceStatus(
                        name=name,
                        label=SOURCE_LABELS.get(name, name),
                        ok=False,
                        error="离线模式已开启，跳过联网检索",
                        skipped=True,
                    )
                    for name in names
                ],
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        # P1-12：按每个源的 max_results 取，不再"pubmed 的值套所有源"。
        async def per_source_for(name: str) -> int:
            if per_source_limit is not None:
                return int(per_source_limit)
            cfg = self.config.sources.get(name)
            return int(cfg.max_results) if cfg else 50

        async def run(name: str) -> tuple[str, list[Paper] | Exception, int]:
            t0 = time.perf_counter()
            # P1-11：Bulkhead 包裹；同源并发请求只能 1 个，其他等位。
            async with self._bulkheads[name]:
                try:
                    client = await self.client(name)
                    if not client.enabled():
                        return name, [], int((time.perf_counter() - t0) * 1000)
                    papers = await client.search(
                        query, limit=await per_source_for(name), filters=filters
                    )
                    return name, papers, int((time.perf_counter() - t0) * 1000)
                except Exception as exc:  # 单源失败必须被隔离
                    logger.info("数据源 %s 检索失败：%s", name, exc)
                    return name, exc, int((time.perf_counter() - t0) * 1000)

        # P1-11：gather 加 wait_for；任一源超时不会让整轮等死。
        # 总超时：取各源 timeout 的 2 倍 + 30s（留给串联重试）。
        max_source_timeout = max(
            (self.config.sources.get(n).timeout for n in names if self.config.sources.get(n)),
            default=30.0,
        )
        timeout_total = max_source_timeout * 2 + 30.0
        try:
            results = await asyncio.wait_for(
                asyncio.gather(*(run(name) for name in names)),
                timeout=timeout_total,
            )
        except asyncio.TimeoutError:
            # P1-11：超时上报为各源 SourceError（timeout 类别）
            statuses = [
                SourceStatus(
                    name=name,
                    label=SOURCE_LABELS.get(name, name),
                    ok=False,
                    error=f"整体超时（{timeout_total:.0f}s）",
                    duration_ms=int((time.perf_counter() - started) * 1000),
                )
                for name in names
            ]
            return SearchOutcome(
                query=query,
                statuses=statuses,
                duration_ms=int((time.perf_counter() - started) * 1000),
            )

        statuses: list[SourceStatus] = []
        bucket: list[Paper] = []
        for name, payload, elapsed in results:
            label = SOURCE_LABELS.get(name, name)
            if isinstance(payload, Exception):
                statuses.append(
                    SourceStatus(
                        name=name,
                        label=label,
                        ok=False,
                        error=str(payload) if str(payload) else type(payload).__name__,
                        duration_ms=elapsed,
                    )
                )
                continue
            statuses.append(
                SourceStatus(
                    name=name, label=label, ok=True, count=len(payload), duration_ms=elapsed
                )
            )
            bucket.extend(payload)

        merged = merge_papers(bucket)[: max(limit * 2, limit)]

        return SearchOutcome(
            query=query,
            papers=merged,
            statuses=statuses,
            duration_ms=int((time.perf_counter() - started) * 1000),
            raw_count=len(bucket),
        )

    # ---------------------------------------------------------------- 收尾
    async def close(self) -> None:
        for client in list(self._clients.values()):
            try:
                await client.close()
            except Exception as exc:  # pragma: no cover
                logger.debug("关闭数据源 %s 失败：%s", client.name, exc)
        self._clients.clear()


# ------------------------------------------------------------ 全局单例
_REGISTRY: SourceRegistry | None = None


def get_registry(config: AppConfig | None = None) -> SourceRegistry:
    """获取全局数据源注册表（复用连接池）。"""
    global _REGISTRY
    if _REGISTRY is None:
        _REGISTRY = SourceRegistry(config=config)
    return _REGISTRY


async def close_registry() -> None:
    global _REGISTRY
    if _REGISTRY is not None:
        await _REGISTRY.close()
        _REGISTRY = None


async def search_all(
    query: str,
    *,
    sources: Sequence[str] | None = None,
    limit: int = 20,
    per_source_limit: int | None = None,
    filters: SearchFilters | None = None,
    config: AppConfig | None = None,
    offline: bool | None = None,
) -> SearchOutcome:
    """便捷函数：用全局注册表做一次跨库检索。"""
    registry = get_registry(config)
    return await registry.search(
        query,
        sources=sources,
        limit=limit,
        per_source_limit=per_source_limit,
        filters=filters,
        offline=offline,
    )
