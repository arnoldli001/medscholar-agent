"""嵌入生成管道（需求文档「指令3」的落地实现）。

职责：

1. 读取 ``papers`` 表中**尚无向量**的文献（增量，不重复计算）
2. 用配置的嵌入模型对「标题 + MeSH + 摘要」生成 768 维向量
3. 批量写入 ``paper_embeddings``（vec0 虚拟表或回退表）
4. 提供查询侧嵌入 :func:`embed_query`，供混合检索使用

非线性流程：**嵌入失败绝不阻断入库**，只记录到报告里，下次运行会重试。
"""

from __future__ import annotations

import asyncio
import logging
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from typing import Any, Sequence

from ..config import AppConfig, get_config
from ..db.connect import Database, get_db
# 直接依赖数据层的具体模块，而不是 db.repo 门面：门面是给历史调用点用的，
# 新代码走子模块可以让依赖图更精确（也让架构校验看得清真实边界）。
from ..db.repositories.embeddings import papers_missing_embeddings, store_embeddings
from ..db.repositories.papers import get_paper
from .providers import EmbeddingProvider, get_provider

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingReport",
    "embed_paper",
    "embed_papers",
    "embed_query",
    "run_embedding_pipeline",
    "run_embedding_pipeline_async",
    "embedding_status",
]


@dataclass(slots=True)
class EmbeddingReport:
    """一次嵌入批处理的统计。"""

    requested: int = 0
    embedded: int = 0
    skipped: int = 0
    failed: int = 0
    duration_ms: int = 0
    provider: str = ""
    model: str = ""
    dim: int = 0
    errors: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "requested": self.requested,
            "embedded": self.embedded,
            "skipped": self.skipped,
            "failed": self.failed,
            "duration_ms": self.duration_ms,
            "provider": self.provider,
            "model": self.model,
            "dim": self.dim,
            "errors": self.errors[:10],
        }

    def summary(self) -> str:
        return (
            f"嵌入完成：成功 {self.embedded} / 跳过 {self.skipped} / 失败 {self.failed}"
            f"（{self.provider}/{self.model}，{self.dim} 维，{self.duration_ms} ms）"
        )


def _get_provider(config: AppConfig | None = None) -> EmbeddingProvider:
    return get_provider(config or get_config())


# ---------------------------------------------------------------- 核心批处理
async def _embed_ids(
    ids: Sequence[int],
    *,
    db: Database,
    provider: EmbeddingProvider,
    batch_size: int | None = None,
    config: AppConfig | None = None,
) -> EmbeddingReport:
    cfg = config or get_config()
    report = EmbeddingReport(
        requested=len(ids),
        provider=provider.name,
        model=provider.model,
        dim=provider.dim,
    )
    if not ids:
        return report

    size = batch_size or cfg.embedding.batch_size
    started = time.perf_counter()

    for start in range(0, len(ids), size):
        chunk = list(ids[start : start + size])
        texts: list[str] = []
        valid: list[int] = []
        for paper_id in chunk:
            paper = get_paper(paper_id, db=db)
            if paper is None:
                report.skipped += 1
                continue
            text = paper.embed_text
            if not text:
                report.skipped += 1
                continue
            texts.append(text)
            valid.append(paper_id)

        if not texts:
            continue

        try:
            vectors = await provider.embed(texts)
        except Exception as exc:
            report.failed += len(texts)
            message = f"{type(exc).__name__}: {exc}"
            report.errors.append(message)
            logger.warning("嵌入批次失败（%d 篇）：%s", len(texts), message)
            continue

        if len(vectors) != len(valid):
            report.failed += len(texts)
            report.errors.append(
                f"向量数量不匹配：请求 {len(valid)}，返回 {len(vectors)}"
            )
            continue

        store_embeddings(list(zip(valid, vectors)), db=db)
        report.embedded += len(valid)

        if cfg.embedding.idle_seconds > 0:
            await asyncio.sleep(cfg.embedding.idle_seconds)

    report.duration_ms = int((time.perf_counter() - started) * 1000)
    return report


async def _sync_embedding_dim(
    database: Database, provider: EmbeddingProvider, cfg: AppConfig
) -> None:
    """探测模型实际维度并与向量表对齐。

    换嵌入模型（例如从 768 维的 ``nomic-embed-text`` 换成 1024 维的 ``bge-m3``）
    时只需改 ``embedding.model``，维度会自动校正并重建向量表，
    不会出现"维度不符"的报错。
    """
    if database.get_meta("dim_probe_done") == f"{provider.name}:{provider.model}":
        return
    try:
        vector = await provider.embed_one("维度探测")
    except Exception as exc:
        logger.debug("维度探测跳过（%s）：%s", provider.model, exc)
        return
    actual = len(vector)
    if actual and actual != int(cfg.embedding.dim):
        logger.warning(
            "嵌入模型 %s 实际维度 %d 与配置 %d 不一致，自动对齐",
            provider.model,
            actual,
            cfg.embedding.dim,
        )
        cfg.embedding.dim = actual
        database.config.embedding.dim = actual
        database.sync_embedding_dim(actual)
    database.set_meta("dim_probe_done", f"{provider.name}:{provider.model}")


async def run_embedding_pipeline_async(
    *,
    ids: Sequence[int] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    db: Database | None = None,
    config: AppConfig | None = None,
    force: bool = False,
) -> EmbeddingReport:
    """增量嵌入管道（异步）。

    Args:
        ids: 只处理这些文献；``None`` 表示扫描全库缺失向量的文献。
        limit: 单次最多处理多少篇（防止首次跑满 CPU）。
        force: 为 ``True`` 时忽略"已有向量"，对 ``ids`` 强制重算。
    """
    database = db or get_db()
    cfg = config or get_config()
    provider = _get_provider(cfg)
    await _sync_embedding_dim(database, provider, cfg)

    if ids is not None and force:
        targets = [int(i) for i in ids]
    else:
        targets = papers_missing_embeddings(
            limit=limit or 10_000, ids=ids, db=database
        )

    return await _embed_ids(
        targets, db=database, provider=provider, batch_size=batch_size, config=cfg
    )


def run_embedding_pipeline(
    *,
    ids: Sequence[int] | None = None,
    limit: int | None = None,
    batch_size: int | None = None,
    db: Database | None = None,
    config: AppConfig | None = None,
    force: bool = False,
) -> EmbeddingReport:
    """同步入口；在已有事件循环中调用时自动切到独立线程执行。"""
    kwargs = {
        "ids": ids,
        "limit": limit,
        "batch_size": batch_size,
        "db": db,
        "config": config,
        "force": force,
    }
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(run_embedding_pipeline_async(**kwargs))

    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="embed") as pool:
        return pool.submit(lambda: asyncio.run(run_embedding_pipeline_async(**kwargs))).result()


# ---------------------------------------------------------------- 单篇/查询
async def embed_papers(
    ids: Sequence[int],
    *,
    db: Database | None = None,
    config: AppConfig | None = None,
    force: bool = True,
) -> EmbeddingReport:
    """为指定文献生成（或重算）向量。"""
    database = db or get_db()
    cfg = config or get_config()
    targets = [int(i) for i in ids]
    if not force:
        targets = papers_missing_embeddings(limit=len(targets), ids=targets, db=database)
    return await _embed_ids(targets, db=database, provider=_get_provider(cfg), config=cfg)


def embed_paper(
    paper_id: int, *, db: Database | None = None, config: AppConfig | None = None
) -> EmbeddingReport:
    """同步为单篇文献生成向量（供入库后自动调用）。"""
    return run_embedding_pipeline(ids=[paper_id], db=db, config=config, force=True)


async def embed_query(
    text: str, *, config: AppConfig | None = None
) -> list[float] | None:
    """把检索词转成查询向量；提供方不可用时返回 ``None``（调用方退化为纯 BM25）。

    **带缓存**：同一个检索词会被反复嵌入（用户在左栏改一个词、切换筛选、
    或者点两次检索），而每次嵌入都是一次真实的模型/网络调用。
    缓存按 ``(provider, model, text)`` 做键 —— 必须带上模型名，
    否则换了嵌入模型之后会拿到上一个模型的向量，而**维度相同、数值不同**的向量
    不会报错，只会让检索结果悄悄变差。

    失效策略：TTL 默认 1 小时（上面的键已经含模型名，所以换模型天然不会串），
    且进程重启即失效 —— 查询向量便宜、且没有跨进程一致性问题。
    """
    from ..platform.cache import cache_registry

    text = (text or "").strip()
    if not text:
        return None
    provider = _get_provider(config)
    cache = cache_registry("embed_query", ttl=3600.0, maxsize=256)
    cache_key = f"{provider.name}|{provider.model}|{text}"

    cached_vector = cache.get(cache_key)
    if cached_vector is not None:
        return list(cached_vector)

    try:
        vector = await provider.embed_one(text)
    except Exception as exc:
        logger.warning("查询向量生成失败，将退化为纯关键词检索：%s", exc)
        return None
    if vector:
        cache.set(cache_key, list(vector))
    return vector


async def embedding_status(config: AppConfig | None = None) -> dict[str, Any]:
    """探测嵌入后端可用性（供 /api/health 与前端左栏状态展示）。"""
    cfg = config or get_config()
    provider = _get_provider(cfg)
    ok, message = await provider.probe()
    return {
        "ok": ok,
        "provider": provider.name,
        "model": provider.model,
        "dim": provider.dim,
        "message": message,
    }
