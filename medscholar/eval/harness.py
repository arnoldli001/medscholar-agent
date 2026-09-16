"""评测框架：把语料装进临时库，按多种检索配置跑一遍并对比。

**最重要的设计约束：评测必须跑在生产代码路径上。**
所以这里直接调用 ``search_fts`` / ``search_vector`` / ``rrf_fuse`` ——
也就是 ``hybrid_search`` 内部所用的同一组原语，并配一个一致性测试断言
"评测里的 ``production`` 配置 == ``hybrid_search`` 的输出"。
否则就是在评测一个自己重写的检索器，指标再漂亮也没有意义。

另一个约束：**结果必须可复现**。语料被装进临时数据库，
文献 id 用 1..N 的确定性序号（``source_id``），与用户真实库无关；
嵌入用同一模型时结果逐位一致。CI 里用 ``hashing`` 提供方，
因此完全离线、无随机性。
"""

from __future__ import annotations

import logging
import shutil
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

from ..config import AppConfig, get_config
from ..db.connect import Database
from ..db.repo import (
    insert_paper,
    rrf_fuse,
    search_fts,
    search_vector,
    store_embeddings,
)
from ..embedding.providers import get_provider
from .dataset import EvalDataset
from .metrics import QueryMetrics, aggregate, evaluate_query

logger = logging.getLogger(__name__)

__all__ = [
    "RetrievalConfig",
    "CONFIGS",
    "AblationResult",
    "EvalReport",
    "index_corpus",
    "evaluate_configs",
]


@dataclass(slots=True)
class RetrievalConfig:
    """一种检索配置（消融实验的一行）。"""

    name: str
    use_fts: bool = True
    use_vector: bool = True
    rrf_k: int = 60
    fts_weight: float = 1.0
    vector_weight: float = 1.0
    description: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "use_fts": self.use_fts,
            "use_vector": self.use_vector,
            "rrf_k": self.rrf_k,
            "fts_weight": self.fts_weight,
            "vector_weight": self.vector_weight,
            "description": self.description,
        }


#: 消融矩阵。刻意包含"只用一路"与"两路融合"，以及 RRF 的 k 值扫描 ——
#: 因为"k=60 是论文推荐值"这句话应该被数据检验，而不是被引用。
CONFIGS: tuple[RetrievalConfig, ...] = (
    RetrievalConfig(
        name="bm25-only", use_fts=True, use_vector=False,
        description="只走 FTS5 BM25（关键词）",
    ),
    RetrievalConfig(
        name="vector-only", use_fts=False, use_vector=True,
        description="只走向量 KNN（语义）",
    ),
    RetrievalConfig(
        name="production", use_fts=True, use_vector=True, rrf_k=60,
        description="生产配置：BM25 ⊕ 向量 → RRF(k=60)",
    ),
    RetrievalConfig(
        name="rrf-k10", use_fts=True, use_vector=True, rrf_k=10,
        description="RRF k=10（更强调头部名次）",
    ),
    RetrievalConfig(
        name="rrf-k100", use_fts=True, use_vector=True, rrf_k=100,
        description="RRF k=100（名次影响更平缓）",
    ),
    RetrievalConfig(
        name="weighted-fts2", use_fts=True, use_vector=True, rrf_k=60,
        fts_weight=2.0, vector_weight=1.0,
        description="RRF 加权：BM25 权重 ×2",
    ),
    RetrievalConfig(
        name="weighted-vec2", use_fts=True, use_vector=True, rrf_k=60,
        fts_weight=1.0, vector_weight=2.0,
        description="RRF 加权：向量权重 ×2",
    ),
)


@dataclass(slots=True)
class AblationResult:
    """一种配置的整体结果。"""

    config: RetrievalConfig
    summary: dict[str, Any] = field(default_factory=dict)
    per_query: list[QueryMetrics] = field(default_factory=list)
    by_source: dict[str, dict[str, Any]] = field(default_factory=dict)
    #: 阴性对照观察：对"本来就没有相关文献"的查询返回了多少条
    empty_control: dict[str, Any] = field(default_factory=dict)
    elapsed_s: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "config": self.config.to_dict(),
            "summary": dict(self.summary),
            "by_source": {k: dict(v) for k, v in self.by_source.items()},
            "empty_control": dict(self.empty_control),
            "per_query": [m.to_dict() for m in self.per_query],
            "elapsed_s": round(self.elapsed_s, 3),
        }


@dataclass(slots=True)
class EvalReport:
    """一次完整评测的报告。"""

    dataset: str = ""
    k: int = 10
    embed_provider: str = ""
    embed_dim: int = 0
    corpus_size: int = 0
    results: list[AblationResult] = field(default_factory=list)
    baseline: str = "bm25-only"

    # -------------------------------------------------------------- 对比视图
    def table(self) -> list[dict[str, Any]]:
        """消融对比表（含与基线之差，便于一眼看出改动是否有效）。"""
        base = next((r for r in self.results if r.config.name == self.baseline), None)
        rows: list[dict[str, Any]] = []
        for result in self.results:
            summary = result.summary
            row: dict[str, Any] = {
                "config": result.config.name,
                "recall": summary.get("recall"),
                "ndcg": summary.get("ndcg"),
                "mrr": summary.get("mrr"),
                "map": summary.get("map"),
            }
            if base is not None and result is not base:
                for key in ("recall", "ndcg", "mrr"):
                    current = summary.get(key)
                    origin = base.summary.get(key)
                    row[f"d_{key}"] = (
                        None if current is None or origin is None else current - origin
                    )
            rows.append(row)
        return rows

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "k": self.k,
            "embed_provider": self.embed_provider,
            "embed_dim": self.embed_dim,
            "corpus_size": self.corpus_size,
            "baseline": self.baseline,
            "results": [r.to_dict() for r in self.results],
        }


def index_corpus(
    dataset: EvalDataset,
    *,
    config: AppConfig,
    workdir: Path | None = None,
    embed: bool = True,
) -> tuple[Database, Path]:
    """把语料装进一个**临时**数据库（幂等、确定性）。返回 (db, workdir)。

    id 由语料顺序决定（``source_id`` = 1..N），并通过 ``insert_paper`` 真实入库，
    因此走的是与生产完全一样的去重/索引/落库路径。
    """
    tmp = Path(workdir) if workdir else Path(tempfile.mkdtemp(prefix="medscholar-eval-"))
    tmp.mkdir(parents=True, exist_ok=True)
    db = Database(tmp / "eval.db", config=config)

    papers = dataset.papers()
    for paper in papers:
        insert_paper(paper, db=db)

    if embed:
        provider = get_provider(config)
        texts = [_embed_text(p) for p in papers]
        vectors = _run(provider.embed(texts))
        dim = len(vectors[0]) if vectors else 0
        if dim:
            db.sync_embedding_dim(dim)
        pairs = [
            (int(p.source_id), vector)
            for p, vector in zip(papers, vectors)
            if p.source_id and vector
        ]
        store_embeddings(pairs, db=db)
    return db, tmp


def _embed_text(paper: Any) -> str:
    """嵌入用的文本：标题 + 摘要 + 关键词（与生产一致的口径）。"""
    parts = [paper.title or "", paper.abstract or ""]
    if paper.keywords:
        parts.append(" ".join(paper.keywords))
    return "\n".join(p for p in parts if p)


def _run(coro: Any) -> Any:
    """在同步上下文里跑一个协程（哈希提供方是同步的，Ollama 提供方需要网络）。"""
    import asyncio

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    # 已在事件循环里（例如从测试调用）：用独立线程跑，避免嵌套
    import concurrent.futures

    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        return pool.submit(asyncio.run, coro).result()


def _rank(
    query: str,
    *,
    embedding: list[float] | None,
    cfg: RetrievalConfig,
    dataset_cfg: Any,
    db: Database,
    limit: int,
) -> list[str]:
    """按配置返回融合后的 id 排名（字符串）。"""
    fts_hits: list[tuple[int, float]] = []
    vec_hits: list[tuple[int, float]] = []

    if cfg.use_fts:
        fts_hits = search_fts(
            query, limit=max(dataset_cfg.fts_candidates, limit * 3), db=db
        )
    if cfg.use_vector and embedding:
        vec_hits = search_vector(
            embedding, limit=max(dataset_cfg.vector_candidates, limit * 3), db=db
        )

    if not fts_hits and not vec_hits:
        return []
    if not vec_hits:
        return [str(pid) for pid, _ in fts_hits[:limit]]
    if not fts_hits:
        return [str(pid) for pid, _ in vec_hits[:limit]]

    fused = rrf_fuse(
        [fts_hits, vec_hits],
        k=cfg.rrf_k,
        weights=[cfg.fts_weight, cfg.vector_weight],
    )
    return [str(pid) for pid, _ in fused[:limit]]


def evaluate_configs(
    dataset: EvalDataset,
    *,
    config: AppConfig | None = None,
    configs: Sequence[RetrievalConfig] = CONFIGS,
    k: int = 10,
    baseline: str = "bm25-only",
    workdir: Path | None = None,
    cleanup: bool = True,
) -> EvalReport:
    """跑完整消融评测。"""
    cfg = config or get_config()
    problems = dataset.validate()
    if problems:
        raise ValueError("评测集本身有问题：\n  - " + "\n  - ".join(problems))

    provider = get_provider(cfg)
    db, tmp = index_corpus(dataset, config=cfg, workdir=workdir)
    report = EvalReport(
        dataset=dataset.name,
        k=k,
        embed_provider=getattr(provider, "name", "unknown"),
        embed_dim=getattr(provider, "dim", 0),
        corpus_size=len(dataset.corpus),
        baseline=baseline,
    )

    try:
        # 查询向量只算一次，所有配置共用（否则会重复调用模型）
        query_vectors: list[list[float]] = []
        if any(c.use_vector for c in configs):
            query_vectors = _run(provider.embed([c.query for c in dataset.cases]))

        for cfg_item in configs:
            started = time.perf_counter()
            per_query: list[QueryMetrics] = []
            empty_returned: list[int] = []
            for index, case in enumerate(dataset.cases):
                embedding = (
                    list(query_vectors[index])
                    if query_vectors and index < len(query_vectors)
                    else None
                )
                ranked = _rank(
                    case.query,
                    embedding=embedding,
                    cfg=cfg_item,
                    dataset_cfg=cfg.retrieval,
                    db=db,
                    limit=k,
                )
                if case.expect_no_relevant:
                    # 阴性对照：不参与召回类指标，只记录"硬凑了多少条"
                    empty_returned.append(len(ranked))
                per_query.append(
                    evaluate_query(
                        case.query, ranked, case.relevant, k=k, config=cfg_item.name
                    )
                )
            result = AblationResult(
                config=cfg_item,
                summary=aggregate(per_query),
                per_query=per_query,
                elapsed_s=time.perf_counter() - started,
            )
            result.by_source = _group_by_source(dataset, per_query)
            result.empty_control = {
                "cases": len(empty_returned),
                "avg_returned": (
                    sum(empty_returned) / len(empty_returned) if empty_returned else None
                ),
                "max_returned": max(empty_returned) if empty_returned else None,
                "note": (
                    "RRF 融合天然总会返回 top_k，因此对无关查询也会给出结果。"
                    "这个数字应当被显式盯住：它越大，说明系统越倾向于"
                    "「宁滥勿缺」，需要靠相关性阈值或引用校验兜住。"
                ),
            }
            report.results.append(result)
    finally:
        try:
            db.close()
        except Exception:  # pragma: no cover
            pass
        if cleanup:
            shutil.rmtree(tmp, ignore_errors=True)

    return report


def _group_by_source(
    dataset: EvalDataset, per_query: Sequence[QueryMetrics]
) -> dict[str, dict[str, Any]]:
    """按查询来源分组聚合 —— 让"known-item 虚高"这件事在报告里显形。"""
    grouped: dict[str, list[QueryMetrics]] = {}
    for case, metrics in zip(dataset.cases, per_query):
        key = case.source or "(未标注来源)"
        grouped.setdefault(key, []).append(metrics)
    return {name: aggregate(items) for name, items in grouped.items()}


def check_regression(
    report: EvalReport,
    *,
    thresholds: Mapping[str, float],
    config_name: str = "production",
) -> list[str]:
    """对照阈值检查是否退化。返回违规说明（空表示通过）。

    阈值写在 CI 里，**只在低于下限时才失败**，因此不会因为"变得更好"而报错。
    """
    result = next((r for r in report.results if r.config.name == config_name), None)
    if result is None:
        return [f"报告里没有配置 {config_name}，无法做回归判定"]

    failures: list[str] = []
    for metric, floor in thresholds.items():
        value = result.summary.get(metric)
        if value is None:
            failures.append(f"{config_name}.{metric} 没有数值（是否所有查询都被跳过了？）")
            continue
        if value < floor:
            failures.append(
                f"{config_name}.{metric} = {value:.4f} 低于下限 {floor:.4f}"
                f"（差 {floor - value:.4f}）"
            )
    return failures


def verify_production_parity(
    dataset: EvalDataset, *, config: AppConfig | None = None, k: int = 10
) -> list[int]:
    """一致性自检：评测里的 ``production`` 配置是否与 ``hybrid_search`` 一致。

    这是"没有评测自己重写的检索器"这句声明的**可执行证据**。
    返回不一致的查询下标（空列表表示完全一致）。
    """
    from ..db.repo import hybrid_search

    cfg = config or get_config()
    db, tmp = index_corpus(dataset, config=cfg)
    mismatched: list[int] = []
    try:
        provider = get_provider(cfg)
        vectors = _run(provider.embed([c.query for c in dataset.cases]))
        prod = next(c for c in CONFIGS if c.name == "production")
        for index, case in enumerate(dataset.cases):
            mine = _rank(
                case.query,
                embedding=list(vectors[index]) if index < len(vectors) else None,
                cfg=prod,
                dataset_cfg=cfg.retrieval,
                db=db,
                limit=k,
            )
            theirs = [
                str(item.paper.paper_id)
                for item in hybrid_search(
                    case.query,
                    embedding=list(vectors[index]) if index < len(vectors) else None,
                    top_k=k,
                    db=db,
                )
            ]
            if mine != theirs:
                mismatched.append(index)
    finally:
        try:
            db.close()
        except Exception:  # pragma: no cover
            pass
        shutil.rmtree(tmp, ignore_errors=True)
    return mismatched
