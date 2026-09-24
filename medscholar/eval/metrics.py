"""检索评测指标（纯函数，无 IO、无依赖）。

错误的指标比没有指标更糟：它会把检索改动调向错误方向，而且看起来一切正常。
所以这里的每个函数都在 `tests/test_eval_metrics.py` 里用手算结果钉死，
包括边界情形（无相关文献、k 大于结果数、并列名次）。

指标定义（沿用信息检索的标准定义）：

* ``recall@k``      = |相关 ∩ 前k| / |相关|
* ``precision@k``   = |相关 ∩ 前k| / k
* ``reciprocal_rank`` = 1 / 第一个相关结果的名次（没有则 0）
* ``average_precision`` = 相关位置上的 precision 的平均（没有相关则 0）
* ``ndcg@k``        = DCG@k / IDCG@k，支持二元或分级相关性

约定：没有标注相关文献的查询不参与聚合（返回 ``None`` 而非 0）。
把"没标注"当成"检索失败"会污染整体指标。
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "retrieved_ids",
    "recall_at_k",
    "precision_at_k",
    "reciprocal_rank",
    "average_precision",
    "dcg_at_k",
    "ndcg_at_k",
    "QueryMetrics",
    "evaluate_query",
    "aggregate",
    "format_table",
]


def _dedupe_keep_order(items: Iterable[Any]) -> list[Any]:
    """去重但保留首次出现顺序 —— 排名列表里出现重复项时不能让指标虚高。"""
    seen: set[Any] = set()
    out: list[Any] = []
    for item in items:
        if item in seen:
            continue
        seen.add(item)
        out.append(item)
    return out


def retrieved_ids(ranked: Sequence[Any]) -> list[str]:
    """把任意排名的结果统一成 ``["1", "2", ...]`` 形式的 id 列表。

    统一成字符串是为了让"数据集里的 id"与"检索返回的 paper_id"
    能用同一个比较口径（避免 int/str 混用导致的静默不匹配）。
    """
    out: list[str] = []
    for item in ranked:
        if isinstance(item, (int, str)):
            out.append(str(item))
        else:
            # 支持带 .paper_id 的对象（如 ScoredPaper）
            paper = getattr(item, "paper", None)
            value = getattr(paper, "paper_id", None) if paper is not None else None
            if value is None:
                value = getattr(item, "paper_id", None)
            if value is None:
                raise TypeError(f"无法从 {type(item).__name__} 取出 paper_id")
            out.append(str(value))
    return _dedupe_keep_order(out)


def recall_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """召回率。``relevant`` 为空时返回 ``None``（该查询不参与聚合）。

    >>> recall_at_k(["1", "2", "3"], {"2"}, 3)
    1.0
    >>> recall_at_k(["1", "2", "3"], {"2", "9"}, 3)
    0.5
    """
    rel = set(relevant)
    if not rel:
        return None
    if k <= 0:
        return 0.0
    hit = len(set(retrieved[:k]) & rel)
    return hit / len(rel)


def precision_at_k(retrieved: Sequence[str], relevant: Iterable[str], k: int) -> float | None:
    """精确率，分母固定为 ``k``（标准定义）。``relevant`` 为空返回 ``None``。

    >>> precision_at_k(["1", "2", "3"], {"2"}, 2)
    0.5
    """
    rel = set(relevant)
    if not rel:
        return None
    if k <= 0:
        return 0.0
    hit = len(set(retrieved[:k]) & rel)
    return hit / k


def reciprocal_rank(retrieved: Sequence[str], relevant: Iterable[str]) -> float | None:
    """第一个相关结果的名次的倒数（MRR 的单查询分量）。未命中返回 0。

    >>> reciprocal_rank(["a", "b", "c"], {"b"})
    0.5
    >>> reciprocal_rank(["a"], {"z"})
    0.0
    """
    rel = set(relevant)
    if not rel:
        return None
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            return 1.0 / index
    return 0.0


def average_precision(retrieved: Sequence[str], relevant: Iterable[str]) -> float | None:
    """AP：在**每个相关结果的位置**上取 precision，再对相关总数取平均。

    注意分母是「相关文献总数」而不是「命中的数量」—— 漏掉的相关文献会拉低 AP，
    这正是我们想要的行为。

    >>> average_precision(["1", "2", "3", "4"], {"1", "3"})
    0.8333333333333333
    """
    rel = set(relevant)
    if not rel:
        return None
    hits = 0
    total = 0.0
    for index, doc_id in enumerate(retrieved, start=1):
        if doc_id in rel:
            hits += 1
            total += hits / index
    return total / len(rel)


def dcg_at_k(relevances: Sequence[float], k: int) -> float:
    """折损累计增益。折损用 ``log2(rank + 1)``（rank 从 1 开始）。

    >>> round(dcg_at_k([3, 2, 1], 3), 6)
    4.76186
    """
    score = 0.0
    for index, gain in enumerate(relevances[:k], start=1):
        score += float(gain) / math.log2(index + 1)
    return score


def ndcg_at_k(
    retrieved: Sequence[str],
    relevance: Iterable[str] | Mapping[str, float],
    k: int,
) -> float | None:
    """归一化折损累计增益。支持二元（集合）或分级（字典）相关性。

    >>> round(ndcg_at_k(["1", "2", "3"], {"1"}, 3), 6)
    1.0
    >>> round(ndcg_at_k(["3", "1"], {"1"}, 2), 6)
    0.63093
    >>> ndcg_at_k(["1"], {"1": 3}, 3)
    1.0
    """
    if isinstance(relevance, Mapping):
        rel_map = {str(key): float(value) for key, value in relevance.items()}
    else:
        rel_map = {str(key): 1.0 for key in relevance}
    if not rel_map:
        return None
    if k <= 0:
        return 0.0

    gains = [rel_map.get(doc_id, 0.0) for doc_id in retrieved[:k]]
    ideal = sorted(rel_map.values(), reverse=True)[:k]
    idcg = dcg_at_k(ideal, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(gains, k) / idcg


@dataclass(slots=True)
class QueryMetrics:
    """单个查询的评测结果（保留逐查询明细，便于定位失败而不是只看均值）。"""

    query: str = ""
    config: str = ""
    num_relevant: int = 0
    num_retrieved: int = 0
    recall: float | None = None
    precision: float | None = None
    mrr: float | None = None
    ap: float | None = None
    ndcg: float | None = None
    #: 命中 / 漏掉 / 多出来的文献，用于人工复核检索失败的原因
    hit_ids: list[str] = field(default_factory=list)
    missed_ids: list[str] = field(default_factory=list)
    noise_ids: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "query": self.query,
            "config": self.config,
            "num_relevant": self.num_relevant,
            "num_retrieved": self.num_retrieved,
            "recall": self.recall,
            "precision": self.precision,
            "mrr": self.mrr,
            "ap": self.ap,
            "ndcg": self.ndcg,
            "hit_ids": list(self.hit_ids),
            "missed_ids": list(self.missed_ids),
            "noise_ids": list(self.noise_ids[:10]),
        }


def evaluate_query(
    query: str,
    retrieved: Sequence[Any],
    relevant: Iterable[str] | Mapping[str, float],
    *,
    k: int = 10,
    config: str = "",
) -> QueryMetrics:
    """评测单个查询。``retrieved`` 可以是 id 列表，也可以是 ScoredPaper 列表。"""
    # 注意：先按 k 截断再交给各指标函数，各函数内部也会再截一次（幂等）
    ranked = retrieved_ids(list(retrieved)[:k])
    rel_set = set(relevance_keys(relevant))
    rel_for_ndcg = relevant

    return QueryMetrics(
        query=query,
        config=config,
        num_relevant=len(rel_set),
        num_retrieved=len(ranked),
        recall=recall_at_k(ranked, rel_set, k),
        precision=precision_at_k(ranked, rel_set, k),
        mrr=reciprocal_rank(ranked, rel_set),
        ap=average_precision(ranked, rel_set),
        ndcg=ndcg_at_k(ranked, rel_for_ndcg, k),
        hit_ids=sorted(set(ranked) & rel_set),
        missed_ids=sorted(rel_set - set(ranked)),
        noise_ids=[doc_id for doc_id in ranked if doc_id not in rel_set],
    )


def relevance_keys(relevant: Iterable[str] | Mapping[str, float]) -> list[str]:
    if isinstance(relevant, Mapping):
        return [str(key) for key in relevant]
    return [str(key) for key in relevant]


def _mean(values: Sequence[float | None]) -> float | None:
    clean = [v for v in values if v is not None]
    if not clean:
        return None
    return sum(clean) / len(clean)


def aggregate(per_query: Sequence[QueryMetrics]) -> dict[str, Any]:
    """聚合多个查询。只对"有标注"的查询求均值。

    另外单独报告 ``skipped``（没有标注相关文献而被跳过的查询数），
    避免"用跳过的方式把指标做漂亮"。
    """
    scored = [m for m in per_query if m.num_relevant > 0]
    skipped = len(per_query) - len(scored)
    return {
        "queries": len(per_query),
        "scored_queries": len(scored),
        "skipped_queries": skipped,
        "recall": _mean([m.recall for m in scored]),
        "precision": _mean([m.precision for m in scored]),
        "mrr": _mean([m.mrr for m in scored]),
        "map": _mean([m.ap for m in scored]),
        "ndcg": _mean([m.ndcg for m in scored]),
    }


def format_table(rows: Sequence[Mapping[str, Any]], columns: Sequence[tuple[str, str]]) -> str:
    """把若干行渲染成等宽表格（不引入 tabulate 之类依赖）。"""
    def cell(row: Mapping[str, Any], key: str) -> str:
        value = row.get(key)
        if value is None:
            return "—"
        if isinstance(value, float):
            return f"{value:.4f}"
        return str(value)

    header = [title for _, title in columns]
    body = [[cell(row, key) for key, _ in columns] for row in rows]
    widths = [
        max(len(header[i]), *(len(line[i]) for line in body)) if body else len(header[i])
        for i in range(len(header))
    ]

    def line(cells: Sequence[str]) -> str:
        return "  ".join(text.ljust(widths[i]) for i, text in enumerate(cells))

    out = [line(header), "  ".join("-" * w for w in widths)]
    out.extend(line(cells) for cells in body)
    return "\n".join(out)
