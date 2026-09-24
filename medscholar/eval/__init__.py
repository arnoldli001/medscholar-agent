"""检索评测：metrics（recall@k / nDCG@k / MRR / MAP，手算用例钉死）、
dataset（黄金集读写校验）、harness（多配置消融报告）。

两条硬约束：评测直接调用 search_fts/search_vector/rrf_fuse 等生产原语，
不另写检索器；未标注相关文献的查询不参与聚合并单独计数，不用"跳过"美化指标。
"""

from __future__ import annotations

from .dataset import EvalCase, EvalDataset, load_dataset, save_dataset
from .harness import (
    AblationResult,
    EvalReport,
    RetrievalConfig,
    CONFIGS,
    evaluate_configs,
    index_corpus,
)

__all__ = [
    "EvalCase",
    "EvalDataset",
    "load_dataset",
    "save_dataset",
    "RetrievalConfig",
    "CONFIGS",
    "AblationResult",
    "EvalReport",
    "evaluate_configs",
    "index_corpus",
]
