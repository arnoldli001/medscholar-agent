"""检索评测：指标、数据集、消融实验框架。

这个包回答一个具体问题："检索改动让结果变好了吗？"

三部分：

* :mod:`medscholar.eval.metrics` —— 纯函数指标（recall@k / nDCG@k / MRR / MAP / AP），
  用手算用例钉死，因为错误的指标比没有指标更糟；
* :mod:`medscholar.eval.dataset` —— 黄金数据集的读写与校验；
* :mod:`medscholar.eval.harness` —— 把语料装进临时库、按多种检索配置跑一遍、
  输出可比对的报告（含消融矩阵与逐查询失败明细）。

两条硬约束：

1. 评测必须跑在生产代码路径上。框架直接调用 ``search_fts`` / ``search_vector`` /
   ``rrf_fuse``——即 ``hybrid_search`` 内部所用的同一组原语，
   并用一致性测试断言"评测里的生产配置 == hybrid_search 的结果"。
   否则就是在评测一个自己重写的检索器，数字再好看也没意义。
2. 没标注相关文献的查询不参与聚合，并单独报告被跳过的数量，
   避免用"跳过"把指标做漂亮。
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
