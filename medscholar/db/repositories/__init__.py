"""仓储层实现包，按业务边界拆分：papers/search/embeddings/fulltext/citations/
library/runs，共享零件在 _common（依赖方向无环：_common ← papers ← search/fulltext/library，
embeddings ← search）。medscholar.db.repo 是保留给历史调用点的门面。"""

from __future__ import annotations

from . import _common, citations, embeddings, fulltext, library, papers, runs, search

__all__ = [
    "_common",
    "papers",
    "embeddings",
    "search",
    "fulltext",
    "citations",
    "library",
    "runs",
]
