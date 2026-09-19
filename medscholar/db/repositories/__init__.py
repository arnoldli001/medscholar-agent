"""仓储层实现包：按业务边界拆分的 db/repositories/*。

原来的 ``medscholar/db/repo.py`` 是一个 1400+ 行、67 个函数的单文件，把 10 个业务边界
混在一起（文献、检索、向量、全文、引用、检索日志、课题、会话、运行、产物），
改一处要在整文件里滚屏。这里按边界切成小模块，``medscholar.db.repo`` 保留为稳定门面。

模块与依赖方向（箭头指向被依赖方，全图无环）::

    _common            ← 所有模块（数据库句柄、FTS 同步、过滤 SQL、日志器）
    papers             ← search / fulltext / library（papers 是最底层的数据边界）
    embeddings         ← search（向量序列化与归一化是检索与嵌入管道共用的表示）
    search             内部依赖 papers + embeddings
    fulltext           只依赖 papers
    citations / library / runs   互相独立

新代码请直接 import 具体子模块（例如
``from medscholar.db.repositories.papers import insert_paper``），
``medscholar.db.repo`` 只服务尚未迁移的历史调用点。
"""

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
