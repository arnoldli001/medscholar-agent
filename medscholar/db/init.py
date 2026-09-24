"""数据库初始化入口（需求文档「指令1：SQLite数据库初始化」的落地实现）。

用法::

    from medscholar.db import init, insert_paper, search_fts, search_vector, hybrid_search

    db = init()                       # 建库建表 + 建 FTS5 + 建 768 维向量表
    paper_id, created = insert_paper(paper)
    hits = hybrid_search("加速rTMS 卒中后抑郁", embedding=vec, top_k=20)

命令行::

    python -m medscholar.db.init            # 初始化（幂等）
    python -m medscholar.db.init --stats    # 查看统计
    python -m medscholar.db.init --rebuild  # 重建向量表
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from ..config import AppConfig, get_config
from .connect import Database, close_db, get_db, vec_backend_name
from .repo import (
    add_citation,
    add_citations,
    add_message,
    add_papers_to_project,
    clear_embeddings,
    count_papers,
    create_project,
    create_session,
    delete_papers,
    delete_project,
    delete_session,
    find_paper_id,
    get_artifact,
    get_citing_papers,
    get_fulltext,
    get_paper,
    get_papers_by_ids,
    get_project,
    get_references,
    hybrid_search,
    insert_paper,
    insert_papers,
    list_artifacts,
    list_messages,
    list_papers,
    list_project_papers,
    list_projects,
    list_sessions,
    log_search,
    papers_missing_embeddings,
    recent_searches,
    remove_paper_from_project,
    rename_session,
    rrf_fuse,
    save_artifact,
    save_fulltext,
    search_fts,
    search_fulltext,
    search_vector,
    store_embedding,
    store_embeddings,
)

__all__ = [
    "init_database",
    "init",
    "reset_database",
    "Database",
    "get_db",
    "close_db",
    "vec_backend_name",
    # 需求文档点名的四个核心函数
    "insert_paper",
    "search_fts",
    "search_vector",
    "hybrid_search",
    # 其余仓储函数
    "insert_papers",
    "get_paper",
    "get_papers_by_ids",
    "find_paper_id",
    "list_papers",
    "count_papers",
    "delete_papers",
    "store_embedding",
    "store_embeddings",
    "papers_missing_embeddings",
    "clear_embeddings",
    "save_fulltext",
    "get_fulltext",
    "search_fulltext",
    "add_citation",
    "add_citations",
    "get_references",
    "get_citing_papers",
    "log_search",
    "recent_searches",
    "create_project",
    "list_projects",
    "get_project",
    "delete_project",
    "add_papers_to_project",
    "remove_paper_from_project",
    "list_project_papers",
    "create_session",
    "list_sessions",
    "rename_session",
    "delete_session",
    "add_message",
    "list_messages",
    "save_artifact",
    "get_artifact",
    "list_artifacts",
    "rrf_fuse",
]


def init_database(
    db_path: str | Path | None = None, *, config: AppConfig | None = None
) -> Database:
    """初始化（或打开）本地学术数据库。

    幂等：重复调用不会丢失数据。会自动创建

    * ``papers`` / ``citations`` / ``search_logs`` / ``projects`` / ``chat_*`` / ``artifacts``
    * ``papers_fts``（FTS5，标题+摘要+MeSH+关键词，含中文逐字切分）
    * ``fulltext_fts``（FTS5，开放获取全文）
    * ``paper_embeddings``（768 维；优先 vec0 虚拟表，否则纯 Python 回退表）
    """
    cfg = config or get_config()
    cfg.ensure_dirs()
    path = Path(db_path) if db_path else cfg.db_path
    return Database(path, config=cfg)


#: 需求文档中的简称
init = init_database


def reset_database(*, config: AppConfig | None = None, confirm: bool = False) -> Path:
    """删除并重建数据库文件（会丢失全部数据）。返回被删除的路径。"""
    cfg = config or get_config()
    path = cfg.db_path
    if not confirm:
        raise RuntimeError("reset_database 需要 confirm=True 才会执行删除")
    close_db()
    for suffix in ("", "-wal", "-shm"):
        candidate = Path(str(path) + suffix)
        if candidate.exists():
            candidate.unlink()
    init_database(config=cfg)
    return path


def _main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="medscholar-db", description="MedScholar Agent 数据库初始化"
    )
    parser.add_argument("--db", help="数据库文件路径（默认取配置）")
    parser.add_argument("--stats", action="store_true", help="打印知识库统计")
    parser.add_argument("--rebuild", action="store_true", help="清空并重建向量索引")
    parser.add_argument("--optimize", action="store_true", help="优化 FTS5 索引")
    parser.add_argument("--vacuum", action="store_true", help="压缩数据库文件")
    parser.add_argument("--reset", action="store_true", help="删除并重建整个数据库（危险）")
    args = parser.parse_args(argv)

    cfg = get_config()
    if args.reset:
        path = reset_database(config=cfg, confirm=True)
        print(f"已重建数据库：{path}")

    database = init_database(args.db, config=cfg)

    if args.rebuild:
        removed = clear_embeddings(db=database)
        print(f"已清空 {removed} 条向量索引，下次检索/入库将自动重建")

    if args.optimize:
        database.optimize_fts()
        print("FTS5 索引已优化")

    if args.vacuum:
        database.vacuum()
        print("数据库已压缩")

    print(f"数据库文件 : {database.path}")
    print(f"向量后端   : {vec_backend_name()} {database.vec_note}".rstrip())
    print(json.dumps(database.stats(), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":  # pragma: no cover
    sys.exit(_main())
