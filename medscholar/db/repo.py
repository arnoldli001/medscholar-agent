"""数据库仓储层门面：实现按业务边界拆分在 repositories/ 子包
（papers/search/embeddings/fulltext/citations/library/runs/_common），此处仅显式重导出。

名字/签名/SQL/行为与拆分前一致，含 _db 等下划线私有名（外部已有依赖）；
``X as X`` 是 PEP 484 显式重导出写法。新代码请直接 import 对应子模块。
"""

from __future__ import annotations

from .repositories._common import (
    _db as _db,
    _build_filters as _build_filters,
    _FILTER_SQL as _FILTER_SQL,
    _fts_delete as _fts_delete,
    _fts_sync as _fts_sync,
    _FTS_COLUMNS as _FTS_COLUMNS,
    logger as logger,
)
from .repositories.citations import (
    add_citation as add_citation,
    add_citations as add_citations,
    get_citing_papers as get_citing_papers,
    get_references as get_references,
)
from .repositories.embeddings import (
    clear_embeddings as clear_embeddings,
    deserialize_vector as deserialize_vector,
    normalize_vector as normalize_vector,
    papers_missing_embeddings as papers_missing_embeddings,
    serialize_vector as serialize_vector,
    store_embedding as store_embedding,
    store_embeddings as store_embeddings,
)
from .repositories.fulltext import (
    clear_fulltext_attempts as clear_fulltext_attempts,
    fulltext_candidates as fulltext_candidates,
    fulltext_failure_stats as fulltext_failure_stats,
    get_fulltext as get_fulltext,
    record_fulltext_attempt as record_fulltext_attempt,
    save_fulltext as save_fulltext,
)
from .repositories.library import (
    add_message as add_message,
    add_papers_to_project as add_papers_to_project,
    create_project as create_project,
    create_session as create_session,
    delete_project as delete_project,
    delete_session as delete_session,
    get_project as get_project,
    get_session as get_session,
    list_messages as list_messages,
    list_project_papers as list_project_papers,
    list_projects as list_projects,
    list_sessions as list_sessions,
    remove_paper_from_project as remove_paper_from_project,
    rename_session as rename_session,
    update_project as update_project,
)
from .repositories.papers import (
    _decode_list as _decode_list,
    _find_existing_id as _find_existing_id,
    _merge_into as _merge_into,
    _PAPER_COLUMNS as _PAPER_COLUMNS,
    _row_to_paper as _row_to_paper,
    count_papers as count_papers,
    delete_papers as delete_papers,
    find_paper_id as find_paper_id,
    get_paper as get_paper,
    get_papers_by_ids as get_papers_by_ids,
    insert_paper as insert_paper,
    insert_papers as insert_papers,
    list_papers as list_papers,
)
from .repositories.runs import (
    delete_artifact as delete_artifact,
    delete_run as delete_run,
    get_artifact as get_artifact,
    get_run as get_run,
    get_run_step as get_run_step,
    latest_run_for_session as latest_run_for_session,
    list_artifacts as list_artifacts,
    list_run_steps as list_run_steps,
    list_runs as list_runs,
    mark_interrupted_runs as mark_interrupted_runs,
    run_step_phases as run_step_phases,
    save_artifact as save_artifact,
    save_run_step as save_run_step,
    upsert_run as upsert_run,
)
from .repositories.search import (
    _apply_vector_filters as _apply_vector_filters,
    hybrid_search as hybrid_search,
    log_search as log_search,
    recent_searches as recent_searches,
    rrf_fuse as rrf_fuse,
    search_fts as search_fts,
    search_fulltext as search_fulltext,
    search_vector as search_vector,
)

# 与拆分前的 repo.__all__ 逐字一致；runs 函数等靠模块属性访问的名字照旧导出但不在此列出。
__all__ = [
    "insert_paper",
    "insert_papers",
    "get_paper",
    "get_papers_by_ids",
    "find_paper_id",
    "list_papers",
    "delete_papers",
    "count_papers",
    "search_fts",
    "search_vector",
    "hybrid_search",
    "rrf_fuse",
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
    "update_project",
    "delete_project",
    "add_papers_to_project",
    "remove_paper_from_project",
    "list_project_papers",
    "create_session",
    "list_sessions",
    "get_session",
    "rename_session",
    "delete_session",
    "add_message",
    "list_messages",
    "save_artifact",
    "get_artifact",
    "list_artifacts",
    "delete_artifact",
]
