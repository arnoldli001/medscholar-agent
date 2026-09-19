"""课题、会话与消息：项目的组织单位（projects / project_papers）和对话记录（chat_sessions / chat_messages）。

单独拆出来是因为这两组表服务的是"用户怎么组织自己的工作"，而不是文献或检索本身：
课题只是给文献打分组标签，会话只是消息流水，它们不需要懂去重、向量或检索策略。
"""

from __future__ import annotations

import json
from typing import Any, Mapping, Sequence

from ..connect import Database
from ...models import Paper
from ._common import _db
from .papers import _decode_list, list_papers

__all__ = [
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
]


def create_project(
    name: str, *, description: str = "", keywords: Sequence[str] | None = None,
    db: Database | None = None,
) -> int:
    database = _db(db)
    with database.transaction() as conn:
        conn.execute(
            "INSERT INTO projects(name, description, keywords) VALUES (?, ?, ?) "
            "ON CONFLICT(name) DO UPDATE SET description = excluded.description, "
            "keywords = excluded.keywords, updated_at = datetime('now')",
            (name.strip(), description, json.dumps(list(keywords or []), ensure_ascii=False)),
        )
        row = conn.execute("SELECT id FROM projects WHERE name = ?", (name.strip(),)).fetchone()
        return int(row["id"])


def list_projects(*, db: Database | None = None) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT pr.id, pr.name, pr.description, pr.keywords, pr.created_at, pr.updated_at, "
        "       (SELECT COUNT(*) FROM project_papers pp WHERE pp.project_id = pr.id) AS paper_count "
        "FROM projects pr ORDER BY pr.updated_at DESC, pr.id DESC"
    )
    out = []
    for r in rows:
        item = dict(r)
        item["keywords"] = _decode_list(item.get("keywords"))
        out.append(item)
    return out


def get_project(project_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one("SELECT * FROM projects WHERE id = ?", (project_id,))
    if not row:
        return None
    item = dict(row)
    item["keywords"] = _decode_list(item.get("keywords"))
    return item


def update_project(
    project_id: int,
    *,
    name: str | None = None,
    description: str | None = None,
    keywords: Sequence[str] | None = None,
    db: Database | None = None,
) -> bool:
    sets, params = [], []
    if name is not None:
        sets.append("name = ?")
        params.append(name)
    if description is not None:
        sets.append("description = ?")
        params.append(description)
    if keywords is not None:
        sets.append("keywords = ?")
        params.append(json.dumps(list(keywords), ensure_ascii=False))
    if not sets:
        return False
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            f"UPDATE projects SET {', '.join(sets)}, updated_at = datetime('now') WHERE id = ?",
            [*params, project_id],
        )
        return (cur.rowcount or 0) > 0


def delete_project(project_id: int, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM projects WHERE id = ?", (project_id,))
        return (cur.rowcount or 0) > 0


def add_papers_to_project(
    project_id: int, paper_ids: Sequence[int], *, note: str = "", db: Database | None = None
) -> int:
    if not paper_ids:
        return 0
    database = _db(db)
    added = 0
    with database.transaction() as conn:
        for pid in paper_ids:
            cur = conn.execute(
                "INSERT OR IGNORE INTO project_papers(project_id, paper_id, note) VALUES (?, ?, ?)",
                (project_id, pid, note),
            )
            added += cur.rowcount or 0
        conn.execute(
            "UPDATE projects SET updated_at = datetime('now') WHERE id = ?", (project_id,)
        )
    return added


def remove_paper_from_project(
    project_id: int, paper_id: int, *, db: Database | None = None
) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "DELETE FROM project_papers WHERE project_id = ? AND paper_id = ?",
            (project_id, paper_id),
        )
        return (cur.rowcount or 0) > 0


def list_project_papers(
    project_id: int, *, limit: int = 200, db: Database | None = None
) -> list[Paper]:
    return list_papers(project_id=project_id, limit=limit, db=db)


def create_session(
    *, title: str = "新会话", project_id: int | None = None, topic: str = "",
    db: Database | None = None,
) -> int:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO chat_sessions(title, project_id, topic) VALUES (?, ?, ?)",
            (title[:200], project_id, topic),
        )
        return int(cur.lastrowid or 0)


def list_sessions(*, limit: int = 50, db: Database | None = None) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT s.id, s.title, s.project_id, s.topic, s.created_at, s.updated_at, "
        "       (SELECT COUNT(*) FROM chat_messages m WHERE m.session_id = s.id) AS message_count "
        "FROM chat_sessions s ORDER BY s.updated_at DESC, s.id DESC LIMIT ?",
        (limit,),
    )
    return [dict(r) for r in rows]


def get_session(session_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    row = _db(db).query_one("SELECT * FROM chat_sessions WHERE id = ?", (session_id,))
    return dict(row) if row else None


def rename_session(session_id: int, title: str, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "UPDATE chat_sessions SET title = ?, updated_at = datetime('now') WHERE id = ?",
            (title[:200], session_id),
        )
        return (cur.rowcount or 0) > 0


def delete_session(session_id: int, *, db: Database | None = None) -> bool:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute("DELETE FROM chat_sessions WHERE id = ?", (session_id,))
        return (cur.rowcount or 0) > 0


def add_message(
    session_id: int,
    role: str,
    content: str,
    *,
    meta: Mapping[str, Any] | None = None,
    db: Database | None = None,
) -> int:
    database = _db(db)
    with database.transaction() as conn:
        cur = conn.execute(
            "INSERT INTO chat_messages(session_id, role, content, meta) VALUES (?, ?, ?, ?)",
            (session_id, role, content, json.dumps(dict(meta or {}), ensure_ascii=False)),
        )
        conn.execute(
            "UPDATE chat_sessions SET updated_at = datetime('now') WHERE id = ?", (session_id,)
        )
        return int(cur.lastrowid or 0)


def list_messages(
    session_id: int, *, limit: int = 200, db: Database | None = None
) -> list[dict[str, Any]]:
    rows = _db(db).query(
        "SELECT id, session_id, role, content, meta, created_at FROM chat_messages "
        "WHERE session_id = ? ORDER BY id LIMIT ?",
        (session_id, limit),
    )
    out = []
    for r in rows:
        item = dict(r)
        try:
            item["meta"] = json.loads(item.get("meta") or "{}")
        except (TypeError, ValueError):
            item["meta"] = {}
        out.append(item)
    return out
