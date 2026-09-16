"""诊断：为什么「综述草稿」页签是空的。

检查 Agent 运行是否真的产出了 artifact，以及会话/消息的落库情况。
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from medscholar.db.connect import get_db  # noqa: E402

db = get_db()

print("=" * 78)
print("1) 产物表 artifacts")
print("=" * 78)
rows = db.query(
    "SELECT id, session_id, kind, title, fmt, length(content) AS chars, "
    "       created_at, updated_at FROM artifacts ORDER BY id DESC LIMIT 10"
)
if rows:
    for r in rows:
        print(f"  #{r['id']}  session={r['session_id']}  kind={r['kind']}  "
              f"{r['chars']} 字  {r['created_at']}")
        print(f"      {r['title']}")
else:
    print("  （空 —— 数据库里一个产物都没有）")

print()
print("=" * 78)
print("2) 会话")
print("=" * 78)
for r in db.query(
    "SELECT id, title, topic, created_at, updated_at FROM chat_sessions ORDER BY id DESC LIMIT 6"
):
    n = db.scalar("SELECT COUNT(*) FROM chat_messages WHERE session_id = ?", (r["id"],), default=0)
    print(f"  #{r['id']}  {n} 条消息  {r['created_at']}  {r['title'][:40]}")
    if r["topic"]:
        print(f"      课题: {r['topic'][:70]}")

print()
print("=" * 78)
print("3) 各会话的消息角色分布")
print("=" * 78)
for r in db.query(
    "SELECT session_id, role, COUNT(*) AS n, MAX(length(content)) AS maxlen "
    "FROM chat_messages GROUP BY session_id, role ORDER BY session_id DESC LIMIT 12"
):
    print(f"  session {r['session_id']:<3} {r['role']:<10} {r['n']:>3} 条，最长 {r['maxlen']} 字")

print()
print("=" * 78)
print("4) 检索历史（判断运行是否真的执行过）")
print("=" * 78)
for r in db.query(
    "SELECT query, source, result_count, created_at FROM search_logs ORDER BY id DESC LIMIT 6"
):
    print(f"  [{r['created_at']}] {r['source']:<12} {r['result_count']:>3} 篇  {r['query'][:46]}")

print()
print("=" * 78)
print("5) 知识库规模")
print("=" * 78)
stats = db.stats()
print(f"  文献 {stats['papers']} 篇 / 全文 {stats['fulltext']} 篇 / 向量 {stats['embedded']} 条")
