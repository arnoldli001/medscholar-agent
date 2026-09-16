"""查看当前知识库内容（只读）。"""

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
stats = db.stats()
print(f"  文献 {stats['papers']} 篇 / 已嵌入 {stats['embedded']} / 全文 {stats['fulltext']}")
print(f"  会话 {stats['sessions']} / 产物见 artifacts 表")
print(f"  年份跨度 {stats['year_min']}–{stats['year_max']} / 向量后端 {stats['vector_backend']}")
print(f"  数据库 {stats['db_size_mb']} MB")

sources = db.query(
    "SELECT source, COUNT(*) AS n FROM papers GROUP BY source ORDER BY n DESC"
)
if sources:
    print("  来源分布：" + "、".join(f"{r['source']}={r['n']}" for r in sources))

print("  最近入库 5 篇：")
for row in db.query(
    "SELECT title, pub_year, source FROM papers ORDER BY paper_id DESC LIMIT 5"
):
    title = (row["title"] or "")[:58]
    print(f"    · [{row['pub_year']}] {title}  ({row['source']})")

projects = db.query("SELECT name, id FROM projects LIMIT 5")
if projects:
    print("  课题：" + "、".join(r["name"] for r in projects))

artifacts = db.query("SELECT COUNT(*) AS n FROM artifacts")[0]["n"]
print(f"  已生成产物（综述草稿等）：{artifacts} 份")
