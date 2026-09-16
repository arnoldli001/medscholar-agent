"""全文抓取最终状态汇总。"""

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
total = db.scalar("SELECT COUNT(*) FROM papers", default=0)
have = db.scalar("SELECT COUNT(*) FROM paper_fulltext", default=0)

print("=" * 76)
print("全文抓取最终状态")
print("=" * 76)
print(f"  文献总数        {total:>5} 篇")
print(f"  已入库全文      {have:>5} 篇  ({have / total * 100:.1f}%)")

attempts = db.query(
    "SELECT COUNT(*) AS n, SUM(permanent) AS p FROM fulltext_attempts"
)[0]
print(f"  尝试过但失败    {attempts['n'] or 0:>5} 篇（其中已标记不再重试 {(attempts['p'] or 0)} 篇）")

remaining = db.scalar(
    "SELECT COUNT(*) FROM papers p "
    "WHERE (p.is_open_access = 1 OR (p.pmcid IS NOT NULL AND p.pmcid <> '')) "
    "  AND p.paper_id NOT IN (SELECT paper_id FROM paper_fulltext) "
    "  AND p.paper_id NOT IN (SELECT paper_id FROM fulltext_attempts WHERE permanent = 1)",
    default=0,
)
print(f"  仍可尝试        {remaining:>5} 篇")

print()
print("  全文来源分布：")
for row in db.query(
    "SELECT origin, COUNT(*) AS n, SUM(char_count) AS chars "
    "FROM paper_fulltext GROUP BY origin ORDER BY n DESC"
):
    origin = row["origin"] or "未知"
    label = {"europepmc": "Europe PMC JATS 全文",
             "pubmed-pmc": "PubMed PMC",
             "pdf": "开放获取 PDF",
             "cache": "缓存"}.get(origin, origin)
    print(f"    {label:<24} {row['n']:>4} 篇   共 {int(row['chars'] or 0) / 10000:.1f} 万字")

print()
print("  已确认「永久取不到」的原因分布：")
rows = db.query(
    "SELECT last_error, COUNT(*) AS n FROM fulltext_attempts "
    "WHERE permanent = 1 GROUP BY last_error ORDER BY n DESC LIMIT 8"
)
for row in rows:
    print(f"    {row['n']:>4} 篇   {(row['last_error'] or '')[:70]}")

print()
print("  抽样：已入库全文的文献")
for row in db.query(
    "SELECT p.title, f.origin, f.char_count FROM paper_fulltext f "
    "JOIN papers p ON p.paper_id = f.paper_id ORDER BY f.char_count DESC LIMIT 5"
):
    print(f"    {row['char_count']:>7} 字  [{row['origin']}]  {(row['title'] or '')[:46]}")
