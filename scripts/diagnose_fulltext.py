"""诊断：知识库里有多少文献具备获取开放获取全文的条件。

回答"为什么补齐全文抓不到东西"：需要区分三种情况
  * 有 PMCID        → Europe PMC / PMC 有 JATS 全文，最可靠
  * 有 OA 全文链接   → 可下载 PDF（需要 PyMuPDF 解析）
  * 只有 DOI        → 需要 Unpaywall 之类的服务再找一次合法 OA 副本
  * 都没有          → 非开放获取，只能保留元数据与出版商链接
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
total = db.scalar("SELECT COUNT(*) FROM papers", default=0)
print(f"总文献数：{total}")

rows = {
    "有 PMCID": "pmcid IS NOT NULL AND pmcid <> ''",
    "标记为开放获取": "is_open_access = 1",
    "有 OA 全文链接": "full_text_url IS NOT NULL AND full_text_url <> ''",
    "有 DOI": "doi IS NOT NULL AND doi <> ''",
    "有摘要": "abstract IS NOT NULL AND abstract <> ''",
    "已入库全文": "paper_id IN (SELECT paper_id FROM paper_fulltext)",
}
for label, where in rows.items():
    n = db.scalar(f"SELECT COUNT(*) FROM papers WHERE {where}", default=0)
    pct = (n / total * 100) if total else 0
    print(f"  {label:<16} {n:>5} 篇  ({pct:5.1f}%)")

print()
print("组合情况（决定能不能抓到全文）：")
combo = db.query(
    """
    SELECT
      CASE
        WHEN pmcid IS NOT NULL AND pmcid <> '' THEN 'A 有 PMCID（Europe PMC 可取）'
        WHEN full_text_url IS NOT NULL AND full_text_url <> '' THEN 'B 有 OA 链接（可下 PDF）'
        WHEN is_open_access = 1 THEN 'C 标称 OA 但无链接（需 Unpaywall 再找）'
        WHEN doi IS NOT NULL AND doi <> '' THEN 'D 仅有 DOI（需 Unpaywall 再找）'
        ELSE 'E 无任何线索（只能留元数据）'
      END AS kind,
      COUNT(*) AS n
    FROM papers GROUP BY kind ORDER BY n DESC
    """
)
for r in combo:
    print(f"  {r['kind']:<34} {r['n']:>5} 篇")

print()
print("开放获取标记来源分布（说明 is_open_access 是怎么来的）：")
for r in db.query(
    "SELECT source, COUNT(*) AS n, SUM(is_open_access) AS oa "
    "FROM papers GROUP BY source ORDER BY n DESC"
):
    oa = r["oa"] or 0
    print(f"  {r['source']:<14} {r['n']:>4} 篇，其中标记 OA {oa:>4} 篇")

print()
print("抽样：标记为开放获取、但没有 PMCID 的 5 篇（看看有什么可用线索）：")
for r in db.query(
    "SELECT paper_id, title, doi, full_text_url, pmcid FROM papers "
    "WHERE is_open_access = 1 AND (pmcid IS NULL OR pmcid = '') LIMIT 5"
):
    print(f"  #{r['paper_id']} {(r['title'] or '')[:48]}")
    print(f"      doi={r['doi'] or '无'}  full_text_url={r['full_text_url'] or '无'}")
