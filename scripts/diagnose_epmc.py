"""诊断：为什么有 PMCID 却拿不到 Europe PMC 全文。"""

from __future__ import annotations

import asyncio
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

logging.basicConfig(level=logging.DEBUG, format="%(levelname)s %(name)s: %(message)s")

EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


async def main() -> int:
    import httpx

    from medscholar.db.connect import get_db

    db = get_db()
    rows = db.query(
        "SELECT paper_id, title, pmid, pmcid FROM papers "
        "WHERE pmcid IS NOT NULL AND pmcid <> '' LIMIT 4"
    )

    async with httpx.AsyncClient(timeout=60.0, follow_redirects=True,
                                 headers={"User-Agent": "MedScholar/1.0"}) as client:
        for row in rows:
            pmcid = row["pmcid"]
            pmid = row["pmid"]
            print("=" * 76)
            print(f"#{row['paper_id']}  {pmcid}  pmid={pmid}")
            print(f"  标题: {(row['title'] or '')[:60]}")

            # 变体 1：不带 source 段（当前实现）
            url1 = f"{EPMC}/{pmcid}/fullTextXML"
            # 变体 2：带 source 段
            url2 = f"{EPMC}/PMC/{pmcid}/fullTextXML"
            # 变体 3：先查这篇文献在 Europe PMC 里的元数据，看它到底有没有全文
            meta = f"{EPMC}/search"
            for label, url, params in (
                ("无 source 段", url1, None),
                ("带 PMC 段", url2, None),
                ("元数据查询", meta, {"query": f"PMCID:{pmcid}", "format": "json",
                                      "resultType": "core"}),
            ):
                try:
                    r = await client.get(url, params=params)
                    body = r.text
                    note = ""
                    if label == "元数据查询" and r.status_code == 200:
                        try:
                            res = (r.json().get("resultList") or {}).get("result") or []
                            if res:
                                hit = res[0]
                                note = (f"  isOpenAccess={hit.get('isOpenAccess')} "
                                        f"inEPMC={hit.get('inEPMC')} inPMC={hit.get('inPMC')} "
                                        f"hasTextMinedTerms={hit.get('hasTextMinedTerms')} "
                                        f"fullTextIdList={hit.get('fullTextIdList')}")
                            else:
                                note = "  该 PMCID 在 Europe PMC 中查不到记录"
                        except Exception as exc:
                            note = f"  解析失败 {exc}"
                    elif r.status_code == 200:
                        note = f"  含 <body>={('<body' in body)}  长度 {len(body)}"
                    print(f"    {label:<12} HTTP {r.status_code}  {note}")
                except Exception as exc:
                    print(f"    {label:<12} {type(exc).__name__}: {exc}")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
