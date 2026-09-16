"""开放获取全文获取路径探查。

Europe PMC 的 ``fullTextXML`` 在不同 PMCID 形态下 URL 不一致，
本脚本枚举候选形式，找出真正可用的那一个；同时验证 PubMed 的 PMC 通路。
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

import httpx  # noqa: E402

UA = {"User-Agent": "MedScholarAgent/1.0 (research tool)"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"
EUTILS = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils"

# 已知开放获取、且 Europe PMC 有全文的经典论文
SAMPLES = [
    ("PMC7431489", "32849235"),  # NoTSAD 2020, Front Neurol
    ("PMC10239808", "37283739"),  # 2023 Front Immunol
    ("PMC5334499", "28242750"),  # 2017 高被引 OA
]


async def main() -> int:
    async with httpx.AsyncClient(timeout=45.0, follow_redirects=True, headers=UA) as client:
        print("=" * 78)
        print("Europe PMC fullTextXML URL 形态枚举")
        for pmcid, pmid in SAMPLES:
            print(f"\n--- {pmcid} (pmid={pmid})")
            variants = [
                f"{EPMC}/PMC/{pmcid}/fullTextXML",
                f"{EPMC}/{pmcid}/fullTextXML",
                f"{EPMC}/MED/{pmid}/fullTextXML",
                f"{EPMC}/PMC/{pmcid}/textMinedTerms",
            ]
            for url in variants:
                try:
                    r = await client.get(url)
                except Exception as exc:
                    print(f"    {url.replace(EPMC, ''):<44} 异常 {type(exc).__name__}")
                    continue
                body = r.text
                marker = ""
                if r.status_code == 200:
                    if "<article" in body[:5000] or "<body" in body[:20000]:
                        marker = "  ← 含 JATS 正文"
                    elif len(body) < 400:
                        marker = f"  ← 内容过短：{body[:80]!r}"
                print(f"    {url.replace(EPMC, ''):<44} HTTP {r.status_code} {len(body):>8} 字节{marker}")

        print("\n" + "=" * 78)
        print("PubMed E-utilities db=pmc 通路")
        for pmcid, _ in SAMPLES[:2]:
            r = await client.get(f"{EUTILS}/efetch.fcgi", params={
                "db": "pmc", "id": pmcid, "retmode": "xml", "tool": "medscholar-agent",
            })
            body = r.text
            has_body = "<body" in body
            print(f"  {pmcid}: HTTP {r.status_code}  {len(body)} 字节  含<body>={has_body}")
            if has_body:
                import re
                text = " ".join(re.sub(r"<[^>]+>", " ", body).split())
                print(f"      纯文本预览：{text[:150]}")

        print("\n" + "=" * 78)
        print("Europe PMC references 接口现状")
        r = await client.get(f"{EPMC}/MED/32849235/references", params={"format": "json"})
        print(f"  HTTP {r.status_code}  {r.text[:150]}")

    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
