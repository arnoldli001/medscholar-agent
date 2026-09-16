"""端点原始响应探查（一次性诊断工具）。

用于确认实际返回结构，避免凭猜测写解析器：

    .python\\python.exe scripts\\probe_endpoints.py
"""

from __future__ import annotations

import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
CACHE = ROOT / ".cache"
CACHE.mkdir(exist_ok=True)

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

import httpx  # noqa: E402

UA = {"User-Agent": "MedScholarAgent/1.0 (research tool)"}
EPMC = "https://www.ebi.ac.uk/europepmc/webservices/rest"


def peek(obj, depth: int = 0, max_depth: int = 3) -> str:
    """打印 JSON 结构骨架（键名 + 类型），不打印大段内容。"""
    pad = "  " * depth
    if depth > max_depth:
        return f"{pad}..."
    if isinstance(obj, dict):
        lines = []
        for key, value in list(obj.items())[:14]:
            if isinstance(value, (dict, list)):
                lines.append(f"{pad}{key}: {type(value).__name__}")
                lines.append(peek(value, depth + 1, max_depth))
            else:
                text = str(value)
                lines.append(f"{pad}{key}: {text[:70]!r}")
        return "\n".join(lines)
    if isinstance(obj, list):
        if not obj:
            return f"{pad}[]（空）"
        return f"{pad}[{len(obj)} 项] 首项：\n" + peek(obj[0], depth + 1, max_depth)
    return f"{pad}{str(obj)[:70]!r}"


async def probe_epmc(client: httpx.AsyncClient) -> None:
    print("=" * 78)
    print("Europe PMC —— 先取一篇有引用的经典文献")
    # Lefaucheur 2019 rTMS 指南：高被引，参考文献充足
    r = await client.get(f"{EPMC}/search", params={
        "query": 'DOI:"10.1016/j.clinph.2019.11.002"', "format": "json", "resultType": "core",
    })
    hits = (r.json().get("resultList") or {}).get("result") or []
    if not hits:
        print("  未取到样本文献")
        return
    hit = hits[0]
    pmid, pmcid = hit.get("pmid"), hit.get("pmcid")
    print(f"  pmid={pmid} pmcid={pmcid} refCount={hit.get('referenceCount')} "
          f"citedBy={hit.get('citedByCount')}")

    for label, url in (
        ("references", f"{EPMC}/MED/{pmid}/references"),
        ("citations", f"{EPMC}/MED/{pmid}/citations"),
    ):
        r = await client.get(url, params={"format": "json", "pageSize": 5})
        print(f"\n--- {label}  HTTP {r.status_code}  {url}")
        if r.status_code == 200:
            data = r.json()
            print(peek(data))
        else:
            print("  " + r.text[:200])

    # 全文：分别试 MED 与 PMC 两种 source
    for source, ident in (("MED", pmid), ("PMC", pmcid)):
        if not ident:
            continue
        url = f"{EPMC}/{source}/{ident}/fullTextXML"
        r = await client.get(url)
        print(f"\n--- fullTextXML {source}/{ident}  HTTP {r.status_code}  长度 {len(r.text)}")
        if r.status_code == 200 and len(r.text) > 200:
            print("  " + r.text[:300].replace("\n", " "))


async def probe_cnki(client: httpx.AsyncClient) -> None:
    print("\n" + "=" * 78)
    print("CNKI 检索页结构探查")
    headers = {
        **UA,
        "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
        "Accept-Language": "zh-CN,zh;q=0.9",
    }
    candidates = [
        ("Result(content,type,order)", "https://search.cnki.com.cn/Search/Result",
         {"content": "卒中后抑郁", "type": "0", "order": "1"}),
        ("Result(kw)", "https://search.cnki.com.cn/Search/Result", {"kw": "卒中后抑郁"}),
        ("ListResult", "https://search.cnki.com.cn/Search/ListResult",
         {"content": "卒中后抑郁", "type": "0"}),
    ]
    for label, url, params in candidates:
        try:
            r = await client.get(url, params=params, headers=headers)
        except Exception as exc:
            print(f"\n--- {label}: 请求异常 {exc}")
            continue
        print(f"\n--- {label}  HTTP {r.status_code}  最终URL {r.url}")
        print(f"  长度 {len(r.text)}  Content-Type {r.headers.get('content-type')}")
        text = r.text
        out = CACHE / f"cnki_{label.split('(')[0]}.html"
        out.write_text(text, encoding="utf-8", errors="replace")
        print(f"  已保存 {out}")

        # 找出页面里出现的 class 名，帮助定位结果容器
        import re
        classes: dict[str, int] = {}
        for match in re.finditer(r'class="([^"]{1,80})"', text):
            for token in match.group(1).split():
                classes[token] = classes.get(token, 0) + 1
        top = sorted(classes.items(), key=lambda kv: -kv[1])[:25]
        print("  class 频次 Top25：" + ", ".join(f"{k}({v})" for k, v in top))
        if "kns.cnki.net" in text or "/Detail" in text:
            print("  ✓ 含文献详情链接")
        else:
            print("  ✗ 未发现文献详情链接（可能需要 JS 渲染或接口已变更）")


async def probe_arxiv(client: httpx.AsyncClient) -> None:
    print("\n" + "=" * 78)
    print("arXiv 查询语法对比（同一课题）")
    phrase = "accelerated rTMS post-stroke depression"
    variants = {
        'all:"phrase"': f'all:"{phrase}"',
        'AND 分词': " AND ".join(f'all:{w}' for w in phrase.split() if len(w) > 2),
        'abs:phrase': f'abs:"{phrase}"',
    }
    for label, expr in variants.items():
        r = await client.get(
            "http://export.arxiv.org/api/query",
            params={"search_query": expr, "max_results": 3},
            headers=UA,
        )
        import xml.etree.ElementTree as ET

        try:
            root = ET.fromstring(r.text)
            titles = [
                " ".join(e.findtext("{http://www.w3.org/2005/Atom}title", "").split())
                for e in root.findall("{http://www.w3.org/2005/Atom}entry")
            ]
        except ET.ParseError:
            titles = ["<解析失败>"]
        print(f"\n--- {label}: {expr[:70]}")
        for title in titles:
            print(f"    · {title[:76]}")


async def main() -> int:
    async with httpx.AsyncClient(timeout=40.0, follow_redirects=True, headers=UA) as client:
        await probe_epmc(client)
        await probe_arxiv(client)
        await probe_cnki(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
