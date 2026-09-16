"""中文文献获取路径探查。

背景：``search.cnki.com.cn`` 的检索结果已改为 **前端 JS 渲染**，
服务端返回的 HTML 里没有任何文献条目（已实测确认）。

本脚本验证两条路线：

A. CNKI 的 AJAX 接口是否仍可用（需要正确的 Referer / X-Requested-With）
B. 稳定的免费替代通路：
   * PubMed ``chinese[la]``      —— 被 PubMed 收录的中文刊（带英文摘要）
   * OpenAlex ``language:zh``    —— 中文语种过滤
   * Crossref                    —— 注册了 DOI 的中文期刊
"""

from __future__ import annotations

import asyncio
import json
import re
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

UA = "MedScholarAgent/1.0 (academic research tool)"
QUERY = "卒中后抑郁"


def count_items(obj) -> int:
    if isinstance(obj, list):
        return len(obj)
    if isinstance(obj, dict):
        for key in ("data", "result", "list", "items", "records", "results"):
            if key in obj:
                return count_items(obj[key])
        return len(obj)
    return 0


async def try_cnki_ajax(client: httpx.AsyncClient) -> None:
    print("=" * 78)
    print("A. CNKI AJAX 接口尝试")
    # 注意：HTTP 头必须是 ASCII，中文查询词必须先百分号编码
    from urllib.parse import quote

    referer = f"https://search.cnki.com.cn/Search/Result?content={quote(QUERY)}"
    headers = {
        "User-Agent": UA,
        "Accept": "application/json, text/javascript, */*; q=0.01",
        "X-Requested-With": "XMLHttpRequest",
        "Referer": referer,
        "Origin": "https://search.cnki.com.cn",
        "Accept-Language": "zh-CN,zh;q=0.9",
        "Content-Type": "application/x-www-form-urlencoded; charset=UTF-8",
    }
    attempts = [
        ("GET  ListResult", "GET", "https://search.cnki.com.cn/Search/ListResult",
         {"content": QUERY, "type": "0", "page": "1"}, None),
        ("POST ListResult", "POST", "https://search.cnki.com.cn/Search/ListResult",
         None, {"content": QUERY, "type": "0", "page": "1"}),
        ("GET  Result&page=2", "GET", "https://search.cnki.com.cn/Search/Result",
         {"content": QUERY, "type": "0", "order": "1", "page": "2"}, None),
        ("GET  kns defaultresult", "GET", "https://kns.cnki.net/kns8/defaultresult/index",
         {"kw": QUERY}, None),
    ]
    for label, method, url, params, data in attempts:
        try:
            r = await client.request(method, url, params=params, data=data,
                                     headers=headers, timeout=25.0)
        except Exception as exc:
            print(f"  {label:<22} 异常 {type(exc).__name__}: {str(exc)[:70]}")
            continue
        body = r.text
        print(f"  {label:<22} HTTP {r.status_code}  {len(body)} 字节  "
              f"{r.headers.get('content-type', '')[:40]}")
        if r.status_code == 200 and body.strip():
            print(f"      预览：{body[:140].strip()}")
            try:
                payload = r.json()
                print(f"      JSON 顶层键：{list(payload)[:10]}  条目数≈{count_items(payload)}")
                (CACHE / "cnki_ajax.json").write_text(
                    json.dumps(payload, ensure_ascii=False)[:20000], encoding="utf-8")
            except ValueError:
                (CACHE / "cnki_ajax.html").write_text(body[:80000], encoding="utf-8")
        # 关键判据：页面里是否有真正的文献条目链接
        has_items = bool(re.search(r'(kns\.cnki\.net|/KCMS/detail|dbcode=|filename=)', body, re.I))
        print(f"      含文献条目链接：{'是' if has_items else '否'}")
    print()


async def try_pubmed_chinese(client: httpx.AsyncClient) -> None:
    print("=" * 78)
    print("B1. PubMed —— 中文语种过滤 chinese[la]")
    url = "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi"
    r = await client.get(url, params={
        "db": "pubmed",
        "term": f'("{QUERY}"[Title/Abstract] OR "post-stroke depression"[Title/Abstract]) AND chinese[la]',
        "retmax": 5, "retmode": "json", "sort": "relevance", "tool": "medscholar-agent",
    }, timeout=30.0)
    ids = r.json().get("esearchresult", {}).get("idlist", [])
    print(f"  命中 PMID：{ids}")
    if ids:
        r2 = await client.get(url.replace("esearch", "esummary"), params={
            "db": "pubmed", "id": ",".join(ids), "retmode": "json",
        }, timeout=30.0)
        result = r2.json().get("result", {})
        for pmid in ids[:4]:
            item = result.get(pmid, {})
            print(f"    · [{item.get('pubdate','')[:4]}] {item.get('title','')[:70]}")
            print(f"      {item.get('fulljournalname','')} | lang={item.get('lang', [])}")
    print()


async def try_openalex_zh(client: httpx.AsyncClient) -> None:
    print("=" * 78)
    print("B2. OpenAlex —— language:zh 过滤")
    r = await client.get("https://api.openalex.org/works", params={
        "search": QUERY, "filter": "language:zh", "per-page": 5,
        "select": "id,doi,title,display_name,publication_year,language,cited_by_count,primary_location",
    }, timeout=30.0)
    data = r.json()
    print(f"  meta.count = {data.get('meta', {}).get('count')}")
    for item in data.get("results", [])[:5]:
        src = ((item.get("primary_location") or {}).get("source") or {}).get("display_name")
        print(f"    · [{item.get('publication_year')}] "
              f"{(item.get('title') or item.get('display_name') or '')[:70]}")
        print(f"      {src} | lang={item.get('language')} | 被引={item.get('cited_by_count')}")
    print()


async def try_crossref(client: httpx.AsyncClient) -> None:
    print("=" * 78)
    print("B3. Crossref —— 中文期刊 DOI 元数据")
    r = await client.get("https://api.crossref.org/works", params={
        "query": QUERY, "rows": 5, "mailto": "medscholar@example.org",
    }, timeout=30.0)
    data = r.json()
    message = data.get("message") if isinstance(data, dict) else None
    if not isinstance(message, dict):
        print(f"  非预期响应结构：{str(data)[:200]}")
        print()
        return
    items = message.get("items", [])
    print(f"  total-results = {message.get('total-results')}")
    cjk = 0
    for item in items:
        title = (item.get("title") or [""])[0]
        journal = (item.get("container-title") or [""])[0]
        year = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
        has_cjk = any("\u4e00" <= ch <= "\u9fff" for ch in title)
        cjk += int(has_cjk)
        print(f"    · [{year}] {title[:64]}  {'[中文]' if has_cjk else ''}")
        print(f"      {journal[:50]} | lang={item.get('language')} | 被引={item.get('is-referenced-by-count')}")
    print(f"  含中文标题：{cjk}/{len(items)}")
    print()


async def main() -> int:
    async with httpx.AsyncClient(follow_redirects=True, headers={"User-Agent": UA}) as client:
        await try_cnki_ajax(client)
        await try_pubmed_chinese(client)
        await try_openalex_zh(client)
        await try_crossref(client)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
