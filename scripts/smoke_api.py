"""数据源联调测试：真实调用各学术 API 并打印诊断。

    .python\\python.exe scripts\\smoke_api.py

这是**联网**测试，用于验证：

* 各数据源的请求参数、限流、重试是否按预期工作
* 响应解析是否与实际返回结构一致（数据结构变更能第一时间暴露）
* 跨库去重是否把同一篇文献正确合并

CNKI 属于尽力而为的数据源，解析失败是预期内的结果之一。
"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

from medscholar.api import (  # noqa: E402
    ArxivClient,
    CnkiClient,
    CrossrefClient,
    EuropePMCClient,
    OpenAlexClient,
    PubMedClient,
    SearchFilters,
    SemanticScholarClient,
    SourceRegistry,
)
from medscholar.config import AppConfig  # noqa: E402

EN_QUERY = "accelerated rTMS post-stroke depression"
ZH_QUERY = "加速rTMS治疗卒中后抑郁"


async def probe(client_cls, query: str, *, limit: int = 5):
    client = client_cls(config=AppConfig())
    t0 = time.perf_counter()
    try:
        await client.start()
        papers = await client.search(query, limit=limit, filters=SearchFilters())
        elapsed = int((time.perf_counter() - t0) * 1000)
        print(f"\n=== {client.label} ({client.name}) — {len(papers)} 篇 / {elapsed} ms ===")
        for paper in papers[:3]:
            year = paper.pub_year or "----"
            doi = paper.doi or "-"
            print(f"  [{year}] {paper.title[:78]}")
            print(
                f"        作者 {len(paper.authors)} | 摘要 {len(paper.abstract)} 字 | "
                f"被引 {paper.cited_by_count} | OA={paper.is_open_access} | doi={doi[:40]}"
            )
            if paper.pmid or paper.pmcid:
                print(f"        pmid={paper.pmid} pmcid={paper.pmcid}")
        return papers
    except Exception as exc:
        elapsed = int((time.perf_counter() - t0) * 1000)
        print(f"\n=== {client_cls.label} ({client_cls.name}) — 失败 / {elapsed} ms ===")
        print(f"  {type(exc).__name__}: {exc}")
        return []
    finally:
        await client.close()


async def test_abstract_reconstruction() -> None:
    client = OpenAlexClient(config=AppConfig())
    await client.start()
    try:
        papers = await client.search(EN_QUERY, limit=3)
        sample = next((p for p in papers if p.abstract), None)
        print("\n=== OpenAlex 倒排索引还原 ===")
        print("  " + (sample.abstract[:220] if sample else "（未取到摘要）"))
    finally:
        await client.close()


async def test_reference_graph() -> None:
    """引用图谱 + 开放获取全文（全文只在 PMCID 下提供）。"""
    client = EuropePMCClient(config=AppConfig())
    await client.start()
    try:
        papers = await client.search(
            "transcranial magnetic stimulation stroke rehabilitation", limit=15
        )
        with_pmc = next((p for p in papers if p.pmcid), None)
        if with_pmc:
            print(f"\n=== Europe PMC 开放获取全文（{with_pmc.pmcid}） ===")
            text = await client.fulltext(with_pmc)
            print(f"  {with_pmc.title[:60]}")
            print(f"  全文长度：{len(text)} 字")
            if text:
                print(f"  预览：{text[:200]}")

        print("\n=== Europe PMC 引用图谱 ===")
        for target in papers[:3]:
            if not target.pmid:
                continue
            cites = await client.citations(target)
            refs = await client.references(target)
            print(f"  · {target.title[:52]}")
            print(f"      参考文献 {len(refs)} 条 | 被引 {len(cites)} 条")
            if refs:
                break
    finally:
        await client.close()


async def test_registry() -> None:
    cfg = AppConfig()
    registry = SourceRegistry(config=cfg)
    print("\n=== 跨库并发检索 + 去重 ===")
    for label, query in (("英文", EN_QUERY), ("中文", ZH_QUERY)):
        outcome = await registry.search(query, limit=40, per_source_limit=10)
        print(f"\n--- {label}课题：{query} ---")
        print(outcome.summary())
    await registry.close()


async def test_parser_unit() -> None:
    """不联网的解析器自检：确保 XML/HTML 解析逻辑本身可用。"""
    print("\n=== 解析器离线自检 ===")
    from medscholar.api.openalex_client import reconstruct_abstract
    from medscholar.textutil import build_match_query, segment_cjk

    assert reconstruct_abstract({"Post-stroke": [0], "depression": [1]}) == "Post-stroke depression"
    assert segment_cjk("加速rTMS治疗") == "加 速 rTMS 治 疗"
    print("  segment_cjk       :", segment_cjk("加速rTMS治疗卒中后抑郁"))
    print("  build_match_query :", build_match_query("加速rTMS治疗卒中后抑郁"))
    print("  abstract 还原     :", reconstruct_abstract({"Post-stroke": [0], "depression": [1]}))
    print("  离线自检通过")


async def main() -> int:
    print("MedScholar Agent — 数据源联调")
    print("=" * 78)

    await test_parser_unit()

    for cls, query in (
        (PubMedClient, EN_QUERY),
        (EuropePMCClient, EN_QUERY),
        (OpenAlexClient, EN_QUERY),
        (CrossrefClient, EN_QUERY),
        (SemanticScholarClient, EN_QUERY),
        (ArxivClient, "transcranial magnetic stimulation depression"),
        (CnkiClient, ZH_QUERY),
    ):
        await probe(cls, query, limit=5)

    await test_abstract_reconstruction()
    await test_reference_graph()
    await test_registry()
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
