"""端到端验证（**联网 + 真实 LLM**）。

按需求文档 5.4 的验收要求跑真实课题「加速rTMS治疗卒中后抑郁」：

1. 嵌入后端实测（Ollama nomic-embed-text，768 维）
2. LLM 后端实测与延迟测量
3. 多源真实检索
4. 入库 + 真实向量生成
5. 混合检索验证（BM25 + 向量 + RRF，验证向量路真的参与）
6. 完整四节点工作流：Plan → Execute → Reflect → Synthesize → Review → 成稿
7. 引用格式与导出验证

    .python\\python.exe -u scripts\\e2e.py
    .python\\python.exe -u scripts\\e2e.py --model qwen3.5:4b
    .python\\python.exe -u scripts\\e2e.py --skip-run       # 只跑到检索+嵌入
"""

from __future__ import annotations

import argparse
import asyncio
import os
import shutil
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

TOPIC_ZH = "加速rTMS治疗卒中后抑郁"
TOPIC_EN = "accelerated rTMS post-stroke depression"

FAILED: list[str] = []


def ok(label: str, condition: bool, detail: str = "") -> None:
    mark = "ok  " if condition else "FAIL"
    print(f"  {mark} {label}" + (f"  [{detail}]" if detail else ""), flush=True)
    if not condition:
        FAILED.append(label)


def section(title: str) -> None:
    print("\n" + "=" * 74, flush=True)
    print(title, flush=True)
    print("=" * 74, flush=True)


async def main() -> int:
    parser = argparse.ArgumentParser(description="MedScholar 端到端验证")
    parser.add_argument("--model", default=None, help="覆盖 LLM 模型")
    parser.add_argument("--fresh", action="store_true", help="清空数据目录后重跑")
    parser.add_argument("--skip-run", action="store_true", help="跳过完整工作流")
    parser.add_argument("--sources", nargs="*", default=None)
    args = parser.parse_args()

    data_dir = ROOT / "data" / "e2e"
    if args.fresh and data_dir.exists():
        shutil.rmtree(data_dir, ignore_errors=True)
    data_dir.mkdir(parents=True, exist_ok=True)
    os.environ["MEDSCHOLAR_HOME"] = str(data_dir)

    if args.model:
        os.environ["MEDSCHOLAR_LLM_MODEL"] = args.model

    from medscholar.config import get_config

    cfg = get_config()
    if args.model:
        cfg.llm.model = args.model

    print(f"数据目录 : {cfg.home}")
    print(f"数据库   : {cfg.db_path}")
    print(f"LLM      : {cfg.llm.provider} / {cfg.llm.model}")
    print(f"嵌入模型 : {cfg.embedding.provider} / {cfg.embedding.model}")

    # ------------------------------------------------------------ 1 嵌入
    section("1) 嵌入后端实测")
    from medscholar.embedding.providers import get_provider

    provider = get_provider(cfg)
    samples = [
        "加速重复经颅磁刺激治疗卒中后抑郁的临床疗效观察",              # 0 中文·rTMS·抑郁
        "accelerated repetitive transcranial magnetic stimulation for post-stroke depression",  # 1 英文·同主题
        "糖尿病患者血糖控制与饮食干预的随机对照研究",                    # 2 中文·无关主题
    ]
    started = time.perf_counter()
    try:
        vectors = await asyncio.wait_for(provider.embed(samples), timeout=180)
    except Exception as exc:
        ok("嵌入调用", False, f"{type(exc).__name__}: {exc}")
        vectors = []
    else:
        elapsed = time.perf_counter() - started
        ok("嵌入调用成功", bool(vectors), f"{elapsed:.1f}s / {len(vectors)} 条")
        if vectors:
            dim = len(vectors[0])
            ok(f"向量维度 = {cfg.embedding.dim}", dim == cfg.embedding.dim, f"实际 {dim}")

            def cosine(a, b):
                dot = sum(x * y for x, y in zip(a, b))
                na = sum(x * x for x in a) ** 0.5
                nb = sum(y * y for y in b) ** 0.5
                return dot / (na * nb) if na and nb else 0.0

            same_topic = cosine(vectors[0], vectors[1])   # 中文rTMS ↔ 英文rTMS（同主题跨语言）
            unrelated = cosine(vectors[0], vectors[2])    # 中文rTMS ↔ 中文糖尿病（无关）
            print(f"       同主题·跨语言（rTMS 中↔英） : {same_topic:.4f}")
            print(f"       无关主题·同语言（rTMS↔糖尿病）: {unrelated:.4f}")
            print(f"       差值                          : {same_topic - unrelated:+.4f}")
            ok(
                "语义可用：同主题跨语言 > 无关主题同语言",
                same_topic > unrelated,
                f"{same_topic:.4f} vs {unrelated:.4f} —— "
                "若此项失败，说明该嵌入模型不适合本场景的跨语言检索，建议改用 bge-m3",
            )

    # ------------------------------------------------------------ 2 LLM
    section("2) LLM 后端实测")
    from medscholar.llm.client import get_llm

    llm = get_llm(cfg)
    started = time.perf_counter()
    try:
        reply = await asyncio.wait_for(
            llm.chat(
                [{"role": "user", "content": "用一句话说明 rTMS 是什么。"}],
                temperature=0.2,
                max_tokens=120,
            ),
            timeout=300,
        )
        elapsed = time.perf_counter() - started
        ok("LLM 调用成功", bool(reply.strip()), f"{elapsed:.1f}s")
        print(f"       回复：{reply.strip()[:160]}")
    except Exception as exc:
        ok("LLM 调用成功", False, f"{type(exc).__name__}: {str(exc)[:120]}")

    # ------------------------------------------------------ 3 真实多源检索
    section("3) 多源真实检索")
    from medscholar.api import SearchFilters, SourceRegistry

    sources = args.sources or ["pubmed", "europepmc", "openalex", "crossref"]
    registry = SourceRegistry(config=cfg)
    outcome = await registry.search(
        TOPIC_EN, sources=sources, limit=40, per_source_limit=15,
        filters=SearchFilters(year_from=2015),
    )
    print(outcome.summary())
    ok("跨库检索有结果", len(outcome.papers) >= 10, f"{len(outcome.papers)} 篇")
    ok("去重生效（原始数 ≥ 去重后）", outcome.raw_count >= len(outcome.papers),
       f"raw={outcome.raw_count} merged={len(outcome.papers)}")

    zh_outcome = await registry.search(
        TOPIC_ZH, sources=["openalex", "crossref"], limit=20, per_source_limit=15
    )
    print(f"\n中文课题检索：{len(zh_outcome.papers)} 篇")
    for paper in zh_outcome.papers[:3]:
        print(f"   · [{paper.pub_year}] {paper.title[:64]}")
    ok("中文文献通路可用（OpenAlex language:zh）", len(zh_outcome.papers) >= 1,
       f"{len(zh_outcome.papers)} 篇")

    # ------------------------------------------------------ 4 入库 + 嵌入
    section("4) 入库与真实向量生成")
    from medscholar.db import repo
    from medscholar.db.connect import get_db

    db = get_db()
    all_papers = outcome.papers + zh_outcome.papers
    started = time.perf_counter()

    # 关键回归路径：FastAPI 的 /api/search/live 与 /api/maintenance/embed 都通过
    # asyncio.to_thread 调用这个同步函数，而它会在线程里新起一个事件循环。
    # 如果嵌入提供方跨事件循环复用 httpx 客户端，这里会**永久挂起**。
    try:
        saved = await asyncio.wait_for(
            asyncio.to_thread(repo.insert_papers, all_papers, embed=True), timeout=600
        )
    except asyncio.TimeoutError:
        ok("入库 + 嵌入（worker 线程内起新事件循环）", False, "超时 600s —— 疑似跨事件循环复用 HTTP 客户端")
        return 1
    elapsed = time.perf_counter() - started
    print(f"入库 {saved['total']} 篇：新增 {saved['new']}，更新 {saved['updated']}（{elapsed:.1f}s）")
    embedded = saved.get("embedded") or {}
    print(f"向量：{embedded.get('embedded', 0)} 成功 / {embedded.get('failed', 0)} 失败")
    ok("入库成功", saved["new"] + saved["updated"] >= 10, str(saved["total"]))
    ok("向量生成成功（回归：不得跨事件循环挂起）", embedded.get("embedded", 0) >= 5,
       f"embedded={embedded.get('embedded')} errors={embedded.get('errors')}")

    stats = db.stats()
    print(f"知识库：{stats['papers']} 篇 / 向量覆盖 {stats['embedding_coverage']:.0%} / 后端 {stats['vector_backend']}")
    ok("向量覆盖率 > 50%", stats["embedding_coverage"] > 0.5, f"{stats['embedding_coverage']:.2f}")

    # ------------------------------------------------------ 5 混合检索
    section("5) 混合检索（BM25 + 向量 + RRF）")
    from medscholar.retrieval import search_knowledge_base

    hits = await search_knowledge_base("rTMS 治疗卒中后抑郁的疗效", top_k=8)
    ok("混合检索有结果", bool(hits), f"{len(hits)} 条")
    for hit in hits[:5]:
        print(
            f"   · score={hit.score:.5f} [{hit.matched_by:<12}] "
            f"bm25#{hit.fts_rank} vec#{hit.vector_rank}  {hit.paper.title[:52]}"
        )
    ok("至少一条被两路同时命中（融合生效）",
       any(h.matched_by == "bm25+vector" for h in hits),
       str([h.matched_by for h in hits[:6]]))

    # ------------------------------------------------------ 6 完整工作流
    if not args.skip_run:
        section("6) 完整 Agent 工作流（Plan → Execute → Reflect → Synthesize → Review）")
        from medscholar.agent.runtime import get_runtime

        runtime = get_runtime(cfg, db)
        handle = await runtime.start(
            topic=TOPIC_ZH,
            sources=sources,
            citation_style="gb7714",
            require_approval=False,
            offline=False,
        )
        print(f"run_id={handle.run_id}  session_id={handle.state.session_id}\n")

        started = time.perf_counter()
        phases: list[str] = []
        artifact_content = ""
        artifact_entries: list[dict] = []
        review: dict = {}
        critique: dict = {}
        tokens = 0
        async for event in runtime.stream(handle.run_id, timeout=3600):
            kind, data = event.type, event.data
            if kind == "phase":
                phases.append(data.get("phase", ""))
                print(f"\n▶ [{time.perf_counter() - started:6.1f}s] {data.get('label')}", flush=True)
            elif kind == "status":
                print(f"    · {data.get('message')}", flush=True)
            elif kind == "search_result":
                counts = ", ".join(
                    f"{s['label']}={s['count']}" if s["ok"] else f"{s['label']}=✗"
                    for s in data.get("sources", [])
                )
                print(f"    · 「{data.get('query')}」→ {data.get('count')} 篇 ({counts})", flush=True)
            elif kind == "papers":
                print(f"    · 候选池：{data.get('count')} 篇", flush=True)
            elif kind == "critique":
                critique = data
                print(f"    · 证据质量：{data.get('overall', {}).get('evidence_quality')}", flush=True)
            elif kind == "token":
                tokens += len(data.get("text", ""))
            elif kind == "review":
                review = data
                print(f"    · 审查：{data.get('verdict')} {data.get('score')}/10 "
                      f"（{len(data.get('issues', []))} 个问题）", flush=True)
            elif kind == "artifact":
                artifact_content = data.get("content", "")
                artifact_entries = data.get("reference_entries", []) or []
                print(f"    · 产物已保存 artifact_id={data.get('artifact_id')}", flush=True)
            elif kind == "error":
                print(f"    ! {data.get('message')}", flush=True)
            elif kind == "done":
                print(f"\n完成，耗时 {time.perf_counter() - started:.1f}s", flush=True)

        ok("走完 Plan 阶段", "plan" in phases, str(phases))
        ok("走完 Execute 阶段", "execute" in phases, str(phases))
        ok("走完 Reflect 阶段", "reflect" in phases, str(phases))
        ok("走完 Synthesize 阶段", "synthesize" in phases, str(phases))
        ok("走完 Review 阶段", "review" in phases, str(phases))
        ok("生成了综述草稿", len(artifact_content) > 800, f"{len(artifact_content)} 字")
        ok("流式输出有内容（token 事件）", tokens > 500, f"{tokens} 字")
        ok("Critic 给出了评估", bool(critique.get("assessments")), str(len(critique.get("assessments", []))))
        ok("Review 给出了结论", bool(review), str(review.get("verdict")))

        # 引用一致性：正文 [n] 必须都能在参考文献里找到
        from medscholar.agent.formatter import FormatterAgent

        entries = sorted(handle.state.citation_map.items())
        report = FormatterAgent.validate_citations(artifact_content, [i for i, _ in entries])
        print(f"\n     引用校验：{report.summary()}")
        ok("正文引用编号全部合法", report.ok, str(sorted(set(report.missing_from_list))))

        print("\n" + "-" * 74)
        print("综述草稿（前 2600 字）：")
        print("-" * 74)
        print(artifact_content[:2600])
        print("-" * 74)

        if artifact_entries:
            print("\n参考文献（前 5 条）：")
            for entry in artifact_entries[:5]:
                print(f"   {entry.get('text', '')[:150]}")

        # 产物落库校验
        if handle.state.artifact_id:
            stored = await asyncio.to_thread(repo.get_artifact, handle.state.artifact_id)
            ok("产物已落库", bool(stored and stored.get("content")),
               str(handle.state.artifact_id))

    # ------------------------------------------------------ 7 引用与导出
    section("7) 引用格式与导出")
    from medscholar.cite import format_citation, to_bibtex
    from medscholar.export.exporters import export_csv, export_json, write_export

    papers = await asyncio.to_thread(repo.list_papers, limit=5)
    for style in ("gb7714", "vancouver", "apa7"):
        print(f"\n[{style}]")
        for index, paper in enumerate(papers[:2], start=1):
            print("  " + format_citation(paper, style, index=index))
    print("\n[bibtex]")
    print(to_bibtex(papers[0]))

    for fmt, content in (
        ("bibtex", "\n\n".join(to_bibtex(p) for p in papers)),
        ("gb7714", "\n".join(format_citation(p, "gb7714", index=i)
                             for i, p in enumerate(papers, 1))),
        ("csv", export_csv(papers)),
        ("json", export_json(papers)),
    ):
        path = write_export(content, name="e2e验证", fmt=fmt, config=cfg)
        ok(f"导出 {fmt}", path.exists() and path.stat().st_size > 0, path.name)

    section(f"结果：{'全部通过' if not FAILED else f'{len(FAILED)} 项失败'}")
    for item in FAILED:
        print(f"  - {item}")
    print("E2E:", "PASS" if not FAILED else "FAIL")
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
