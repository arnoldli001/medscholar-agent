"""MedScholar 命令行界面。

::

    medscholar serve                    启动 Web 工作台
    medscholar doctor                   环境自检（诊断为什么跑不起来）
    medscholar search "rTMS 卒中后抑郁"   联网检索并入库
    medscholar kb "rTMS 抑郁"            本地知识库混合检索
    medscholar run "加速rTMS治疗卒中后抑郁" 跑完整 Agent 工作流（终端流式输出）
    medscholar summarize 12             单篇文献速读
    medscholar cite 1 2 3 --style apa7  生成参考文献
    medscholar export 1 2 3 -f bibtex   导出到文件
    medscholar stats                    知识库统计
    medscholar mcp                      启动 MCP Server
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import sys
from typing import Any, Sequence

from . import __version__
from .config import get_config

__all__ = ["main", "build_parser"]


def _setup_stdout() -> None:
    """Windows 控制台默认 GBK，中文输出会乱码；同时确保实时刷出。

    ``line_buffering=True`` 很关键：输出被重定向到文件或管道时（例如
    ``run.bat > log.txt``），Python 默认是块缓冲，启动横幅要等到进程退出
    才会出现，看起来像"什么都没打印"。在真实控制台上本来就无缓冲，
    但显式设置可以避免这类困惑。
    """
    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace", line_buffering=True)
        except (AttributeError, ValueError, OSError):  # pragma: no cover
            pass


def _print_json(data: Any) -> None:
    print(json.dumps(data, ensure_ascii=False, indent=2, default=str))


def _source_badge(status: dict[str, Any]) -> str:
    """把单个数据源的状态渲染成 ``标签=结果``。

    必须区分「跳过」与「失败」—— 离线模式下所有数据源都是跳过，
    早期版本一律显示成「失败」，会让人误以为网络出了问题。
    """
    label = status.get("label") or status.get("name") or "?"
    if status.get("skipped"):
        return f"{label}=跳过"
    if status.get("ok"):
        return f"{label}={status.get('count', 0)}"
    return f"{label}=失败"


def _wait_for_port(host: str, port: int, timeout: float = 40.0) -> bool:
    """轮询直到端口真的开始接受连接。

    不要用固定 sleep 再开浏览器。 早期实现是 ``threading.Timer(1.5, open)``，
    但 uvicorn 从启动到监听通常要 2~4 秒（要导入模块、初始化数据库与向量表）。
    浏览器在服务还没监听时就打开，前端首屏的每个请求都会 "Failed to fetch"，
    用户看到的是满屏"无法连接后端服务"。实测确认过这个时序问题。
    """
    import socket
    import time

    target = "127.0.0.1" if host in {"0.0.0.0", "::", ""} else host
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(0.5)
            if sock.connect_ex((target, port)) == 0:
                return True
        time.sleep(0.2)
    return False


def _open_browser_when_ready(url: str, host: str, port: int) -> None:
    """等端口就绪后再打开浏览器（就绪后额外留 0.4s 让 uvicorn 稳定）。"""
    import time
    import webbrowser

    if _wait_for_port(host, port):
        time.sleep(0.4)
        webbrowser.open(url)
        print(f"  已在浏览器中打开 {url}")
    else:
        print(f"  ! 等待端口 {port} 就绪超时，请手动访问 {url}")


# ===================================================================== serve
def cmd_serve(args: argparse.Namespace) -> int:
    import threading

    import uvicorn

    from .server.app import app

    cfg = get_config()
    host = args.host or cfg.server.host
    port = args.port or cfg.server.port
    url = f"http://{'127.0.0.1' if host in {'0.0.0.0', '::'} else host}:{port}"

    print("=" * 72)
    print(f"  {cfg.app_name} v{__version__}")
    print("=" * 72)
    print(f"  工作台地址 : {url}")
    print(f"  接口文档   : {url}/docs")
    print(f"  数据目录   : {cfg.home}")
    print(f"  数据库     : {cfg.db_path}")
    print(f"  LLM        : {cfg.llm.provider} / {cfg.llm.model}")
    print(f"  嵌入模型   : {cfg.embedding.provider} / {cfg.embedding.model}（{cfg.embedding.dim} 维）")
    print("=" * 72)
    print("  按 Ctrl+C 停止服务")
    print()

    if cfg.server.open_browser and not args.no_browser:
        threading.Thread(
            target=_open_browser_when_ready,
            args=(url, host, port),
            daemon=True,
            name="open-browser",
        ).start()

    uvicorn.run(app, host=host, port=port, log_level=args.log_level)
    return 0


# ==================================================================== doctor
async def _doctor_async(args: argparse.Namespace) -> int:
    import platform
    import sqlite3

    import httpx

    from .db.connect import get_db
    from .embedding.providers import get_provider
    from .llm.client import get_llm

    cfg = get_config()
    problems: list[str] = []

    def line(label: str, ok: bool | None, detail: str) -> None:
        mark = "✓" if ok else ("✗" if ok is False else "·")
        print(f"  {mark} {label:<22} {detail}")

    print("=" * 72)
    print("MedScholar 环境自检")
    print("=" * 72)

    print("\n[运行环境]")
    line("Python", sys.version_info >= (3, 10), f"{platform.python_version()} @ {sys.executable}")
    line("操作系统", None, f"{platform.system()} {platform.release()} ({platform.machine()})")
    line("SQLite", tuple(int(x) for x in sqlite3.sqlite_version.split(".")) >= (3, 35, 0),
         sqlite3.sqlite_version)
    try:
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        line("FTS5 全文索引", True, "可用")
    except sqlite3.Error as exc:
        line("FTS5 全文索引", False, f"不可用：{exc}")
        problems.append("SQLite 缺少 FTS5 支持，关键词检索无法工作。")

    print("\n[数据目录]")
    cfg.ensure_dirs()
    line("数据目录", cfg.home.is_dir(), str(cfg.home))
    line("导出目录", cfg.export_dir.is_dir(), str(cfg.export_dir))

    print("\n[数据库]")
    try:
        db = get_db()
        stats = db.stats()
        line("数据库文件", db.path.exists(), str(db.path))
        line("向量后端", db.vec_available, db.vec_note or ("sqlite-vec" if db.vec_available else ""))
        if not db.vec_available:
            problems.append(
                "sqlite-vec 未能加载，已退化为纯 Python 向量检索（功能可用，但文献量很大时会变慢）。"
            )
        line("文献总数", None, str(stats["papers"]))
        line("向量覆盖", None, f"{stats['embedded']}/{stats['papers']}（{stats['embedding_coverage']:.0%}）")
        line("全文记录", None, str(stats["fulltext"]))
    except Exception as exc:
        line("数据库", False, f"初始化失败：{exc}")
        problems.append(f"数据库初始化失败：{exc}")

    print("\n[LLM 后端]")
    try:
        llm = get_llm(cfg)
        ok, message = await asyncio.wait_for(llm.health(), timeout=60.0)
        line(f"{cfg.llm.provider}", ok, message)
        if not ok:
            problems.append(
                f"LLM 不可用：{message}\n"
                "      → 本地模式：确认已运行 `ollama serve` 并 `ollama pull "
                f"{cfg.llm.model}`\n"
                "      → 云端模式：在 config.yaml 配置 llm.provider=deepseek 与 llm.api_key"
            )
    except (asyncio.TimeoutError, TimeoutError):
        # 注意：asyncio.TimeoutError 的字符串表示是空串，
        # 早期实现 f"LLM 探测失败：{exc}" 会输出"LLM 探测失败："后面什么都没有，
        # 让人完全无法判断发生了什么。
        line(cfg.llm.provider, False, "探测超时（60 秒无响应）")
        problems.append(
            "LLM 探测超时（60 秒无响应）。\n"
            f"      → 本地模型：{cfg.llm.model} 可能正在冷加载（实测 8B 模型要 20~35 秒），"
            "或 Ollama 正被其他程序占用\n"
            "      → 先手动确认一次：ollama run "
            f"{cfg.llm.model} \"你好\"\n"
            "      → 若长期无响应，换更小的模型（如 qwen3.5:4b）"
        )
    except Exception as exc:
        detail = f"{type(exc).__name__}: {exc}" if str(exc).strip() else type(exc).__name__
        line(cfg.llm.provider, False, detail)
        problems.append(f"LLM 探测失败：{detail}")

    print("\n[嵌入后端]")
    try:
        provider = get_provider(cfg)
        ok, message = await asyncio.wait_for(provider.probe(), timeout=60.0)
        line(f"{provider.name}/{provider.model}", ok, message)
        if not ok:
            problems.append(
                f"嵌入模型不可用：{message}\n"
                f"      → 执行 `ollama pull {cfg.embedding.model}`，"
                "或把 embedding.provider 改为 hashing 做无模型冒烟测试"
            )
    except Exception as exc:
        line(cfg.embedding.provider, False, str(exc))

    print("\n[学术数据源]")
    async with httpx.AsyncClient(timeout=20.0, follow_redirects=True) as client:
        probes = [
            ("PubMed", "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/esearch.fcgi?db=pubmed&term=rTMS&retmax=1&retmode=json"),
            ("Europe PMC", "https://www.ebi.ac.uk/europepmc/webservices/rest/search?query=rTMS&format=json&pageSize=1"),
            ("OpenAlex", "https://api.openalex.org/works?search=rTMS&per-page=1"),
            ("Crossref", "https://api.crossref.org/works?query=rTMS&rows=1"),
            ("Semantic Scholar", "https://api.semanticscholar.org/graph/v1/paper/search?query=rTMS&limit=1"),
            ("arXiv", "http://export.arxiv.org/api/query?search_query=all:rTMS&max_results=1"),
        ]
        for label, url in probes:
            try:
                response = await client.get(url)
                note = ""
                if response.status_code == 429:
                    note = "（限流；配置免费 API Key 可解除）"
                line(label, response.status_code < 400, f"HTTP {response.status_code}{note}")
            except Exception as exc:
                line(label, False, f"{type(exc).__name__}: {str(exc)[:60]}")

    print("\n[配置]")
    from .config import config_path

    line("配置文件", config_path().exists(), str(config_path()) + ("" if config_path().exists() else "（不存在，使用默认值）"))
    for name, settings in cfg.sources.as_dict().items():
        if name == "cnki" and not settings.enabled:
            line(f"  {name}", None, "已禁用（CNKI 公开检索页改为 JS 渲染，见文档）")
            continue
        line(f"  {name}", None, f"启用={settings.enabled} 限速={settings.rps}/s Key={'有' if settings.api_key else '无'}")

    print("\n" + "=" * 72)
    if problems:
        print(f"发现 {len(problems)} 个需要关注的问题：")
        for index, problem in enumerate(problems, start=1):
            print(f"  {index}. {problem}")
    else:
        print("全部检查通过，可以开始使用。")
    print("=" * 72)
    return 0 if not problems else 0


def cmd_doctor(args: argparse.Namespace) -> int:
    return asyncio.run(_doctor_async(args))


# ==================================================================== search
def cmd_search(args: argparse.Namespace) -> int:
    async def run() -> int:
        from .api import SearchFilters, close_registry, search_all
        from .db import repo

        filters = SearchFilters(year_from=args.year_from, year_to=args.year_to,
                                open_access_only=args.oa)
        outcome = await search_all(
            args.query, sources=args.sources or None, limit=args.limit, filters=filters
        )
        print(outcome.summary())
        print()

        for index, paper in enumerate(outcome.papers[: args.limit], start=1):
            year = paper.pub_year or "----"
            print(f"[{index}] ({year}) {paper.title}")
            print(f"     {paper.short_authors} | {paper.journal or '—'} | 被引 {paper.cited_by_count} | {paper.source}")
            if paper.doi:
                print(f"     DOI: {paper.doi}")
            print()

        if args.save and outcome.papers:
            result = repo.insert_papers(outcome.papers, embed=not args.no_embed)
            print(f"入库：新增 {result['new']} 篇，更新 {result['updated']} 篇")
            embedded = result.get("embedded") or {}
            if embedded:
                print(f"向量：成功 {embedded.get('embedded', 0)} / 失败 {embedded.get('failed', 0)}")
        if args.json:
            _print_json([p.to_dict() for p in outcome.papers])
        await close_registry()
        return 0

    return asyncio.run(run())


# ======================================================================== kb
def cmd_kb(args: argparse.Namespace) -> int:
    async def run() -> int:
        from .retrieval import search_knowledge_base

        hits = await search_knowledge_base(args.query, top_k=args.limit)
        if not hits:
            print(f"本地知识库中没有与「{args.query}」匹配的文献。")
            print("提示：先用 `medscholar search \"...\"` 检索并入库。")
            return 1
        print(f"本地知识库混合检索：{len(hits)} 条（BM25 + 向量 + RRF 融合）\n")
        for hit in hits:
            paper = hit.paper
            print(f"[{paper.paper_id}] score={hit.score:.5f} matched={hit.matched_by}")
            print(f"    ({paper.pub_year or '----'}) {paper.title}")
            print(f"    {paper.short_authors} | {paper.journal or '—'} | 被引 {paper.cited_by_count}")
            print()
        return 0

    return asyncio.run(run())


# ==================================================================== import
def cmd_import(args: argparse.Namespace) -> int:
    """导入题录文件。"""
    import json as _json

    from .importers import import_paths
    from .importers.parsers import SUPPORTED_FORMATS

    report = import_paths(
        args.files,
        source=args.source,
        embed=not args.no_embed,
        dry_run=args.dry_run,
    )

    if args.json:
        print(_json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        return 0 if (report.parsed or report.errors) else 1

    print("=" * 68)
    print("  导入题录文件")
    print("=" * 68)
    for name in report.files:
        print(f"  文件：{name}")
    if report.format:
        print(f"  识别格式：{report.format}")
    print()
    print(f"  {report.summary()}")
    if args.dry_run:
        print("  （--dry-run：没有写入数据库）")
    if report.sample_titles:
        print("\n  解析到的前几条：")
        for title in report.sample_titles[:5]:
            print(f"    · {title[:78]}")
    if report.errors:
        print("\n  问题：")
        for message in report.errors[:8]:
            print(f"    ! {message}")
    if not report.parsed and not report.errors:
        print(f"\n  支持的文件格式：{'、'.join(SUPPORTED_FORMATS)}")
        print("  Web of Science / Scopus / Embase / CNKI / 万方 的导出文件都可以。")
    return 0 if report.parsed else 1


# ==================================================================== zotero
def cmd_zotero(args: argparse.Namespace) -> int:
    """导入本机 Zotero 库。"""
    import asyncio

    from .importers import import_from_zotero

    async def run() -> int:
        try:
            report = await import_from_zotero(
                args.data_dir,
                limit=args.limit,
                embed=not args.no_embed,
                index_pdfs=not args.no_pdf,
            )
        except FileNotFoundError as exc:
            print(f"错误：{exc}", file=sys.stderr)
            return 1

        print("=" * 68)
        print("  导入 Zotero 本地库")
        print("=" * 68)
        print(f"  {report.summary()}")
        if report.sample_titles:
            print("\n  前几条：")
            for title in report.sample_titles[:5]:
                print(f"    · {title[:78]}")
        if report.errors:
            print("\n  问题：")
            for message in report.errors[:5]:
                print(f"    ! {message}")
        return 0 if report.parsed else 1

    return asyncio.run(run())


# ====================================================================== bulk
def _print_bulk(title: str, report) -> None:
    print("=" * 68)
    print(f"  {title}")
    print("=" * 68)
    for name in report.files[:10]:
        print(f"  数据包：{name}")
    if len(report.files) > 10:
        print(f"  …另有 {len(report.files) - 10} 个文件")
    print()
    print(f"  {report.summary()}")
    if report.sample_titles:
        print("\n  命中的前几条：")
        for item in report.sample_titles[:5]:
            print(f"    · {item[:78]}")
    if report.errors:
        print("\n  问题：")
        for message in report.errors[:5]:
            print(f"    ! {message}")


def cmd_bulk_pubmed(args: argparse.Namespace) -> int:
    import asyncio
    import json as _json

    from .bulk import import_pubmed_baseline

    async def run() -> int:
        report = await import_pubmed_baseline(
            args.path,
            terms=args.match,
            year_from=args.year_from,
            year_to=args.year_to,
            limit=args.limit,
            require_abstract=not args.allow_no_abstract,
            dry_run=args.dry_run,
            embed=not args.no_embed,
        )
        if args.json:
            print(_json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_bulk("导入 PubMed baseline", report)
        return 0 if (report.matched or report.errors) else 1

    return asyncio.run(run())


def cmd_bulk_pmc(args: argparse.Namespace) -> int:
    import asyncio
    import json as _json

    from .bulk import import_pmc_oa

    async def run() -> int:
        report = await import_pmc_oa(
            args.path,
            terms=args.match,
            year_from=args.year_from,
            year_to=args.year_to,
            limit=args.limit,
            dry_run=args.dry_run,
            embed=not args.no_embed,
        )
        if args.json:
            print(_json.dumps(report.to_dict(), ensure_ascii=False, indent=2))
        else:
            _print_bulk("导入 PMC Open Access Subset", report)
        return 0 if (report.matched or report.errors) else 1

    return asyncio.run(run())


# ======================================================================= run
def cmd_run(args: argparse.Namespace) -> int:
    async def run() -> int:
        from .agent.runtime import get_runtime

        runtime = get_runtime()
        handle = await runtime.start(
            topic=args.topic,
            sources=args.sources or None,
            citation_style=args.style,
            require_approval=not args.yes,
            offline=args.offline,
        )
        print(f"运行 ID：{handle.run_id}   会话 ID：{handle.state.session_id}")
        print("=" * 72)

        approved = args.yes
        async for event in runtime.stream(handle.run_id):
            kind, data = event.type, event.data
            if kind == "phase":
                print(f"\n▶ {data.get('label', data.get('phase'))}")
            elif kind == "status":
                print(f"  · {data.get('message', '')}")
            elif kind == "plan" and not approved:
                plan = data.get("plan", {})
                print("\n检索计划：")
                print(f"  中文课题：{plan.get('topic_zh')}")
                print(f"  英文课题：{plan.get('topic_en')}")
                for query in plan.get("queries", []):
                    print(f"    - {query.get('query')}  [{', '.join(query.get('sources') or [])}]")
                print("  大纲：")
                for section in plan.get("outline", []):
                    print(f"    - {section.get('title')}")
                answer = input("\n是否按此计划执行？[Y/n] ").strip().lower()
                if answer in {"n", "no", "否"}:
                    feedback = input("请给出修改意见（直接回车表示取消）：").strip()
                    if feedback:
                        await runtime.approve(handle.run_id, "revise", feedback)
                    else:
                        await runtime.approve(handle.run_id, "cancel", "")
                else:
                    await runtime.approve(handle.run_id, "approve", "")
                approved = True
            elif kind == "awaiting_approval" and not approved:
                pass  # 已在上面的 plan 分支处理
            elif kind == "search_result":
                counts = ", ".join(
                    _source_badge(s) for s in data.get("sources", [])
                )
                print(f"  · 检索「{data.get('query')}」→ {data.get('count')} 篇（{counts}）")
            elif kind == "critique":
                overall = data.get("overall", {})
                print(f"  · 证据质量：{overall.get('evidence_quality')}")
                for item in data.get("assessments", [])[:8]:
                    print(
                        f"      [{item['index']}] 相关 {item['relevance']:.1f} / "
                        f"质量 {item['quality']:.1f} · {item['evidence_level']}"
                    )
            elif kind == "token":
                sys.stdout.write(data.get("text", ""))
                sys.stdout.flush()
            elif kind == "review":
                print(f"\n  · 审查结论：{data.get('verdict')}（{data.get('score')}/10）")
                for issue in data.get("issues", [])[:5]:
                    print(f"      - [{issue.get('severity')}] {issue.get('detail')}")
            elif kind == "error":
                print(f"\n  ! {data.get('message')}")
            elif kind == "artifact":
                print(f"\n\n{'=' * 72}\n产物已保存（artifact_id={data.get('artifact_id')}）")
            elif kind == "done":
                summary = data.get("summary", {})
                print("\n" + "=" * 72)
                print(
                    f"完成：文献 {summary.get('papers')} 篇，"
                    f"引用 {summary.get('citations')} 条，"
                    f"耗时 {summary.get('elapsed_ms', 0) / 1000:.1f}s"
                )
                if summary.get("errors"):
                    print("警告：" + "；".join(summary["errors"][:3]))
        return 0

    return asyncio.run(run())


# ================================================================ summarize
def cmd_summarize(args: argparse.Namespace) -> int:
    async def run() -> int:
        from .agent.writer import WriterAgent
        from .db import repo

        paper = repo.get_paper(args.paper_id)
        if paper is None:
            print(f"未找到文献 paper_id={args.paper_id}", file=sys.stderr)
            return 1
        print(f"{paper.title}\n{paper.journal} {paper.pub_year}\n" + "-" * 72)
        text = await WriterAgent().summarize_paper(paper, topic=args.topic or "")
        print(text)
        return 0

    return asyncio.run(run())


# ====================================================================== cite
def cmd_cite(args: argparse.Namespace) -> int:
    from .cite import STYLE_LABELS, detect_style, format_records
    from .db import repo

    mapping = repo.get_papers_by_ids(args.paper_ids)
    papers = [mapping[i] for i in args.paper_ids if i in mapping]
    if not papers:
        print("指定的文献不存在于本地库。", file=sys.stderr)
        return 1
    style = detect_style(args.style)
    print(f"# {STYLE_LABELS.get(style, style)}\n")
    print(format_records(papers, style))
    return 0


# ==================================================================== export
def cmd_export(args: argparse.Namespace) -> int:
    from .cite import format_records
    from .db import repo
    from .export.exporters import export_csv, export_json, write_export

    mapping = repo.get_papers_by_ids(args.paper_ids) if args.paper_ids else {}
    papers = [mapping[i] for i in args.paper_ids if i in mapping]
    if not papers:
        from .db.repo import list_papers

        papers = list_papers(limit=args.all_limit)
        print(f"未指定 paper_id，导出最近 {len(papers)} 篇。")
    if not papers:
        print("知识库为空，无可导出内容。", file=sys.stderr)
        return 1

    fmt = args.format.lower()
    if fmt == "csv":
        content = export_csv(papers)
    elif fmt == "json":
        content = export_json(papers)
    else:
        content = format_records(papers, fmt)
    path = write_export(content, name=args.name, fmt=fmt)
    print(f"已导出 {len(papers)} 篇 → {path}")
    return 0


# ===================================================================== stats
def cmd_stats(args: argparse.Namespace) -> int:
    from .db import init as db_init

    db = db_init.init_database()
    stats = db.stats()
    _print_json(stats)
    from .db import recent_searches

    recent = recent_searches(limit=args.recent)
    if recent:
        print(f"\n最近 {len(recent)} 次检索：")
        for item in recent:
            status = "失败" if item.get("error") else f"{item['result_count']} 篇"
            print(f"  [{item['created_at']}] {item['source']:<16} {status:<10} {item['query'][:50]}")
    return 0


# ======================================================================= mcp
def cmd_mcp(args: argparse.Namespace) -> int:
    from .mcp.server import main as mcp_main

    argv = []
    if args.http:
        argv += ["--http", "--host", args.host, "--port", str(args.port)]
    return mcp_main(argv)


# ====================================================================== dbinit
def cmd_db(args: argparse.Namespace) -> int:
    from .db.init import _main as db_main

    argv: list[str] = []
    if args.stats:
        argv.append("--stats")
    if args.rebuild:
        argv.append("--rebuild")
    if args.optimize:
        argv.append("--optimize")
    if args.vacuum:
        argv.append("--vacuum")
    if args.reset:
        argv.append("--reset")
    if args.db:
        argv += ["--db", args.db]
    return db_main(argv)


# ==================================================================== parser
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medscholar",
        description="MedScholar Agent — 面向医学研究者的本地化学术智能体",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "示例：\n"
            "  medscholar doctor\n"
            "  medscholar search \"accelerated rTMS post-stroke depression\" --limit 30\n"
            "  medscholar run \"加速rTMS治疗卒中后抑郁\" --yes\n"
            "  medscholar kb \"rTMS 抑郁\"\n"
            "  medscholar serve\n"
        ),
    )
    parser.add_argument("--version", action="version", version=f"MedScholar Agent {__version__}")
    parser.add_argument("-v", "--verbose", action="store_true", help="输出调试日志")
    sub = parser.add_subparsers(dest="command", metavar="<命令>")

    p = sub.add_parser("serve", help="启动本地 Web 工作台")
    p.add_argument("--host", default=None)
    p.add_argument("--port", type=int, default=None)
    p.add_argument("--no-browser", action="store_true", help="不自动打开浏览器")
    p.add_argument("--log-level", default="info")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="环境自检（排查为什么跑不起来）")
    p.set_defaults(func=cmd_doctor)

    p = sub.add_parser("search", help="联网检索文献并可选入库")
    p.add_argument("query")
    p.add_argument("-s", "--sources", nargs="*", help="数据源（pubmed europepmc openalex crossref s2 arxiv cnki）")
    p.add_argument("-n", "--limit", type=int, default=20)
    p.add_argument("--year-from", type=int)
    p.add_argument("--year-to", type=int)
    p.add_argument("--oa", action="store_true", help="仅开放获取")
    p.add_argument("--save", action="store_true", default=True)
    p.add_argument("--no-save", dest="save", action="store_false")
    p.add_argument("--no-embed", action="store_true", help="入库时不生成向量")
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_search)

    p = sub.add_parser("kb", help="本地知识库混合检索")
    p.add_argument("query")
    p.add_argument("-n", "--limit", type=int, default=15)
    p.set_defaults(func=cmd_kb)

    p = sub.add_parser(
        "import",
        help="导入题录文件（RIS / BibTeX / EndNote 标记文本 / WoS 纯文本 / CSV）",
    )
    p.add_argument("files", nargs="+", help="一个或多个题录文件")
    p.add_argument(
        "--dry-run", action="store_true", help="只解析不写库（先看看能识别多少条）"
    )
    p.add_argument("--no-embed", action="store_true", help="入库时不生成向量")
    p.add_argument(
        "--source",
        default="import",
        help="标记这些文献的来源（默认 import；可写 wos/scopus/cnki/embase 等）",
    )
    p.add_argument("--json", action="store_true")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("zotero", help="导入本机 Zotero 库（含本地 PDF 全文索引）")
    p.add_argument("--data-dir", default=None, help="Zotero 数据目录；默认自动探测")
    p.add_argument("--limit", type=int, default=None, help="最多导入多少条")
    p.add_argument("--no-pdf", action="store_true", help="只导题录，不解析本地 PDF")
    p.add_argument("--no-embed", action="store_true", help="入库时不生成向量")
    p.set_defaults(func=cmd_zotero)

    p = sub.add_parser("bulk", help="导入官方批量数据包（PubMed baseline / PMC OA）")
    bulk_sub = p.add_subparsers(dest="bulk_command", metavar="<数据包>")
    bp = bulk_sub.add_parser("pubmed", help="PubMed baseline XML（文件或目录）")
    bp.add_argument("path")
    bp.add_argument("--match", default="", help="只收同时含这些词的记录，逗号分隔")
    bp.add_argument("--year-from", type=int)
    bp.add_argument("--year-to", type=int)
    bp.add_argument("--limit", type=int)
    bp.add_argument("--allow-no-abstract", action="store_true", help="也收没有摘要的记录")
    bp.add_argument("--dry-run", action="store_true")
    bp.add_argument("--no-embed", action="store_true")
    bp.add_argument("--json", action="store_true")
    bp.set_defaults(func=cmd_bulk_pubmed)

    bm = bulk_sub.add_parser("pmc-oa", help="PMC Open Access Subset 的 .tar.gz")
    bm.add_argument("path")
    bm.add_argument("--match", default="")
    bm.add_argument("--year-from", type=int)
    bm.add_argument("--year-to", type=int)
    bm.add_argument("--limit", type=int)
    bm.add_argument("--dry-run", action="store_true")
    bm.add_argument("--no-embed", action="store_true")
    bm.add_argument("--json", action="store_true")
    bm.set_defaults(func=cmd_bulk_pmc)

    p = sub.add_parser("run", help="执行完整 Agent 工作流并生成综述草稿")
    p.add_argument("topic")
    p.add_argument("-s", "--sources", nargs="*")
    p.add_argument("--style", default="gb7714", help="引用格式（gb7714/vancouver/apa7）")
    p.add_argument("-y", "--yes", action="store_true", help="跳过审批，直接执行")
    p.add_argument("--offline", action="store_true", help="离线模式（只用本地库）")
    p.set_defaults(func=cmd_run)

    p = sub.add_parser("summarize", help="单篇文献速读笔记")
    p.add_argument("paper_id", type=int)
    p.add_argument("--topic", default="")
    p.set_defaults(func=cmd_summarize)

    p = sub.add_parser("cite", help="生成参考文献")
    p.add_argument("paper_ids", type=int, nargs="+")
    p.add_argument("--style", default="gb7714")
    p.set_defaults(func=cmd_cite)

    p = sub.add_parser("export", help="导出文献到文件")
    p.add_argument("paper_ids", type=int, nargs="*")
    p.add_argument("-f", "--format", default="bibtex",
                   help="bibtex/ris/apa7/vancouver/gb7714/csv/json")
    p.add_argument("--name", default="medscholar")
    p.add_argument("--all-limit", type=int, default=200, help="未指定 ID 时导出最近 N 篇")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("stats", help="知识库统计与最近检索")
    p.add_argument("--recent", type=int, default=10)
    p.set_defaults(func=cmd_stats)

    p = sub.add_parser("db", help="数据库初始化与维护")
    p.add_argument("--db", default=None, help="数据库文件路径")
    p.add_argument("--stats", action="store_true")
    p.add_argument("--rebuild", action="store_true", help="清空向量索引")
    p.add_argument("--optimize", action="store_true", help="优化 FTS5")
    p.add_argument("--vacuum", action="store_true")
    p.add_argument("--reset", action="store_true", help="删除并重建数据库（危险）")
    p.set_defaults(func=cmd_db)

    p = sub.add_parser("mcp", help="启动 MCP Server")
    p.add_argument("--http", action="store_true")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8761)
    p.set_defaults(func=cmd_mcp)

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    _setup_stdout()
    parser = build_parser()
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.DEBUG if getattr(args, "verbose", False) else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    if not getattr(args, "command", None):
        parser.print_help()
        return 0
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        print("\n已中断。", file=sys.stderr)
        return 130
    except Exception as exc:
        print(f"错误：{type(exc).__name__}: {exc}", file=sys.stderr)
        if getattr(args, "verbose", False):
            import traceback

            traceback.print_exc()
        return 1


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
