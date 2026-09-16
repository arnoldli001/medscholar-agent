"""开发自检：语法编译 + 模块导入 + 关键纯函数断言（不联网）。

    .python\\python.exe scripts\\check.py
"""

from __future__ import annotations

import importlib
import py_compile
import sys
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

MODULES = [
    "medscholar",
    "medscholar.config",
    "medscholar.models",
    "medscholar.textutil",
    "medscholar.dedupe",
    "medscholar.retrieval",
    "medscholar.db",
    "medscholar.db.connect",
    "medscholar.db.init",
    "medscholar.db.repo",
    "medscholar.api",
    "medscholar.api.base",
    "medscholar.api.registry",
    "medscholar.api.pubmed_client",
    "medscholar.api.europepmc_client",
    "medscholar.api.semantic_scholar_client",
    "medscholar.api.openalex_client",
    "medscholar.api.crossref_client",
    "medscholar.api.arxiv_client",
    "medscholar.api.cnki_client",
    "medscholar.embedding",
    "medscholar.embedding.providers",
    "medscholar.embedding.pipeline",
    "medscholar.llm",
    "medscholar.llm.client",
    "medscholar.llm.prompts",
    "medscholar.cite",
    "medscholar.cite.styles",
    "medscholar.export",
    "medscholar.export.exporters",
    "medscholar.agent",
    "medscholar.agent.state",
    "medscholar.agent.scout",
    "medscholar.agent.reader",
    "medscholar.agent.critic",
    "medscholar.agent.writer",
    "medscholar.agent.formatter",
    "medscholar.agent.graph",
    "medscholar.agent.runtime",
]


def check_syntax() -> int:
    print("=" * 72)
    print("1) 语法编译")
    bad = 0
    for path in sorted((ROOT / "medscholar").rglob("*.py")):
        try:
            py_compile.compile(str(path), doraise=True, quiet=2)
        except py_compile.PyCompileError as exc:
            bad += 1
            print(f"  FAIL {path.relative_to(ROOT)}: {exc}")
    print(f"  {len(list((ROOT / 'medscholar').rglob('*.py')))} 个文件，失败 {bad} 个")
    return bad


def check_imports() -> int:
    print("=" * 72)
    print("2) 模块导入")
    bad = 0
    for name in MODULES:
        try:
            importlib.import_module(name)
        except Exception as exc:
            bad += 1
            print(f"  FAIL {name} -> {type(exc).__name__}: {exc}")
            traceback.print_exc(limit=2)
    print(f"  {len(MODULES)} 个模块，失败 {bad} 个")
    return bad


def check_unit() -> int:
    """关键纯函数的行为断言（离线、确定性）。"""
    print("=" * 72)
    print("3) 纯函数断言")
    from medscholar.agent.writer import extract_citations, _sanitize
    from medscholar.api.openalex_client import reconstruct_abstract
    from medscholar.cite import format_citation, format_reference_list, to_bibtex, to_ris
    from medscholar.models import Paper
    from medscholar.textutil import build_match_query, segment_cjk, to_family_first

    failures: list[str] = []

    def expect(label: str, actual, expected) -> None:
        if actual != expected:
            failures.append(f"{label}\n      期望 {expected!r}\n      实际 {actual!r}")
            print(f"  FAIL {label}")
        else:
            print(f"  ok   {label}")

    expect("segment_cjk", segment_cjk("加速rTMS治疗"), "加 速 rTMS 治 疗")
    expect(
        "build_match_query(phrase)",
        build_match_query("加速rTMS治疗卒中后抑郁"),
        '"加 速" AND "rTMS" AND "治 疗 卒 中 后 抑 郁"',
    )
    expect(
        "build_match_query(bigram)",
        build_match_query("卒中后抑郁", cjk="bigram"),
        '("卒 中" AND "中 后" AND "后 抑" AND "抑 郁")',
    )
    expect(
        "build_match_query(bigram+or)",
        build_match_query("卒中后抑郁", mode="or", cjk="bigram"),
        '("卒 中" OR "中 后" OR "后 抑" OR "抑 郁")',
    )
    expect("to_family_first(名 姓)", to_family_first("Wei Zhang"), "Zhang Wei")
    expect("to_family_first(姓名两点)", to_family_first("Jean-Pierre Dupont"), "Dupont Jean-Pierre")
    expect("to_family_first(带前缀姓氏)", to_family_first("van der Berg Jan"), "van der Berg Jan")
    expect("to_family_first(逗号)", to_family_first("Zhang, Wei"), "Zhang Wei")
    expect("to_family_first(中文)", to_family_first("王伟"), "王伟")
    expect(
        "reconstruct_abstract",
        reconstruct_abstract({"Post-stroke": [0], "depression": [1], "trial": [2]}),
        "Post-stroke depression trial",
    )
    expect("extract_citations", extract_citations("见 [1] 与 [2,3] 以及 [5-7]"), [1, 2, 3, 5, 6, 7])
    expect("_sanitize 剔除越界", _sanitize("结论 [1] 与 [9]", [1, 2, 3]), "结论 [1] 与")
    expect("_sanitize 组内过滤", _sanitize("结果 [1,9]", [1, 2, 3]), "结果 [1]")

    papers = [
        Paper(
            title="Accelerated rTMS for post-stroke depression",
            authors=["Zhang Wei", "Li Ming"],
            journal="Brain Stimulation",
            pub_year=2023,
            source="pubmed",
            doi="10.1016/j.brs.2023.001",
            volume="16", issue="2", pages="100-108",
            publication_type="Journal Article",
        ),
        Paper(
            title="加速rTMS治疗卒中后抑郁的临床疗效观察",
            authors=["王伟", "李静", "张强", "赵敏"],
            journal="中国康复医学杂志",
            pub_year=2022,
            source="cnki",
            volume="37", issue="4", pages="512-516",
        ),
    ]

    apa = format_citation(papers[0], "apa7")
    expect(
        "APA7 含作者与年份",
        all(token in apa for token in ("Zhang, W.", "(2023)", "Brain Stimulation", "https://doi.org/")),
        True,
    )
    van = format_citation(papers[0], "vancouver", index=1)
    expect("Vancouver 前缀编号", van.startswith("1. "), True)
    gb_en = format_citation(papers[0], "gb7714", index=1)
    expect("GB/T 英文 [J]", "[J]" in gb_en and gb_en.startswith("[1]"), True)
    gb_zh = format_citation(papers[1], "gb7714", index=2)
    expect("GB/T 中文 4 作者用等", "等" in gb_zh, True)
    bib = to_bibtex(papers[0])
    expect("BibTeX 条目", bib.startswith("@article{") and "doi" in bib, True)
    ris = to_ris(papers[0])
    expect("RIS 条目", ris.startswith("TY  - JOUR") and ris.rstrip().endswith("ER  -"), True)
    refs = format_reference_list(papers, "gb7714")
    expect("参考文献表两行", len([ln for ln in refs.splitlines() if ln.strip()]), 2)

    print(f"  断言失败 {len(failures)} 项")
    for item in failures:
        print("   -", item)
    return len(failures)


def main() -> int:
    bad = check_syntax() + check_imports() + check_unit()
    print("=" * 72)
    print("CHECK:", "PASS" if bad == 0 else f"FAIL（{bad} 项）")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
