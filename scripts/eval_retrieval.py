"""检索评测 CLI：跑消融实验、出报告、做回归门禁。

三种用法：

1. **内置回归集（CI 用，完全离线）**
       .python\\python.exe scripts\\eval_retrieval.py --dataset regression --k 10

2. **在你自己的库上跑真实评测（推荐，需要 Ollama）**
       .python\\python.exe scripts\\eval_retrieval.py --from-library --limit 40
   用你库里的真实文献构造语料，用真实嵌入模型算指标。
   ``known-item`` 模式的查询就是文献标题，所以**绝对指标会偏高**（词面重叠大），
   它的价值在于比较不同检索配置与跟踪改动方向。

3. **回归门禁**
       .python\\python.exe scripts\\eval_retrieval.py --dataset regression --check
   低于阈值时退出码为 1，供 CI 使用。

为什么要有 1 和 3：CI 里没有 Ollama，所以用确定性的哈希嵌入 + 合成语料，
它验证的是**指标与融合逻辑没被改坏**；真实质量必须在本机用真实模型测。
把这两件事混为一谈是评测里最常见的自欺。
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

from medscholar.config import AppConfig, get_config  # noqa: E402
from medscholar.eval import (  # noqa: E402
    CONFIGS,
    EvalCase,
    EvalDataset,
    evaluate_configs,
    load_dataset,
)
from medscholar.eval.dataset import DATASET_DIR, corpus_from_papers  # noqa: E402
from medscholar.eval.harness import check_regression, verify_production_parity  # noqa: E402
from medscholar.eval.metrics import format_table  # noqa: E402

#: CI 门禁的默认下限。**按 hashing（确定性）模式的实测基线标定**，
#: 留约 8~10% headroom：门禁要抓"把检索改坏了"，不是抓"没到理想水平"。
#: 定得太紧会因为无关改动频繁误报，然后就会被人习惯性忽略 —— 那样的门禁等于没有。
#:
#: 标定基线（regression 数据集 @10，hashing 256 维）：
#:     production: recall=0.8889  nDCG=0.8678  MRR=0.8611
DEFAULT_THRESHOLDS = {
    "recall": 0.80,
    "ndcg": 0.78,
    "mrr": 0.77,
}


def build_dataset_from_library(
    *, limit: int, k: int, mode: str, config: AppConfig
) -> EvalDataset:
    """用本地库的真实文献构造评测集。"""
    from medscholar.db.repo import list_papers

    papers = list_papers(limit=limit, order_by="cited_desc", db=None)
    if not papers:
        raise SystemExit(
            "本地库是空的，无法构造评测集。\n"
            "先用 `medscholar search \"...\"` 检索入库，或用 `medscholar import` 导入题录。"
        )

    corpus = corpus_from_papers(papers)
    cases: list[EvalCase] = []
    for index, paper in enumerate(papers, start=1):
        if mode == "known-item":
            # 查询 = 标题。词面重叠极大，绝对指标会偏高，仅用于横向比较。
            query = paper.title
        else:  # title-terms：只取标题里最长的若干实词，削掉一些词面泄漏
            words = [w for w in (paper.title or "").replace(":", " ").split() if len(w) > 3]
            query = " ".join(words[:6]) or paper.title
        if not query.strip():
            continue
        cases.append(
            EvalCase(
                query=query,
                relevant={str(index): 1.0},
                source="known-item",
                notes=f"来源：本地库第 {index} 篇（标题派生，词面重叠偏高）",
                tags=["from-library", mode],
            )
        )

    return EvalDataset(
        name=f"library-{mode}",
        description=f"从本地库取的 {len(papers)} 篇文献构造的评测集",
        provenance=(
            "语料取自已入库的真实文献元数据（不入版本控制）。"
            "查询由标题派生，因此词面重叠显著高于真实使用场景 —— "
            "**绝对指标不可用于对外宣称质量**，只用于比较不同检索配置、"
            "以及跟踪同一配置随时间的相对变化。"
        ),
        corpus=corpus,
        cases=cases,
    )


def render_report(report, *, k: int, dataset_path: str) -> str:
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  检索质量评测（消融实验）")
    lines.append("=" * 78)
    lines.append(f"  数据集      : {dataset_path}")
    lines.append(f"  语料规模    : {report.corpus_size} 篇")
    lines.append(f"  评测查询    : {report.results[0].summary['queries'] if report.results else 0} 条")
    lines.append(f"  指标位置    : @{k}")
    lines.append(f"  嵌入提供方  : {report.embed_provider}（{report.embed_dim} 维）")

    if report.embed_provider == "hashing":
        lines.append(
            "  ⚠️ 当前用的是哈希嵌入（无外部依赖，仅反映词面重叠），"
            "因此 vector 一路的语义能力被严重低估。"
        )
        lines.append("     要测真实语义检索，请在本机（有 Ollama）跑 --from-library。")

    lines.append("")
    lines.append("— 消融对比（基线 = %s）%s" % (report.baseline, "-" * 30))
    rows = report.table()
    columns = [
        ("config", "配置"),
        ("recall", f"recall@{k}"),
        ("ndcg", f"nDCG@{k}"),
        ("mrr", "MRR"),
        ("map", "MAP"),
    ]
    if any("d_recall" in row for row in rows):
        columns.extend([("d_recall", "Δrecall"), ("d_ndcg", "ΔnDCG"), ("d_mrr", "ΔMRR")])
    lines.append("  " + format_table(rows, columns).replace("\n", "\n  "))

    lines.append("")
    lines.append("— 按查询来源分组（用于识别标注偏差）" + "-" * 33)
    production = next((r for r in report.results if r.config.name == "production"), None)
    if production and production.by_source:
        source_rows = [
            {
                "source": name,
                "queries": summary.get("scored_queries"),
                "recall": summary.get("recall"),
                "ndcg": summary.get("ndcg"),
                "mrr": summary.get("mrr"),
            }
            for name, summary in sorted(production.by_source.items())
        ]
        lines.append(
            "  "
            + format_table(
                source_rows,
                [("source", "来源"), ("queries", "条数"), ("recall", "recall"),
                 ("ndcg", "nDCG"), ("mrr", "MRR")],
            ).replace("\n", "\n  ")
        )

    # 阴性对照：说明"对无关查询会不会硬凑结果"
    if production and production.empty_control.get("cases"):
        control = production.empty_control
        lines.append("")
        lines.append("— 阴性对照（本来就没有相关文献的查询）" + "-" * 31)
        lines.append(f"  条数            : {control['cases']}")
        lines.append(f"  平均返回条数    : {control['avg_returned']}")
        lines.append(f"  最多返回条数    : {control['max_returned']}")
        lines.append(f"  说明            : {control['note']}")

    return "\n".join(lines)


def render_failures(report, *, k: int, top: int = 8) -> str:
    """列出失败最严重的查询 —— 均值只能说明"变差了"，明细才能说明"为什么"。"""
    production = next((r for r in report.results if r.config.name == "production"), None)
    if not production:
        return ""
    failed = [m for m in production.per_query if m.num_relevant and (m.recall or 0) < 1.0]
    failed.sort(key=lambda m: (m.recall or 0, m.mrr or 0))
    if not failed:
        return ""
    lines = ["", "— 未完全召回的查询（按 recall 升序，便于定位问题）" + "-" * 28]
    for metrics in failed[:top]:
        lines.append(f"  [{metrics.recall:.2f}] {metrics.query[:66]}")
        if metrics.missed_ids:
            lines.append(f"         漏掉：{', '.join(metrics.missed_ids[:6])}")
        if metrics.noise_ids:
            lines.append(f"         噪声：{', '.join(metrics.noise_ids[:6])}")
    if len(failed) > top:
        lines.append(f"  …另有 {len(failed) - top} 条")
    return "\n".join(lines)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval-retrieval", description="检索质量评测与回归门禁"
    )
    parser.add_argument(
        "--dataset", default="regression",
        help="内置数据集名（regression）或 .json/.jsonl 路径",
    )
    parser.add_argument("--k", type=int, default=10, help="评测位置（默认 10）")
    parser.add_argument("--json", default=None, help="把完整报告写到该文件（JSON）")
    parser.add_argument("--check", action="store_true", help="按阈值做回归判定")
    parser.add_argument(
        "--threshold", action="append", default=None,
        help="覆盖阈值，形如 recall=0.6（可多次）",
    )
    parser.add_argument(
        "--from-library", action="store_true",
        help="用本地库真实文献构造评测集（需要 Ollama 提供真实嵌入）",
    )
    parser.add_argument("--limit", type=int, default=40, help="--from-library 的语料篇数")
    parser.add_argument(
        "--mode", choices=["known-item", "title-terms"], default="known-item",
        help="--from-library 的查询构造方式",
    )
    parser.add_argument(
        "--config", action="append", default=None,
        help="只跑指定配置（可多次），默认跑全部消融配置",
    )
    parser.add_argument(
        "--embed-provider", default=None,
        choices=["ollama", "hashing", "sentence-transformers"],
        help="临时覆盖嵌入提供方。CI 用 hashing（确定性、离线），"
             "本地测真实语义用 ollama",
    )
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args(argv)

    config = get_config()

    # CI 里没有 Ollama：显式覆盖成哈希嵌入，保证评测完全离线且可复现。
    # 这也让"确定性回归"与"真实质量评测"成为两条明确分开的路径。
    if args.embed_provider:
        config = config.model_copy(deep=True)
        config.embedding = config.embedding.model_copy(
            update={"provider": args.embed_provider}
        )
        if args.embed_provider == "hashing" and config.embedding.dim > 512:
            # 哈希嵌入不需要 768 维，缩小一点能显著加快确定性回归
            config.embedding = config.embedding.model_copy(update={"dim": 256})

    # ------------------------------------------------------------ 数据集
    if args.from_library:
        dataset = build_dataset_from_library(
            limit=args.limit, k=args.k, mode=args.mode, config=config
        )
        dataset_path = f"<本地库 {args.limit} 篇 / {args.mode}>"
        # 真实评测用真实嵌入，不要被 CI 的哈希配置影响
        if config.embedding.provider == "hashing":
            print("提示：当前嵌入提供方是 hashing，语义检索能力会被严重低估。")
            print("     如需真实语义评测，请把 config.yaml 的 embedding.provider 设为 ollama，")
            print("     或去掉 --embed-provider 参数。")
    else:
        path = Path(args.dataset)
        if not path.is_file():
            path = DATASET_DIR / f"{args.dataset}.json"
            if not path.is_file():
                path = DATASET_DIR / f"{args.dataset}.golden.jsonl"
        dataset = load_dataset(path)
        dataset_path = str(path.relative_to(ROOT)) if path.is_relative_to(ROOT) else str(path)

    # ------------------------------------------------------------ 配置筛选
    configs = CONFIGS
    if args.config:
        wanted = {name.strip() for name in args.config}
        configs = tuple(c for c in CONFIGS if c.name in wanted)
        if not configs:
            print(f"错误：--config 没有匹配到任何配置。可选：{[c.name for c in CONFIGS]}")
            return 2

    report = evaluate_configs(dataset, config=config, configs=configs, k=args.k)

    if not args.quiet:
        print(render_report(report, k=args.k, dataset_path=dataset_path))
        failures = render_failures(report, k=args.k)
        if failures:
            print(failures)

    # -------------------------------------------------- 一致性自检（重要）
    if not args.quiet and any(c.name == "production" for c in configs):
        mismatched = verify_production_parity(dataset, config=config, k=args.k)
        if mismatched:
            print(
                f"\n⚠️ 一致性自检失败：评测里的 production 配置与 hybrid_search 在 "
                f"{len(mismatched)} 条查询上排名不一致。"
                "\n   这说明评测框架与生产代码已经分叉，指标不再可信 —— 请先修这个。"
            )
        else:
            print("\n一致性自检：评测的 production 配置与 hybrid_search 排名完全一致 ✓")

    if args.json:
        out = Path(args.json)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        if not args.quiet:
            print(f"\n完整报告已写入 {out}")

    # ------------------------------------------------------------ 回归判定
    if args.check:
        thresholds = dict(DEFAULT_THRESHOLDS)
        for item in args.threshold or []:
            key, _, value = item.partition("=")
            thresholds[key.strip()] = float(value)
        problems = check_regression(report, thresholds=thresholds)
        if problems:
            print("\n" + "=" * 78)
            print("  回归判定：未通过")
            print("=" * 78)
            for problem in problems:
                print(f"  ✗ {problem}")
            print("\n  阈值（下限）：" + ", ".join(f"{k}>={v}" for k, v in thresholds.items()))
            return 1
        if not args.quiet:
            print("\n回归判定：通过（" + ", ".join(f"{k}>={v}" for k, v in thresholds.items()) + "）")
    return 0


if __name__ == "__main__":
    sys.exit(main())
