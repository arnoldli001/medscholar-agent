"""引用支持性核查 CLI：``[n]`` 到底支不支持那句话。

三种用法：

1. **核查一篇已生成的综述**（最常用）
       .python\\python.exe scripts\\eval_faithfulness.py --from-db
       .python\\python.exe scripts\\eval_faithfulness.py --from-db 3      # 指定 artifact id
   从数据库里取出草稿与它的引用映射，用被引文献的摘要/全文逐条核对。

2. **核查一个 Markdown 文件**（可用于任何草稿，不限于本工具生成）
       .python\\python.exe scripts\\eval_faithfulness.py --draft draft.md --sources sources.json

3. **评估校验器本身**（evaluate the evaluator）
       .python\\python.exe scripts\\eval_faithfulness.py --labels
   在人工标注集上算 precision/recall —— 一个从没被衡量过的校验器不可信。

Tier 1（LLM 逐条判定）默认关闭，加 ``--llm`` 打开；它对跨语言引用是**必需**的，
因为 Tier 0 的词面代理在跨语言对上没有信号（会如实 abstain）。
"""

from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

from medscholar.config import get_config  # noqa: E402
from medscholar.eval.faithfulness import analyse_draft, verify_claims_llm  # noqa: E402
from medscholar.eval.faithfulness_eval import (  # noqa: E402
    evaluate_rules,
    render_faithfulness_report,
)


def sources_from_artifact(artifact: dict) -> tuple[str, dict[int, str], dict[int, dict]]:
    """从产物里取出 ``(草稿, {编号: 文本}, {编号: 元数据})``。

    引用编号 → 文献的映射来自产物 meta 里的 ``references``（写作时序号已重排过），
    因此这里**不能**用 paper_id 直接当编号。
    """
    from medscholar.db.repo import get_fulltext, get_paper

    draft = str(artifact.get("content") or "")
    meta = artifact.get("meta") or {}
    if isinstance(meta, str):
        try:
            meta = json.loads(meta)
        except json.JSONDecodeError:
            meta = {}

    sources: dict[int, str] = {}
    source_meta: dict[int, dict] = {}
    for ref in meta.get("references") or []:
        try:
            index = int(ref.get("index"))
        except (TypeError, ValueError):
            continue
        paper_id = ref.get("paper_id")
        text = ""
        if paper_id:
            # 优先全文（证据更充分），没有就退回摘要
            try:
                text = get_fulltext(int(paper_id)) or ""
            except Exception:
                text = ""
            if not text.strip():
                paper = get_paper(int(paper_id))
                if paper is not None:
                    text = paper.abstract or ""
                    source_meta[index] = {
                        "publication_type": paper.publication_type or "",
                        "title": paper.title or "",
                        "journal": paper.journal or "",
                        "pub_year": paper.pub_year,
                    }
        sources[index] = text.strip()
    return draft, sources, source_meta


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="eval-faithfulness", description="引用支持性核查（claim-level faithfulness）"
    )
    group = parser.add_mutually_exclusive_group()
    group.add_argument("--from-db", nargs="?", const="latest", default=None,
                       help="核查数据库里的产物（默认最新一份；也可给 artifact id）")
    group.add_argument("--draft", default=None, help="核查一个 Markdown 文件")
    group.add_argument("--labels", action="store_true",
                       help="在人工标注集上评估校验器本身（precision/recall）")
    parser.add_argument("--sources", default=None,
                        help="配合 --draft：JSON 文件，形如 {\"1\": \"文献文本\"}")
    parser.add_argument("--llm", action="store_true", help="开启 Tier 1（LLM 逐条判定）")
    parser.add_argument("--self-consistency", action="store_true",
                        help="Tier 1 每条问两次，不一致标为 uncertain（更慢但更诚实）")
    parser.add_argument("--max-claims", type=int, default=40, help="Tier 1 最多核查多少条")
    parser.add_argument("--json", default=None, help="把完整报告写到该文件")
    #: 校验器自评估的门禁下限。**召回必须为 1.0** —— 漏报一个"结论被说反了"
    #: 可能直接进论文；误报只是多让人核对一次，因此对精确率宽容一些。
    parser.add_argument("--min-recall", type=float, default=1.0,
                        help="--labels 时二分类召回下限（默认 1.0：不允许漏报真问题）")
    parser.add_argument("--min-precision", type=float, default=0.85,
                        help="--labels 时二分类精确率下限（默认 0.85）")
    args = parser.parse_args(argv)

    # ---------------------------------------------------- 校验器自评估
    if args.labels:
        report = evaluate_rules()
        print("=" * 78)
        print("  校验器自评估（Tier 0 规则 vs 人工标注集）")
        print("=" * 78)
        print(f"  标注案例        : {report.total}")
        print(f"  判定完全一致    : {report.verdict_match}/{report.total}"
              f" = {report.verdict_match / report.total:.1%}")
        print(f"  预期规则命中    : {report.rule_hit}/{report.rule_total}")
        print()
        print("  — 逐类 precision / recall " + "-" * 48)
        for label, metrics in report.per_class.items():
            if metrics["support"]:
                print(f"    {label:<14} P={metrics['precision']:.2f}  "
                      f"R={metrics['recall']:.2f}  n={int(metrics['support'])}")
        binary = report.binary
        print()
        print("  — 二分类（有问题 / 没问题）" + "-" * 48)
        print(f"    precision={binary['precision']:.2f}  recall={binary['recall']:.2f}"
              f"  accuracy={binary['accuracy']:.2f}")
        print(f"    tp={binary['tp']} fp={binary['fp']} tn={binary['tn']} fn={binary['fn']}")
        if report.failures:
            print()
            print("  — 未一致的案例（含刻意的已知盲区）" + "-" * 38)
            for item in report.failures:
                print(f"    {item['id']:<24} gold={item['gold']:<12} pred={item['predicted']}")
        if report.notes:
            print()
            for note in report.notes:
                print(f"  {note}")
        if args.json:
            Path(args.json).write_text(
                json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
            )
            print(f"\n  报告已写入 {args.json}")

        # 门禁：召回必须达标（不允许漏报真问题），精确率也要守住
        problems: list[str] = []
        if binary["recall"] < args.min_recall:
            problems.append(
                f"召回 {binary['recall']:.2f} 低于下限 {args.min_recall:.2f} "
                f"（有 {binary['fn']} 条真问题被漏报）"
            )
        if binary["precision"] < args.min_precision:
            problems.append(
                f"精确率 {binary['precision']:.2f} 低于下限 {args.min_precision:.2f} "
                f"（有 {binary['fp']} 条误报）"
            )
        if problems:
            print()
            print("  门禁未通过：")
            for item in problems:
                print(f"    ✗ {item}")
            return 1
        print()
        print(f"  门禁通过：recall>={args.min_recall:.2f}，precision>={args.min_precision:.2f}")
        return 0

    # ---------------------------------------------------- 取草稿与文献
    if args.from_db:
        from medscholar.db.repo import get_artifact, list_artifacts

        if args.from_db == "latest":
            items = list_artifacts(limit=1)
            if not items:
                print("数据库里还没有产物。先生成一份综述，或用 --draft 指定文件。", file=sys.stderr)
                return 1
            artifact_id = items[0]["id"]
        else:
            artifact_id = int(args.from_db)
        artifact = get_artifact(artifact_id)
        if artifact is None:
            print(f"找不到产物 id={artifact_id}", file=sys.stderr)
            return 1
        draft, sources, source_meta = sources_from_artifact(artifact)
        label = f"产物 #{artifact_id}：{artifact.get('title') or ''}"
    elif args.draft:
        draft_path = Path(args.draft)
        if not draft_path.is_file():
            print(f"文件不存在：{draft_path}", file=sys.stderr)
            return 1
        draft = draft_path.read_text(encoding="utf-8")
        sources: dict[int, str] = {}
        source_meta: dict[int, dict] = {}
        if args.sources:
            raw = json.loads(Path(args.sources).read_text(encoding="utf-8"))
            sources = {int(k): str(v) for k, v in raw.items()}
        else:
            print("提示：没有 --sources，所有引用都会被标为「无法核实」。", file=sys.stderr)
        label = str(draft_path)
    else:
        parser.print_help()
        return 0

    if not draft.strip():
        print("草稿是空的。", file=sys.stderr)
        return 1

    print(f"核查对象：{label}")
    print(f"可用文献文本：{sum(1 for v in sources.values() if v.strip())}/{len(sources)} 条")

    llm_verdicts = None
    tier1_model = ""
    if args.llm:
        from medscholar.eval.faithfulness import extract_claims

        claims = extract_claims(draft)
        print(f"Tier 1：对前 {min(len(claims), args.max_claims)} 条论断逐条判定"
              f"（自一致性={'开' if args.self_consistency else '关'}）…")
        config = get_config()
        tier1_model = f"{config.llm.provider}/{config.llm.model}"
        llm_verdicts = asyncio.run(
            verify_claims_llm(
                claims,
                sources,
                config=config,
                max_claims=args.max_claims,
                self_consistency=args.self_consistency,
            )
        )
        print(f"  Tier 1 完成：{len(llm_verdicts)} 条")

    report = analyse_draft(
        draft,
        sources,
        source_meta=source_meta,
        valid_ids=list(sources) or None,
        llm_verdicts=llm_verdicts,
        tier1_model=tier1_model,
    )
    print()
    print(render_faithfulness_report(report))

    if args.json:
        Path(args.json).write_text(
            json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8"
        )
        print(f"\n完整报告已写入 {args.json}")

    # 退出码：有 contradicted（结论被说反）时非零，便于脚本化拦截
    contradicted = report.by_verdict.get("contradicted", 0)
    if contradicted:
        print(f"\n⚠️ 有 {contradicted} 条论断与被引文献结论方向相反，请务必逐条复核。")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
