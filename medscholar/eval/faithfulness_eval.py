"""衡量校验器本身准不准（evaluate the evaluator）与报告渲染。

为什么必须做这一步：一个从没被衡量过的校验器，和"没有校验器"在可信度上
差别不大 —— 你不知道它的误报会把你带偏多少。所以这里：

1. 用**人工标注集**（`datasets/faithfulness.json`）跑校验器；
2. 输出**混淆矩阵 + 每类 precision/recall**，而不是一个笼统的"准确率"——
   对这类任务来说，漏报（说 supported 但实际是 contradicted）与误报
   （说 contradicted 但其实没问题）后果完全不同；
3. **如实记录已知盲区**：标注集里特意放了语义改写导致词面重合低的条目，
   用来度量 Tier 0 代理指标的固有误报，而不是把它藏起来。

诚实声明（也写进报告）：标注集由本项目作者编写、条目边界清晰，
因此它证明的是"规则在无歧义案例上可靠"，**不能**声称达到人类一致水平。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Sequence

from .faithfulness import (
    Claim,
    FaithfulnessReport,
    VERDICTS,
    check_claim_rules,
    _verdict_from_problems,
)

logger = logging.getLogger(__name__)

__all__ = [
    "LabeledCase",
    "load_labeled_cases",
    "evaluate_rules",
    "RuleEvalReport",
    "render_faithfulness_report",
    "LABELS_PATH",
]

LABELS_PATH = Path(__file__).resolve().parent / "datasets" / "faithfulness.json"

#: 判定归类：把 overclaim 视作"证据强度不匹配"而非"编造"，
#: 因此二分类的"有问题"= 非 supported 且非 unverifiable
_PROBLEM_VERDICTS = {"unsupported", "overclaim", "contradicted"}


@dataclass(slots=True)
class LabeledCase:
    """一条人工标注的核查案例。"""

    id: str
    claim_text: str
    citations: list[int]
    sources: dict[str, str]
    gold: str
    expected_rule: str | None = None
    note: str = ""
    source_meta: dict[int, dict[str, Any]] = field(default_factory=dict)
    #: 明确声明哪些编号"存在"。用于区分「越界引用」与「编号存在但拿不到全文」——
    #: 这是两件后果完全不同的事，必须能分开标注。
    valid_ids: list[int] | None = None

    def claim(self) -> Claim:
        return Claim(text=self.claim_text, citations=list(self.citations))

    def source_map(self) -> dict[int, str]:
        return {int(k): v for k, v in self.sources.items()}


def load_labeled_cases(path: str | Path | None = None) -> list[LabeledCase]:
    target = Path(path) if path else LABELS_PATH
    raw = json.loads(target.read_text(encoding="utf-8"))
    cases: list[LabeledCase] = []
    for item in raw.get("cases") or []:
        raw_valid = item.get("valid_ids")
        cases.append(
            LabeledCase(
                id=str(item.get("id") or ""),
                claim_text=str(item.get("claim") or ""),
                citations=[int(c) for c in (item.get("citations") or [])],
                sources={str(k): str(v) for k, v in (item.get("sources") or {}).items()},
                gold=str(item.get("gold") or ""),
                expected_rule=item.get("expected_rule"),
                note=str(item.get("note") or ""),
                source_meta={
                    int(k): dict(v)
                    for k, v in (item.get("source_meta") or {}).items()
                },
                valid_ids=[int(v) for v in raw_valid] if raw_valid else None,
            )
        )
    return cases


@dataclass(slots=True)
class RuleEvalReport:
    """校验器的自评估报告。"""

    total: int = 0
    #: 判定完全一致的数量
    verdict_match: int = 0
    #: 预期规则被命中的数量（仅统计 expected_rule 非空的案例）
    rule_hit: int = 0
    rule_total: int = 0
    #: 逐类混淆：gold -> {predicted: count}
    confusion: dict[str, dict[str, int]] = field(default_factory=dict)
    #: 每类的 precision / recall
    per_class: dict[str, dict[str, float]] = field(default_factory=dict)
    #: 二分类（有问题 / 没问题）指标 —— 这个更贴近实际用途
    binary: dict[str, float] = field(default_factory=dict)
    failures: list[dict[str, Any]] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)
    dataset: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "dataset": self.dataset,
            "total": self.total,
            "verdict_match": self.verdict_match,
            "verdict_accuracy": (self.verdict_match / self.total) if self.total else None,
            "rule_hit": self.rule_hit,
            "rule_total": self.rule_total,
            "confusion": self.confusion,
            "per_class": self.per_class,
            "binary": self.binary,
            "failures": self.failures,
            "notes": self.notes,
        }


def evaluate_rules(
    cases: Sequence[LabeledCase] | None = None,
    *,
    dataset_name: str = "faithfulness-labelled",
) -> RuleEvalReport:
    """在标注集上评估 Tier 0 规则。"""
    items = list(cases) if cases is not None else load_labeled_cases()
    report = RuleEvalReport(total=len(items), dataset=dataset_name)

    labels = list(VERDICTS)
    confusion: dict[str, dict[str, int]] = {g: {p: 0 for p in labels} for g in labels}
    # 二分类计数
    tp = fp = tn = fn = 0

    for case in items:
        problems = check_claim_rules(
            case.claim(),
            case.source_map(),
            source_meta=case.source_meta or None,
            valid_ids=case.valid_ids,
        )
        predicted = _verdict_from_problems(problems)
        gold = case.gold

        if predicted == gold:
            report.verdict_match += 1

        if case.expected_rule:
            report.rule_total += 1
            if any(p.get("rule") == case.expected_rule for p in problems):
                report.rule_hit += 1

        if gold in confusion:
            confusion[gold][predicted] = confusion[gold].get(predicted, 0) + 1

        gold_problem = gold in _PROBLEM_VERDICTS
        pred_problem = predicted in _PROBLEM_VERDICTS
        if gold_problem and pred_problem:
            tp += 1
        elif not gold_problem and pred_problem:
            fp += 1
        elif not gold_problem and not pred_problem:
            tn += 1
        else:
            fn += 1

        if predicted != gold:
            report.failures.append({
                "id": case.id,
                "gold": gold,
                "predicted": predicted,
                "expected_rule": case.expected_rule,
                "rules_fired": [p.get("rule") for p in problems],
                "claim": case.claim_text[:80],
                "note": case.note,
            })

    report.confusion = confusion

    # 每类 precision / recall
    for label in labels:
        tp_c = confusion.get(label, {}).get(label, 0)
        predicted_c = sum(confusion.get(g, {}).get(label, 0) for g in labels)
        actual_c = sum(confusion.get(label, {}).values())
        report.per_class[label] = {
            "precision": (tp_c / predicted_c) if predicted_c else 0.0,
            "recall": (tp_c / actual_c) if actual_c else 0.0,
            "support": actual_c,
        }

    report.binary = {
        "tp": tp, "fp": fp, "tn": tn, "fn": fn,
        "precision": (tp / (tp + fp)) if (tp + fp) else 0.0,
        "recall": (tp / (tp + fn)) if (tp + fn) else 0.0,
        "accuracy": ((tp + tn) / (tp + fp + tn + fn)) if (tp + fp + tn + fn) else 0.0,
    }

    report.notes = [
        "标注集由本项目作者编写、条目边界清晰，因此这里衡量的是"
        "「规则在无歧义案例上是否可靠」，**不能**据此声称达到人类一致水平。",
        "二分类指标（有问题 / 没问题）比逐类准确率更贴近实际用途："
        "使用者真正关心的是「会不会漏掉真问题」与「会不会喊狼来了」。",
    ]
    if report.failures:
        blind = [f for f in report.failures if f["gold"] == "supported"]
        if blind:
            report.notes.append(
                f"其中有 {len(blind)} 条是**误报**（标注为 supported 却被报警），"
                "主要来自语义改写导致的词面重合过低 —— 这是 Tier 0 词面代理指标的固有盲区，"
                "也正是需要 Tier 1（LLM/NLI 蕴含判断）的原因。"
            )
    return report


# ============================================================ 报告渲染
def render_faithfulness_report(report: FaithfulnessReport, *, top: int = 15) -> str:
    """把支持性报告渲染成可读文本（CLI 用）。"""
    lines: list[str] = []
    lines.append("=" * 78)
    lines.append("  引用支持性核查（claim-level faithfulness）")
    lines.append("=" * 78)
    lines.append(f"  带引用的论断    : {report.claims} 条")
    lines.append(f"  句子数          : {report.sentences}（其中带引用 {report.cited_sentences}）")
    if report.sentences:
        lines.append(f"  引用覆盖面      : {report.cited_sentences / report.sentences:.0%}")
    rate = report.supported_rate
    lines.append(f"  未触发规则的占比: {'—' if rate is None else f'{rate:.1%}'}")
    lines.append(f"  Tier 1（LLM）   : {'已运行 · ' + (report.tier1_model or '模型未知') if report.tier1_ran else '未运行'}")

    if report.by_verdict:
        lines.append("")
        lines.append("— 判定分布 " + "-" * 55)
        for verdict, count in sorted(
            report.by_verdict.items(), key=lambda kv: (-kv[1], kv[0])
        ):
            lines.append(f"  {verdict:<14} {count:>4}")
    if report.by_rule:
        lines.append("")
        lines.append("— 规则命中 " + "-" * 55)
        for rule, count in sorted(report.by_rule.items(), key=lambda kv: -kv[1]):
            lines.append(f"  {rule:<16} {count:>4}")

    flagged = [d for d in report.details if d.verdict != "supported"]
    flagged.sort(key=lambda d: -d.severity)
    if flagged:
        lines.append("")
        lines.append(f"— 需要核对的论断（共 {len(flagged)} 条，按严重程度排序）" + "-" * 24)
        for item in flagged[:top]:
            lines.append(f"  [{item.verdict}] {item.claim.text[:64]}")
            lines.append(f"        引用：{item.claim.citations}")
            for problem in item.problems[:3]:
                lines.append(f"        · {problem.get('rule')}: {str(problem.get('detail'))[:96]}")
            if item.evidence:
                lines.append(f"        证据：{item.evidence[:90]}")
        if len(flagged) > top:
            lines.append(f"  …另有 {len(flagged) - top} 条")

    if report.notes:
        lines.append("")
        lines.append("— 说明 " + "-" * 59)
        for note in report.notes:
            lines.append(f"  {note}")
    return "\n".join(lines)
