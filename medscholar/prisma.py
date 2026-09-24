"""PRISMA 流程数据与报告规范清单（系统评价/Meta 分析的投稿硬要求）。

项目各环节本就留有检索日志、去重记录、评估与全文获取结果，据此自动产出
PRISMA 各环节数字且可复现——换检索式数字自动更新，替代手工在 Excel 里重数。
边界：

* 只做计数与文本生成，不替用户做纳入/排除决定——那是研究者的学术判断，工具只让决定可审计；
* 术语对齐 PRISMA 2020；排除理由分类沿用项目 Critic 的实际输出，不硬编码标准分类；
* 不生成图片（渲染涉及字体排版，交给绘图工具/期刊模板），只输出可直接填模板的数字与英文短句。
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

__all__ = [
    "PRISMA_CHECKLIST",
    "PrismaFlow",
    "build_prisma_flow",
    "prisma_checklist_status",
    "render_prisma_text",
]


@dataclass
class PrismaFlow:
    """PRISMA 2020 流程图的数字骨架（字段名与官方流程图一致）。"""

    #: 各数据库/登记平台检索到的记录数：``{"pubmed": 120, "openalex": 88}``
    identified: dict[str, int] = field(default_factory=dict)
    duplicates_removed: int = 0
    #: 题目/摘要筛选的记录数
    screened: int = 0
    #: 题目/摘要阶段排除数
    excluded_at_screening: int = 0
    #: 寻求全文的报告数
    sought_for_retrieval: int = 0
    #: 未获取到全文的报告数（与"排除"分开：拿不到全文 ≠ 不合格）
    not_retrieved: int = 0
    #: 评估全文的报告数
    assessed_for_eligibility: int = 0
    #: 全文阶段排除：``{"研究设计不符": 12, "无全文": 5}``
    excluded_at_fulltext: dict[str, int] = field(default_factory=dict)
    included: int = 0
    #: 其他来源（引文追踪、专家推荐、灰色文献）单独列，PRISMA 要求区分
    identified_from_other: int = 0
    #: 纳入研究来自哪些来源，便于核对
    included_sources: list[str] = field(default_factory=list)

    # ---------------------------------------------------------------- 派生量
    @property
    def identified_total(self) -> int:
        """数据库检索总数（不含其他来源）。"""
        return sum(self.identified.values())

    @property
    def records_after_dedup(self) -> int:
        return max(0, self.identified_total + self.identified_from_other - self.duplicates_removed)

    @property
    def excluded_total(self) -> int:
        return self.excluded_at_screening + sum(self.excluded_at_fulltext.values())

    def to_dict(self) -> dict[str, Any]:
        return {
            "identified_total": self.identified_total,
            "identified": dict(self.identified),
            "identified_from_other": self.identified_from_other,
            "duplicates_removed": self.duplicates_removed,
            "records_after_dedup": self.records_after_dedup,
            "screened": self.screened,
            "excluded_at_screening": self.excluded_at_screening,
            "sought_for_retrieval": self.sought_for_retrieval,
            "not_retrieved": self.not_retrieved,
            "assessed_for_eligibility": self.assessed_for_eligibility,
            "excluded_at_fulltext": dict(self.excluded_at_fulltext),
            "included": self.included,
            "excluded_total": self.excluded_total,
            "included_sources": list(self.included_sources),
        }

    def warnings(self) -> list[str]:
        """数字不自洽时给出的中文提示。

        PRISMA 数字必须内部自洽，否则审稿人一眼看穿；价值在于在做图之前
        就拦住"检索 120、筛选 100、纳入 30"这类对不上的情况。
        """
        problems: list[str] = []
        if self.identified_total == 0 and self.identified_from_other == 0:
            problems.append("没有任何检索记录：请先在「检索」页完成一次跨库检索。")
        if self.duplicates_removed > self.identified_total + self.identified_from_other:
            problems.append(
                f"去重数（{self.duplicates_removed}）超过检索总数"
                f"（{self.identified_total + self.identified_from_other}），数字不可能成立。"
            )
        if self.screened > self.records_after_dedup:
            problems.append(
                f"进入筛选的记录数（{self.screened}）大于去重后的记录数"
                f"（{self.records_after_dedup}）。"
            )
        if self.sought_for_retrieval > self.screened - self.excluded_at_screening:
            problems.append("寻求全文的报告数大于「通过题目/摘要筛选」的数量。")
        if self.assessed_for_eligibility > self.sought_for_retrieval - self.not_retrieved:
            problems.append("评估全文的报告数大于实际获取到全文的数量。")
        if self.included > self.assessed_for_eligibility - sum(self.excluded_at_fulltext.values()):
            problems.append("纳入研究数大于「评估后剩下」的数量。")
        return problems


#: PRISMA 2020 清单：官方 27 个条目，若干条目带子项（10a/10b、13a~13c、
#: 16a/16b、20a/20b、23a/23b、24a~24c 等），共 35 个编号；保留官方编号
#: 是为了让研究者直接对着投稿要求逐条打勾。
PRISMA_CHECKLIST: tuple[tuple[str, str, str], ...] = (
    ("1", "Title", "标题中注明是系统评价/Meta 分析"),
    ("2", "Abstract", "结构化摘要（含注册号）"),
    ("3", "Rationale", "研究背景与理由"),
    ("4", "Objectives", "明确的研究问题（PICO）"),
    ("5", "Eligibility criteria", "纳入排除标准"),
    ("6", "Information sources", "检索的信息来源"),
    ("7", "Search strategy", "完整检索式（可复现）"),
    ("8", "Selection process", "筛选流程与执行者"),
    ("9", "Data collection process", "数据提取流程"),
    ("10a", "Data items (outcomes)", "提取的变量与结局"),
    ("10b", "Data items (other)", "其他变量"),
    ("11", "Risk of bias assessment", "偏倚风险评估工具"),
    ("12", "Effect measures", "效应量指标"),
    ("13a", "Synthesis methods (eligibility)", "合并方法（可合并性判断）"),
    ("13b", "Synthesis methods (heterogeneity)", "异质性处理"),
    ("13c", "Synthesis methods (sensitivity)", "敏感性分析"),
    ("14", "Reporting bias assessment", "发表偏倚评估"),
    ("15", "Certainty assessment", "证据确信度（GRADE）"),
    ("16a", "Study selection results", "筛选结果与 PRISMA 流程图"),
    ("16b", "Excluded studies", "被排除研究的清单与理由"),
    ("17", "Study characteristics", "纳入研究特征表"),
    ("18", "Risk of bias in studies", "各研究偏倚风险"),
    ("19", "Results of individual studies", "各研究结果"),
    ("20a", "Results of syntheses (summary)", "合并结果"),
    ("20b", "Results of syntheses (heterogeneity)", "异质性结果"),
    ("21", "Reporting biases", "发表偏倚结果"),
    ("22", "Certainty of evidence", "证据确信度结论"),
    ("23a", "Discussion (interpretation)", "结果解释"),
    ("23b", "Discussion (limitations)", "局限（含证据确信度）"),
    ("24a", "Registration", "注册信息"),
    ("24b", "Protocol access", "方案获取方式"),
    ("24c", "Protocol amendments", "方案修改说明"),
    ("25", "Support", "资金来源"),
    ("26", "Competing interests", "利益冲突"),
    ("27", "Availability of data", "数据与代码可获取性"),
)

#: 本工具能自动填数字/草稿的条目（其余需要研究者的学术判断）
_AUTO_ITEMS: frozenset[str] = frozenset({"6", "7", "16a", "16b", "17", "24a"})


def prisma_checklist_status(
    flow: PrismaFlow | None = None, *, covered: Iterable[str] | None = None
) -> list[dict[str, Any]]:
    """返回清单各条目状态：``auto``（工具能填）/ ``manual``（需要人写）/ ``done``。

    共 35 个编号（见 :data:`PRISMA_CHECKLIST`），目前只有 6 个能自动填
    （信息来源、检索式、筛选结果、被排除清单、纳入特征、注册信息），其余靠研究者——
    不要把这条能力对外说成"自动生成 PRISMA 清单"。
    ``covered`` 传入"正文里已写到的条目号"，即可把清单当投稿自检表用。
    """
    already = set(covered or ())
    rows: list[dict[str, Any]] = []
    for code, name, note in PRISMA_CHECKLIST:
        if code in already:
            status = "done"
        elif code in _AUTO_ITEMS and flow is not None and flow.identified_total > 0:
            status = "auto"
        else:
            status = "manual"
        rows.append({"code": code, "name": name, "note": note, "status": status})
    return rows


def build_prisma_flow(
    *,
    identified: Mapping[str, int] | None = None,
    duplicates_removed: int = 0,
    excluded_at_screening: int = 0,
    not_retrieved: int = 0,
    excluded_at_fulltext: Mapping[str, int] | None = None,
    included: int = 0,
    identified_from_other: int = 0,
    assessed_for_eligibility: int | None = None,
    included_sources: Sequence[str] | None = None,
) -> PrismaFlow:
    """从各环节原始计数装配 PRISMA 流程。

    不做推断：检索/去重/排除数都由调用方从真实数据取。唯一允许推导的是
    ``screened`` / ``sought_for_retrieval`` 这类恒等式（进入筛选数 = 去重后全部记录），
    让调用方再传一遍只会制造不一致的机会。
    """
    flow = PrismaFlow(
        identified=dict(identified or {}),
        duplicates_removed=max(0, int(duplicates_removed)),
        excluded_at_screening=max(0, int(excluded_at_screening)),
        not_retrieved=max(0, int(not_retrieved)),
        excluded_at_fulltext=dict(excluded_at_fulltext or {}),
        included=max(0, int(included)),
        identified_from_other=max(0, int(identified_from_other)),
        included_sources=list(included_sources or []),
    )
    flow.screened = flow.records_after_dedup
    flow.sought_for_retrieval = max(0, flow.screened - flow.excluded_at_screening)
    if assessed_for_eligibility is None:
        flow.assessed_for_eligibility = max(0, flow.sought_for_retrieval - flow.not_retrieved)
    else:
        flow.assessed_for_eligibility = max(0, int(assessed_for_eligibility))
    return flow


def render_prisma_text(flow: PrismaFlow) -> str:
    """渲染成流程图各框的英文短句（可直接填进期刊模板）。

    用英文而非中文：PRISMA 是国际投稿要求，模板与审稿意见都用英文字段名，
    中英对照反而容易在翻译时把数字放错框。
    """
    lines: list[str] = []
    lines.append("Identification")
    if flow.identified:
        for source, count in sorted(flow.identified.items(), key=lambda kv: -kv[1]):
            lines.append(f"  Records identified from {source} (n = {count})")
    if flow.identified_from_other:
        lines.append(f"  Records identified from other sources (n = {flow.identified_from_other})")
    lines.append("  Records removed before screening:")
    lines.append(f"    Duplicate records removed (n = {flow.duplicates_removed})")
    lines.append("")
    lines.append("Screening")
    lines.append(f"  Records screened (n = {flow.screened})")
    lines.append(f"  Records excluded at title/abstract (n = {flow.excluded_at_screening})")
    lines.append(f"  Reports sought for retrieval (n = {flow.sought_for_retrieval})")
    lines.append(f"  Reports not retrieved (n = {flow.not_retrieved})")
    lines.append(f"  Reports assessed for eligibility (n = {flow.assessed_for_eligibility})")
    for reason, count in sorted(flow.excluded_at_fulltext.items(), key=lambda kv: -kv[1]):
        lines.append(f"    Reports excluded: {reason} (n = {count})")
    lines.append("")
    lines.append("Included")
    lines.append(f"  Studies included in review (n = {flow.included})")

    problems = flow.warnings()
    if problems:
        lines.append("")
        lines.append("Digital inconsistencies (fix before drawing the diagram):")
        for problem in problems:
            lines.append(f"  - {problem}")
    return "\n".join(lines)
