"""PRISMA 流程数据与自检清单的测试。

重点不是"函数能跑"，而是**数字不自洽时必须报警** ——
PRISMA 数字要交给审稿人看，一处对不上就会被质疑整篇的可信度，
所以在做图之前就要拦住。
"""

from __future__ import annotations

import pytest

from medscholar.prisma import (
    PRISMA_CHECKLIST,
    build_prisma_flow,
    prisma_checklist_status,
    render_prisma_text,
)


class TestFlowArithmetic:
    def test_identified_total_sums_sources(self):
        flow = build_prisma_flow(identified={"pubmed": 120, "openalex": 88, "cnki": 30})
        assert flow.identified_total == 238

    def test_screened_equals_after_dedup(self):
        """进入筛选的记录数 = 检索总数 − 去重数（恒等式，不让调用方再传一遍）。"""
        flow = build_prisma_flow(identified={"pubmed": 100}, duplicates_removed=25)
        assert flow.screened == 75

    def test_other_sources_counted(self):
        """PRISMA 要求把引文追踪等其他来源单独列出来，不能混进数据库检索数。"""
        flow = build_prisma_flow(identified={"pubmed": 100}, identified_from_other=7)
        assert flow.identified_total == 100
        assert flow.records_after_dedup == 107

    def test_sought_for_retrieval_subtracts_screening_exclusions(self):
        flow = build_prisma_flow(
            identified={"pubmed": 100}, duplicates_removed=10, excluded_at_screening=60
        )
        assert flow.screened == 90
        assert flow.sought_for_retrieval == 30

    def test_not_retrieved_is_separate_from_excluded(self):
        """拿不到全文 ≠ 不合格，PRISMA 里是两个不同的框。"""
        flow = build_prisma_flow(
            identified={"pubmed": 50},
            excluded_at_screening=10,
            not_retrieved=4,
            excluded_at_fulltext={"研究设计不符": 6},
            included=30,
        )
        assert flow.sought_for_retrieval == 40
        assert flow.assessed_for_eligibility == 36
        assert flow.excluded_total == 16

    def test_explicit_assessed_overrides_derivation(self):
        flow = build_prisma_flow(
            identified={"pubmed": 50}, not_retrieved=5, assessed_for_eligibility=41
        )
        assert flow.assessed_for_eligibility == 41

    def test_negative_inputs_clamped(self):
        """外部计数可能来自用户的表格（含负数），不能让它变成负的记录数。"""
        flow = build_prisma_flow(identified={"pubmed": 10}, duplicates_removed=-5, included=-3)
        assert flow.duplicates_removed == 0
        assert flow.included == 0

    def test_to_dict_is_complete(self):
        payload = build_prisma_flow(identified={"pubmed": 10}, included=1).to_dict()
        for key in (
            "identified_total",
            "duplicates_removed",
            "records_after_dedup",
            "screened",
            "sought_for_retrieval",
            "not_retrieved",
            "assessed_for_eligibility",
            "excluded_at_fulltext",
            "included",
            "excluded_total",
        ):
            assert key in payload


class TestConsistencyWarnings:
    def test_clean_flow_has_no_warnings(self):
        flow = build_prisma_flow(
            identified={"pubmed": 100},
            duplicates_removed=20,
            excluded_at_screening=50,
            not_retrieved=5,
            excluded_at_fulltext={"无全文": 5},
            included=20,
        )
        assert flow.warnings() == []

    def test_empty_search_warns(self):
        assert any("检索" in w for w in build_prisma_flow().warnings())

    def test_duplicates_exceeding_total_warns(self):
        flow = build_prisma_flow(identified={"pubmed": 10}, duplicates_removed=50)
        assert any("去重数" in w for w in flow.warnings())

    def test_screening_exceeding_records_warns(self):
        flow = build_prisma_flow(identified={"pubmed": 10})
        flow.screened = 99  # 人为破坏不变量
        assert any("进入筛选" in w for w in flow.warnings())

    def test_included_exceeding_eligible_warns(self):
        flow = build_prisma_flow(
            identified={"pubmed": 10},
            excluded_at_screening=2,
            not_retrieved=0,
            excluded_at_fulltext={"设计不符": 6},
            included=9,
        )
        assert any("纳入研究数" in w for w in flow.warnings())

    def test_sought_exceeding_screened_warns(self):
        flow = build_prisma_flow(identified={"pubmed": 10}, excluded_at_screening=-0)
        flow.sought_for_retrieval = 99
        assert any("寻求全文" in w for w in flow.warnings())


class TestRender:
    def test_text_contains_official_sections(self):
        flow = build_prisma_flow(identified={"pubmed": 100}, duplicates_removed=20, included=30)
        text = render_prisma_text(flow)
        for section in ("Identification", "Screening", "Included"):
            assert section in text

    def test_text_lists_each_source_with_count(self):
        flow = build_prisma_flow(identified={"pubmed": 120, "openalex": 88})
        text = render_prisma_text(flow)
        assert "Records identified from pubmed (n = 120)" in text
        assert "Records identified from openalex (n = 88)" in text

    def test_text_orders_sources_by_count_desc(self):
        flow = build_prisma_flow(identified={"small": 1, "big": 99})
        text = render_prisma_text(flow)
        assert text.index("big") < text.index("small")

    def test_text_includes_exclusion_reasons(self):
        flow = build_prisma_flow(
            identified={"pubmed": 50},
            excluded_at_fulltext={"研究设计不符": 12, "样本量过小": 3},
        )
        text = render_prisma_text(flow)
        assert "Reports excluded: 研究设计不符 (n = 12)" in text
        assert "Reports excluded: 样本量过小 (n = 3)" in text

    def test_text_surfaces_inconsistencies_instead_of_hiding_them(self):
        """不自洽时要在输出里显式提示 —— 直接把错数字交给审稿人是最坏的结果。"""
        flow = build_prisma_flow(identified={"pubmed": 10}, duplicates_removed=99)
        text = render_prisma_text(flow)
        assert "inconsisten" in text.lower()


class TestChecklist:
    def test_checklist_has_prisma_2020_items(self):
        assert len(PRISMA_CHECKLIST) >= 27
        codes = {code for code, _, _ in PRISMA_CHECKLIST}
        for required in ("1", "7", "16a", "17", "24a", "27"):
            assert required in codes

    def test_status_marks_automatable_items(self):
        flow = build_prisma_flow(identified={"pubmed": 10})
        rows = {row["code"]: row for row in prisma_checklist_status(flow)}
        assert rows["6"]["status"] == "auto", "信息来源可由检索记录自动填"
        assert rows["7"]["status"] == "auto", "检索式是系统产出的，可自动填"
        assert rows["16a"]["status"] == "auto", "PRISMA 流程数字可自动填"
        assert rows["11"]["status"] == "manual", "偏倚风险评估必须由研究者决定"

    def test_without_flow_nothing_is_auto(self):
        """没有真实检索记录时不该声称能自动填 —— 那只是空壳。"""
        rows = prisma_checklist_status(None)
        assert all(row["status"] == "manual" for row in rows)

    def test_covered_items_become_done(self):
        flow = build_prisma_flow(identified={"pubmed": 10})
        rows = {row["code"]: row for row in prisma_checklist_status(flow, covered=["5", "9"])}
        assert rows["5"]["status"] == "done"
        assert rows["9"]["status"] == "done"
        assert rows["11"]["status"] == "manual"

    def test_every_row_has_note(self):
        for row in prisma_checklist_status(None):
            assert row["note"].strip(), f"{row['code']} 缺少中文说明"


@pytest.mark.parametrize("reason_count", [0, 1, 5])
def test_render_handles_any_exclusion_reason_count(reason_count: int):
    reasons = {f"理由{i}": i + 1 for i in range(reason_count)}
    flow = build_prisma_flow(identified={"pubmed": 100}, excluded_at_fulltext=reasons)
    text = render_prisma_text(flow)
    assert "Screening" in text
    for reason in reasons:
        assert reason in text
