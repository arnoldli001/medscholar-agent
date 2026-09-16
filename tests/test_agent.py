"""Agent 层测试：引用纪律、证据评估、格式校验、状态机与运行时。

不依赖 LLM：所有 LLM 相关路径都验证**降级行为**（LLM 不可用时必须仍然工作）。
"""

from __future__ import annotations

import asyncio

import pytest

from medscholar.agent.critic import (
    CriticAgent,
    detect_evidence_level,
    extract_sample_size,
    heuristic_assessment,
)
from medscholar.agent.formatter import FormatterAgent
from medscholar.agent.state import (
    AgentState,
    CritiqueResult,
    PaperAssessment,
    Phase,
    PlanQuery,
    PlanSection,
    ResearchPlan,
    ReviewResult,
)
from medscholar.agent.writer import (
    _default_outline,
    _sanitize,
    extract_citations,
)
from medscholar.models import Paper


# ======================================================== 引用标记提取与清洗
class TestExtractCitations:
    @pytest.mark.parametrize(
        "text,expected",
        [
            ("见 [1]", [1]),
            ("见 [1,2]", [1, 2]),
            ("见 [1，2]", [1, 2]),
            ("见 [1-3]", [1, 2, 3]),
            ("见 【1】", [1]),
            ("[1][2][3]", [1, 2, 3]),
            ("无引用", []),
            ("", []),
        ],
    )
    def test_cases(self, text, expected):
        assert extract_citations(text) == expected

    def test_range_guard_against_absurd(self):
        assert extract_citations("[1-999]") == []

    def test_mixed(self):
        assert extract_citations("A [1] B [3,4] C [6-8]") == [1, 3, 4, 6, 7, 8]


class TestSanitize:
    def test_removes_out_of_range(self):
        assert _sanitize("结论 [1] 与 [99]", [1, 2, 3]).strip() == "结论 [1] 与"

    def test_filters_within_group(self):
        assert _sanitize("结果 [1,99,2]", [1, 2, 3]) == "结果 [1,2]"

    def test_range_clipped_to_valid(self):
        assert _sanitize("见 [1-5]", [1, 2, 3]) == "见 [1,2,3]"

    def test_keeps_valid_untouched(self):
        text = "加速 rTMS 有效 [2][5]"
        assert _sanitize(text, [1, 2, 3, 4, 5]) == text

    def test_drops_fully_invalid_group(self):
        assert "[9]" not in _sanitize("无关 [9]", [1, 2])

    def test_no_dangling_spaces(self):
        assert "  。" not in _sanitize("结论 [9]。", [1])


# ================================================================== 证据评估
class TestEvidenceLevel:
    @pytest.mark.parametrize(
        "abstract,expected",
        [
            ("This randomized controlled trial enrolled 120 patients", "RCT"),
            ("A systematic review and meta-analysis of 20 trials", "Meta分析/系统评价"),
            ("This prospective cohort study followed 500 subjects", "队列研究"),
            ("A case-control study was conducted", "病例对照"),
            ("This cross-sectional survey analysed data", "横断面"),
            ("We report a case report of a 45-year-old man", "病例报告"),
            ("Sprague-Dawley rats were used", "动物实验"),
            ("Clinical practice guideline recommendations", "指南/共识"),
            ("This narrative review summarises", "综述"),
            ("Some text with no design markers", "其他"),
        ],
    )
    def test_detection(self, abstract, expected):
        paper = Paper(title="Study", source="pubmed", abstract=abstract)
        assert detect_evidence_level(paper) == expected

    def test_animal_overrides_design(self):
        """动物 RCT 仍应判为动物实验，而不是 RCT。"""
        paper = Paper(
            title="Effects in rats",
            source="pubmed",
            abstract="randomized controlled trial in mice",
        )
        assert detect_evidence_level(paper) == "动物实验"

    def test_chinese_design_markers(self):
        paper = Paper(title="研究", source="cnki", abstract="采用随机对照方法，将60例患者分组。")
        assert detect_evidence_level(paper) == "RCT"


class TestSampleSize:
    @pytest.mark.parametrize(
        "abstract,expected",
        [
            ("n = 120 participants", 120),
            ("60 patients were enrolled", 60),
            ("共纳入 88 例患者", 88),
            ("选取 45 名受试者", 45),
            ("1200 subjects", 1200),
        ],
    )
    def test_extraction(self, abstract, expected):
        assert extract_sample_size(Paper(title="T", source="pubmed", abstract=abstract)) == expected

    def test_none_when_absent(self):
        assert extract_sample_size(Paper(title="T", source="pubmed", abstract="No numbers here.")) is None

    def test_ignores_implausible_small_numbers(self):
        assert extract_sample_size(
            Paper(title="T", source="pubmed", abstract="3 patients dropped out")
        ) is None


class TestHeuristicAssessment:
    def test_rct_scores_higher_than_case_report(self):
        rct = Paper(
            title="A randomized controlled trial of rTMS",
            source="pubmed",
            abstract="randomized controlled trial, n = 500 patients",
            pub_year=2024,
            cited_by_count=200,
        )
        case = Paper(
            title="A case report",
            source="pubmed",
            abstract="case report of a single patient",
            pub_year=2024,
        )
        high = heuristic_assessment(rct, topic="rTMS depression", index=1)
        low = heuristic_assessment(case, topic="rTMS depression", index=2)
        assert high.quality > low.quality

    def test_relevance_rewards_topic_overlap(self):
        on_topic = Paper(
            title="Accelerated rTMS for post-stroke depression",
            source="pubmed",
            abstract="rTMS improved depression scores after stroke.",
        )
        off_topic = Paper(
            title="Gut microbiome in colorectal cancer",
            source="pubmed",
            abstract="Sequencing of faecal samples.",
        )
        a = heuristic_assessment(on_topic, topic="accelerated rTMS post-stroke depression", index=1)
        b = heuristic_assessment(off_topic, topic="accelerated rTMS post-stroke depression", index=2)
        assert a.relevance > b.relevance
        assert a.use_in_review is True

    def test_missing_abstract_penalised(self):
        with_abstract = Paper(
            title="T", source="pubmed", abstract="randomized controlled trial " * 5
        )
        without = Paper(title="T", source="pubmed", abstract="")
        assert (
            heuristic_assessment(with_abstract, topic="T", index=1).quality
            > heuristic_assessment(without, topic="T", index=2).quality
        )

    def test_small_sample_flagged(self):
        paper = Paper(title="T", source="pubmed", abstract="randomized trial, n = 12 patients")
        assessment = heuristic_assessment(paper, topic="T", index=1)
        assert "样本量偏小" in assessment.limitation

    def test_scores_within_bounds(self):
        paper = Paper(title="T", source="pubmed", abstract="randomized controlled trial n = 99999")
        assessment = heuristic_assessment(paper, topic="T", index=1)
        assert 0.0 <= assessment.quality <= 10.0
        assert 0.0 <= assessment.relevance <= 10.0

    def test_combined_weights_relevance_more(self):
        assessment = PaperAssessment(index=1, relevance=10, quality=0)
        assert assessment.combined == 6.0


class TestCriticAgent:
    async def test_falls_back_to_heuristic_without_llm(self, config, sample_papers):
        """离线/LLM 不可用时必须给出完整评估，而不是空结果。"""
        agent = CriticAgent(config=config)
        entries = list(enumerate(sample_papers, start=1))
        result = await agent.assess("加速rTMS治疗卒中后抑郁", entries, use_llm=False)
        assert len(result.assessments) == len(entries)
        assert result.evidence_quality in {"高", "中", "低", "极低"}
        assert result.used_llm is False
        assert all(a.source == "heuristic" for a in result.assessments)

    async def test_suggestions_generated(self, config, sample_papers):
        agent = CriticAgent(config=config)
        result = await agent.assess("rTMS", list(enumerate(sample_papers, 1)), use_llm=False)
        assert result.suggestions

    async def test_assessment_carries_paper_id(self, config, sample_papers):

        agent = CriticAgent(config=config)
        entries = [(i, p) for i, p in enumerate(sample_papers, 1)]
        entries[0][1].paper_id = 101
        result = await agent.assess("rTMS", entries, use_llm=False)
        assert any(a.paper_id == 101 for a in result.assessments)

    async def test_heuristic_covers_all_when_llm_limited(self, config):
        """LLM 只点评前 N 篇时，其余文献仍必须有评估结果（不能漏）。"""
        from medscholar.models import Paper as P

        config.agent.critique_max_papers = 2
        papers = [
            P(title=f"rTMS study {i}", source="pubmed",
              abstract="randomized controlled trial of rTMS for depression, n = 100 patients")
            for i in range(8)
        ]
        entries = [(i, p) for i, p in enumerate(papers, 1)]
        agent = CriticAgent(config=config)
        # use_llm=False 走纯启发式；这里主要验证条目数与编号完整性
        result = await agent.assess("rTMS depression", entries, use_llm=False)
        assert len(result.assessments) == 8
        assert sorted(a.index for a in result.assessments) == list(range(1, 9))


# ================================================================ 格式校验
class TestFormatterValidation:
    def test_all_valid(self, sample_papers):
        report = FormatterAgent.validate_citations("结论 [1] 与 [2]", [1, 2, 3])
        assert report.ok
        assert not report.missing_from_list

    def test_detects_out_of_range(self, sample_papers):
        report = FormatterAgent.validate_citations("结论 [9]", [1, 2, 3])
        assert not report.ok
        assert 9 in report.missing_from_list

    def test_detects_never_cited(self, sample_papers):
        report = FormatterAgent.validate_citations("结论 [1]", [1, 2, 3])
        assert sorted(set(report.never_cited)) == [2, 3]

    def test_summary_readable(self):
        report = FormatterAgent.validate_citations("x [9]", [1])
        assert "引用校验" in report.summary()

    def test_build_references_only_cited(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        text = agent.build_references(entries, "gb7714", only_cited="正文只引用了 [2]")
        assert sample_papers[1].title[:15] in text
        assert sample_papers[0].title[:15] not in text

    def test_reference_entries_shape(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        rows = agent.reference_entries(entries, "gb7714")
        assert len(rows) == 3
        assert rows[0]["index"] == 1
        for key in ("text", "inline", "citation_label", "title"):
            assert key in rows[0]

    def test_reference_entries_text_is_full_reference(self, sample_papers):
        """回归：右栏引用列表里必须是**完整参考文献**，不能只剩一个 `[1]`。"""
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        for style in ("gb7714", "vancouver", "apa7"):
            rows = agent.reference_entries(entries, style)
            text = rows[0]["text"]
            assert len(text) > 40, f"{style} 的 entry.text 过短：{text!r}"
            assert sample_papers[0].title[:20] in text, f"{style} 的 entry.text 缺少标题"
            assert sample_papers[0].journal in text

    def test_reference_entries_inline_is_short_marker(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        rows = agent.reference_entries(entries, "gb7714")
        assert rows[0]["inline"] == "[1]"
        apa_rows = agent.reference_entries(entries, "apa7")
        assert apa_rows[0]["inline"].startswith("(")

    def test_restyle_numeric_unchanged(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        assert agent.restyle_inline("x [1]", entries, "gb7714") == "x [1]"

    def test_restyle_author_year(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        text = agent.restyle_inline("rTMS 有效 [1]。", entries, "apa7")
        assert "[1]" not in text
        assert "(Zhang et al., 2023)" in text

    def test_selfcheck_flags_empty_section(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        check = agent.selfcheck("# 标题\n\n## 引言\n\n## 方法\n内容 [1]", entries)
        assert any(i["type"] == "空章节" for i in check["issues"])

    def test_selfcheck_verdict_revise_on_high(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        check = agent.selfcheck("## 方法\n引用不存在的 [9]", entries)
        assert check["verdict"] == "revise"

    def test_selfcheck_pass(self, sample_papers):
        agent = FormatterAgent()
        entries = list(enumerate(sample_papers, start=1))
        body = "本研究显示加速 rTMS 可显著改善抑郁评分 [1]。" * 5
        check = agent.selfcheck(f"## 结果\n{body}", entries[:1])
        assert check["verdict"] == "pass"


# ================================================================ 计划与状态
class TestPlanModel:
    def test_roundtrip(self):
        plan = ResearchPlan(
            topic_zh="加速rTMS治疗卒中后抑郁",
            topic_en="accelerated rTMS post-stroke depression",
            pico={"population": "卒中后抑郁患者"},
            queries=[PlanQuery(query="rTMS", sources=["pubmed"], rationale="主检索式")],
            mesh_terms=["Depression"],
            year_from=2015,
            key_questions=["有效吗"],
            outline=[PlanSection(title="1 引言", points=["背景"])],
        )
        restored = ResearchPlan.from_dict(plan.to_dict())
        assert restored.topic_zh == plan.topic_zh
        assert restored.queries[0].sources == ["pubmed"]
        assert restored.outline[0].points == ["背景"]
        assert restored.year_from == 2015

    def test_tolerates_missing_fields(self):
        plan = ResearchPlan.from_dict({})
        assert plan.queries == [] and plan.outline == []

    def test_tolerates_llm_garbage(self):
        plan = ResearchPlan.from_dict(
            {"queries": [{"query": ""}, {"query": "ok"}, "not-a-dict"], "outline": [{"title": ""}]}
        )
        assert [q.query for q in plan.queries] == ["ok"]
        assert plan.outline == []

    def test_string_year_coerced(self):
        assert ResearchPlan.from_dict({"year_from": "2015"}).year_from == 2015
        assert ResearchPlan.from_dict({"year_from": "abc"}).year_from is None

    def test_source_string_split(self):
        plan = ResearchPlan.from_dict({"queries": [{"query": "x", "sources": "pubmed, openalex"}]})
        assert plan.queries[0].sources == ["pubmed", "openalex"]


class TestAgentState:
    def test_renumber_is_continuous(self, sample_papers):
        state = AgentState(topic="T")
        state.citation_map = {i: p for i, p in enumerate(sample_papers, start=1)}
        state.renumber([(2, sample_papers[1]), (3, sample_papers[2])])
        assert sorted(state.citation_map) == [1, 2]
        assert state.citation_map[1] is sample_papers[1]

    def test_select_papers_respects_critique(self, sample_papers):
        state = AgentState(topic="T")
        state.citation_map = {i: p for i, p in enumerate(sample_papers, start=1)}
        state.critique = CritiqueResult(
            assessments=[
                PaperAssessment(index=1, relevance=9, quality=9, use_in_review=True),
                PaperAssessment(index=2, relevance=1, quality=1, use_in_review=False),
                PaperAssessment(index=3, relevance=7, quality=7, use_in_review=True),
            ]
        )
        selected = state.select_papers(max_papers=10)
        assert 2 not in [index for index, _ in selected], "被判定不纳入的文献必须被过滤掉"

    def test_select_papers_without_critique(self, sample_papers):
        state = AgentState(topic="T")
        state.citation_map = {i: p for i, p in enumerate(sample_papers, start=1)}
        assert len(state.select_papers(max_papers=2)) == 2

    def test_summary_shape(self):
        state = AgentState(topic="T")
        state.phase = Phase.DONE
        payload = state.summary()
        for key in ("run_id", "topic", "phase", "papers", "citations", "elapsed_ms"):
            assert key in payload

    def test_add_error_dedupes(self):
        state = AgentState(topic="T")
        state.add_error("boom")
        state.add_error("boom")
        assert state.errors == ["boom"]

    def test_phase_labels(self):
        assert Phase.PLAN.label == "规划检索策略"
        assert Phase.AWAIT_APPROVAL.label == "等待确认"


# ==================================================================== 写作
class TestWriterHelpers:
    def test_default_outline_shape(self):
        outline = _default_outline()
        assert len(outline) >= 4
        assert all(section.title and section.points for section in outline)


class TestReviewResult:
    def test_passed_when_no_high_issues(self):
        result = ReviewResult(verdict="pass", issues=[{"severity": "low"}])
        assert result.passed

    def test_failed_on_high_issue(self):
        result = ReviewResult(verdict="pass", issues=[{"severity": "high"}])
        assert not result.passed

    def test_failed_on_revise_verdict(self):
        assert not ReviewResult(verdict="revise").passed


# ================================================================== 运行时
class TestRuntime:
    async def test_offline_run_reaches_done(self, tmp_path):
        """离线模式下的完整链路：start → stream → done。"""
        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        config.agent.require_approval = False
        database = Database(tmp_path / "rt.db", config=config)
        runtime = AgentRuntime(config=config, db=database)

        handle = await runtime.start(topic="测试课题", offline=True, require_approval=False)
        events = [event.type async for event in runtime.stream(handle.run_id, timeout=60)]
        assert "done" in events
        assert handle.state.phase in {Phase.DONE, Phase.ERROR}

    async def test_approval_round_trip(self, tmp_path):
        """人类审批节点：必须真的暂停，审批后才继续。"""
        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        config.agent.require_approval = True
        database = Database(tmp_path / "rt2.db", config=config)
        runtime = AgentRuntime(config=config, db=database)

        handle = await runtime.start(topic="审批测试", offline=True, require_approval=True)
        seen: list[str] = []

        async def consume():
            async for event in runtime.stream(handle.run_id, timeout=60):
                seen.append(event.type)

        task = asyncio.create_task(consume())

        for _ in range(200):
            if "awaiting_approval" in seen:
                break
            await asyncio.sleep(0.05)
        assert "awaiting_approval" in seen, "应当暂停等待审批"
        assert handle.status == "awaiting_approval"

        assert await runtime.approve(handle.run_id, "approve") is True
        await asyncio.wait_for(task, timeout=60)
        assert "done" in seen

    async def test_double_approve_rejected(self, tmp_path):
        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        config.agent.require_approval = True
        database = Database(tmp_path / "rt3.db", config=config)
        runtime = AgentRuntime(config=config, db=database)
        handle = await runtime.start(topic="重复审批", offline=True, require_approval=True)

        assert await runtime.approve(handle.run_id, "approve") is True
        assert await runtime.approve(handle.run_id, "approve") is False

    async def test_cancel_path(self, tmp_path):
        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        config.agent.require_approval = True
        database = Database(tmp_path / "rt4.db", config=config)
        runtime = AgentRuntime(config=config, db=database)
        handle = await runtime.start(topic="取消测试", offline=True, require_approval=True)

        assert await runtime.approve(handle.run_id, "cancel") is True
        events = []
        async for event in runtime.stream(handle.run_id, timeout=60):
            events.append(event.type)
            if event.type == "done":
                break
        assert "done" in events

    async def test_stream_unknown_run(self):
        from medscholar.agent.runtime import AgentRuntime

        runtime = AgentRuntime()
        events = [event.type async for event in runtime.stream("nonexistent", timeout=5)]
        assert events[0] == "error"
        assert "done" in events

    async def test_list_runs(self, tmp_path):
        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        database = Database(tmp_path / "rt5.db", config=config)
        runtime = AgentRuntime(config=config, db=database)
        handle = await runtime.start(topic="列表测试", offline=True, require_approval=False)
        runs = runtime.list_runs()
        assert any(r["run_id"] == handle.run_id for r in runs)
        assert runs[0]["topic"] == "列表测试"


# ================================================ 规划 JSON 归一化（回归）
class TestCoercePlanPayload:
    """模型返回顶层数组时，不能整次规划降级成模板大纲。"""

    def test_passthrough_mapping(self):
        from medscholar.agent.state import coerce_plan_payload

        data = {"topic_zh": "rTMS", "queries": []}
        assert coerce_plan_payload(data) == data

    def test_unwraps_single_element_list(self):
        from medscholar.agent.state import coerce_plan_payload

        got = coerce_plan_payload([{"topic_zh": "rTMS", "topic_en": "rTMS"}])
        assert got["topic_zh"] == "rTMS"

    def test_merges_split_object(self):
        """模型把一份计划拆成多个对象放进数组。"""
        from medscholar.agent.state import coerce_plan_payload

        got = coerce_plan_payload(
            [
                {"topic_zh": "rTMS 治疗卒中后抑郁", "topic_en": "rTMS for PSD"},
                {"queries": [{"query": "rTMS AND depression"}]},
                {"outline": [{"title": "引言", "points": ["背景"]}]},
            ]
        )
        assert got["topic_zh"] == "rTMS 治疗卒中后抑郁"
        assert got["queries"] == [{"query": "rTMS AND depression"}]
        assert got["outline"] == [{"title": "引言", "points": ["背景"]}]

    def test_bare_query_list(self):
        from medscholar.agent.state import coerce_plan_payload

        got = coerce_plan_payload([{"query": "rTMS"}, {"query": "depression"}])
        assert got == {"queries": [{"query": "rTMS"}, {"query": "depression"}]}

    def test_bare_outline_list(self):
        from medscholar.agent.state import coerce_plan_payload

        got = coerce_plan_payload([{"title": "引言"}, {"title": "方法"}])
        assert got == {"outline": [{"title": "引言"}, {"title": "方法"}]}

    def test_string_list_without_syntax_is_key_questions(self):
        """回归：qwen3:8b 真的返回过 pico.outcomes 这样的裸字符串数组。

        不能当成章节标题（否则综述大纲会变成「抑郁症状改善」这类结局指标），
        也不该硬塞进 queries；放进 key_questions 并让 outline 留空，
        好让 graph 的兜底补上标准综述大纲。
        """
        from medscholar.agent.state import coerce_plan_payload

        got = coerce_plan_payload(
            ["抑郁症状改善（如HAMD评分）", "治疗反应率", "不良反应发生率", "长期随访结果"]
        )
        assert got == {
            "key_questions": [
                "抑郁症状改善（如HAMD评分）",
                "治疗反应率",
                "不良反应发生率",
                "长期随访结果",
            ]
        }
        assert "outline" not in got and "queries" not in got

    def test_bare_string_list_leaves_outline_for_fallback(self):
        """裸字符串列表经 from_dict 后 outline 仍为空 → 触发模板大纲兜底。"""
        from medscholar.agent.state import ResearchPlan, coerce_plan_payload

        plan = ResearchPlan.from_dict(coerce_plan_payload(["疗效", "安全性"]))
        assert plan.outline == []
        assert plan.queries == []
        assert plan.key_questions == ["疗效", "安全性"]

    def test_string_list_with_search_syntax_is_queries(self):
        from medscholar.agent.state import coerce_plan_payload

        got = coerce_plan_payload(["rTMS AND depression", "TMS OR rTMS [Title]"])
        assert got == {"queries": [{"query": "rTMS AND depression"}, {"query": "TMS OR rTMS [Title]"}]}

    def test_non_container_payloads_become_empty(self):
        from medscholar.agent.state import coerce_plan_payload

        for bad in ("抱歉，我无法回答", 42, None, True, [], [None, ""], {}):
            assert coerce_plan_payload(bad) == {}, bad

    def test_from_dict_survives_list_payload(self):
        """回归：'list' object has no attribute 'get'。"""
        from medscholar.agent.state import ResearchPlan

        plan = ResearchPlan.from_dict(
            [
                {
                    "topic_zh": "加速rTMS治疗卒中后抑郁的疗效试验与安全性",
                    "queries": [{"query": "rTMS AND depression"}],
                    "outline": [{"title": "引言"}],
                }
            ]
        )
        assert plan.topic_zh == "加速rTMS治疗卒中后抑郁的疗效试验与安全性"
        assert plan.queries and plan.queries[0].query == "rTMS AND depression"
        assert plan.outline and plan.outline[0].title == "引言"

    def test_from_dict_survives_garbage_payload(self):
        from medscholar.agent.state import ResearchPlan

        for bad in ("不是 JSON", 3.14, None, ["a", "b"]):
            plan = ResearchPlan.from_dict(bad)  # 不得抛异常
            assert plan.topic_zh == ""


# ================================================ 规划降级必须可被前端识别
class TestPlanDegradedFlag:
    """兜底方案不能伪装成"规划成功"，否则用户会批准一个注定低命中的方案。"""

    async def test_offline_plan_is_marked_degraded(self, tmp_path):
        from medscholar.agent.graph import ResearchGraph
        from medscholar.agent.state import AgentState
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "d1.db", config=config)
        graph = ResearchGraph(config=config, db=db)
        state = AgentState(topic="加速rTMS治疗卒中后抑郁", offline=True)

        plan = await graph.plan(state)
        assert plan.degraded is True
        assert plan.degraded_reason
        assert plan.to_dict()["degraded"] is True
        assert plan.to_dict()["degraded_reason"] == plan.degraded_reason

    async def test_llm_failure_marks_degraded_with_reason(self, tmp_path, monkeypatch):
        """LLM 挂了 → 兜底方案 + 明确原因（截图里"LLM 不可用"那条就是它）。"""
        from medscholar.agent import graph as graph_mod
        from medscholar.agent.graph import ResearchGraph
        from medscholar.agent.state import AgentState
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.llm.client import LLMError

        class BoomClient:
            async def start(self):
                return None

            async def chat_json(self, *a, **k):
                raise LLMError("Ollama 未运行")

        monkeypatch.setattr(graph_mod, "get_llm", lambda *a, **k: BoomClient())

        config = AppConfig(data_dir=str(tmp_path), offline=False)
        db = Database(tmp_path / "d2.db", config=config)
        graph = ResearchGraph(config=config, db=db)
        state = AgentState(topic="加速rTMS治疗卒中后抑郁", offline=False)

        plan = await graph.plan(state)
        assert plan.degraded is True
        assert "Ollama 未运行" in plan.degraded_reason
        assert plan.queries, "兜底方案仍须给出可检索的检索式"
        assert plan.outline, "兜底方案仍须给出大纲"
        assert any("规划" in e for e in state.errors)

    async def test_successful_plan_is_not_degraded(self, tmp_path, monkeypatch):
        """正常规划不得被误标为降级。"""
        from medscholar.agent import graph as graph_mod
        from medscholar.agent.graph import ResearchGraph
        from medscholar.agent.state import AgentState
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        class GoodClient:
            async def start(self):
                return None

            async def chat_json(self, *a, **k):
                return {
                    "topic_zh": "加速rTMS治疗卒中后抑郁",
                    "topic_en": "Accelerated rTMS for post-stroke depression",
                    "queries": [{"query": "rTMS AND depression", "sources": ["pubmed"]}],
                    "outline": [{"title": "引言", "points": ["背景"]}],
                }

        monkeypatch.setattr(graph_mod, "get_llm", lambda *a, **k: GoodClient())

        config = AppConfig(data_dir=str(tmp_path), offline=False)
        db = Database(tmp_path / "d3.db", config=config)
        graph = ResearchGraph(config=config, db=db)
        state = AgentState(topic="加速rTMS治疗卒中后抑郁", offline=False)

        plan = await graph.plan(state)
        assert plan.degraded is False
        assert plan.degraded_reason == ""
        assert plan.to_dict()["degraded"] is False


# ====================================================== 占位符课题拦截（回归）
class TestPlaceholderTopic:

    def test_detects_placeholder_prefixes(self):
        from medscholar.agent.runtime import looks_like_placeholder

        for text in (
            "例如：加速rTMS治疗卒中后抑郁的疗效与安全性",
            "例如:加速rTMS",
            "示例：rTMS 治疗抑郁",
            "示例:rTMS",
            "比如：rTMS",
            "比如:rTMS",
            "请输入研究课题",
            "请在此输入课题",
            "输入研究课题",
            "输入课题",
            "",
            "   ",
        ):
            assert looks_like_placeholder(text), text

    def test_accepts_real_topics(self):
        from medscholar.agent.runtime import looks_like_placeholder

        for text in (
            "加速rTMS治疗卒中后抑郁的疗效与安全性",
            "rTMS加速重复经颅磁刺激治疗卒中后抑郁的初步疗效观察",
            "针刺治疗失眠的系统评价",
            # 只是句中出现示例字样，不算占位符
            "比较例如rTMS与舍曲林治疗抑郁的疗效",
        ):
            assert not looks_like_placeholder(text), text

    async def test_start_rejects_placeholder(self, tmp_path):
        import pytest

        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        config.agent.require_approval = False
        database = Database(tmp_path / "ph.db", config=config)
        runtime = AgentRuntime(config=config, db=database)

        with pytest.raises(ValueError, match="示例提示"):
            await runtime.start(
                topic="例如：加速rTMS治疗卒中后抑郁的疗效与安全性",
                offline=True,
                require_approval=False,
            )
        # 拒绝后不应留下任何运行记录
        assert runtime.list_runs() == []

    async def test_start_rejects_blank(self, tmp_path):
        import pytest

        from medscholar.agent.runtime import AgentRuntime
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        database = Database(tmp_path / "ph2.db", config=config)
        runtime = AgentRuntime(config=config, db=database)

        with pytest.raises(ValueError, match="课题不能为空"):
            await runtime.start(topic="   ", offline=True, require_approval=False)
