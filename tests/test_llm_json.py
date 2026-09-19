"""JSON 容错解析：形状感知 + 截断补全。

回归来源：qwen3:8b 用真实课题「加速rTMS治疗卒中后抑郁的疗效试验与安全性」规划时，
要么把 queries 里的 JSON 写坏，要么在 queries 中间陷入空白循环烧完 token，
而旧的 extract_json 会把残缺对象里第一个配平的 [...]（即 pico.outcomes）
当成整份计划返回，导致 topic_zh / queries / outline 全丢。
"""

from __future__ import annotations

import json

import pytest

from medscholar.llm.client import extract_json as client_extract_json
from medscholar.llm.json_parsing import LLMError, _close_truncated, _json_candidates, extract_json

# 真实抓取的截断输出（模型在 "query": "(" 之后一直输出空白直到预算耗尽）
TRUNCATED_PLAN = (
    "{\n"
    '  "topic_zh": "重复经颅磁刺激加速治疗卒中后抑郁的疗效与安全性研究",\n'
    '  "topic_en": "Efficacy and Safety of Accelerated rTMS for Post-Stroke Depression",\n'
    '  "pico": {\n'
    '    "population": "卒中后抑郁患者",\n'
    '    "intervention": "加速重复经颅磁刺激（accelerated rTMS）",\n'
    '    "comparator": "常规rTMS、药物治疗、安慰剂",\n'
    '    "outcomes": [\n'
    '      "抑郁症状改善（如HAMD评分）",\n'
    '      "安全性指标（如不良事件发生率）",\n'
    '      "治疗依从性",\n'
    '      "长期疗效维持"\n'
    "    ]\n"
    "  },\n"
    '  "queries": [\n'
    "    {\n"
    '      "query": "("  \n'
    + "\n" * 60
)

# 真实抓取的另一种走样：对象合法配平，但 queries 里有个未转义的双引号
MALFORMED_PLAN = (
    "{\n"
    '  "topic_zh": "加速rTMS治疗卒中后抑郁",\n'
    '  "queries": [\n'
    '    {"query": "("加速rTMS" OR "rTMS")", "sources": ["pubmed"]}\n'
    "  ],\n"
    '  "outline": [{"title": "引言"}]\n'
    "}"
)


class TestExtractJsonShape:
    def test_broken_object_does_not_yield_nested_array(self):
        """核心回归：残缺对象不得把嵌套数组当成答案交给调用方。"""
        with pytest.raises(LLMError):
            extract_json(MALFORMED_PLAN, expect="object")

    def test_broken_object_never_yields_nested_array(self):
        """形状感知不依赖 expect：以 { 开头就绝不把嵌套数组当答案。

        残缺对象里第一个配平的 [...] 是 pico.outcomes，把它当答案会静默丢掉
        topic_zh / queries / outline。
        """
        assert all(not c.lstrip().startswith("[") for c in _json_candidates(MALFORMED_PLAN))
        with pytest.raises(LLMError):
            extract_json(MALFORMED_PLAN)

    def test_leading_array_is_still_returned(self):
        raw = '["疗效", "安全性"]'
        assert extract_json(raw) == ["疗效", "安全性"]
        assert extract_json(raw, expect="array") == ["疗效", "安全性"]

    def test_leading_array_rejected_when_object_expected(self):
        with pytest.raises(LLMError):
            extract_json('["疗效", "安全性"]', expect="object")

    def test_object_rejected_when_array_expected(self):
        with pytest.raises(LLMError):
            extract_json('{"a": 1}', expect="array")

    def test_plain_object(self):
        assert extract_json('{"a": 1}', expect="object") == {"a": 1}

    def test_fenced_object(self):
        raw = '这是结果：\n```json\n{"a": 1, "b": [1, 2]}\n```\n以上。'
        assert extract_json(raw, expect="object") == {"a": 1, "b": [1, 2]}

    def test_trailing_comma_repair(self):
        assert extract_json('{"a": [1, 2,],}', expect="object") == {"a": [1, 2]}

    def test_chinese_quotes_repair(self):
        assert extract_json('{“a”: 1}', expect="object") == {"a": 1}

    def test_empty_raises(self):
        with pytest.raises(LLMError):
            extract_json("")


class TestCloseTruncated:
    def test_recovers_complete_prefix_fields(self):
        """截断也要把已经生成好的 topic_zh / pico 捞回来。"""
        recovered = _close_truncated(TRUNCATED_PLAN)
        assert recovered is not None
        parsed = json.loads(recovered)
        assert parsed["topic_zh"] == "重复经颅磁刺激加速治疗卒中后抑郁的疗效与安全性研究"
        assert parsed["pico"]["population"] == "卒中后抑郁患者"
        assert parsed["pico"]["outcomes"][0] == "抑郁症状改善（如HAMD评分）"

    def test_truncated_plan_parses_via_extract_json(self):
        got = extract_json(TRUNCATED_PLAN, expect="object")
        assert isinstance(got, dict)
        assert got["topic_en"].startswith("Efficacy and Safety")

    def test_already_balanced_returns_none(self):
        assert _close_truncated('{"a": 1}') is None
        assert _close_truncated("[1, 2]") is None

    def test_unterminated_string_returns_none(self):
        assert _close_truncated('{"a": "没有结束') is None

    def test_empty_and_garbage(self):
        assert _close_truncated("") is None
        assert _close_truncated("{") is None

    def test_nested_truncation_closes_all_levels(self):
        """截断在第 2 个元素中间：无法区分 2 与 20，保守丢弃未完成的标量。"""
        recovered = _close_truncated('{"a": {"b": [1, 2')
        assert recovered is not None
        assert json.loads(recovered) == {"a": {"b": [1]}}

    def test_truncation_keeps_last_complete_element(self):
        recovered = _close_truncated('{"a": {"b": [1, 2,')
        assert recovered is not None
        assert json.loads(recovered) == {"a": {"b": [1, 2]}}


class TestTruncatedPlanEndToEnd:
    """截断输出经完整链路后，仍应得到可用的计划（而不是空壳）。"""

    def test_plan_from_truncated_and_malformed(self):
        from medscholar.agent.state import (
            ResearchPlan,
            coerce_plan_payload,
            _usable_query,
        )

        payload = extract_json(TRUNCATED_PLAN, expect="object")
        plan = ResearchPlan.from_dict(coerce_plan_payload(payload))
        assert plan.topic_zh.startswith("重复经颅磁刺激")
        assert plan.pico["intervention"].startswith("加速重复经颅磁")
        # "(" 这类纯标点检索式必须被丢掉，好让 graph 退回用课题检索
        assert all(_usable_query(q.query) for q in plan.queries)

    def test_punctuation_only_query_is_dropped(self):
        from medscholar.agent.state import ResearchPlan

        plan = ResearchPlan.from_dict(
            {"queries": [{"query": "("}, {"query": "  "}, {"query": "rTMS AND depression"}]}
        )
        assert [q.query for q in plan.queries] == ["rTMS AND depression"]

    def test_dropping_bad_queries_leaves_empty_for_fallback(self):
        from medscholar.agent.state import ResearchPlan

        plan = ResearchPlan.from_dict({"queries": [{"query": "("}]})
        assert plan.queries == []

class TestReExport:
    """client.extract_json 必须仍可用：它是 medscholar.llm.__init__ 与
    既有调用方在用的公开名字（搬迁不应改变对外 API）。"""

    def test_client_reexports_same_function(self):
        assert client_extract_json is extract_json
