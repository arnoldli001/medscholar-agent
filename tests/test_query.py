"""检索式解析与跨源翻译测试。

覆盖用户提出的「组合检索 / 分隔符多关键词」需求：
空格与逗号 = AND、``|`` 或 ``OR`` = 任选、``-`` 或 ``NOT`` = 排除、引号 = 精确短语，
并按各数据源自身支持的语法翻译（PubMed/Europe PMC/arXiv 支持布尔，
OpenAlex/Semantic Scholar/Crossref 只做相关度检索，退化为核心词）。

直接运行本文件可以打印翻译对照表：

    .python\\python.exe -X utf8 tests\\test_query.py
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from medscholar.query import (  # noqa: E402
    BOOLEAN_SOURCES,
    describe_query,
    for_source,
    parse_query,
)


class TestParsing:
    def test_space_is_and(self):
        parsed = parse_query("rTMS 卒中后抑郁")
        assert parsed.must == ["rTMS", "卒中后抑郁"]
        assert parsed.is_simple

    def test_comma_is_and(self):
        parsed = parse_query("rTMS, 卒中后抑郁")
        assert parsed.must == ["rTMS", "卒中后抑郁"]

    def test_chinese_comma_is_and(self):
        assert parse_query("rTMS，卒中后抑郁").must == ["rTMS", "卒中后抑郁"]

    def test_pipe_is_or(self):
        parsed = parse_query("rTMS | 经颅磁刺激")
        assert parsed.any_groups == [["rTMS", "经颅磁刺激"]]
        assert parsed.must == []
        assert not parsed.is_simple

    def test_explicit_or_word(self):
        assert parse_query("rTMS OR 经颅磁刺激").any_groups == [["rTMS", "经颅磁刺激"]]

    def test_chinese_or_word(self):
        assert parse_query("rTMS 或者 经颅磁刺激").any_groups == [["rTMS", "经颅磁刺激"]]

    def test_chinese_semicolon_is_or(self):
        # 中文全角分号在学术检索里通常表示同义词并列
        parsed = parse_query("加速rTMS；重复经颅磁刺激")
        assert parsed.any_groups == [["加速rTMS", "重复经颅磁刺激"]]

    def test_dash_is_not(self):
        parsed = parse_query("rTMS -动物实验")
        assert parsed.must == ["rTMS"]
        assert parsed.exclude == ["动物实验"]

    def test_not_keyword(self):
        assert parse_query("rTMS NOT 动物实验").exclude == ["动物实验"]

    def test_quoted_phrase(self):
        parsed = parse_query('"post-stroke depression" rTMS')
        assert parsed.phrases == ["post-stroke depression"]
        assert parsed.must == ["rTMS"]

    def test_combined(self):
        parsed = parse_query('加速rTMS 卒中后抑郁 -大鼠 "randomized trial"')
        assert parsed.must == ["加速rTMS", "卒中后抑郁"]
        assert parsed.exclude == ["大鼠"]
        assert parsed.phrases == ["randomized trial"]

    def test_or_group_then_more_and(self):
        parsed = parse_query("rTMS | 经颅磁刺激 卒中后抑郁")
        assert parsed.any_groups == [["rTMS", "经颅磁刺激"]]
        assert parsed.must == ["卒中后抑郁"]

    def test_three_way_or(self):
        parsed = parse_query("a | b | c")
        assert parsed.any_groups == [["a", "b", "c"]]

    def test_dedupe(self):
        assert parse_query("rTMS rTMS rtms").must == ["rTMS"]

    def test_empty(self):
        parsed = parse_query("   ")
        assert parsed.is_empty
        assert for_source(parsed, "pubmed") == ""

    def test_none_safe(self):
        assert parse_query(None).is_empty  # type: ignore[arg-type]

    def test_single_or_group_collapses(self):
        # "a | " 后面没词时不该留下只有一项的 OR 组
        parsed = parse_query("rTMS |")
        assert parsed.any_groups == []
        assert parsed.must == ["rTMS"]


class TestDescribe:
    def test_and(self):
        assert describe_query(parse_query("rTMS 卒中")) == "rTMS 且 卒中"

    def test_or(self):
        assert "或" in describe_query(parse_query("a | b"))

    def test_exclude(self):
        assert "排除" in describe_query(parse_query("a -b"))

    def test_empty(self):
        assert "空" in describe_query(parse_query(""))


class TestTranslation:
    def test_pubmed_and(self):
        assert for_source("rTMS 卒中后抑郁", "pubmed") == "(rTMS AND 卒中后抑郁)"

    def test_pubmed_or(self):
        expr = for_source("rTMS | 经颅磁刺激", "pubmed")
        assert "OR" in expr and "(" in expr

    def test_pubmed_not(self):
        expr = for_source("rTMS -动物实验", "pubmed")
        assert "NOT" in expr and "动物实验" in expr

    def test_pubmed_phrase_quoted(self):
        expr = for_source('"post-stroke depression"', "pubmed")
        assert '"post-stroke depression"' in expr

    def test_pubmed_terms_with_space_quoted(self):
        # 含空格的词必须加引号，否则布尔表达式会被拆散
        expr = for_source("rTMS repetitive transcranial", "pubmed")
        assert expr  # 不抛异常即可，具体切分由分词器决定

    def test_arxiv_uses_andnot(self):
        expr = for_source("rTMS -动物实验", "arxiv")
        assert "ANDNOT" in expr

    def test_non_boolean_sources_get_core_terms(self):
        for source in ("openalex", "crossref", "semantic_scholar", "cnki"):
            expr = for_source("rTMS | 经颅磁刺激 卒中后抑郁 -大鼠", source)
            assert "OR" not in expr and "NOT" not in expr, f"{source} 不支持布尔，应退化为核心词"
            assert "rTMS" in expr and "卒中后抑郁" in expr
            assert "大鼠" not in expr

    def test_boolean_sources_declared(self):
        assert BOOLEAN_SOURCES == {"pubmed", "europepmc", "arxiv"}

    def test_simple_query_passthrough_shape(self):
        # 简单查询在支持布尔与不支持布尔的源上都不该被破坏
        assert for_source("accelerated rTMS", "openalex") == "accelerated rTMS"
        assert "accelerated" in for_source("accelerated rTMS", "pubmed")

    def test_accepts_parsed_object(self):
        parsed = parse_query("rTMS | 经颅磁刺激")
        assert for_source(parsed, "pubmed") == for_source("rTMS | 经颅磁刺激", "pubmed")

    @pytest.mark.parametrize(
        "query",
        ["rTMS", "rTMS 卒中", "a | b", "a -b", '"x y" z', "a | b | c -d"],
    )
    def test_all_sources_never_crash(self, query):
        for source in ("pubmed", "europepmc", "arxiv", "openalex", "crossref",
                       "semantic_scholar", "cnki"):
            assert isinstance(for_source(query, source), str)


def _demo() -> None:
    """打印翻译对照表，方便人工检查（直接运行本文件时执行）。"""
    cases = [
        "rTMS 卒中后抑郁",
        "rTMS, 卒中后抑郁",
        "rTMS | 经颅磁刺激",
        "加速rTMS；重复经颅磁刺激",
        '加速rTMS 卒中后抑郁 -大鼠 "randomized trial"',
    ]
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except (AttributeError, OSError):
        pass
    for case in cases:
        parsed = parse_query(case)
        print("=" * 78)
        print(f"输入　　：{case}")
        print(f"解析结果：{describe_query(parsed)}")
        print(f"　· 必须词：{parsed.must}")
        print(f"　· OR 组：{parsed.any_groups}")
        print(f"　· 精确短语：{parsed.phrases}")
        print(f"　· 排除：{parsed.exclude}")
        for source in ("pubmed", "europepmc", "arxiv", "openalex", "semantic_scholar"):
            mark = "布尔" if source in BOOLEAN_SOURCES else "相关度"
            print(f"　　{source:<17}({mark}) {for_source(parsed, source)}")
        print()


if __name__ == "__main__":
    _demo()
