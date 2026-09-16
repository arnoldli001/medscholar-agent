"""文本处理与检索式构造测试。

这里覆盖的是整个中文检索能力的地基：SQLite FTS5 的 unicode61 分词器会把
连续汉字当作单个 token，因此必须有逐字切分 + 短语/二元组策略。
"""

from __future__ import annotations

import pytest

from medscholar.textutil import (
    build_match_query,
    clean_abstract,
    has_cjk,
    is_cjk,
    normalize_doi,
    normalize_text,
    normalize_title_key,
    segment_cjk,
    title_fingerprint,
    to_family_first,
    truncate,
)


class TestSegmentCjk:
    def test_pure_chinese(self):
        assert segment_cjk("加速治疗") == "加 速 治 疗"

    def test_mixed_chinese_and_ascii(self):
        assert segment_cjk("加速rTMS治疗") == "加 速 rTMS 治 疗"

    def test_trailing_chinese(self):
        assert segment_cjk("rtms治疗") == "rtms 治 疗"

    def test_no_chinese_untouched(self):
        assert segment_cjk("deep brain stimulation") == "deep brain stimulation"

    def test_empty(self):
        assert segment_cjk("") == ""
        assert segment_cjk(None) == ""

    def test_whitespace_collapsed(self):
        assert "  " not in segment_cjk("加速  治疗")

    def test_has_cjk(self):
        assert has_cjk("卒中后抑郁") is True
        assert has_cjk("post-stroke") is False
        assert has_cjk("") is False

    def test_is_cjk_boundaries(self):
        assert is_cjk("中") is True
        assert is_cjk("a") is False
        assert is_cjk("") is False


class TestBuildMatchQuery:
    def test_english_and(self):
        assert build_match_query("rTMS depression") == '"rTMS" AND "depression"'

    def test_english_or(self):
        assert build_match_query("rTMS depression", mode="or") == '"rTMS" OR "depression"'

    def test_cjk_phrase_is_contiguous(self):
        # phrase 策略把整段汉字合成一个短语 → 等价于连续子串匹配
        assert build_match_query("卒中后抑郁") == '"卒 中 后 抑 郁"'

    def test_cjk_bigram(self):
        # 二元组必须是「两个单字 token 组成的短语」：索引侧把汉字切成了单字，
        # 写成双字 token（"卒中"）在索引里根本不存在，永远匹配不到。
        assert (
            build_match_query("卒中后抑郁", cjk="bigram")
            == '("卒 中" AND "中 后" AND "后 抑" AND "抑 郁")'
        )

    def test_cjk_bigram_or(self):
        assert (
            build_match_query("卒中后抑郁", mode="or", cjk="bigram")
            == '("卒 中" OR "中 后" OR "后 抑" OR "抑 郁")'
        )

    def test_cjk_single_char(self):
        assert build_match_query("痛", cjk="bigram") == '"痛"'

    def test_cjk_two_chars_bigram(self):
        assert build_match_query("抑郁", cjk="bigram") == '"抑 郁"'

    def test_mixed_tokens_join(self):
        expr = build_match_query("加速rTMS治疗")
        assert expr == '"加 速" AND "rTMS" AND "治 疗"'

    def test_quote_injection_is_escaped(self):
        # 用户输入的引号不能让 MATCH 表达式失衡
        expr = build_match_query('rTMS " OR "x')
        assert expr.count('"') % 2 == 0

    def test_empty_input(self):
        assert build_match_query("") == ""
        assert build_match_query("   ") == ""

    def test_prefix_mode_adds_wildcard(self):
        assert build_match_query("rehabilitation", prefix=True) == '"rehabilitation"*'

    def test_prefix_not_applied_to_short_tokens(self):
        assert build_match_query("rT", prefix=True) == '"rT"'

    @pytest.mark.parametrize("query", ["卒中后抑郁", "加速rTMS治疗卒中后抑郁", "rTMS 抑郁"])
    def test_produces_valid_fts5_syntax(self, query, db):
        """构造出的表达式必须真的能被 FTS5 解析（防止语法错误导致检索崩溃）。"""
        import sqlite3

        expression = build_match_query(query)
        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.execute("INSERT INTO t VALUES (?)", (segment_cjk(query),))
        rows = conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (expression,)).fetchall()
        assert rows, f"{expression!r} 应能命中自身"

    def test_bigram_matches_across_inserted_modifier(self, db):
        """核心回归：中文分句中间插入修饰语时，放宽策略必须能命中。

        「治疗卒中后抑郁」不该因为原文是「治疗脑卒中后抑郁」而完全检索不到。
        二元组 AND 会因「疗 卒」不存在而失败，因此最终兜底必须是 OR 级
        （repo.search_fts 正是按 phrase → bigram → or 逐级放宽的）。
        """
        import sqlite3

        conn = sqlite3.connect(":memory:")
        conn.execute("CREATE VIRTUAL TABLE t USING fts5(x)")
        conn.execute("INSERT INTO t VALUES (?)", (segment_cjk("针刺治疗脑卒中后抑郁机制研究进展"),))

        phrase = build_match_query("治疗卒中后抑郁", cjk="phrase")
        bigram_and = build_match_query("治疗卒中后抑郁", cjk="bigram")
        loose = build_match_query("治疗卒中后抑郁", mode="or", cjk="bigram")

        hits_phrase = conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (phrase,)).fetchall()
        hits_loose = conn.execute("SELECT rowid FROM t WHERE t MATCH ?", (loose,)).fetchall()

        assert hits_phrase == [], "phrase 策略本就匹配不到（这正是需要放宽策略的原因）"
        assert hits_loose, f"OR 级放宽策略应能命中：{loose}"
        assert "AND" in bigram_and, "二元组 AND 级也应当生成（更精确的中间档）"


class TestNormalization:
    def test_normalize_doi_strips_prefixes(self):
        assert normalize_doi("https://doi.org/10.1/ABC") == "10.1/abc"
        assert normalize_doi("doi:10.1/abc") == "10.1/abc"
        assert normalize_doi("  10.1/abc.  ") == "10.1/abc"

    def test_normalize_doi_rejects_invalid(self):
        assert normalize_doi("not-a-doi") is None
        assert normalize_doi("") is None
        assert normalize_doi(None) is None

    def test_clean_abstract_strips_jats(self):
        raw = "<h4>BACKGROUND</h4>Post-stroke &amp; depression is common."
        cleaned = clean_abstract(raw)
        assert "<" not in cleaned and "&amp;" not in cleaned
        assert "Post-stroke & depression" in cleaned

    def test_normalize_text_nfkc(self):
        assert normalize_text("ＡＢＣ　１２３") == "ABC 123"

    def test_title_key_ignores_case_and_punctuation(self):
        assert normalize_title_key("Accelerated rTMS: A Trial!") == normalize_title_key(
            "accelerated rtms a trial"
        )

    def test_title_fingerprint_stable(self):
        assert title_fingerprint("Hello World") == title_fingerprint("hello, world!")
        assert title_fingerprint("A") != title_fingerprint("B")
        assert title_fingerprint("") == ""

    def test_truncate(self):
        assert truncate("abcdefghij", 5) == "abcd…"
        assert truncate("abc", 5) == "abc"
        assert truncate("", 5) == ""


class TestFamilyFirst:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Wei Zhang", "Zhang Wei"),
            ("Zhang, Wei", "Zhang Wei"),
            ("Jean-Pierre Dupont", "Dupont Jean-Pierre"),
            ("van der Berg Jan", "van der Berg Jan"),
            ("王伟", "王伟"),
            ("Malone", "Malone"),
        ],
    )
    def test_cases(self, raw, expected):
        assert to_family_first(raw) == expected

    def test_empty(self):
        assert to_family_first("") == ""
        assert to_family_first(None) == ""
