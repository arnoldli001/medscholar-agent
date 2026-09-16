"""引用格式化引擎测试。

重点验证两件事：
1. 六种样式的输出结构符合各自规范；
2. 作者姓氏提取正确 —— 这是 APA「Zhang, W.」与 Vancouver「Zhang W」的前提，
   依赖各数据源客户端把姓名统一成「姓 名」顺序。
"""

from __future__ import annotations

import pytest

from medscholar.cite import (
    STYLE_LABELS,
    STYLES,
    citation_key,
    detect_style,
    format_authors,
    format_citation,
    format_inline,
    format_records,
    format_reference_list,
    split_author,
    to_bibtex,
    to_ris,
)


class TestSplitAuthor:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("Zhang Wei", ("Zhang", "Wei")),
            ("Malone DA", ("Malone", "DA")),
            ("王伟", ("王伟", "")),
            ("van der Berg Jan", ("van der Berg", "Jan")),
            ("Smith", ("Smith", "")),
        ],
    )
    def test_cases(self, raw, expected):
        assert split_author(raw) == expected

    def test_empty(self):
        assert split_author("") == ("", "")


class TestDetectStyle:
    @pytest.mark.parametrize(
        "raw,expected",
        [
            ("APA", "apa7"), ("apa7", "apa7"), ("APA 7th", "apa7"),
            ("Vancouver", "vancouver"), ("GB/T 7714", "gb7714"),
            ("gbt7714", "gb7714"), ("国标", "gb7714"),
            ("bibtex", "bibtex"), ("RIS", "ris"),
        ],
    )
    def test_aliases(self, raw, expected):
        assert detect_style(raw) == expected

    def test_unknown_falls_back(self):
        assert detect_style("harvard") == "gb7714"
        assert detect_style(None, default="apa7") == "apa7"

    def test_all_styles_have_labels(self):
        for style in STYLES:
            assert style in STYLE_LABELS


class TestFormatAuthors:
    def test_apa_two_authors(self):
        assert format_authors(["Zhang Wei", "Li Ming"], "apa7") == "Zhang, W., & Li, M."

    def test_apa_three_authors(self):
        result = format_authors(["Zhang Wei", "Li Ming", "Chen Hua"], "apa7")
        assert result == "Zhang, W., Li, M., & Chen, H."

    def test_vancouver_six_no_et_al(self):
        names = ["Zhang Wei", "Li Ming", "Chen Hua", "Wang Lei", "Liu Yang", "Zhao Min"]
        assert format_authors(names, "vancouver") == "Zhang W, Li M, Chen H, Wang L, Liu Y, Zhao M"

    def test_vancouver_seven_uses_et_al(self):
        names = ["Zhang Wei", "Li Ming", "Chen Hua", "Wang Lei", "Liu Yang", "Zhao Min", "Sun Qi"]
        assert format_authors(names, "vancouver").endswith(", et al.")

    def test_gb7714_three_then_deng(self):
        assert format_authors(["王伟", "李静", "张强"], "gb7714") == "王伟, 李静, 张强"
        assert format_authors(["王伟", "李静", "张强", "赵敏"], "gb7714").endswith(", 等")

    def test_empty_authors(self):
        assert format_authors([], "apa7") == "Anonymous"
        assert format_authors([], "gb7714") == "佚名"


class TestCitationOutput:
    def test_apa7_structure(self, sample_papers):
        text = format_citation(sample_papers[0], "apa7")
        assert "Zhang, W., Li, M., & Chen, H." in text
        assert "(2023)." in text
        assert "Brain Stimulation, 16(2), 100-108" in text
        assert "https://doi.org/10.1016/j.brs.2023.001" in text

    def test_apa7_no_leading_number(self, sample_papers):
        assert not format_citation(sample_papers[0], "apa7").startswith("[")

    def test_vancouver_numbered(self, sample_papers):
        text = format_citation(sample_papers[0], "vancouver", index=3)
        assert text.startswith("3. ")
        assert "Zhang W" in text
        assert "doi:10.1016/j.brs.2023.001" in text

    def test_vancouver_unnumbered(self, sample_papers):
        assert not format_citation(sample_papers[0], "vancouver").startswith("1.")

    def test_gb7714_english_has_type_marker(self, sample_papers):
        text = format_citation(sample_papers[0], "gb7714", index=1)
        assert text.startswith("[1] ")
        assert "[J]" in text

    def test_gb7714_chinese_format(self, sample_papers):
        text = format_citation(sample_papers[1], "gb7714", index=2)
        assert "[J]" in text
        assert "中国康复医学杂志" in text
        assert "2022" in text

    def test_gb7714_dissertation_marker(self, sample_papers):
        from medscholar.models import Paper

        thesis = Paper(title="某学位论文", source="manual", publication_type="Dissertation")
        assert "[D]" in format_citation(thesis, "gb7714")

    def test_chicago_no_double_period(self, sample_papers):
        text = format_citation(sample_papers[0], "chicago")
        assert ".." not in text

    def test_chicago_year_after_author(self, sample_papers):
        text = format_citation(sample_papers[0], "chicago")
        assert text.startswith("Zhang, W., Li, M., & Chen, H. 2023.")

    def test_chicago_volume_issue_together(self, sample_papers):
        text = format_citation(sample_papers[0], "chicago")
        assert "16(2): 100-108" in text

    def test_all_styles_produce_nonempty(self, sample_papers):
        for style in ("apa7", "vancouver", "gb7714", "chicago"):
            for paper in sample_papers:
                assert len(format_citation(paper, style, index=1)) > 20


class TestInline:
    @pytest.mark.parametrize("style", ["vancouver", "gb7714", "bibtex", "ris"])
    def test_numeric_styles(self, style, sample_papers):
        assert format_inline(sample_papers[0], style, index=5) == "[5]"

    def test_apa_three_authors_et_al(self, sample_papers):
        assert format_inline(sample_papers[0], "apa7", index=1) == "(Zhang et al., 2023)"

    def test_apa_two_authors_ampersand(self, sample_papers):
        from medscholar.models import Paper

        two = Paper(title="T", source="manual", authors=["Zhang Wei", "Li Ming"], pub_year=2020)
        assert format_inline(two, "apa7") == "(Zhang & Li, 2020)"

    def test_apa_single_author(self, sample_papers):
        assert format_inline(sample_papers[2], "apa7") == "(Malone, 2019)"


class TestReferenceList:
    def test_numbered_and_ordered(self, sample_papers):
        text = format_reference_list(sample_papers, "gb7714")
        lines = [ln for ln in text.splitlines() if ln.strip()]
        assert len(lines) == 3
        assert lines[0].startswith("[1]")
        assert lines[2].startswith("[3]")

    def test_cited_order_preserved(self, sample_papers):
        text = format_reference_list(sample_papers, "gb7714", sort="cited")
        assert sample_papers[0].title[:20] in text.splitlines()[0]

    def test_year_sort(self, sample_papers):
        text = format_reference_list(sample_papers, "gb7714", sort="year")
        first_line = text.splitlines()[0]
        assert "2023" in first_line, "按年份降序时 2023 应排第一"

    def test_empty(self):
        assert format_reference_list([], "gb7714") == ""


class TestBibtex:
    def test_entry_shape(self, sample_papers):
        entry = to_bibtex(sample_papers[0])
        assert entry.startswith("@article{")
        assert entry.rstrip().endswith("}")
        assert "author" in entry and "Zhang Wei and Li Ming and Chen Hua" in entry
        assert "100--108" in entry, "BibTeX 的页码范围必须用双连字符"

    def test_special_chars_escaped(self):
        from medscholar.models import Paper

        paper = Paper(title="A & B: 100% _test_", source="manual", journal="J#1")
        entry = to_bibtex(paper)
        assert r"\&" in entry and r"\%" in entry and r"\_" in entry and r"\#" in entry

    def test_citation_key_shape(self, sample_papers):
        key = citation_key(sample_papers[0])
        assert key.startswith("zhang2023")
        assert key.isascii() and " " not in key

    def test_citation_key_collision_suffix(self, sample_papers):
        taken: set[str] = set()
        first = citation_key(sample_papers[0], taken=taken)
        second = citation_key(sample_papers[0], taken=taken)
        assert first != second

    def test_chinese_author_key_falls_back(self, sample_papers):
        key = citation_key(sample_papers[1])
        assert key  # 中文姓氏无法转拼音，但键必须仍然合法


class TestRis:
    def test_entry_shape(self, sample_papers):
        entry = to_ris(sample_papers[0])
        assert entry.startswith("TY  - JOUR")
        assert entry.rstrip().endswith("ER  -")
        assert "TI  - Accelerated rTMS" in entry
        assert "DO  - 10.1016/j.brs.2023.001" in entry

    def test_pages_split(self, sample_papers):
        from medscholar.models import Paper

        paper = Paper(title="T", source="manual", pages="512-516")
        entry = to_ris(paper)
        assert "SP  - 512" in entry and "EP  - 516" in entry

    def test_multiple_authors_each_au_line(self, sample_papers):
        entry = to_ris(sample_papers[1])
        assert entry.count("AU  - ") == 4


class TestFormatRecords:
    def test_bibtex_multiple_separated(self, sample_papers):
        text = format_records(sample_papers, "bibtex")
        assert text.count("@article{") == 3

    def test_ris_multiple(self, sample_papers):
        text = format_records(sample_papers, "ris")
        assert text.count("TY  - JOUR") == 3

    def test_other_styles_delegate(self, sample_papers):
        assert format_records(sample_papers, "gb7714") == format_reference_list(
            sample_papers, "gb7714"
        )
