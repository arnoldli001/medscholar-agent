"""题录文件导入：RIS / BibTeX / EndNote 标记 / CSV。

样例取自各数据库实际导出的样式（Web of Science、Scopus、CNKI、万方、
Zotero/Google Scholar），因为真实导出文件总有各种小怪癖。
"""

from __future__ import annotations

import pytest

from medscholar.importers.parsers import (
    detect_format,
    parse_any,
    parse_bibtex,
    parse_csv,
    parse_ris,
    parse_tagged,
    parse_wos_plain,
)
from medscholar.models import Paper

# Web of Science 的 RIS 导出（savedrecs.ris）。注意 WoS 用 SO 表示期刊名、
# BP/EP 表示页码、DE 表示关键词 —— 这些标签和 Scopus 不一样。
WOS_RIS = """FN Clarivate Analytics Web of Science
VR 1.0
PT J
AU Zhang, Wei
   Li, Ming
   Chen, Hua
TI Efficacy and safety of accelerated rTMS for post-stroke depression: a randomized trial
SO JOURNAL OF AFFECTIVE DISORDERS
VL 285
IS 3
BP 112
EP 119
PY 2021
DI 10.1016/j.jad.2021.02.045
AB This randomized controlled trial evaluated accelerated repetitive transcranial
   magnetic stimulation in patients with post-stroke depression.
LA English
DT Article
DE rTMS; post-stroke depression; accelerated
SN 0165-0327
UR https://www.webofscience.com/wos/record/1
ER

PT J
AU Wang, Li
TI Accelerated rTMS in stroke rehabilitation
SO BRAIN STIMULATION
PY 2020
DI 10.1016/j.brs.2020.01.001
AB A pilot study.
ER

EF
"""

# WoS 的 RIS 导出（真的带短横线的那种）
WOS_RIS_REAL = """TY  - JOUR
AU  - Zhang, Wei
AU  - Li, Ming
AU  - Chen, Hua
TI  - Efficacy and safety of accelerated rTMS for post-stroke depression
SO  - JOURNAL OF AFFECTIVE DISORDERS
VL  - 285
IS  - 3
BP  - 112
EP  - 119
PY  - 2021
DI  - 10.1016/j.jad.2021.02.045
AB  - A randomized trial of accelerated rTMS.
LA  - English
DT  - Article
DE  - rTMS; post-stroke depression; accelerated
SN  - 0165-0327
UR  - https://www.webofscience.com/wos/record/1
ER  -
"""

# Scopus 的 RIS 用 T1/JO，且页码在 SP/EP
SCOPUS_RIS = """TY  - JOUR
T1  - Repetitive transcranial magnetic stimulation for depression
AU  - Smith, John A.
AU  - Doe, Jane
JO  - Lancet Psychiatry
PY  - 2019
SP  - 100
EP  - 108
DO  - 10.1016/S2215-0366(19)30001-2
AB  - Background: rTMS is effective.
KW  - rTMS
KW  - depression
ER  -
"""

BIBTEX_WOS = """@article{Zhang2021,
\tAuthor = {Zhang, Wei and Li, Ming and Chen, Hua},
\tTitle = {Efficacy and safety of accelerated rTMS for post-stroke depression},
\tJournal = {Journal of Affective Disorders},
\tYear = {2021},
\tVolume = {285},
\tNumber = {3},
\tPages = {112--119},
\tDoi = {10.1016/j.jad.2021.02.045},
\tAbstract = {A randomized trial.},
\tKeywords = {rTMS; depression},
\tPmid = {33512345}
}

@inproceedings{Wang2020,
\tAuthor = {Wang, Li},
\tTitle = {Accelerated {rTMS} in stroke rehabilitation},
\tBooktitle = {Proceedings of the Brain Stimulation Conference},
\tYear = {2020},
\tDoi = {10.1000/conf.2020.1}
}
"""

# CNKI 的 EndNote 导出（中文，GB 编码在读取层处理）
CNKI_TAGGED = """%0 Journal Article
%A 张伟
%A 李明
%T 加速重复经颅磁刺激治疗卒中后抑郁的疗效观察
%J 中华神经科杂志
%D 2021
%V 54
%N 3
%P 245-250
%K 卒中后抑郁; 重复经颅磁刺激; 加速治疗
%X 目的 探讨加速rTMS治疗卒中后抑郁的疗效与安全性。
%R 10.3760/cma.j.cn113694-20200812-00631
%U https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD
%L 中文
%W CNKI
"""

SCOPUS_CSV = """Authors,Title,Year,Source title,Volume,Issue,Page start,DOI,Abstract,Language,Document Type
"Zhang W; Li M","Efficacy of accelerated rTMS",2021,"Journal of Affective Disorders",285,3,112,10.1016/j.jad.2021.02.045,"A trial.","English","Article"
"Wang L","rTMS in stroke rehabilitation",2020,"Brain Stimulation",13,1,10,10.1016/j.brs.2020.01.001,"A pilot.","English","Article"
"""


class TestFormatDetection:
    @pytest.mark.parametrize(
        "text,filename,expected",
        [
            (WOS_RIS_REAL, "savedrecs.ris", "ris"),
            (WOS_RIS, "savedrecs.txt", "wos"),
            (SCOPUS_RIS, "scopus.ris", "ris"),
            (BIBTEX_WOS, "export.bib", "bibtex"),
            (BIBTEX_WOS, "export.txt", "bibtex"),
            (CNKI_TAGGED, "cnki.enw", "tagged"),
            (CNKI_TAGGED, "cnki.txt", "tagged"),
            (SCOPUS_CSV, "scopus.csv", "csv"),
            ("完全是别的东西", "x.txt", ""),
            ("", "x.txt", ""),
        ],
    )
    def test_detection(self, text, filename, expected):
        assert detect_format(text, filename) == expected

    def test_wos_plain_text_is_not_mistaken_for_csv(self):
        """踩过的坑：WoS 纯文本存成 .txt，首行含逗号+title，曾被误判为 CSV。"""
        assert detect_format(WOS_RIS, "savedrecs.txt") == "wos"
        assert detect_format(WOS_RIS, "no-extension") == "wos"


class TestWosPlainText:
    """WoS 默认的纯文本导出（无短横线）—— 用户最常用的导出方式。"""

    def test_two_records(self):
        papers = parse_wos_plain(WOS_RIS)
        assert len(papers) == 2
        first = papers[0]
        assert first.title.startswith("Efficacy and safety of accelerated rTMS")
        assert first.authors == ["Zhang, Wei", "Li, Ming", "Chen, Hua"]
        assert first.journal == "JOURNAL OF AFFECTIVE DISORDERS"
        assert first.pub_year == 2021
        assert first.doi == "10.1016/j.jad.2021.02.045"
        assert first.volume == "285"
        assert first.pages == "112-119"
        assert first.keywords == ["rTMS", "post-stroke depression", "accelerated"]

    def test_indented_continuation_joined(self):
        paper = parse_wos_plain(WOS_RIS)[0]
        assert "magnetic stimulation in patients" in paper.abstract

    def test_publication_type_mapped_from_PT(self):
        papers = parse_wos_plain(WOS_RIS)
        assert papers[0].publication_type == "journal-article"


class TestRis:
    def test_wos_ris_export(self):
        papers = parse_ris(WOS_RIS_REAL)
        assert len(papers) == 1
        first = papers[0]
        assert first.title.startswith("Efficacy and safety of accelerated rTMS")
        assert first.authors == ["Zhang, Wei", "Li, Ming", "Chen, Hua"]
        assert first.journal == "JOURNAL OF AFFECTIVE DISORDERS"
        assert first.pub_year == 2021
        assert first.doi == "10.1016/j.jad.2021.02.045"
        assert first.volume == "285"
        assert first.issue == "3"
        assert first.pages == "112-119"
        assert first.keywords == ["rTMS", "post-stroke depression", "accelerated"]

    def test_multiline_abstract_continuation_is_joined(self):
        paper = parse_wos_plain(WOS_RIS)[0]
        assert "magnetic stimulation" in paper.abstract
        assert "\n" not in paper.abstract

    def test_scopus_export_uses_t1_jo(self):
        papers = parse_ris(SCOPUS_RIS)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "Repetitive transcranial magnetic stimulation for depression"
        assert paper.journal == "Lancet Psychiatry"
        assert paper.pages == "100-108"
        assert paper.keywords == ["rTMS", "depression"]

    def test_url_falls_back_to_doi(self):
        paper = parse_ris(SCOPUS_RIS)[0]
        assert paper.url == "https://doi.org/10.1016/S2215-0366(19)30001-2"

    def test_record_without_title_skipped(self):
        text = "TY  - JOUR\nAU  - Nobody\nER  - \n"
        assert parse_ris(text) == []

    def test_empty_input(self):
        assert parse_ris("") == []

    def test_garbage_does_not_crash(self):
        assert parse_ris("这不是 RIS\n随便写点\n") == []


class TestBibtex:
    def test_wos_bibtex_export(self):
        papers = parse_bibtex(BIBTEX_WOS)
        assert len(papers) == 2
        first = papers[0]
        assert first.title == "Efficacy and safety of accelerated rTMS for post-stroke depression"
        assert first.authors == ["Zhang, Wei", "Li, Ming", "Chen, Hua"]
        assert first.journal == "Journal of Affective Disorders"
        assert first.pub_year == 2021
        assert first.pages == "112--119"
        assert first.doi == "10.1016/j.jad.2021.02.045"
        assert first.pmid == "33512345"
        assert first.keywords == ["rTMS", "depression"]

    def test_nested_braces_in_title_are_handled(self):
        paper = parse_bibtex(BIBTEX_WOS)[1]
        assert paper.title == "Accelerated rTMS in stroke rehabilitation"
        assert paper.publication_type == "conference-paper"

    def test_latex_accents_converted(self):
        text = '@article{a, title={M{\\"u}ller and {\\"O}zt{\\"u}rk study}, year={2020}}'
        paper = parse_bibtex(text)[0]
        assert paper.title == "Müller and Öztürk study"

    def test_quoted_values(self):
        text = '@article{a, title = "Quoted title", year = "2019", doi = "10.1/x"}'
        paper = parse_bibtex(text)[0]
        assert paper.title == "Quoted title"
        assert paper.pub_year == 2019

    def test_entry_without_title_skipped(self):
        assert parse_bibtex("@article{a, year={2020}}") == []

    def test_unterminated_entry_does_not_hang(self):
        assert parse_bibtex("@article{a, title={Cut off") == []


class TestTagged:
    def test_cnki_export(self):
        papers = parse_tagged(CNKI_TAGGED)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "加速重复经颅磁刺激治疗卒中后抑郁的疗效观察"
        assert paper.authors == ["张伟", "李明"]
        assert paper.journal == "中华神经科杂志"
        assert paper.pub_year == 2021
        assert paper.doi == "10.3760/cma.j.cn113694-20200812-00631"
        assert paper.pages == "245-250"
        assert paper.language == "中文"
        assert "卒中后抑郁" in paper.keywords

    def test_multiple_records_split_by_percent_zero(self):
        text = CNKI_TAGGED + "\n" + CNKI_TAGGED.replace("张伟", "王芳")
        papers = parse_tagged(text)
        assert len(papers) == 2
        assert papers[0].authors == ["张伟", "李明"]
        assert papers[1].authors == ["王芳", "李明"]
        assert papers[1].journal == "中华神经科杂志"

    def test_records_without_percent_zero_are_merged(self):
        text = "%A 张伟\n%T 标题一\n%A 李明\n%T 标题二\n"
        papers = parse_tagged(text)
        assert len(papers) == 1


class TestCsv:
    def test_scopus_csv_export(self):
        papers = parse_csv(SCOPUS_CSV)
        assert len(papers) == 2
        first = papers[0]
        assert first.title == "Efficacy of accelerated rTMS"
        assert first.authors == ["Zhang W", "Li M"]
        assert first.pub_year == 2021
        assert first.journal == "Journal of Affective Disorders"
        assert first.doi == "10.1016/j.jad.2021.02.045"
        assert first.publication_type == "Article"

    def test_chinese_headers(self):
        text = "标题,作者,期刊,年份,DOI\n卒中后抑郁研究,张三;李四,中华医学杂志,2022,10.1/cn\n"
        papers = parse_csv(text)
        assert len(papers) == 1
        assert papers[0].title == "卒中后抑郁研究"
        assert papers[0].authors == ["张三", "李四"]

    def test_csv_without_title_column_returns_empty(self):
        assert parse_csv("foo,bar\n1,2\n") == []

    def test_bom_is_stripped(self):
        text = "\ufeff标题,作者\n标题A,作者B\n"
        assert parse_csv(text)[0].title == "标题A"

    def test_tab_separated(self):
        text = "Title\tAuthors\tYear\nT1\tA B\t2020\n"
        papers = parse_csv(text)
        assert papers and papers[0].title == "T1"


class TestParseAny:
    def test_returns_format_alongside_papers(self):
        papers, fmt = parse_any(WOS_RIS_REAL, filename="a.ris")
        assert fmt == "ris" and len(papers) == 1
        papers2, fmt2 = parse_any(WOS_RIS, filename="savedrecs.txt")
        assert fmt2 == "wos" and len(papers2) == 2

    def test_all_formats_parse_via_parse_any(self):
        for text, name in (
            (WOS_RIS_REAL, "a.ris"),
            (WOS_RIS, "savedrecs.txt"),
            (BIBTEX_WOS, "a.bib"),
            (CNKI_TAGGED, "a.enw"),
            (SCOPUS_CSV, "a.csv"),
        ):
            papers, fmt = parse_any(text, filename=name)
            assert papers, f"{name} 没有解析出条目（识别为 {fmt or '未知'}）"
            assert all(isinstance(p, Paper) for p in papers)
            assert all(p.title for p in papers)


class TestImportReport:
    async def test_unknown_format_gives_actionable_error(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.importers import import_text

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "i.db", config=config)
        report = await import_text("随便写点东西", filename="x.txt", config=config, db=db)
        assert report.parsed == 0
        assert report.errors and "无法识别文件格式" in report.errors[0]
        assert "RIS" in report.errors[0]

    async def test_dry_run_does_not_write(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.importers import import_text

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "i2.db", config=config)
        report = await import_text(
            WOS_RIS, filename="a.ris", dry_run=True, config=config, db=db
        )
        assert report.parsed == 2 and report.unique == 2
        assert report.created == 0
        from medscholar.db.repo import count_papers

        assert count_papers(db=db) == 0

    async def test_real_import_lands_in_database(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers, list_papers
        from medscholar.importers import import_text

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "i3.db", config=config)
        report = await import_text(
            WOS_RIS, filename="a.ris", embed=False, config=config, db=db
        )
        assert report.created == 2, report.errors
        assert count_papers(db=db) == 2
        titles = [p.title for p in list_papers(limit=10, db=db)]
        assert any("accelerated rTMS" in t for t in titles)

    async def test_reimport_is_idempotent(self, tmp_path):
        """重复导入同一个文件不应产生重复文献。"""
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers
        from medscholar.importers import import_text

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "i4.db", config=config)
        first = await import_text(WOS_RIS, filename="a.ris", embed=False, config=config, db=db)
        second = await import_text(WOS_RIS, filename="a.ris", embed=False, config=config, db=db)
        assert first.created == 2
        assert second.created == 0, "第二次导入不应新增"
        assert count_papers(db=db) == 2

    async def test_imported_papers_are_searchable(self, tmp_path):
        """导入的文献必须能被本地 FTS 搜到 —— 否则导入等于没导。"""
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import search_fts
        from medscholar.importers import import_text

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "i5.db", config=config)
        await import_text(WOS_RIS, filename="a.ris", embed=False, config=config, db=db)
        hits = search_fts("rTMS", limit=10, db=db)
        assert hits, "导入后应当能检索到"

    async def test_cross_format_dedupe(self, tmp_path):
        """同一篇文献从 RIS 和 BibTeX 各导一次，应当合并成一条。"""
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers
        from medscholar.importers import import_text

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "i6.db", config=config)
        await import_text(SCOPUS_RIS, filename="a.ris", embed=False, config=config, db=db)
        bib = (
            "@article{x, title={Repetitive transcranial magnetic stimulation for depression},"
            " author={Smith, John A. and Doe, Jane}, journal={Lancet Psychiatry},"
            " year={2019}, doi={10.1016/S2215-0366(19)30001-2}}"
        )
        report = await import_text(bib, filename="b.bib", embed=False, config=config, db=db)
        assert report.created == 0
        assert count_papers(db=db) == 1


class TestFileReading:
    def test_gbk_encoded_file_is_readable(self, tmp_path):
        """中文数据库的老导出常是 GBK，不能因此整份导入失败。"""
        from medscholar.importers import _read_text

        path = tmp_path / "cnki.enw"
        path.write_bytes(CNKI_TAGGED.encode("gb18030"))
        text = _read_text(path)
        assert "加速重复经颅磁刺激" in text

    def test_utf8_with_bom(self, tmp_path):
        from medscholar.importers import _read_text

        path = tmp_path / "a.csv"
        path.write_text("\ufeff标题,作者\nT,A\n", encoding="utf-8")
        assert _read_text(path).startswith("标题")
