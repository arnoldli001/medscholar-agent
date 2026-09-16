"""官方批量数据包导入：PubMed baseline XML 与 PMC OA tar.gz。

全部用合成的小样本，避免测试依赖几十 GB 的真实数据包。
重点验证：流式解析、字段抽取、过滤、以及不会把内存吃爆的处理方式。
"""

from __future__ import annotations

import gzip
import io
import tarfile


from medscholar.bulk import (
    BulkReport,
    import_pmc_oa,
    import_pubmed_baseline,
    iter_pmc_oa_tar,
    iter_pubmed_xml,
    parse_jats,
    parse_pubmed_article,
    parse_terms,
)

PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">33512345</PMID>
      <Article>
        <Journal>
          <Title>Journal of Affective Disorders</Title>
          <JournalIssue>
            <Volume>285</Volume><Issue>3</Issue>
            <PubDate><Year>2021</Year></PubDate>
          </JournalIssue>
        </Journal>
        <ArticleTitle>Efficacy and safety of accelerated rTMS for post-stroke depression</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Post-stroke depression is common.</AbstractText>
          <AbstractText Label="METHODS">A randomized controlled trial of 120 patients.</AbstractText>
        </Abstract>
        <AuthorList>
          <Author><LastName>Zhang</LastName><ForeName>Wei</ForeName></Author>
          <Author><LastName>Li</LastName><ForeName>Ming</ForeName></Author>
          <Author><CollectiveName>PSD Study Group</CollectiveName></Author>
        </AuthorList>
        <Pagination><MedlinePgn>112-119</MedlinePgn></Pagination>
        <ELocationID EIdType="doi">10.1016/j.jad.2021.02.045</ELocationID>
        <PublicationTypeList><PublicationType>Randomized Controlled Trial</PublicationType></PublicationTypeList>
        <Language>eng</Language>
      </Article>
      <MeshHeadingList>
        <MeshHeading><DescriptorName>Depression</DescriptorName></MeshHeading>
        <MeshHeading><DescriptorName>Transcranial Magnetic Stimulation</DescriptorName></MeshHeading>
      </MeshHeadingList>
      <KeywordList><Keyword>rTMS</Keyword><Keyword>stroke</Keyword></KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">33512345</ArticleId>
        <ArticleId IdType="pmc">PMC7890123</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">99999999</PMID>
      <Article>
        <Journal><Title>Some Journal</Title><JournalIssue><PubDate><Year>1998</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>An unrelated old paper about cardiology</ArticleTitle>
        <Abstract><AbstractText>Heart stuff.</AbstractText></Abstract>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID Version="1">88888888</PMID>
      <Article>
        <Journal><Title>No Abstract Journal</Title><JournalIssue><PubDate><Year>2021</Year></PubDate></JournalIssue></Journal>
        <ArticleTitle>Editorial without an abstract about rTMS</ArticleTitle>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""

JATS_NXML = """<?xml version="1.0" encoding="UTF-8"?>
<article>
  <front>
    <journal-meta><journal-title-group><journal-title>Brain Stimulation</journal-title></journal-title-group></journal-meta>
    <article-meta>
      <article-id pub-id-type="pmc">PMC1234567</article-id>
      <article-id pub-id-type="doi">10.1016/j.brs.2020.01.001</article-id>
      <title-group><article-title>Accelerated rTMS in post-stroke depression</article-title></title-group>
      <contrib-group>
        <contrib contrib-type="author"><name><surname>Wang</surname><given-names>Li</given-names></name></contrib>
        <contrib contrib-type="author"><collab>Stroke Group</collab></contrib>
      </contrib-group>
      <pub-date><year>2020</year></pub-date>
      <abstract><p>We studied accelerated rTMS.</p></abstract>
    </article-meta>
  </front>
  <body>
    <sec><title>Introduction</title><p>Stroke is a leading cause of disability.</p></sec>
    <sec><title>Methods</title><p>We enrolled 60 patients and gave accelerated rTMS.</p></sec>
  </body>
</article>
"""


def write_pubmed(tmp_path, text: str = PUBMED_XML, *, gz: bool = False):
    path = tmp_path / ("pubmed.xml.gz" if gz else "pubmed25n0001.xml")
    if gz:
        path.write_bytes(gzip.compress(text.encode("utf-8")))
    else:
        path.write_text(text, encoding="utf-8")
    return path


def write_pmc_tar(tmp_path, *, name: str = "pmc_oa.tar.gz"):
    path = tmp_path / name
    with tarfile.open(path, "w:gz") as tar:
        data = JATS_NXML.encode("utf-8")
        info = tarfile.TarInfo("PMC1234567.nxml")
        info.size = len(data)
        tar.addfile(info, io.BytesIO(data))
        # 顺带放一个 PDF 成员，确认不会被当成 JATS 解析
        pdf = b"%PDF-1.4 fake"
        info2 = tarfile.TarInfo("PMC1234567.pdf")
        info2.size = len(pdf)
        tar.addfile(info2, io.BytesIO(pdf))
    return path


class TestParseTerms:
    def test_splits_on_common_separators(self):
        assert parse_terms("rTMS,depression") == ["rTMS", "depression"]
        assert parse_terms("a；b，c;d") == ["a", "b", "c", "d"]

    def test_none_and_empty(self):
        assert parse_terms(None) == []
        assert parse_terms("") == []
        assert parse_terms(["x", " y "]) == ["x", "y"]


class TestPubmedParsing:
    def test_iterates_only_articles(self, tmp_path):
        path = write_pubmed(tmp_path)
        papers = list(iter_pubmed_xml(path))
        assert len(papers) == 3

    def test_field_extraction(self, tmp_path):
        path = write_pubmed(tmp_path)
        paper = list(iter_pubmed_xml(path))[0]
        assert paper.title == "Efficacy and safety of accelerated rTMS for post-stroke depression"
        assert paper.pmid == "33512345"
        assert paper.pmcid == "PMC7890123"
        assert paper.doi == "10.1016/j.jad.2021.02.045"
        assert paper.journal == "Journal of Affective Disorders"
        assert paper.pub_year == 2021
        assert paper.volume == "285" and paper.issue == "3" and paper.pages == "112-119"
        assert paper.source == "bulk"
        # 结构化摘要要带 Label
        assert "BACKGROUND: Post-stroke depression is common." in paper.abstract
        assert "METHODS: A randomized controlled trial" in paper.abstract
        # 作者：个人 + 团体
        assert paper.authors == ["Zhang, Wei", "Li, Ming", "PSD Study Group"]
        assert "Depression" in paper.mesh_terms
        assert paper.keywords == ["rTMS", "stroke"]
        assert paper.publication_type == "Randomized Controlled Trial"
        assert paper.language == "eng"
        assert paper.url == "https://pubmed.ncbi.nlm.nih.gov/33512345/"

    def test_gz_file_supported(self, tmp_path):
        path = write_pubmed(tmp_path, gz=True)
        assert len(list(iter_pubmed_xml(path))) == 3

    def test_truncated_xml_does_not_raise(self, tmp_path):
        """分卷文件偶尔截断，不能因此让整批导入失败。"""
        path = tmp_path / "truncated.xml"
        path.write_text(PUBMED_XML[: len(PUBMED_XML) // 2], encoding="utf-8")
        papers = list(iter_pubmed_xml(path))
        assert isinstance(papers, list)  # 不抛异常即可

    def test_element_without_title_skipped(self):
        import xml.etree.ElementTree as ET

        elem = ET.fromstring(
            "<PubmedArticle><MedlineCitation><PMID>1</PMID><Article>"
            "<Journal><Title>J</Title></Journal></Article></MedlineCitation></PubmedArticle>"
        )
        assert parse_pubmed_article(elem) is None


class TestJatsParsing:
    def test_metadata_and_fulltext(self):
        parsed = parse_jats(JATS_NXML)
        assert parsed is not None
        paper, fulltext = parsed
        assert paper.title == "Accelerated rTMS in post-stroke depression"
        assert paper.journal == "Brain Stimulation"
        assert paper.pub_year == 2020
        assert paper.doi == "10.1016/j.brs.2020.01.001"
        assert paper.pmcid == "PMC1234567"
        assert paper.authors == ["Wang, Li", "Stroke Group"]
        assert paper.is_open_access is True
        # 正文要带章节标题
        assert "## Introduction" in fulltext
        assert "We enrolled 60 patients" in fulltext

    def test_malformed_xml_returns_none(self):
        assert parse_jats(b"<not xml") is None

    def test_without_title_returns_none(self):
        assert parse_jats("<article><front><article-meta></article-meta></front></article>") is None


class TestPmcTar:
    def test_streams_nxml_and_skips_pdf(self, tmp_path):
        path = write_pmc_tar(tmp_path)
        results = list(iter_pmc_oa_tar(path))
        assert len(results) == 1, "PDF 成员不应产出条目"
        paper, fulltext = results[0]
        assert paper.pmcid == "PMC1234567"
        assert "Introduction" in fulltext

    def test_broken_tar_does_not_raise(self, tmp_path):
        path = tmp_path / "broken.tar.gz"
        path.write_bytes(b"not a tar at all")
        assert list(iter_pmc_oa_tar(path)) == []


class TestImportPubmed:
    async def test_filters_by_terms_and_year(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path / "ms"), offline=True)
        db = Database(tmp_path / "ms" / "b.db", config=config)
        report = await import_pubmed_baseline(
            write_pubmed(tmp_path),
            terms="rTMS,depression",
            year_from=2015,
            embed=False,
            config=config,
            db=db,
        )
        # 只有第 1 条同时命中 rTMS+depression 且年份 >= 2015
        assert report.seen == 3
        assert report.matched == 1, report.errors
        assert report.created == 1
        assert "accelerated rTMS" in report.sample_titles[0]

    async def test_require_abstract_drops_editorials(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path / "ms2"), offline=True)
        db = Database(tmp_path / "ms2" / "b2.db", config=config)
        report = await import_pubmed_baseline(
            write_pubmed(tmp_path), terms="rTMS", embed=False, config=config, db=db
        )
        # 第 3 条虽然含 rTMS，但没有摘要 → 默认丢弃
        assert report.matched == 1

    async def test_dry_run_writes_nothing(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers

        config = AppConfig(data_dir=str(tmp_path / "ms3"), offline=True)
        db = Database(tmp_path / "ms3" / "b3.db", config=config)
        report = await import_pubmed_baseline(
            write_pubmed(tmp_path), dry_run=True, embed=False, config=config, db=db
        )
        assert report.matched == 2  # 有摘要的两条
        assert report.created == 0
        assert count_papers(db=db) == 0

    async def test_limit_respected(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path / "ms4"), offline=True)
        db = Database(tmp_path / "ms4" / "b4.db", config=config)
        report = await import_pubmed_baseline(
            write_pubmed(tmp_path), limit=1, embed=False, config=config, db=db
        )
        assert report.matched == 1

    async def test_directory_input(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        write_pubmed(tmp_path)
        write_pubmed(tmp_path, gz=True)
        config = AppConfig(data_dir=str(tmp_path / "ms5"), offline=True)
        db = Database(tmp_path / "ms5" / "b5.db", config=config)
        report = await import_pubmed_baseline(
            tmp_path, embed=False, config=config, db=db
        )
        assert len(report.files) == 2
        assert report.seen == 6

    async def test_missing_file_reports_clearly(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path / "ms6"), offline=True)
        db = Database(tmp_path / "ms6" / "b6.db", config=config)
        report = await import_pubmed_baseline(
            tmp_path / "nope", embed=False, config=config, db=db
        )
        assert report.seen == 0
        assert report.errors and "没有找到 PubMed XML" in report.errors[0]

    async def test_imported_papers_are_searchable(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import search_fts

        config = AppConfig(data_dir=str(tmp_path / "ms7"), offline=True)
        db = Database(tmp_path / "ms7" / "b7.db", config=config)
        await import_pubmed_baseline(
            write_pubmed(tmp_path), terms="rTMS", embed=False, config=config, db=db
        )
        assert search_fts("accelerated", limit=5, db=db), "批量导入的文献必须可检索"


class TestImportPmcOa:
    async def test_imports_with_fulltext(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers, get_fulltext

        config = AppConfig(data_dir=str(tmp_path / "ms"), offline=True)
        db = Database(tmp_path / "ms" / "p.db", config=config)
        report = await import_pmc_oa(
            write_pmc_tar(tmp_path), embed=False, config=config, db=db
        )
        assert report.matched == 1, report.errors
        assert report.created == 1
        assert report.with_fulltext == 1
        assert count_papers(db=db) == 1
        # 全文要真的进库，而不是只存了题录
        papers = db.query("SELECT paper_id FROM papers LIMIT 1")
        text = get_fulltext(int(papers[0]["paper_id"]), db=db)
        assert "We enrolled 60 patients" in text

    async def test_dry_run(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers

        config = AppConfig(data_dir=str(tmp_path / "ms2"), offline=True)
        db = Database(tmp_path / "ms2" / "p2.db", config=config)
        report = await import_pmc_oa(
            write_pmc_tar(tmp_path), dry_run=True, embed=False, config=config, db=db
        )
        assert report.matched == 1 and report.created == 0
        assert count_papers(db=db) == 0

    async def test_missing_archive_reports_clearly(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        config = AppConfig(data_dir=str(tmp_path / "ms3"), offline=True)
        db = Database(tmp_path / "ms3" / "p3.db", config=config)
        report = await import_pmc_oa(tmp_path / "nope", embed=False, config=config, db=db)
        assert report.errors and "tar.gz" in report.errors[0]


class TestBulkReport:
    def test_summary_mentions_dry_run(self):
        report = BulkReport(seen=100, matched=5, dry_run=True)
        assert "演练" in report.summary()

    def test_summary_counts(self):
        report = BulkReport(seen=100, matched=5, created=4, merged=1, with_fulltext=3)
        text = report.summary()
        assert "扫描 100 条" in text and "命中 5 条" in text
        assert "新增 4 篇" in text and "含全文 3 篇" in text

    def test_to_dict_is_json_safe(self):
        import json

        report = BulkReport(seen=1, matched=1, created=1)
        assert json.loads(json.dumps(report.to_dict()))["created"] == 1
