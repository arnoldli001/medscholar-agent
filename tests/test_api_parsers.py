"""数据源解析器测试（完全离线，使用固定的真实响应样本）。

比联网测试更有价值的地方：数据源改了字段结构时，这些测试会**立刻**失败，
而不是等到线上检索悄悄返回空结果。

样本都是从各 API 的真实响应里截取并精简过的。
"""

from __future__ import annotations

import pytest

from medscholar.api import SearchFilters, SourceError
from medscholar.api.arxiv_client import ArxivClient
from medscholar.api.cnki_client import CnkiClient
from medscholar.api.crossref_client import CrossrefClient
from medscholar.api.europepmc_client import EuropePMCClient, _jats_to_text
from medscholar.api.openalex_client import OpenAlexClient, reconstruct_abstract
from medscholar.api.pubmed_client import PubMedClient
from medscholar.api.semantic_scholar_client import SemanticScholarClient

# --------------------------------------------------------------------- PubMed
PUBMED_XML = """<?xml version="1.0" ?>
<PubmedArticleSet>
  <PubmedArticle>
    <MedlineCitation Status="MEDLINE">
      <PMID Version="1">37123456</PMID>
      <Article PubModel="Print">
        <Journal>
          <JournalIssue CitedMedium="Internet">
            <Volume>16</Volume>
            <Issue>2</Issue>
            <PubDate><Year>2023</Year><Month>Mar</Month></PubDate>
          </JournalIssue>
          <Title>Brain Stimulation</Title>
          <ISOAbbreviation>Brain Stimul</ISOAbbreviation>
        </Journal>
        <ArticleTitle>Accelerated <i>rTMS</i> for post-stroke depression: a randomized trial</ArticleTitle>
        <Abstract>
          <AbstractText Label="BACKGROUND">Post-stroke depression is common.</AbstractText>
          <AbstractText Label="METHODS">Sixty patients were randomly assigned.</AbstractText>
          <AbstractText Label="RESULTS">HAMD scores decreased significantly.</AbstractText>
        </Abstract>
        <Pagination><MedlinePgn>100-108</MedlinePgn></Pagination>
        <ELocationID EIdType="doi" ValidYN="Y">10.1016/j.brs.2023.001</ELocationID>
        <AuthorList CompleteYN="Y">
          <Author ValidYN="Y"><LastName>Zhang</LastName><ForeName>Wei</ForeName></Author>
          <Author ValidYN="Y"><LastName>Li</LastName><ForeName>Ming</ForeName></Author>
          <Author ValidYN="Y"><CollectiveName>The NoTSAD Group</CollectiveName></Author>
        </AuthorList>
        <PublicationTypeList>
          <PublicationType UI="D016428">Journal Article</PublicationType>
          <PublicationType UI="D016449">Randomized Controlled Trial</PublicationType>
        </PublicationTypeList>
        <Language>eng</Language>
      </Article>
      <MeshHeadingList>
        <MeshHeading>
          <DescriptorName UI="D003863">Depression</DescriptorName>
          <QualifierName UI="Q000453">etiology</QualifierName>
        </MeshHeading>
        <MeshHeading>
          <DescriptorName UI="D013548">Transcranial Magnetic Stimulation</DescriptorName>
        </MeshHeading>
      </MeshHeadingList>
      <KeywordList Owner="NOTNLM">
        <Keyword MajorTopicYN="N">accelerated protocol</Keyword>
        <Keyword MajorTopicYN="N">stroke</Keyword>
      </KeywordList>
    </MedlineCitation>
    <PubmedData>
      <ArticleIdList>
        <ArticleId IdType="pubmed">37123456</ArticleId>
        <ArticleId IdType="doi">10.1016/j.brs.2023.001</ArticleId>
        <ArticleId IdType="pmc">PMC10239808</ArticleId>
      </ArticleIdList>
    </PubmedData>
  </PubmedArticle>
  <PubmedArticle>
    <MedlineCitation>
      <PMID>32849235</PMID>
      <Article>
        <Journal>
          <JournalIssue><PubDate><MedlineDate>2020 Jul-Aug</MedlineDate></PubDate></JournalIssue>
          <Title>Frontiers in Neurology</Title>
        </Journal>
        <ArticleTitle>Novel TMS for Stroke and Depression</ArticleTitle>
        <AuthorList>
          <Author><LastName>Smith</LastName><Initials>JA</Initials></Author>
        </AuthorList>
      </Article>
    </MedlineCitation>
  </PubmedArticle>
</PubmedArticleSet>
"""


class TestPubMedParser:
    @pytest.fixture()
    def papers(self):
        client = PubMedClient()
        return client._parse_articles(PUBMED_XML)

    def test_parses_both_records(self, papers):
        assert len(papers) == 2

    def test_core_fields(self, papers):
        paper = papers[0]
        assert paper.pmid == "37123456"
        assert paper.doi == "10.1016/j.brs.2023.001"
        assert paper.pmcid == "PMC10239808"
        assert paper.journal == "Brain Stimulation"
        assert paper.pub_year == 2023
        assert paper.volume == "16" and paper.issue == "2" and paper.pages == "100-108"

    def test_nested_markup_in_title_flattened(self, papers):
        assert "Accelerated rTMS for post-stroke depression" in papers[0].title
        assert "<i>" not in papers[0].title

    def test_structured_abstract_labels_kept(self, papers):
        abstract = papers[0].abstract
        assert "BACKGROUND: Post-stroke depression is common." in abstract
        assert "METHODS:" in abstract and "RESULTS:" in abstract

    def test_authors_are_family_first(self, papers):
        assert papers[0].authors[:2] == ["Zhang Wei", "Li Ming"]

    def test_collective_name_kept(self, papers):
        assert "The NoTSAD Group" in papers[0].authors

    def test_initials_used_when_no_forename(self, papers):
        assert papers[1].authors == ["Smith JA"]

    def test_mesh_with_qualifier(self, papers):
        assert "Depression / etiology" in papers[0].mesh_terms
        assert "Transcranial Magnetic Stimulation" in papers[0].mesh_terms

    def test_keywords(self, papers):
        assert papers[0].keywords == ["accelerated protocol", "stroke"]

    def test_publication_types(self, papers):
        assert "Randomized Controlled Trial" in papers[0].publication_type

    def test_open_access_flag_from_pmcid(self, papers):
        assert papers[0].is_open_access is True
        assert papers[1].is_open_access is False

    def test_medline_date_year_parsed(self, papers):
        assert papers[1].pub_year == 2020

    def test_malformed_xml_raises_source_error(self):
        client = PubMedClient()
        with pytest.raises(SourceError):
            client._parse_articles("<not-xml")

    def test_empty_set(self):
        client = PubMedClient()
        assert client._parse_articles("<PubmedArticleSet></PubmedArticleSet>") == []

    def test_term_builder_open_access(self):
        term = PubMedClient._build_term("rTMS", SearchFilters(open_access_only=True))
        assert "free full text[sb]" in term

    def test_term_builder_publication_type(self):
        term = PubMedClient._build_term(
            "rTMS", SearchFilters(publication_types=["Randomized Controlled Trial"])
        )
        assert '"Randomized Controlled Trial"[pt]' in term


# ----------------------------------------------------------------- Europe PMC
EPMC_JSON = {
    "resultList": {
        "result": [
            {
                "id": "37123456",
                "source": "MED",
                "pmid": "37123456",
                "pmcid": "PMC10239808",
                "doi": "10.1016/j.brs.2023.001",
                "title": "Accelerated rTMS for <i>post-stroke</i> depression",
                "authorString": "Zhang W, Li M, Chen H.",
                "authorList": {
                    "author": [
                        {"lastName": "Zhang", "firstName": "Wei"},
                        {"lastName": "Li", "firstName": "Ming"},
                        {"collectiveName": "The NoTSAD Group"},
                        {"fullName": "Hua Chen"},
                    ]
                },
                "journalInfo": {
                    "volume": "16",
                    "issue": "2",
                    "yearOfPublication": 2023,
                    "journal": {"title": "Brain Stimulation"},
                },
                "pubYear": "2023",
                "abstractText": "Post-stroke depression is common. <h4>Methods</h4>60 patients.",
                "citedByCount": 42,
                "isOpenAccess": "Y",
                "pageInfo": "100-108",
                "language": "eng",
                "pubType": "Journal Article",
                "meshHeadingList": {
                    "meshHeading": [
                        {"descriptorName": "Depression"},
                        {"descriptorName": "Stroke"},
                    ]
                },
                "keywordList": {"keyword": ["accelerated protocol", "rTMS"]},
                "fullTextUrlList": {
                    "fullTextUrl": [
                        {"availability": "Open access", "documentStyle": "html",
                         "url": "https://europepmc.org/article/PMC/PMC10239808"},
                        {"availability": "Open access", "documentStyle": "pdf",
                         "url": "https://example.org/paper.pdf"},
                    ]
                },
            }
        ]
    },
    "nextCursorMark": "AoIIP2",
}


class TestEuropePMCParser:
    @pytest.fixture()
    def paper(self):
        client = EuropePMCClient()
        return client._parse_hit(EPMC_JSON["resultList"]["result"][0])

    def test_core_fields(self, paper):
        assert paper.pmid == "37123456"
        assert paper.pmcid == "PMC10239808"
        assert paper.doi == "10.1016/j.brs.2023.001"
        assert paper.journal == "Brain Stimulation"
        assert paper.pub_year == 2023
        assert paper.cited_by_count == 42
        assert paper.is_open_access is True

    def test_html_stripped_from_title_and_abstract(self, paper):
        assert "<i>" not in paper.title
        assert "<h4>" not in paper.abstract
        assert "Methods" in paper.abstract

    def test_structured_authors_preferred(self, paper):
        assert paper.authors[:2] == ["Zhang Wei", "Li Ming"]
        assert "The NoTSAD Group" in paper.authors

    def test_fullname_converted_to_family_first(self, paper):
        assert "Chen Hua" in paper.authors

    def test_oa_landing_page_preferred_over_pdf(self, paper):
        assert paper.full_text_url.endswith("PMC10239808")
        assert not paper.full_text_url.endswith(".pdf")

    def test_mesh_and_keywords(self, paper):
        assert paper.mesh_terms == ["Depression", "Stroke"]
        assert paper.keywords == ["accelerated protocol", "rTMS"]

    def test_query_builder_year_filter(self):
        expr = EuropePMCClient._build_query("rTMS", SearchFilters(year_from=2015, year_to=2023))
        assert "PUB_YEAR:[2015 TO 3000]" in expr
        assert "PUB_YEAR:[1000 TO 2023]" in expr

    def test_query_builder_open_access(self):
        expr = EuropePMCClient._build_query("rTMS", SearchFilters(open_access_only=True))
        assert "OPEN_ACCESS:Y" in expr

    def test_jats_to_text(self):
        xml = (
            "<article><body>"
            "<sec><title>Introduction</title><p>First paragraph.</p></sec>"
            "<sec><title>Methods</title><p>Second paragraph.</p></sec>"
            "</body></article>"
        )
        text = _jats_to_text(xml)
        assert "Introduction" in text and "First paragraph." in text
        assert "<p>" not in text

    def test_jats_to_text_handles_bad_xml(self):
        assert _jats_to_text("<article>") == ""


# ------------------------------------------------------------------- OpenAlex
class TestOpenAlex:
    def test_abstract_reconstruction_orders_words(self):
        inverted = {"depression": [1], "Post-stroke": [0], "is": [2], "common": [3]}
        assert reconstruct_abstract(inverted) == "Post-stroke depression is common"

    def test_abstract_tolerates_missing(self):
        assert reconstruct_abstract(None) == ""
        assert reconstruct_abstract({}) == ""

    def test_abstract_ignores_malformed_positions(self):
        assert reconstruct_abstract({"a": "not-a-list"}) == ""

    def test_work_parsing(self):
        client = OpenAlexClient()
        work = {
            "id": "https://openalex.org/W123",
            "doi": "https://doi.org/10.1016/J.BRS.2023.001",
            "title": "Accelerated rTMS for post-stroke depression",
            "publication_year": 2023,
            "type": "article",
            "language": "en",
            "authorships": [
                {"author": {"display_name": "Wei Zhang"}},
                {"author": {"display_name": "Ming Li"}},
            ],
            "primary_location": {"source": {"display_name": "Brain Stimulation"}},
            "best_oa_location": {"pdf_url": "https://example.org/x.pdf"},
            "open_access": {"is_oa": True},
            "cited_by_count": 42,
            "abstract_inverted_index": {"Accelerated": [0], "rTMS": [1], "works": [2]},
            "mesh": [{"descriptor_name": "Depression"}],
            "keywords": [{"display_name": "transcranial magnetic stimulation"}],
            "concepts": [{"display_name": "Medicine", "score": 0.9}],
            "ids": {"pmid": "https://pubmed.ncbi.nlm.nih.gov/37123456",
                    "pmcid": "https://www.ncbi.nlm.nih.gov/pmc/articles/PMC10239808"},
            "biblio": {"volume": "16", "issue": "2", "first_page": "100", "last_page": "108"},
        }
        paper = client._parse_work(work)
        assert paper is not None
        assert paper.doi == "10.1016/j.brs.2023.001"
        assert paper.pmid == "37123456"
        assert paper.pmcid == "PMC10239808"
        assert paper.journal == "Brain Stimulation"
        assert paper.abstract == "Accelerated rTMS works"
        assert paper.authors == ["Zhang Wei", "Li Ming"], "OpenAlex 的「名 姓」应被翻转"
        assert paper.pages == "100-108"
        assert paper.cited_by_count == 42

    def test_work_parsing_tolerates_garbage(self):
        client = OpenAlexClient()
        assert client._parse_work({}) is not None  # 空记录不崩溃

    def test_filter_builder(self):
        expr = OpenAlexClient._build_filter(
            SearchFilters(year_from=2015, year_to=2023, open_access_only=True, language="zh")
        )
        assert "from_publication_date:2015-01-01" in expr
        assert "to_publication_date:2023-12-31" in expr
        assert "is_oa:true" in expr
        assert "language:zh" in expr


# ------------------------------------------------------------------ Crossref
class TestCrossref:
    def test_item_parsing(self):
        client = CrossrefClient()
        item = {
            "DOI": "10.12677/IJPN.2018.71001",
            "title": ["The Effects of ZTTLT on PSD Animals"],
            "author": [{"family": "林", "given": "菲菲"}, {"family": "Zhang", "given": "Wei"}],
            "container-title": ["International Journal of Psychiatry and Neurology"],
            "issued": {"date-parts": [[2018, 1, 1]]},
            "abstract": "<jats:p>Some <jats:italic>abstract</jats:italic> text.</jats:p>",
            "is-referenced-by-count": 3,
            "volume": "07",
            "issue": "01",
            "page": "1-11",
            "type": "journal-article",
            "subject": ["Psychiatry"],
            "reference": [{"DOI": "10.1/a"}, {"DOI": "10.1/b"}],
            "URL": "https://doi.org/10.12677/ijpn.2018.71001",
        }
        paper = client._parse_item(item)
        assert paper is not None
        assert paper.doi == "10.12677/ijpn.2018.71001"
        assert paper.pub_year == 2018
        assert paper.journal == "International Journal of Psychiatry and Neurology"
        assert paper.pages == "1-11"
        assert paper.cited_by_count == 3
        assert "<jats:" not in paper.abstract
        assert paper.authors[0] == "林 菲菲"

    def test_published_online_fallback_year(self):
        client = CrossrefClient()
        item = {
            "DOI": "10.1/x",
            "title": ["T"],
            "issued": {"date-parts": [[None]]},
            "published-online": {"date-parts": [[2021, 5]]},
        }
        assert client._parse_item(item).pub_year == 2021

    def test_filter_builder(self):
        expr = CrossrefClient._build_filter(SearchFilters(year_from=2015, year_to=2023))
        assert "from-pub-date:2015-01-01" in expr
        assert "until-pub-date:2023-12-31" in expr


# --------------------------------------------------------- Semantic Scholar
class TestSemanticScholar:
    def test_paper_parsing(self):
        client = SemanticScholarClient()
        item = {
            "paperId": "abc123",
            "title": "Accelerated rTMS for post-stroke depression",
            "abstract": "Some abstract.",
            "year": 2023,
            "venue": "Brain Stimulation",
            "journal": {"name": "Brain Stimulation", "volume": "16", "pages": "100-108"},
            "externalIds": {"DOI": "10.1016/j.brs.2023.001", "PubMed": "37123456"},
            "authors": [{"name": "Wei Zhang"}, {"name": "Ming Li"}],
            "citationCount": 42,
            "openAccessPdf": {"url": "https://example.org/x.pdf"},
            "publicationTypes": ["JournalArticle"],
            "fieldsOfStudy": ["Medicine"],
        }
        paper = client._parse_paper(item)
        assert paper is not None
        assert paper.source == "s2"
        assert paper.doi == "10.1016/j.brs.2023.001"
        assert paper.pmid == "37123456"
        assert paper.authors == ["Zhang Wei", "Li Ming"]
        assert paper.is_open_access is True
        assert paper.pages == "100-108"

    def test_tldr_used_when_abstract_missing(self):
        client = SemanticScholarClient()
        paper = client._parse_paper(
            {"paperId": "x", "title": "T", "tldr": {"text": "Short summary."}}
        )
        assert "[TLDR] Short summary." in paper.abstract

    def test_year_range_param(self):
        assert SemanticScholarClient._year_param(SearchFilters(year_from=2015, year_to=2023)) == "2015-2023"
        assert SemanticScholarClient._year_param(SearchFilters(year_from=2015)) == "2015-"
        assert SemanticScholarClient._year_param(SearchFilters(year_to=2023)) == "-2023"
        assert SemanticScholarClient._year_param(None) is None

    def test_reference_list_parsing(self):
        data = {
            "data": [
                {"citedPaper": {"paperId": "r1", "title": "Ref One", "year": 2019,
                                "externalIds": {"DOI": "10.1/r1"}, "venue": "J",
                                "authors": [{"name": "Wei Zhang"}], "citationCount": 5}}
            ]
        }
        refs = SemanticScholarClient._parse_ref_list(data, key="citedPaper")
        assert refs[0]["doi"] == "10.1/r1"
        assert refs[0]["title"] == "Ref One"


# -------------------------------------------------------------------- arXiv
ARXIV_XML = """<?xml version="1.0" encoding="UTF-8"?>
<feed xmlns="http://www.w3.org/2005/Atom" xmlns:arxiv="http://arxiv.org/schemas/atom">
  <entry>
    <id>http://arxiv.org/abs/2301.12345v2</id>
    <published>2023-01-29T18:00:00Z</published>
    <title>Personalized rTMS for Depression: A Review</title>
    <summary>
      We review personalized transcranial magnetic stimulation protocols.
    </summary>
    <author><name>Wei Zhang</name></author>
    <author><name>Ming Li</name></author>
    <arxiv:doi>10.1016/j.brs.2023.001</arxiv:doi>
    <arxiv:journal_ref>Brain Stimulation 16 (2023)</arxiv:journal_ref>
    <category term="q-bio.NC"/>
    <category term="cs.LG"/>
  </entry>
</feed>
"""


class TestArxivParser:
    @pytest.fixture()
    def paper(self):
        client = ArxivClient()
        return client._parse_feed(ARXIV_XML)[0]

    def test_core_fields(self, paper):
        assert paper.source == "arxiv"
        assert paper.pub_year == 2023
        assert paper.doi == "10.1016/j.brs.2023.001"
        assert paper.journal == "Brain Stimulation 16 (2023)"
        assert paper.is_open_access is True

    def test_version_stripped_from_id_and_url(self, paper):
        assert paper.source_id == "2301.12345"
        assert paper.url == "https://arxiv.org/abs/2301.12345"
        assert paper.full_text_url == "https://arxiv.org/pdf/2301.12345"

    def test_whitespace_collapsed(self, paper):
        assert "\n" not in paper.title
        assert "  " not in paper.title

    def test_authors_family_first(self, paper):
        assert paper.authors == ["Zhang Wei", "Li Ming"]

    def test_categories_as_keywords(self, paper):
        assert set(paper.keywords) == {"q-bio.NC", "cs.LG"}

    def test_malformed_feed(self):
        with pytest.raises(SourceError):
            ArxivClient()._parse_feed("<feed")

    def test_candidate_queries_progressive(self):
        candidates = ArxivClient._candidate_queries("accelerated rTMS depression")
        assert candidates[0].startswith('all:"')
        assert len(candidates) >= 2
        assert all("AND" in c or c.startswith('all:"') for c in candidates)

    def test_relevance_filter(self):
        from medscholar.models import Paper as P

        terms = ["rtms", "depression"]
        assert ArxivClient._is_relevant(P(title="rTMS for depression", source="arxiv"), terms)
        assert not ArxivClient._is_relevant(P(title="Cattle genomics", source="arxiv"), terms)

    def test_stopwords_excluded(self):
        terms = ArxivClient._significant_terms("a study of the effects of rTMS")
        assert "the" not in terms and "study" not in terms
        assert "rtms" in terms


# --------------------------------------------------------------------- CNKI
CNKI_JS_SHELL = """<!DOCTYPE html>
<html xmlns="http://www.w3.org/1999/xhtml"><head><title>中国知网</title></head>
<body>
<div class="logo"><a href="//www.cnki.com.cn/" target="_blank"><img src="/images/home/logo.png" /></a></div>
<div class="searchBar">
  <a class="search" href="javascript:void(0)" id="search" onclick="Search()">搜索</a>
  <a href="javascript:void(0)" class="positionA gj kuaijie">快捷搜索</a>
</div>
<p class="p2">京ICP证040431号</p>
</body></html>
"""

CNKI_LEGACY_WITH_RESULTS = """<html><body>
<div class="list-item">
  <p class="tit clearfix"><a href="https://kns.cnki.net/kcms/detail/detail.aspx?dbcode=CJFD&filename=ZGKF202204012" target="_blank">加速rTMS治疗卒中后抑郁的临床疗效观察</a></p>
  <p class="source">中国康复医学杂志-2022年04期</p>
  <p class="info">作者：王伟;李静;</p>
  <p class="summary">目的：探讨加速重复经颅磁刺激治疗卒中后抑郁的临床疗效。</p>
</div>
</body></html>
"""


class TestCnkiClient:
    def test_js_shell_yields_no_results(self):
        """当前线上真实返回的就是这种 JS 空壳 —— 解析器必须返回空而不是抓到导航链接。"""
        client = CnkiClient()
        assert client._parse_results(CNKI_JS_SHELL, limit=10) == []

    def test_legacy_markup_still_parsed(self):
        """若页面结构恢复（或用户指向自建渲染服务），解析器应当仍然work。"""
        client = CnkiClient()
        papers = client._parse_results(CNKI_LEGACY_WITH_RESULTS, limit=10)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.title == "加速rTMS治疗卒中后抑郁的临床疗效观察"
        assert paper.pub_year == 2022
        assert "中国康复医学杂志" in paper.journal
        assert "王伟" in paper.authors
        assert "重复经颅磁刺激" in paper.abstract

    def test_navigation_links_filtered_out(self):
        client = CnkiClient()
        html = '<div class="list-item"><p class="tit"><a href="javascript:void(0)">快捷搜索</a></p></div>'
        assert client._parse_results(html, limit=5) == []

    async def test_non_chinese_query_skipped(self):
        """纯英文查询直接跳过 CNKI，不发请求。"""
        client = CnkiClient()
        assert await client.search("accelerated rTMS") == []

    def test_fulltext_never_returned(self):
        """合规要求：CNKI 客户端不得返回全文。"""
        import asyncio

        from medscholar.models import Paper

        client = CnkiClient()
        assert asyncio.run(client.fulltext(Paper(title="T", source="cnki"))) == ""
