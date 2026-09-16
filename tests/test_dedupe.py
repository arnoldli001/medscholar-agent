"""跨库去重合并测试。

同一篇文献常同时出现在 PubMed / Europe PMC / OpenAlex / Semantic Scholar，
去重质量直接决定知识库里有多少重复条目。
"""

from __future__ import annotations

from medscholar.dedupe import group_by_key, merge_papers, pick_richest
from medscholar.models import Paper


class TestDedupKey:
    def test_doi_priority(self):
        paper = Paper(title="T", source="pubmed", doi="10.1/x", pmid="123")
        assert paper.dedup_key == "doi:10.1/x"

    def test_pmid_fallback(self):
        assert Paper(title="T", source="pubmed", pmid="123").dedup_key == "pmid:123"

    def test_title_fingerprint_fallback(self):
        key = Paper(title="Some Title", source="openalex").dedup_key
        assert key.startswith("title:")

    def test_fingerprint_survives_case_and_punctuation(self):
        a = Paper(title="Accelerated rTMS: A Trial", source="openalex").dedup_key
        b = Paper(title="accelerated rtms a trial", source="crossref").dedup_key
        assert a == b


class TestMerge:
    def test_single_paper_annotated_with_source(self):
        merged = merge_papers([Paper(title="T", source="pubmed", doi="10.1/x")])
        assert len(merged) == 1
        assert merged[0].found_in == ["pubmed"]

    def test_cross_source_dedupe(self):
        papers = [
            Paper(title="Accelerated rTMS for post-stroke depression", source="pubmed",
                  doi="10.1/x", abstract="short", authors=["Zhang Wei"]),
            Paper(title="Accelerated rTMS for post-stroke depression", source="openalex",
                  doi="10.1/x", abstract="a much longer abstract " * 10),
            Paper(title="Accelerated rTMS for post-stroke depression", source="europepmc",
                  doi="10.1/x", pub_year=2023, pmid="37123456", cited_by_count=42),
        ]
        merged = merge_papers(papers)
        assert len(merged) == 1
        assert set(merged[0].found_in) == {"pubmed", "openalex", "europepmc"}

    def test_merge_takes_longest_abstract(self):
        papers = [
            Paper(title="T", source="pubmed", doi="10.1/x", abstract="short"),
            Paper(title="T", source="openalex", doi="10.1/x", abstract="x" * 500),
        ]
        assert len(merge_papers(papers)[0].abstract) == 500

    def test_merge_unions_authors_and_terms(self):
        papers = [
            Paper(title="T", source="pubmed", doi="10.1/x", authors=["Zhang Wei"],
                  mesh_terms=["Stroke"]),
            Paper(title="T", source="openalex", doi="10.1/x", authors=["Zhang Wei", "Li Ming"],
                  keywords=["rTMS"]),
        ]
        merged = merge_papers(papers)[0]
        assert merged.authors == ["Zhang Wei", "Li Ming"]
        assert merged.mesh_terms == ["Stroke"]
        assert merged.keywords == ["rTMS"]

    def test_merge_takes_max_cited(self):
        papers = [
            Paper(title="T", source="pubmed", doi="10.1/x", cited_by_count=5),
            Paper(title="T", source="openalex", doi="10.1/x", cited_by_count=99),
        ]
        assert merge_papers(papers)[0].cited_by_count == 99

    def test_merge_ors_open_access(self):
        papers = [
            Paper(title="T", source="pubmed", doi="10.1/x", is_open_access=False),
            Paper(title="T", source="openalex", doi="10.1/x", is_open_access=True),
        ]
        assert merge_papers(papers)[0].is_open_access is True

    def test_different_identifier_types_not_merged_here(self):
        """``merge_papers`` 只按 dedup_key 归并。

        一条只有 PMID、另一条只有 DOI 时，在这一层不会合并（dedup_key 不同）。
        真正的跨标识符合并发生在**入库时**：``repo._find_existing_id`` 按
        DOI → PMID → 标题指纹 逐级查找，见 test_db 中的对应用例。
        """
        papers = [
            Paper(title="T", source="pubmed", pmid="123"),
            Paper(title="T", source="crossref", doi="10.1/x"),
        ]
        assert len(merge_papers(papers)) == 2

    def test_same_doi_across_sources_merged(self):
        papers = [
            Paper(title="T", source="pubmed", pmid="123", doi="10.1/x"),
            Paper(title="T", source="crossref", doi="10.1/x"),
        ]
        merged = merge_papers(papers)
        assert len(merged) == 1
        assert merged[0].pmid == "123"

    def test_distinct_papers_kept(self):
        papers = [
            Paper(title="Paper A", source="pubmed", doi="10.1/a"),
            Paper(title="Paper B", source="pubmed", doi="10.1/b"),
        ]
        assert len(merge_papers(papers)) == 2

    def test_order_is_stable(self):
        papers = [
            Paper(title="First", source="pubmed", doi="10.1/a"),
            Paper(title="Second", source="pubmed", doi="10.1/b"),
            Paper(title="First", source="openalex", doi="10.1/a"),
        ]
        merged = merge_papers(papers)
        assert [p.title for p in merged] == ["First", "Second"]

    def test_empty(self):
        assert merge_papers([]) == []


class TestHelpers:
    def test_group_by_key(self):
        groups = group_by_key([
            Paper(title="T", source="pubmed", doi="10.1/x"),
            Paper(title="T", source="openalex", doi="10.1/x"),
            Paper(title="U", source="pubmed", doi="10.1/y"),
        ])
        assert len(groups) == 2
        assert len(groups["doi:10.1/x"]) == 2

    def test_pick_richest(self):
        sparse = Paper(title="T", source="pubmed", doi="10.1/x")
        rich = Paper(title="T", source="pubmed", doi="10.1/x", abstract="x" * 200,
                     authors=["A", "B"])
        assert pick_richest([sparse, rich]) is rich
