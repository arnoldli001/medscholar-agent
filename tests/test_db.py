"""数据层测试：建表、入库、去重、FTS5、向量检索、RRF 融合。

全部离线运行，嵌入使用确定性哈希提供方。
"""

from __future__ import annotations

import math

import pytest

from medscholar.db import repo
from medscholar.models import Paper


def fake_vector(seed: int, dim: int = 768) -> list[float]:
    return [math.sin(seed * 0.37 + i * 0.017) for i in range(dim)]


class TestSchema:
    def test_tables_created(self, db):
        names = {
            row["name"]
            for row in db.query("SELECT name FROM sqlite_master WHERE type IN ('table','view')")
        }
        for expected in (
            "papers", "papers_fts", "paper_embeddings", "paper_fulltext", "fulltext_fts",
            "citations", "search_logs", "projects", "project_papers",
            "chat_sessions", "chat_messages", "artifacts", "subscriptions", "meta",
        ):
            assert expected in names, f"缺少表 {expected}"

    def test_fts5_queryable(self, db):
        # 建表即证明 FTS5 可用（SQLite 缺 FTS5 时 CREATE VIRTUAL TABLE 会直接失败）
        assert db.query("SELECT count(*) AS n FROM papers_fts")[0]["n"] == 0

    def test_vector_backend_recorded(self, db):
        backend = db.get_meta("vector_backend")
        assert backend in {"sqlite-vec", "python"}
        assert db.get_meta("embedding_dim") == str(db.config.embedding.dim)
        assert isinstance(db.vec_note, str)


class TestInsert:
    def test_insert_new(self, db, sample_papers):
        paper_id, created = repo.insert_paper(sample_papers[0], db=db)
        assert created is True
        assert paper_id > 0
        stored = repo.get_paper(paper_id, db=db)
        assert stored is not None
        assert stored.title == sample_papers[0].title
        assert stored.mesh_terms == sample_papers[0].mesh_terms

    def test_insert_is_idempotent_via_doi(self, db, sample_papers):
        first_id, created_first = repo.insert_paper(sample_papers[0], db=db)
        second_id, created_second = repo.insert_paper(sample_papers[0], db=db)
        assert created_first is True
        assert created_second is False
        assert first_id == second_id
        assert repo.count_papers(db=db) == 1

    def test_insert_merges_richer_record(self, db, sample_papers):
        """同一篇文献从第二个数据源再取到时，应合并补全而不是插重复。"""
        sparse = Paper(
            title=sample_papers[0].title,
            source="openalex",
            pub_year=2023,
        )
        repo.insert_paper(sparse, db=db)
        rich = Paper(
            title=sample_papers[0].title,
            source="pubmed",
            abstract="A much longer abstract with real content " * 5,
            authors=["Zhang Wei"],
            doi="10.1016/j.brs.2023.001",
            cited_by_count=99,
            is_open_access=True,
        )
        paper_id, created = repo.insert_paper(rich, db=db)
        assert created is False
        stored = repo.get_paper(paper_id, db=db)
        assert stored is not None
        assert len(stored.abstract) > 100
        assert stored.doi == "10.1016/j.brs.2023.001"
        assert stored.cited_by_count == 99
        assert stored.is_open_access is True
        assert repo.count_papers(db=db) == 1

    def test_bulk_insert_counts(self, db, sample_papers):
        result = repo.insert_papers(sample_papers, db=db)
        assert result["new"] == 3
        assert result["updated"] == 0
        assert len(result["ids"]) == 3
        again = repo.insert_papers(sample_papers, db=db)
        assert again["new"] == 0 and again["updated"] == 3

    def test_title_only_dedupe(self, db):
        """无 DOI/PMID 时靠标题指纹去重。"""
        a = Paper(title="Accelerated rTMS for post-stroke depression", source="openalex")
        b = Paper(title="accelerated RTMS for post-stroke depression!", source="crossref")
        repo.insert_paper(a, db=db)
        _, created = repo.insert_paper(b, db=db)
        assert created is False
        assert repo.count_papers(db=db) == 1

    def test_papers_without_title_skipped(self, db):
        result = repo.insert_papers([Paper(title="", source="manual")], db=db)
        assert result["new"] == 0

    def test_delete_cleans_indexes(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embedding(ids[0], fake_vector(1), db=db)
        removed = repo.delete_papers([ids[0]], db=db)
        assert removed == 1
        assert repo.get_paper(ids[0], db=db) is None
        # FTS 索引必须同步清理：已删除的文献不能再出现在检索结果里
        hits = repo.search_fts(sample_papers[0].title, db=db)
        assert all(pid != ids[0] for pid, _ in hits)
        # 向量也必须清掉
        assert db.query_one(
            "SELECT count(*) AS n FROM paper_embeddings WHERE paper_id = ?", (ids[0],)
        )["n"] == 0

    def test_cross_identifier_merge_via_title(self, db):
        """一个只有 PMID、一个只有 DOI 的同一篇文献，入库时应靠标题指纹合并。"""
        repo.insert_paper(Paper(title="Same Title Here", source="pubmed", pmid="123"), db=db)
        _, created = repo.insert_paper(
            Paper(title="Same title here!", source="crossref", doi="10.1/x"), db=db
        )
        assert created is False, "标题指纹应识别为同一篇"
        assert repo.count_papers(db=db) == 1
        stored = repo.list_papers(db=db)[0]
        assert stored.pmid == "123" and stored.doi == "10.1/x"


class TestFtsSearch:
    def test_english_hit(self, db, sample_papers):
        repo.insert_papers(sample_papers, db=db)
        hits = repo.search_fts("accelerated rTMS depression", db=db)
        assert hits
        assert hits[0][1] < 0, "bm25 分数应为负数（越小越相关）"

    def test_chinese_exact(self, db, sample_papers):
        repo.insert_papers(sample_papers, db=db)
        hits = repo.search_fts("卒中后抑郁", db=db)
        assert hits, "中文子串检索必须可用"

    def test_chinese_fragment_inside_longer_phrase(self, db, sample_papers):
        """回归：查询词是原文的一个片段时必须命中。"""
        repo.insert_papers(sample_papers, db=db)
        hits = repo.search_fts("经颅磁刺激", db=db)
        assert hits

    def test_chinese_query_with_extra_modifier(self, db):
        """回归：原文「治疗脑卒中后抑郁」，查询「治疗卒中后抑郁」也应命中（放宽策略）。"""
        repo.insert_paper(
            Paper(title="针刺治疗脑卒中后抑郁机制研究进展", source="openalex"),
            db=db,
        )
        hits = repo.search_fts("治疗卒中后抑郁", db=db)
        assert hits, "逐级放宽策略应兜住这种中间插入修饰语的情况"

    def test_title_weighted_above_abstract(self, db):
        in_title = Paper(
            title="Transcranial magnetic stimulation review",
            abstract="unrelated text about something else entirely",
            source="pubmed",
        )
        in_abstract = Paper(
            title="A completely different topic",
            abstract="transcranial magnetic stimulation appears only here",
            source="pubmed",
        )
        repo.insert_papers([in_title, in_abstract], db=db)
        hits = repo.search_fts("transcranial magnetic stimulation", db=db)
        assert len(hits) == 2
        assert hits[0][0] < hits[1][0], "标题命中的 id 应排在前面（分数更小）"

    def test_no_hit_returns_empty(self, db, sample_papers):
        repo.insert_papers(sample_papers, db=db)
        assert repo.search_fts("quantum chromodynamics lattice", db=db) == []

    def test_filters_applied(self, db, sample_papers):
        repo.insert_papers(sample_papers, db=db)
        hits = repo.search_fts("rTMS depression", db=db, filters={"year_from": 2020})
        ids = [pid for pid, _ in hits]
        for pid in ids:
            assert (repo.get_paper(pid, db=db).pub_year or 0) >= 2020


class TestVectorSearch:
    def test_roundtrip(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embeddings([(pid, fake_vector(pid)) for pid in ids], db=db)
        hits = repo.search_vector(fake_vector(ids[0]), db=db)
        assert hits
        assert hits[0][0] == ids[0], "以自身为查询时，自己应排第一"
        assert hits[0][1] == pytest.approx(0.0, abs=1e-3)

    def test_ordering_is_by_distance(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embeddings([(pid, fake_vector(pid * 7)) for pid in ids], db=db)
        hits = repo.search_vector(fake_vector(ids[0] * 7), db=db)
        distances = [d for _, d in hits]
        assert distances == sorted(distances)

    def test_empty_when_no_embeddings(self, db, sample_papers):
        repo.insert_papers(sample_papers, db=db)
        assert repo.search_vector(fake_vector(1), db=db) == []

    def test_missing_embeddings_incremental(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        assert sorted(repo.papers_missing_embeddings(db=db)) == sorted(ids)
        repo.store_embedding(ids[0], fake_vector(1), db=db)
        assert repo.papers_missing_embeddings(db=db) == sorted(ids[1:])

    def test_l2_distance_zero_for_identical_normalized(self, db, sample_papers):
        """向量写入时会被 L2 归一化，因此「自身对自身」的距离必须为 0。"""
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embedding(ids[0], fake_vector(7), db=db)
        hits = repo.search_vector(fake_vector(7), limit=1, db=db)
        assert hits[0][0] == ids[0]
        assert hits[0][1] == pytest.approx(0.0, abs=1e-4)

    def test_scaling_invariance_after_normalization(self, db, sample_papers):
        """同一个方向、不同模长的向量必须视为同一篇（归一化生效）。"""
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        base = fake_vector(11)
        repo.store_embedding(ids[0], base, db=db)
        scaled = [v * 37.0 for v in base]
        hits = repo.search_vector(scaled, limit=1, db=db)
        assert hits[0][0] == ids[0]
        assert hits[0][1] == pytest.approx(0.0, abs=1e-4)

    def test_overwrite_replaces_embedding(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embedding(ids[0], fake_vector(1), db=db)
        repo.store_embedding(ids[0], fake_vector(2), db=db)
        assert db.query_one("SELECT count(*) AS n FROM paper_embeddings")["n"] == 1


class TestRrf:
    def test_fusion_math(self):
        fused = repo.rrf_fuse([[(1, -3.0), (2, -2.0)], [(2, 0.1), (3, 0.2)]], k=60)
        assert fused[0][0] == 2, "被两路同时命中的文档应排第一"
        assert fused[0][1] == pytest.approx(1 / 62 + 1 / 61)

    def test_weights(self):
        fused = repo.rrf_fuse([[(1, 0)], [(2, 0)]], k=60, weights=[2.0, 1.0])
        assert fused[0][0] == 1

    def test_empty(self):
        assert repo.rrf_fuse([[], []]) == []


class TestHybrid:
    def test_both_paths_contribute(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embeddings([(pid, fake_vector(pid)) for pid in ids], db=db)
        hits = repo.hybrid_search("rTMS depression", embedding=fake_vector(ids[0]), top_k=5, db=db)
        assert hits
        assert any(h.matched_by == "bm25+vector" for h in hits), (
            "融合应产生同时被两路命中的结果：" + str([h.matched_by for h in hits])
        )

    def test_keyword_only_when_no_embedding(self, db, sample_papers):
        repo.insert_papers(sample_papers, db=db)
        hits = repo.hybrid_search("rTMS", embedding=None, top_k=5, db=db)
        assert hits
        assert all(h.vector_rank is None for h in hits)

    def test_chinese_query_uses_fallback_strategy(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embeddings([(pid, fake_vector(pid)) for pid in ids], db=db)
        hits = repo.hybrid_search("rTMS 治疗卒中后抑郁的疗效", embedding=fake_vector(ids[1]), top_k=5, db=db)
        assert hits
        assert any(h.fts_rank is not None for h in hits), "中文查询必须能走通 BM25 路径"

    def test_empty_library(self, db):
        assert repo.hybrid_search("anything", embedding=fake_vector(1), db=db) == []

    def test_scored_paper_dict_shape(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embedding(ids[0], fake_vector(1), db=db)
        hits = repo.hybrid_search("rTMS", embedding=fake_vector(1), db=db)
        payload = hits[0].to_dict()
        for key in ("score", "matched_by", "fts_rank", "vector_rank", "title", "paper_id"):
            assert key in payload


class TestFulltext:
    def test_save_and_search(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.save_fulltext(
            ids[1],
            "目的：观察加速经颅磁刺激的疗效。方法：随机分组。结果：HAMD 评分降低。",
            origin="europepmc",
            db=db,
        )
        assert repo.get_fulltext(ids[1], db=db)
        assert repo.search_fulltext("随机分组", db=db)

    def test_empty_content_ignored(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.save_fulltext(ids[0], "   ", db=db)
        assert repo.get_fulltext(ids[0], db=db) == ""


class TestCitationsAndLogs:
    def test_citation_roundtrip(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.add_citation(ids[0], cited_paper_id=ids[1], source="s2", db=db)
        repo.add_citation(ids[0], cited_external_id="10.9999/external", cited_title="X", db=db)
        refs = repo.get_references(ids[0], db=db)
        assert len(refs) == 2
        assert repo.get_citing_papers(ids[1], db=db)

    def test_add_citations_resolves_local(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        count = repo.add_citations(
            ids[0],
            [{"doi": "10.1016/j.biopsych.2019.05.011"}, {"title": "Unknown work"}],
            db=db,
        )
        assert count == 2

    def test_search_log(self, db):
        from medscholar.models import SearchLogEntry

        repo.log_search(SearchLogEntry(query="rTMS", source="pubmed", result_count=12), db=db)
        recent = repo.recent_searches(db=db)
        assert recent[0]["query"] == "rTMS"
        assert recent[0]["result_count"] == 12


class TestProjectsAndSessions:
    def test_project_lifecycle(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        project_id = repo.create_project("卒中后抑郁", description="测试", keywords=["rTMS"], db=db)
        assert repo.add_papers_to_project(project_id, ids[:2], db=db) == 2
        assert repo.count_papers(project_id=project_id, db=db) == 2
        assert len(repo.list_papers(project_id=project_id, db=db)) == 2
        assert repo.remove_paper_from_project(project_id, ids[0], db=db) is True
        assert repo.count_papers(project_id=project_id, db=db) == 1
        assert repo.delete_project(project_id, db=db) is True

    def test_project_upsert_by_name(self, db):
        first = repo.create_project("课题A", db=db)
        second = repo.create_project("课题A", description="更新", db=db)
        assert first == second
        assert repo.get_project(first, db=db)["description"] == "更新"

    def test_session_and_messages(self, db):
        session_id = repo.create_session(title="会话", topic="rTMS", db=db)
        repo.add_message(session_id, "user", "你好", db=db)
        repo.add_message(session_id, "assistant", "回复", meta={"k": 1}, db=db)
        messages = repo.list_messages(session_id, db=db)
        assert [m["role"] for m in messages] == ["user", "assistant"]
        assert messages[1]["meta"] == {"k": 1}
        assert repo.delete_session(session_id, db=db) is True
        assert repo.list_messages(session_id, db=db) == []

    def test_artifact_lifecycle(self, db):
        artifact_id = repo.save_artifact(title="草稿", content="正文", session_id=None, db=db)
        assert repo.get_artifact(artifact_id, db=db)["content"] == "正文"
        assert repo.list_artifacts(db=db)
        assert repo.delete_artifact(artifact_id, db=db) is True


class TestStats:
    def test_stats_shape(self, db, sample_papers):
        ids = repo.insert_papers(sample_papers, db=db)["ids"]
        repo.store_embedding(ids[0], fake_vector(1), db=db)
        stats = db.stats()
        assert stats["papers"] == 3
        assert stats["embedded"] == 1
        assert stats["embedding_coverage"] == pytest.approx(1 / 3, abs=0.01)
        assert stats["vector_backend"] in {"sqlite-vec", "python"}
        assert stats["year_min"] == 2019 and stats["year_max"] == 2023
