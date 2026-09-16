"""检索评测：指标正确性、数据集校验、消融框架。

**错误的指标比没有指标更糟** —— 它会把检索改动调向错误方向，而且看起来一切正常。
所以这里每个指标都用手算结果钉死，包括边界情形。
"""

from __future__ import annotations

import json
import math

import pytest

from medscholar.eval.dataset import (
    DATASET_DIR,
    EvalCase,
    EvalDataset,
    load_dataset,
    save_dataset,
)
from medscholar.eval.metrics import (
    aggregate,
    average_precision,
    dcg_at_k,
    evaluate_query,
    format_table,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
    retrieved_ids,
)


# ====================================================== 指标：手算钉死
class TestRecall:
    def test_all_relevant_found(self):
        assert recall_at_k(["1", "2", "3"], {"2"}, 3) == 1.0

    def test_half_found(self):
        assert recall_at_k(["1", "2", "3"], {"2", "9"}, 3) == 0.5

    def test_k_cut_off(self):
        # 相关文献排在第 4 位，k=3 时取不到
        assert recall_at_k(["1", "2", "3", "4"], {"4"}, 3) == 0.0

    def test_no_relevant_is_none_not_zero(self):
        """没有标注相关文献 → None（不参与聚合），而不是 0 分。"""
        assert recall_at_k(["1"], set(), 5) is None

    def test_k_zero(self):
        assert recall_at_k(["1"], {"1"}, 0) == 0.0


class TestPrecision:
    def test_denominator_is_k(self):
        # 命中 1 条，k=2 → 0.5（标准定义，分母是 k 不是结果数）
        assert precision_at_k(["1", "2", "3"], {"2"}, 2) == 0.5

    def test_more_results_than_k(self):
        assert precision_at_k(["1", "2", "3"], {"1"}, 1) == 1.0

    def test_none_when_no_relevant(self):
        assert precision_at_k(["1"], set(), 5) is None


class TestReciprocalRank:
    def test_second_position(self):
        assert reciprocal_rank(["a", "b", "c"], {"b"}) == pytest.approx(0.5)

    def test_first_position(self):
        assert reciprocal_rank(["a", "b"], {"a"}) == 1.0

    def test_miss_is_zero(self):
        assert reciprocal_rank(["a"], {"z"}) == 0.0

    def test_none_when_no_relevant(self):
        assert reciprocal_rank(["a"], set()) is None


class TestAveragePrecision:
    def test_hand_computed(self):
        # 相关在位置 1 和 3：AP = (1/1 + 2/3) / 2
        expected = (1.0 + 2 / 3) / 2
        assert average_precision(["1", "2", "3", "4"], {"1", "3"}) == pytest.approx(expected)

    def test_missing_relevant_lowers_ap(self):
        """漏掉的相关文献必须拉低 AP —— 分母是相关总数，不是命中数。"""
        perfect = average_precision(["1", "2"], {"1", "2"})
        partial = average_precision(["1"], {"1", "2"})
        assert perfect == pytest.approx(1.0)
        assert partial < perfect

    def test_all_relevant_first(self):
        assert average_precision(["1", "2", "3"], {"1", "2"}) == pytest.approx(1.0)

    def test_no_hits_is_zero(self):
        assert average_precision(["9"], {"1"}) == 0.0


class TestDcg:
    def test_hand_computed(self):
        # 3/log2(2) + 2/log2(3) + 1/log2(4) = 3 + 1.26186 + 0.5
        expected = 3 / math.log2(2) + 2 / math.log2(3) + 1 / math.log2(4)
        assert dcg_at_k([3, 2, 1], 3) == pytest.approx(expected)

    def test_position_matters(self):
        """把更相关的结果排在前面，DCG 必须更高。"""
        assert dcg_at_k([3, 1], 2) > dcg_at_k([1, 3], 2)

    def test_empty(self):
        assert dcg_at_k([], 5) == 0.0

    def test_k_truncates(self):
        assert dcg_at_k([3, 2, 1], 1) == pytest.approx(3.0)


class TestNdcg:
    def test_perfect_when_relevant_first(self):
        assert ndcg_at_k(["1", "2"], {"1"}, 2) == pytest.approx(1.0)

    def test_hand_computed_binary(self):
        # 相关排在第 2 位：DCG = 1/log2(3)，IDCG = 1/log2(2) = 1
        expected = (1 / math.log2(3)) / 1.0
        assert ndcg_at_k(["9", "1"], {"1"}, 2) == pytest.approx(expected)

    def test_graded_relevance(self):
        # 分级：3 分排第一 → 完美
        assert ndcg_at_k(["1"], {"1": 3}, 3) == pytest.approx(1.0)

    def test_graded_position_penalty(self):
        good = ndcg_at_k(["1", "2"], {"1": 3, "2": 1}, 2)
        bad = ndcg_at_k(["2", "1"], {"1": 3, "2": 1}, 2)
        assert good > bad

    def test_none_when_no_relevant(self):
        assert ndcg_at_k(["1"], set(), 5) is None

    def test_bounded(self):
        for retrieved in (["1", "2"], ["2", "1"], ["9", "9"]):
            value = ndcg_at_k(retrieved, {"1"}, 2)
            assert value is None or 0.0 <= value <= 1.0


class TestRetrievedIds:
    def test_accepts_ints_and_strings(self):
        assert retrieved_ids([1, "2", 3]) == ["1", "2", "3"]

    def test_reads_scored_paper_objects(self):
        from medscholar.models import Paper, ScoredPaper

        items = [ScoredPaper(paper=Paper(title="t", source="x", paper_id=7), score=1.0)]
        assert retrieved_ids(items) == ["7"]

    def test_dedupes_keeping_order(self):
        """重复项不能把指标算高。"""
        assert retrieved_ids(["1", "2", "1"]) == ["1", "2"]

    def test_raises_on_unknown_shape(self):
        with pytest.raises(TypeError):
            retrieved_ids([object()])


# ==================================================== 单查询评测与聚合
class TestEvaluateQuery:
    def test_full_metrics(self):
        metrics = evaluate_query("q", ["1", "2", "3"], {"2", "3"}, k=3)
        assert metrics.recall == 1.0
        assert metrics.precision == pytest.approx(2 / 3)
        assert metrics.mrr == pytest.approx(0.5)
        assert metrics.hit_ids == ["2", "3"]
        assert metrics.missed_ids == []

    def test_records_missed_and_noise_for_diagnosis(self):
        """逐查询明细是定位"为什么没检到"的关键，不能只留均值。"""
        metrics = evaluate_query("q", ["9", "8"], {"1"}, k=2)
        assert metrics.recall == 0.0
        assert metrics.missed_ids == ["1"]
        assert metrics.noise_ids == ["9", "8"]

    def test_truncates_to_k(self):
        metrics = evaluate_query("q", ["1", "2", "3", "4"], {"4"}, k=2)
        assert metrics.recall == 0.0  # 第 4 位被截掉


class TestAggregate:
    def test_skips_unannotated_queries_and_reports_count(self):
        good = evaluate_query("a", ["1"], {"1"}, k=5)
        unannotated = evaluate_query("b", ["2"], set(), k=5)
        summary = aggregate([good, unannotated])
        assert summary["queries"] == 2
        assert summary["scored_queries"] == 1
        assert summary["skipped_queries"] == 1
        # 被跳过的查询不能把均值拉低
        assert summary["recall"] == 1.0

    def test_all_skipped_gives_none(self):
        summary = aggregate([evaluate_query("b", ["2"], set(), k=5)])
        assert summary["recall"] is None
        assert summary["scored_queries"] == 0

    def test_averages(self):
        a = evaluate_query("a", ["1"], {"1"}, k=2)   # recall 1.0
        b = evaluate_query("b", ["9"], {"1"}, k=2)   # recall 0.0
        assert aggregate([a, b])["recall"] == pytest.approx(0.5)


class TestFormatTable:
    def test_renders_columns_and_dashes(self):
        text = format_table(
            [{"config": "a", "recall": 0.5, "ndcg": None}],
            [("config", "配置"), ("recall", "recall"), ("ndcg", "nDCG")],
        )
        assert "配置" in text and "0.5000" in text and "—" in text

    def test_empty_rows_only_header(self):
        text = format_table([], [("config", "配置")])
        assert "配置" in text


# ======================================================== 数据集校验
def make_dataset(**overrides) -> EvalDataset:
    base = dict(
        name="t",
        provenance="测试用",
        corpus=[{"title": "Paper one"}, {"title": "Paper two"}],
        cases=[EvalCase(query="q", relevant={"1": 1.0}, source="manual")],
    )
    base.update(overrides)
    return EvalDataset(**base)


class TestDatasetValidation:
    def test_valid_dataset_passes(self):
        assert make_dataset().validate() == []

    def test_unannotated_case_is_rejected(self):
        """没有标注相关文献的查询必须被拦下 —— 否则会被跳过而虚高指标。"""
        problems = make_dataset(cases=[EvalCase(query="q")]).validate()
        assert any("没有标注相关文献" in p for p in problems)

    def test_explicit_negative_control_is_allowed(self):
        """显式声明为阴性对照的查询是合法的（校验器要能区分"忘标注"与"故意留空"）。"""
        dataset = make_dataset(
            cases=[EvalCase(query="无关查询", expect_no_relevant=True, source="manual")]
        )
        assert dataset.validate() == []
        assert dataset.cases[0].to_dict()["expect_no_relevant"] is True

    def test_negative_control_flag_round_trips(self):
        case = EvalCase.from_dict(
            {"query": "q", "relevant": {}, "expect_no_relevant": True}
        )
        assert case.expect_no_relevant is True
        assert case.to_dict()["expect_no_relevant"] is True

    def test_unknown_corpus_id_is_rejected(self):
        problems = make_dataset(
            cases=[EvalCase(query="q", relevant={"99": 1.0})]
        ).validate()
        assert any("不存在的语料 id" in p for p in problems)

    def test_empty_corpus_rejected(self):
        assert any("语料为空" in p for p in make_dataset(corpus=[]).validate())

    def test_missing_provenance_rejected(self):
        assert any("provenance" in p for p in make_dataset(provenance="").validate())

    def test_unknown_source_rejected(self):
        problems = make_dataset(
            cases=[EvalCase(query="q", relevant={"1": 1.0}, source="随便写的")]
        ).validate()
        assert any("source" in p for p in problems)

    def test_corpus_without_title_rejected(self):
        assert any(
            "没有 title" in p for p in make_dataset(corpus=[{"abstract": "x"}]).validate()
        )


class TestDatasetIO:
    def test_json_round_trip(self, tmp_path):
        dataset = make_dataset()
        path = tmp_path / "d.json"
        path.write_text(json.dumps(dataset.to_dict(), ensure_ascii=False), encoding="utf-8")
        loaded = load_dataset(path)
        assert loaded.corpus == dataset.corpus
        assert loaded.cases[0].relevant == {"1": 1.0}

    def test_jsonl_round_trip_with_corpus_file(self, tmp_path):
        dataset = make_dataset()
        golden, corpus = save_dataset(
            dataset,
            golden_path=tmp_path / "d.golden.jsonl",
            corpus_path=tmp_path / "d.corpus.jsonl",
        )
        assert golden.is_file() and corpus.is_file()
        loaded = load_dataset(golden)
        assert len(loaded.cases) == 1
        assert len(loaded.corpus) == 2

    def test_jsonl_skips_comments_and_blanks(self, tmp_path):
        path = tmp_path / "d.golden.jsonl"
        path.write_text(
            "# 这是注释\n\n" + json.dumps({"query": "q", "relevant": {"1": 1}}) + "\n",
            encoding="utf-8",
        )
        (tmp_path / "d.corpus.jsonl").write_text(
            json.dumps({"title": "t"}) + "\n", encoding="utf-8"
        )
        assert len(load_dataset(path).cases) == 1

    def test_malformed_line_reports_line_number(self, tmp_path):
        path = tmp_path / "bad.golden.jsonl"
        path.write_text('{"query": "ok", "relevant": {"1": 1}}\n{broken\n', encoding="utf-8")
        with pytest.raises(ValueError, match="第 2 行"):
            load_dataset(path)

    def test_missing_file_raises(self, tmp_path):
        with pytest.raises(FileNotFoundError):
            load_dataset(tmp_path / "nope.json")

    def test_relevant_accepts_list_form(self):
        case = EvalCase.from_dict({"query": "q", "relevant": ["1", "2"]})
        assert case.relevant == {"1": 1.0, "2": 1.0}

    def test_case_without_query_rejected(self):
        with pytest.raises(ValueError, match="缺少 query"):
            EvalCase.from_dict({"relevant": {"1": 1}})

    def test_bundled_regression_dataset_is_valid(self):
        """仓库里带的回归数据集必须自身合法（否则 CI 会以看不懂的方式失败）。"""
        dataset = load_dataset(DATASET_DIR / "regression.json")
        assert dataset.validate() == []
        assert len(dataset.cases) >= 20
        assert len(dataset.corpus) >= 30


class TestPapersConversion:
    def test_corpus_becomes_papers_with_sequential_ids(self):
        dataset = make_dataset()
        papers = dataset.papers()
        assert [p.source_id for p in papers] == ["1", "2"]
        assert papers[0].title == "Paper one"
        assert papers[0].source == "eval"
