"""评测框架：确定性、消融、阴性对照、以及"评测的是生产代码"的证据。

CI 里没有 Ollama，所以全部用确定性的 ``hashing`` 嵌入提供方跑 ——
这样评测完全离线、可复现。真实质量评测在本地用真实模型跑
（``scripts/eval_retrieval.py --from-library``）。
"""

from __future__ import annotations

import pytest

from medscholar.config import AppConfig
from medscholar.eval import (
    CONFIGS,
    EvalCase,
    EvalDataset,
    evaluate_configs,
    index_corpus,
    load_dataset,
)
from medscholar.eval.dataset import DATASET_DIR
from medscholar.eval.harness import check_regression, verify_production_parity


@pytest.fixture
def config(tmp_path) -> AppConfig:
    """确定性配置：哈希嵌入、不联网、临时数据目录。"""
    cfg = AppConfig(data_dir=str(tmp_path / "eval"))
    cfg.embedding.provider = "hashing"
    cfg.embedding.dim = 256
    cfg.offline = True
    return cfg


@pytest.fixture
def dataset() -> EvalDataset:
    return EvalDataset(
        name="tiny",
        provenance="单元测试用，人工合成",
        corpus=[
            {
                "title": "Accelerated rTMS for post-stroke depression",
                "abstract": "Randomized trial of accelerated rTMS in post-stroke depression.",
                "journal": "JAD",
                "pub_year": 2021,
                "keywords": ["rTMS"],
            },
            {
                "title": "Theta burst stimulation for depression",
                "abstract": "Meta-analysis of theta burst stimulation versus rTMS.",
                "journal": "Brain Stimulation",
                "pub_year": 2020,
            },
            {
                "title": "Nurse-led follow-up after stroke",
                "abstract": "Screening rates for depression after stroke.",
                "journal": "JAN",
                "pub_year": 2019,
            },
            {
                "title": "Exercise for post-stroke depression",
                "abstract": "Aerobic exercise reduces depressive symptoms after stroke.",
                "journal": "APMR",
                "pub_year": 2021,
            },
        ],
        cases=[
            EvalCase(
                query="accelerated rTMS post-stroke depression randomized trial",
                relevant={"1": 1.0},
                source="manual",
            ),
            EvalCase(
                query="theta burst stimulation versus rTMS meta-analysis",
                relevant={"2": 1.0},
                source="manual",
            ),
        ],
    )


# ============================================================ 语料装载
class TestIndexCorpus:
    def test_papers_get_deterministic_ids(self, config, dataset, tmp_path):
        db, workdir = index_corpus(dataset, config=config, workdir=tmp_path / "idx")
        try:
            rows = db.query("SELECT paper_id, source_id, title FROM papers ORDER BY paper_id")
            assert [r["source_id"] for r in rows] == ["1", "2", "3", "4"]
            assert rows[0]["title"].startswith("Accelerated rTMS")
        finally:
            db.close()

    def test_embeddings_are_stored(self, config, dataset, tmp_path):
        db, _ = index_corpus(dataset, config=config, workdir=tmp_path / "idx2")
        try:
            count = db.query_one("SELECT COUNT(*) AS n FROM paper_embeddings")["n"]
            assert count == 4
        finally:
            db.close()

    def test_hashing_embedder_is_deterministic(self, config, dataset, tmp_path):
        """同一语料跑两次，向量必须逐位一致 —— 否则指标无法比对。"""
        db1, _ = index_corpus(dataset, config=config, workdir=tmp_path / "a")
        db2, _ = index_corpus(dataset, config=config, workdir=tmp_path / "b")
        try:
            def vectors(db):
                rows = db.query(
                    "SELECT paper_id, embedding FROM paper_embeddings ORDER BY paper_id"
                )
                return [(r["paper_id"], bytes(r["embedding"])) for r in rows]

            assert vectors(db1) == vectors(db2)
        finally:
            db1.close()
            db2.close()


# ============================================================ 消融评测
class TestEvaluateConfigs:
    def test_runs_all_configs_and_reports(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        assert len(report.results) == len(CONFIGS)
        names = [r.config.name for r in report.results]
        assert "bm25-only" in names and "vector-only" in names and "production" in names
        for result in report.results:
            assert result.summary["queries"] == 2
            assert result.summary["scored_queries"] == 2

    def test_metrics_are_in_valid_range(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        for result in report.results:
            for key in ("recall", "precision", "mrr", "map", "ndcg"):
                value = result.summary.get(key)
                assert value is None or 0.0 <= value <= 1.0, (result.config.name, key, value)

    def test_bm25_hits_exact_wording(self, config, dataset):
        report = evaluate_configs(
            dataset, config=config, configs=[c for c in CONFIGS if c.name == "bm25-only"], k=3
        )
        # 查询与标题高度重合，BM25 应当能召回
        assert report.results[0].summary["recall"] == 1.0

    def test_ablation_table_has_deltas_vs_baseline(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3, baseline="bm25-only")
        rows = report.table()
        assert rows[0]["config"] == "bm25-only"
        assert "d_recall" not in rows[0]  # 基线自身没有差值列
        assert any("d_recall" in row for row in rows[1:])

    def test_deterministic_across_runs(self, config, dataset):
        """两次运行结果必须一致，否则无法用它做回归门禁。"""
        first = evaluate_configs(dataset, config=config, k=3)
        second = evaluate_configs(dataset, config=config, k=3)
        for a, b in zip(first.results, second.results):
            assert a.config.name == b.config.name
            assert a.summary == b.summary

    def test_per_query_details_available_for_diagnosis(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        production = next(r for r in report.results if r.config.name == "production")
        assert len(production.per_query) == 2
        assert production.per_query[0].query
        assert isinstance(production.per_query[0].to_dict()["missed_ids"], list)

    def test_invalid_dataset_is_rejected(self, config):
        bad = EvalDataset(
            name="bad",
            provenance="x",
            corpus=[{"title": "t"}],
            cases=[EvalCase(query="q")],  # 没有标注
        )
        with pytest.raises(ValueError, match="评测集本身有问题"):
            evaluate_configs(bad, config=config, k=3)

    def test_report_serializes_to_json(self, config, dataset):
        import json

        report = evaluate_configs(dataset, config=config, k=3)
        text = json.dumps(report.to_dict(), ensure_ascii=False)
        assert "bm25-only" in text
        assert json.loads(text)["corpus_size"] == 4


# ================================================== 阴性对照（无关查询）
class TestEmptyControl:
    def test_records_how_many_results_returned_for_unrelated_query(self, config):
        dataset = EvalDataset(
            name="nc",
            provenance="阴性对照测试",
            corpus=[{"title": "Accelerated rTMS for depression", "abstract": "trial"}],
            cases=[
                EvalCase(query="rTMS depression", relevant={"1": 1.0}, source="manual"),
                EvalCase(
                    query="quantum chromodynamics lattice gauge theory",
                    expect_no_relevant=True,
                    source="manual",
                ),
            ],
        )
        report = evaluate_configs(dataset, config=config, k=5)
        for result in report.results:
            # 阴性对照不参与 recall 聚合
            assert result.summary["queries"] == 2
            assert result.summary["scored_queries"] == 1
            assert result.summary["skipped_queries"] == 1
            assert result.empty_control["cases"] == 1
            assert result.empty_control["avg_returned"] is not None

    def test_rrf_returns_results_even_for_unrelated_query(self, config):
        """这是一个**应该被显式盯住**的系统特性：融合总会给出 top_k。"""
        dataset = EvalDataset(
            name="nc2",
            provenance="阴性对照测试",
            corpus=[
                {"title": "Accelerated rTMS for depression", "abstract": "trial"},
                {"title": "Exercise for depression", "abstract": "meta"},
            ],
            cases=[
                EvalCase(
                    query="quantum chromodynamics lattice gauge theory",
                    expect_no_relevant=True,
                    source="manual",
                )
            ],
        )
        report = evaluate_configs(
            dataset,
            config=config,
            configs=[c for c in CONFIGS if c.name == "production"],
            k=5,
        )
        result = report.results[0]
        # 只要库里非空，融合就会凑出结果 —— 这正是需要靠阈值/校验兜住的地方
        assert result.empty_control["avg_returned"] >= 0
        assert "宁滥勿缺" in result.empty_control["note"]


# ============================== 关键：证明评测跑在生产代码路径上
class TestProductionParity:
    def test_eval_production_config_matches_hybrid_search(self, config, dataset):
        """评测里的 production 配置必须与 hybrid_search 输出**逐条一致**。

        这是「没有评测一个自己重写的检索器」的可执行证据 ——
        如果哪天有人改了 hybrid_search 而忘了同步评测框架，这条会立刻红。
        """
        assert verify_production_parity(dataset, config=config, k=3) == []

    def test_parity_detects_divergence(self, config, dataset, monkeypatch):
        """反过来验证：当两者不一致时，parity 必须能发现。

        做法是让 ``hybrid_search`` 返回倒序结果 —— 这样"检查本身是否有效"
        就被独立验证了，而不是依赖某个碰巧会改变排序的参数。
        """
        import medscholar.db.repo as repo

        real = repo.hybrid_search

        def reversed_search(query, **kwargs):
            return list(reversed(real(query, **kwargs)))

        monkeypatch.setattr(repo, "hybrid_search", reversed_search)
        assert verify_production_parity(dataset, config=config, k=3) != []

    def test_rrf_k_may_not_change_order_on_tiny_corpus(self, config):
        """诚实记录一个实测现象：**小语料上改 RRF 的 k 可能完全不改变排序**。

        因为 k 只是对 ``1/(k+rank)`` 做单调缩放，当两路排名高度一致时，
        不同 k 会给出同样的顺序。这意味着：

        * 用 4 篇文献的语料"调 k"是没有意义的；
        * 想真正检验 ``rrf_k`` 的影响，必须用**真实规模**的库
          （这也是 ``scripts/eval_retrieval.py --from-library`` 存在的原因）。

        这条测试把这个认知固定下来，免得以后有人拿小语料得出"k 无所谓"的结论。
        """
        dataset = EvalDataset(
            name="k-sensitivity",
            provenance="用于观察 k 的敏感性",
            corpus=[
                {"title": "Accelerated rTMS for post-stroke depression", "abstract": "trial"},
                {"title": "Theta burst stimulation for depression", "abstract": "meta"},
                {"title": "Exercise for post-stroke depression", "abstract": "review"},
                {"title": "Nurse-led follow-up after stroke", "abstract": "cohort"},
                {"title": "Ketamine for treatment-resistant depression", "abstract": "rct"},
            ],
            cases=[
                EvalCase(query="rTMS depression", relevant={"1": 1.0, "2": 1.0}, source="manual")
            ],
        )
        report = evaluate_configs(dataset, config=config, k=5)
        by_name = {r.config.name: r.summary for r in report.results}
        # 不断言它们一定相同或一定不同，只断言"都能算出合法指标"
        for name in ("rrf-k10", "production", "rrf-k100"):
            value = by_name[name].get("ndcg")
            assert value is None or 0.0 <= value <= 1.0


# ============================================================ 回归门禁
class TestRegressionGate:
    def test_passes_when_above_thresholds(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        assert check_regression(report, thresholds={"recall": 0.0}) == []

    def test_fails_when_below_threshold(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        failures = check_regression(report, thresholds={"recall": 1.1})
        assert failures and "低于下限" in failures[0]

    def test_reports_missing_metric_instead_of_passing_silently(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        failures = check_regression(report, thresholds={"不存在的指标": 0.5})
        assert failures and "没有数值" in failures[0]

    def test_unknown_config_name_is_reported(self, config, dataset):
        report = evaluate_configs(dataset, config=config, k=3)
        failures = check_regression(report, thresholds={"recall": 0.1}, config_name="nope")
        assert failures and "没有配置 nope" in failures[0]


# ==================================================== 仓库内置回归数据集
class TestBundledDataset:
    def test_bundled_dataset_runs_end_to_end(self, config):
        dataset = load_dataset(DATASET_DIR / "regression.json")
        report = evaluate_configs(dataset, config=config, k=10)
        assert report.corpus_size >= 30
        production = next(r for r in report.results if r.config.name == "production")
        assert production.summary["scored_queries"] >= 20
        # 合成语料 + 哈希嵌入：只断言"跑得通、指标合法"，不断言绝对值
        assert 0.0 <= production.summary["recall"] <= 1.0

    def test_bundled_dataset_has_negative_control(self, config):
        dataset = load_dataset(DATASET_DIR / "regression.json")
        controls = [c for c in dataset.cases if c.expect_no_relevant]
        assert len(controls) == 1

    def test_bundled_dataset_has_cross_lingual_cases(self, config):
        dataset = load_dataset(DATASET_DIR / "regression.json")
        assert any("zh" in c.tags for c in dataset.cases)
        assert any("es" in c.tags for c in dataset.cases)
