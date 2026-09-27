"""端到端回归测试 —— 守护第二轮审计修复（P1/P2 批次）。

覆盖：
- P1-1  FTS 三级策略改并集（不再首级非空即短路）
- P1-3  按阶段路由 LLM 客户端
- P1-6  Tier 0 claim-level 校验接进 review 阶段（含分层搬迁）
- P1-9  静态资源缓存策略（不再 no-store 与 ?v= 互斥）
- P1-11 数据源扇出有整体超时 / Bulkhead 接线 / Retry-After 上限
- P1-12 限流不再静默上调 + S2 限速可恢复
- P2-5  min_score 改相对阈值语义
- P2-7  失败/取消的运行不再被报成"已完成"
- P2-8  section 提示词前缀重排（digest 前置，为云端前缀缓存铺路）
- P2-10 search_logs 批量写入
- P2-11 search_logs 保留策略

每个测试对应一个真实跑过的缺陷；回滚修复后会被这条测试抓住。
"""

from __future__ import annotations

import re
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
WEB = ROOT / "medscholar" / "web"


def _src(rel: str) -> str:
    return (ROOT / rel).read_text(encoding="utf-8")


# ---------------------------------------------------------------------------
# P1-1：FTS 三级策略并集
# ---------------------------------------------------------------------------

def test_search_fts_unions_strategies_instead_of_short_circuit():
    """P1-1：三路（phrase / bigram-AND / bigram-OR）必须并集，不能首级命中就返回。

    原实现 `for ...: if rows: return` 让 bigram-OR（实测候选数 5~10 倍于前两级）
    只在 phrase 与 bigram-AND 都空时才启用，等于把宽召回整条丢掉。
    """
    src = _src("medscholar/db/repositories/search.py")
    fn = src[src.index("def search_fts(") :]
    fn = fn[: fn.index("\ndef search_fulltext(")]
    # 不能有"命中即返回"的短路
    assert not re.search(r"if rows:\s*\n\s*return", fn), (
        "search_fts 又出现了 `if rows: return` 短路——宽召回路径会被丢弃"
    )
    # 必须有并集累积结构
    assert "merged" in fn, "search_fts 应把多路结果并集累积到 merged"
    # 三路候选都要跑完
    assert fn.count("database.query(") >= 1


def test_search_fts_returns_union_on_real_db(db):
    """P1-1 端到端：同一查询下，并集结果的召回数应 ≥ 单路 phrase 的召回数。

    构造一篇只含 bigram 片段（不含完整短语）的文献，验证它仍能被召回——
    这正是原先被短路丢掉的路径。
    """
    from medscholar.db.repositories.papers import insert_paper
    from medscholar.db.repositories.search import search_fts
    from medscholar.models import Paper

    # 标题含完整短语的文献
    exact = Paper(
        title="post-stroke depression treatment",
        abstract="A study about post-stroke depression treatment.",
        authors=["A B"], journal="J", pub_year=2023, source="pubmed",
    )
    # 标题只含"被打断"的词序（phrase 级匹配不到，bigram-OR 能命中）
    fragmented = Paper(
        title="depression after post stroke: treatment options",
        abstract="We discuss depression following a post stroke event and its treatment.",
        authors=["C D"], journal="J", pub_year=2023, source="pubmed",
    )
    insert_paper(exact, db=db)
    insert_paper(fragmented, db=db)

    hits = search_fts("post-stroke depression treatment", limit=20, db=db)
    ids = {pid for pid, _ in hits}
    assert len(ids) >= 1, "至少应命中精确短语那篇"
    # 并集语义下，两篇都应有机会进入候选（不再因 phrase 命中就截断）
    assert len(ids) >= 2, (
        f"并集应至少召回 2 篇（精确 + 碎片化），实测 {len(ids)} 篇：{ids}"
    )


def test_search_fts_is_deterministic_on_ties(db):
    """P1-1 附带修复：同分时按 paper_id 兜底排序，保证候选集稳定。"""
    from medscholar.db.repositories.papers import insert_paper
    from medscholar.db.repositories.search import search_fts
    from medscholar.models import Paper

    for i in range(5):
        insert_paper(
            Paper(
                title="rTMS depression trial",
                abstract="rTMS depression trial abstract.",
                authors=["A B"], journal="J", pub_year=2023, source="pubmed",
            ),
            db=db,
        )
    a = [pid for pid, _ in search_fts("rTMS depression trial", limit=3, db=db)]
    b = [pid for pid, _ in search_fts("rTMS depression trial", limit=3, db=db)]
    assert a == b, f"同分候选集不稳定：{a} vs {b}"


# ---------------------------------------------------------------------------
# P1-3：按阶段路由 LLM 客户端
# ---------------------------------------------------------------------------

def test_get_llm_for_stage_falls_back_to_default(config):
    """P1-3：未配置 routing 时，任何阶段都回落到默认客户端（向后兼容）。"""
    from medscholar.llm.client import get_llm, get_llm_for_stage, reset_llm

    reset_llm()
    default = get_llm(config)
    for stage in ("plan", "critique", "section", "review"):
        assert get_llm_for_stage(stage, config) is default, (
            f"未配置 routing 时 {stage} 应回落到默认客户端"
        )


def test_get_llm_for_stage_uses_routed_model(config):
    """P1-3：配置 routing 后，该阶段返回独立客户端（不同 model）。"""
    from medscholar.llm.client import get_llm, get_llm_for_stage, reset_llm

    reset_llm()
    routed = config.llm.model_copy(update={"model": "strong-model"})
    config.llm.routing = {"plan": routed}

    plan_client = get_llm_for_stage("plan", config)
    other_client = get_llm_for_stage("section", config)
    assert plan_client is not other_client, (
        "plan 配了独立模型，不应与未配置阶段共用客户端"
    )
    assert plan_client.settings.model == "strong-model"
    assert other_client is get_llm(config), "未配置阶段应回落到默认"


def test_llm_settings_has_routing_field():
    """P1-3：LLMSettings 必须有 routing 字段，否则上面的路由无从配置。"""
    from medscholar.platform.config import LLMSettings

    settings = LLMSettings()
    assert hasattr(settings, "routing"), "LLMSettings 缺 routing 字段"


# ---------------------------------------------------------------------------
# P1-6：Tier 0 接进 review（含分层搬迁）
# ---------------------------------------------------------------------------

def test_tier0_module_lives_in_domain_layer():
    """P1-6：纯规则下沉到 domain，生成链路才能合法调用（application 不得依赖 eval）。"""
    assert (ROOT / "medscholar" / "domain" / "faithfulness.py").exists(), (
        "Tier 0 应在 medscholar/domain/faithfulness.py"
    )
    src = _src("medscholar/domain/faithfulness.py")
    # domain 层不得依赖 config / llm（架构校验器会红）
    assert "from ..config import" not in src, "domain 不得依赖 config"
    assert "from ..llm" not in src, "domain 不得依赖 llm"
    assert "from ..constants import" not in src, (
        "Tier 0 是纯 stdlib；JUDGE_*/LLM_* 常量都属于 Tier 1，不该出现在这里"
    )


def test_tier1_judge_lives_outside_domain():
    """P1-6：Tier 1（要调模型）不得待在 domain。"""
    domain_src = _src("medscholar/domain/faithfulness.py")
    assert "verify_claims_llm" not in domain_src or "def verify_claims_llm" not in domain_src, (
        "verify_claims_llm 需要 llm.client，不能留在 domain 层"
    )
    eval_src = _src("medscholar/eval/faithfulness.py")
    assert "async def verify_claims_llm" in eval_src, "Tier 1 应留在 eval/"


def test_compat_shell_reexports_same_objects():
    """P1-6：兼容外壳必须重导出**同一个对象**（不存在两份真相）。"""
    import medscholar.domain.faithfulness as domain_mod
    import medscholar.eval.faithfulness as eval_mod

    for name in ("Claim", "ClaimVerdict", "FaithfulnessReport",
                 "extract_claims", "check_claim_rules", "analyse_draft",
                 "VERDICTS", "SEVERITY_ORDER"):
        assert getattr(eval_mod, name) is getattr(domain_mod, name), (
            f"{name} 在外壳与实现里不是同一个对象"
        )


def test_graph_wires_tier0_into_review():
    """P1-6：review 阶段必须真的调用 Tier 0 校验。"""
    src = _src("medscholar/agent/graph.py")
    assert "_tier0_faithfulness" in src, "graph 应有 _tier0_faithfulness 方法"
    assert "from ..domain.faithfulness import analyse_draft" in src, (
        "graph 应从 domain 层 import analyse_draft（不是 eval）"
    )
    # review() 里必须调用
    review_fn = src[src.index("async def review(") :]
    review_fn = review_fn[: review_fn.index("async def _tier0_faithfulness")]
    assert "_tier0_faithfulness" in review_fn, "review() 必须调用 Tier 0 校验"


def test_tier0_runs_offline_and_flags_contradiction():
    """P1-6：Tier 0 离线可跑，且"强主张撞阴性结论"必须被判为高危。

    这是校验器存在的核心理由——结论说反是医学写作最严重的错误之一。
    规则要求同时满足三件事：① 论断含强主张 cue（如"优于/证实"）；
    ② 被引文献含阴性结果 cue（如"无显著差异"）；③ 两边有实词重合（防止引错文献造成假矛盾）。
    """
    from medscholar.domain.faithfulness import analyse_draft

    draft = "加速 rTMS 显著优于假刺激，已证实其疗效 [1]。"
    sources = {
        1: "加速 rTMS 与假刺激比较，抑郁量表评分无显著差异，未证实疗效优势。"
    }
    report = analyse_draft(draft, sources, valid_ids=[1])
    payload = report.to_dict()
    assert payload["claims"] >= 1
    assert payload["by_verdict"].get("contradicted", 0) >= 1, (
        f"方向矛盾未被识别：by_verdict={payload['by_verdict']} by_rule={payload['by_rule']}"
    )
    assert payload["by_rule"].get("direction", 0) >= 1, (
        f"direction 规则未命中：{payload['by_rule']}"
    )


def test_tier0_does_not_cry_wolf_without_overlap():
    """P1-6 反向约束：无实词重合时不得报"方向矛盾"（否则会把"引错文献"误报成"结论说反"）。"""
    from medscholar.domain.faithfulness import analyse_draft

    draft = "加速 rTMS 显著优于假刺激，已证实其疗效 [1]。"
    sources = {1: "Quantum chromodynamics at finite temperature: no significant difference."}
    report = analyse_draft(draft, sources, valid_ids=[1])
    payload = report.to_dict()
    assert payload["by_verdict"].get("contradicted", 0) == 0, (
        f"无实词重合却报了 contradicted：{payload['by_rule']}"
    )


def test_tier0_flags_fabricated_number():
    """P1-6：论断里的数字在被引文献中找不到 → numbers 规则报警（编造数据强信号）。"""
    from medscholar.domain.faithfulness import analyse_draft

    draft = "该研究报告 HAMD 下降 4.2 分 [1]。"
    sources = {1: "该研究观察了 60 例患者，未报告具体量表变化数值。"}
    report = analyse_draft(draft, sources, valid_ids=[1])
    payload = report.to_dict()
    assert payload["by_rule"].get("numbers", 0) >= 1, (
        f"编造数字未被识别：by_rule={payload['by_rule']} by_verdict={payload['by_verdict']}"
    )


def test_tier0_flags_out_of_range_citation():
    """P1-6：引用了不存在的编号 → existence 规则报警（越界引用兜底）。"""
    from medscholar.domain.faithfulness import analyse_draft

    draft = "某结论有据可依 [9]。"
    report = analyse_draft(draft, {1: "无关内容"}, valid_ids=[1, 2, 3])
    payload = report.to_dict()
    assert payload["by_rule"].get("existence", 0) >= 1, (
        f"越界引用未被识别：{payload['by_rule']}"
    )


def test_review_result_carries_faithfulness_field():
    """P1-6：ReviewResult 必须能承载并序列化 Tier 0 报告（含断点续跑还原）。"""
    from medscholar.agent.state import ReviewResult

    r = ReviewResult(verdict="revise", faithfulness={"claims": 3, "verdict": "revise"})
    d = r.to_dict()
    assert d["faithfulness"] == {"claims": 3, "verdict": "revise"}
    # 从快照还原
    restored = ReviewResult.from_dict(d)
    assert restored.faithfulness == {"claims": 3, "verdict": "revise"}
    # 无报告时不炸
    assert ReviewResult.from_dict({}).faithfulness is None
    assert ReviewResult().to_dict()["faithfulness"] is None


# ---------------------------------------------------------------------------
# P1-9：静态资源缓存策略
# ---------------------------------------------------------------------------

def test_static_assets_with_version_are_immutable():
    """P1-9：带 ?v= 的静态资源必须强缓存（原实现与 no-store 互斥，导致每次全量重下）。"""
    src = _src("medscholar/server/app.py")
    assert "immutable" in src, "带版本号的静态资源应使用 immutable 强缓存"
    # 不能再对所有 /static/* 无条件 no-store
    m = re.search(
        r'path\.startswith\("/static/"\)[^}]*?no-store',
        src,
    )
    assert not m, (
        "不能再对 /static/* 无条件 no-store —— 那会让 ?v= 版本号失去意义"
    )


def test_index_html_still_no_store():
    """P1-9 反向约束：HTML 入口仍须 no-store（render_index 每次注入新 ?v=）。"""
    src = _src("medscholar/server/app.py")
    assert "no-store" in src, "HTML 入口必须保留 no-store"
    # no-store 分支应针对 "/" 或 index.html
    assert re.search(r'(path == "/"|index\.html)[\s\S]{0,200}?no-store', src), (
        "no-store 应作用在 HTML 入口上"
    )


# ---------------------------------------------------------------------------
# P1-11：数据源扇出超时 / Bulkhead / Retry-After 上限
# ---------------------------------------------------------------------------

def test_registry_search_has_overall_timeout():
    """P1-11：任一慢源不得让整轮检索无限等待。"""
    src = _src("medscholar/api/registry.py")
    assert "asyncio.wait_for" in src, "registry.search 的 gather 必须包 wait_for"
    assert "TimeoutError" in src, "超时必须被捕获并转成各源失败状态"


def test_registry_has_bulkhead_per_source():
    """P1-11：Bulkhead 必须真的被接线（原先全仓零调用者）。"""
    src = _src("medscholar/api/registry.py")
    assert "Bulkhead" in src and "self._bulkheads" in src, (
        "registry 应给每个数据源建 Bulkhead 并在 run() 里 async with 包裹"
    )


def test_registry_uses_per_source_max_results():
    """P1-12：per-source limit 必须取该源自己的 max_results（原先套用 pubmed 的）。"""
    src = _src("medscholar/api/registry.py")
    assert "per_source_for" in src, "应逐源计算 limit"
    assert 'sources.get("pubmed").max_results' not in src, (
        "不该再把 pubmed 的 max_results 套给所有源"
    )


def test_retry_after_is_capped():
    """P1-11：Retry-After 必须有上限，否则异常响应可让进程睡任意久。"""
    src = _src("medscholar/api/base.py")
    assert re.search(r"min\(retry_after,\s*cap\)", src), (
        "429 分支应对 Retry-After 加上限（min(retry_after, cap)）"
    )


# ---------------------------------------------------------------------------
# P1-12：限流不再被静默上调 + S2 可恢复
# ---------------------------------------------------------------------------

def test_pubmed_rps_not_silently_raised():
    """P1-12：带 api_key 时不得把用户的 rps 上调（原 max() 会覆盖保守设置）。"""
    from medscholar.api.pubmed_client import PubMedClient
    from medscholar.platform.config import AppConfig

    c = PubMedClient(
        AppConfig().sources.pubmed.model_copy(update={"api_key": "k", "rps": 2.0})
    )
    assert c.bucket.rps == 2.0, f"rps 被静默改写为 {c.bucket.rps}"
    high = PubMedClient(
        AppConfig().sources.pubmed.model_copy(update={"api_key": "k", "rps": 99.0})
    )
    assert high.bucket.rps == 10.0, "上限兜底应夹到 10"


def test_semantic_scholar_rps_not_silently_raised():
    """P1-12：S2 同理。"""
    from medscholar.api.semantic_scholar_client import SemanticScholarClient
    from medscholar.platform.config import AppConfig

    c = SemanticScholarClient(
        AppConfig().sources.semantic_scholar.model_copy(
            update={"api_key": "k", "rps": 0.2}
        )
    )
    assert c.bucket.rps == 0.2, f"rps 被静默改写为 {c.bucket.rps}"


def test_semantic_scholar_rate_recovers():
    """P1-12：S2 限速下调必须是可恢复的（原实现单向只降不升）。"""
    src = _src("medscholar/api/semantic_scholar_client.py")
    assert "_consecutive_ok" in src, "应有连续成功计数用于恢复"
    assert re.search(r"_consecutive_ok\s*>=\s*\d+", src), "应有恢复阈值"
    assert "update_rps(target)" in src, "恢复路径应真的更新桶速率"


# ---------------------------------------------------------------------------
# P2-5：min_score 相对阈值
# ---------------------------------------------------------------------------

def test_min_score_uses_relative_threshold():
    """P2-5：RRF 分数恒正，绝对阈值永不触发；必须改为相对 max(rrf) 的比例。"""
    src = _src("medscholar/db/repositories/search.py")
    assert "min_score_ratio" in src, "应使用 min_score_ratio（相对阈值）"
    assert re.search(r"max_rrf\s*\*\s*cfg\.retrieval\.min_score_ratio", src), (
        "阈值应算成 max_rrf * min_score_ratio"
    )


def test_min_score_ratio_default_disabled():
    """P2-5：默认 0 = 关闭，不改变现有行为。"""
    from medscholar.platform.config import RetrievalSettings

    assert RetrievalSettings().min_score_ratio == 0.0


# ---------------------------------------------------------------------------
# P2-7：失败/取消不再被报成"已完成"
# ---------------------------------------------------------------------------

def test_on_done_branches_on_phase():
    """P2-7：onDone 必须读 data.phase 并区分 error / cancelled / done。"""
    js = (WEB / "app.js").read_text(encoding="utf-8", errors="replace")
    fn = js[js.index("function onDone(data)") :]
    fn = fn[: fn.index("function ", fn.index("refreshHealth"))]
    assert "data.phase" in fn, "onDone 必须读 data.phase"
    assert "'error'" in fn and "'cancelled'" in fn, (
        "onDone 必须显式区分 error 与 cancelled"
    )
    # 不能再无条件报"已完成"
    assert "toast('研究流程已完成。', 'ok');" not in fn, (
        "不能在所有分支都 toast 成功"
    )


# ---------------------------------------------------------------------------
# P2-8：section 提示词前缀重排
# ---------------------------------------------------------------------------

def test_section_prompt_puts_digest_first():
    """P2-8：digest（多章节共用）必须排在章节标题之前，才能命中云端前缀缓存。"""
    from medscholar.llm.prompts import section_user

    digest = "【材料块】这是所有章节共用的文献材料。"
    prompt = section_user("课题", "引言", ["背景"], digest, min_chars=800, max_chars=1600)
    digest_pos = prompt.find(digest)
    title_pos = prompt.find("引言")
    assert digest_pos != -1, "digest 必须出现在 section 提示词里"
    assert title_pos != -1, "章节标题必须出现"
    assert digest_pos < title_pos, (
        f"digest 必须前置于章节标题（缓存前缀稳定性）：digest@{digest_pos} title@{title_pos}"
    )


def test_section_prompt_prefix_is_stable_across_sections():
    """P2-8：相邻章节的提示词前缀（digest 部分）必须逐字节相同。"""
    from medscholar.llm.prompts import section_user

    digest = "【材料块】" + "A" * 500
    p1 = section_user("课题", "第一章", ["要点一"], digest, min_chars=800, max_chars=1600)
    p2 = section_user("课题", "第二章", ["要点二"], digest, min_chars=800, max_chars=1600)
    # 公共前缀长度应至少覆盖整个 digest
    common = 0
    for a, b in zip(p1, p2):
        if a != b:
            break
        common += 1
    assert common >= len(digest), (
        f"两章节的公共前缀 {common} 短于 digest 长度 {len(digest)}——缓存无法命中"
    )


def test_section_prompt_keeps_length_clause():
    """P2-8 反向约束：重排不能丢掉字数区间约束。"""
    from medscholar.llm.prompts import section_user

    prompt = section_user("课题", "引言", ["背景"], "材料", min_chars=800, max_chars=1600)
    assert "800" in prompt and "1600" in prompt


# ---------------------------------------------------------------------------
# P2-10：search_logs 批量写入
# ---------------------------------------------------------------------------

def test_log_searches_batch_writes_in_one_transaction(db):
    """P2-10：批量入口一次事务写多条。"""
    from medscholar.db.repositories.search import log_searches
    from medscholar.models import SearchLogEntry

    entries = [
        SearchLogEntry(query=f"q{i}", source="pubmed", result_count=i, duration_ms=10)
        for i in range(5)
    ]
    n = log_searches(entries, db=db)
    assert n == 5, f"应写入 5 条，实测 {n}"
    rows = db.query("SELECT COUNT(*) AS c FROM search_logs")
    assert rows[0]["c"] == 5


def test_log_searches_empty_is_noop(db):
    """P2-10：空列表不应开事务。"""
    from medscholar.db.repositories.search import log_searches

    assert log_searches([], db=db) == 0


def test_scout_uses_batch_logging():
    """P2-10：scout 必须改用批量入口，而不是每条一次事务。"""
    src = _src("medscholar/agent/scout.py")
    assert "log_searches" in src, "scout 应调用 log_searches"
    assert "_pending_log_entries" in src, "scout 应累积待写条目"
    assert not re.search(r"^\s*await\s+asyncio\.to_thread\(\s*log_search\s*,", src, re.M), (
        "scout 不应再逐条 to_thread(log_search, ...)"
    )


# ---------------------------------------------------------------------------
# P2-11：search_logs 保留策略
# ---------------------------------------------------------------------------

def test_purge_old_search_logs_keeps_recent(db):
    """P2-11：保留策略删旧留新，避免无界增长。"""
    from medscholar.db.repositories.search import log_search, purge_old_search_logs
    from medscholar.models import SearchLogEntry

    for i in range(30):
        log_search(
            SearchLogEntry(query=f"q{i}", source="pubmed", result_count=i),
            db=db,
        )
    # 手工把前 20 条改成很久以前
    db.execute("UPDATE search_logs SET created_at = datetime('now', '-400 days') WHERE id <= 20")
    deleted = purge_old_search_logs(older_than_days=90, keep_recent=5, db=db)
    assert deleted >= 20, f"应至少删掉 20 条陈旧记录，实测 {deleted}"
    remaining = db.query("SELECT COUNT(*) AS c FROM search_logs")[0]["c"]
    assert remaining <= 20, f"保留策略后应 ≤ keep_recent + 未过期数，实测 {remaining}"


def test_purge_is_exported_from_repo():
    """P2-11：保留策略应可从 repo 门面调用（供维护命令/CLI 使用）。"""
    from medscholar.db import repo

    assert hasattr(repo, "purge_old_search_logs")
