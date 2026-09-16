"""反馈/质疑/学习闭环，以及基于实验数据的论文生成。

这里最该被测试的是「反馈到底改变了什么行为」——否则所谓强化学习只是界面装饰。
以及论文模块的**数字溯源校验**：医学论文里编一个 P 值就是学术不端，
必须由程序兜住。
"""

from __future__ import annotations

import pytest

from medscholar.feedback import (
    FEEDBACK_CATEGORIES,
    FeedbackEntry,
    correction_memories,
    export_preference_pairs,
    feedback_summary,
    list_feedback,
    memories_as_prompt,
    paper_penalty,
    record_feedback,
    source_penalty,
)
from medscholar.manuscript import (
    IMRAD_SECTIONS,
    ManuscriptBrief,
    build_outline,
    check_number_provenance,
    get_manuscript,
    list_manuscripts,
    save_manuscript,
)


@pytest.fixture
def db(tmp_path):
    from medscholar.config import AppConfig
    from medscholar.db.connect import Database

    config = AppConfig(data_dir=str(tmp_path), offline=True)
    return Database(tmp_path / "fb.db", config=config)


def add(db, verdict="up", **kwargs) -> int:
    entry = FeedbackEntry(verdict=verdict, **kwargs)
    return record_feedback(entry, db=db)


# ===================================================== 反馈的基础行为
class TestFeedbackStorage:
    def test_record_and_list(self, db):
        add(db, verdict="down", comment="结论说反了", topic="rTMS 抑郁", run_id="run1")
        items = list_feedback(db=db)
        assert len(items) == 1
        assert items[0].verdict == "down"
        assert items[0].comment == "结论说反了"

    def test_invalid_verdict_falls_back_to_up(self, db):
        add(db, verdict="胡说")
        assert list_feedback(db=db)[0].verdict == "up"

    def test_invalid_category_is_dropped(self, db):
        add(db, verdict="challenge", category="不存在的分类")
        assert list_feedback(db=db)[0].category == ""

    def test_long_text_is_truncated_not_rejected(self, db):
        add(db, verdict="challenge", corrected_text="修" * 50000)
        assert len(list_feedback(db=db)[0].corrected_text) == 20000

    def test_filter_by_run_and_type(self, db):
        add(db, run_id="a", target_type="message")
        add(db, run_id="b", target_type="paper")
        assert len(list_feedback(run_id="a", db=db)) == 1
        assert len(list_feedback(target_type="paper", db=db)) == 1

    def test_summary_counts(self, db):
        add(db, verdict="up")
        add(db, verdict="down")
        add(db, verdict="challenge", category="fact", corrected_text="正确说法")
        summary = feedback_summary(db=db)
        assert summary["total"] == 3
        assert summary["up"] == 1 and summary["down"] == 1 and summary["challenge"] == 1
        assert summary["active_memories"] == 1
        assert any(c["category"] == "fact" for c in summary["by_category"])

    def test_categories_are_exposed_to_frontend(self, db):
        summary = feedback_summary(db=db)
        keys = {c["key"] for c in summary["categories"]}
        assert {"citation", "fact", "omission", "overreach", "data"} <= keys
        assert set(FEEDBACK_CATEGORIES) == keys


# ==================================== 闭环 1：纠错记忆（即时生效）
class TestCorrectionMemory:
    def test_challenge_with_correction_becomes_memory(self, db):
        add(
            db, verdict="challenge", category="fact", topic="加速rTMS治疗卒中后抑郁",
            comment="结论与原文相反", corrected_text="原文结论是无效，不是有效",
            quoted_text="加速rTMS显著优于对照组",
        )
        memories = correction_memories(topic="加速rTMS治疗卒中后抑郁", db=db)
        assert len(memories) == 1
        assert "无效" in memories[0]["corrected_text"]

    def test_plain_downvote_is_not_a_memory(self, db):
        """只点踩、没给正确说法的反馈没有可复用信息，不该进记忆。"""
        add(db, verdict="down", comment="不好")
        assert correction_memories(db=db) == []

    def test_challenge_without_correction_is_not_a_memory(self, db):
        add(db, verdict="challenge", comment="这里有问题")
        assert correction_memories(db=db) == []

    def test_memory_prompt_is_renderable_and_empty_when_none(self, db):
        assert memories_as_prompt("任意课题", db=db) == ""
        add(db, verdict="challenge", category="fact", topic="T",
            corrected_text="正确说法是 X")
        text = memories_as_prompt("T", db=db)
        assert "必须避免重犯" in text and "正确说法是 X" in text

    def test_same_topic_memories_rank_first(self, db):
        add(db, verdict="challenge", topic="别的课题", corrected_text="通用纠错")
        add(db, verdict="challenge", topic="我的课题", corrected_text="同主题纠错")
        memories = correction_memories(topic="我的课题", limit=1, db=db)
        assert memories[0]["corrected_text"] == "同主题纠错"

    def test_falls_back_to_other_topics_when_none_match(self, db):
        add(db, verdict="challenge", topic="别的课题", corrected_text="通用纠错")
        memories = correction_memories(topic="全新课题", limit=3, db=db)
        assert len(memories) == 1


# ============================== 闭环 2：偏好对导出（离线训练数据）
class TestPreferenceExport:
    def test_pair_requires_both_sides(self, db):
        add(db, verdict="challenge", quoted_text="错误说法", corrected_text="正确说法", topic="课题")
        add(db, verdict="challenge", quoted_text="只有错误说法")  # 缺 chosen
        add(db, verdict="challenge", corrected_text="只有正确说法")  # 缺 rejected
        pairs = export_preference_pairs(db=db)
        assert len(pairs) == 1
        assert pairs[0]["chosen"] == "正确说法"
        assert pairs[0]["rejected"] == "错误说法"
        assert "课题" in pairs[0]["prompt"]

    def test_upvotes_do_not_become_pairs(self, db):
        add(db, verdict="up", quoted_text="x", corrected_text="y")
        assert export_preference_pairs(db=db) == []

    def test_jsonl_export(self, db, tmp_path):
        from medscholar.feedback import export_jsonl

        add(db, verdict="challenge", quoted_text="坏", corrected_text="好", topic="T")
        path = tmp_path / "dpo.jsonl"
        count = export_jsonl(str(path), db=db)
        assert count == 1
        import json

        line = json.loads(path.read_text(encoding="utf-8").strip())
        assert line["chosen"] == "好" and line["rejected"] == "坏"


# ============================== 闭环 3：文献/来源重加权（即时生效）
class TestReweighting:
    def test_repeatedly_challenged_paper_is_penalised(self, db):
        for _ in range(3):
            add(db, verdict="challenge", category="fact", target_type="paper",
                target_id="42", corrected_text="说反了")
        penalties = paper_penalty(db=db)
        assert 42 in penalties
        assert penalties[42] < 1.0

    def test_penalty_has_a_floor(self, db):
        for _ in range(50):
            add(db, verdict="challenge", category="fact", target_type="paper",
                target_id="7", corrected_text="错")
        assert paper_penalty(db=db)[7] >= 0.5

    def test_plain_downvote_does_not_penalise_a_paper(self, db):
        """口味不同不该把有价值的文献永久压下去。"""
        add(db, verdict="down", target_type="paper", target_id="42")
        assert paper_penalty(db=db) == {}

    def test_non_numeric_target_ignored(self, db):
        add(db, verdict="challenge", category="fact", target_type="paper",
            target_id="not-a-number", corrected_text="x")
        assert paper_penalty(db=db) == {}

    def test_source_penalty_needs_enough_samples(self, db):
        for _ in range(3):
            add(db, verdict="down", target_type="search_result", target_id="openalex")
        assert source_penalty(db=db, min_samples=5) == {}, "样本太少不下判断"

    def test_source_penalty_applies_after_enough_bad_feedback(self, db):
        for _ in range(4):
            add(db, verdict="down", target_type="search_result", target_id="s2")
        add(db, verdict="up", target_type="search_result", target_id="s2")
        penalties = source_penalty(db=db, min_samples=5, min_ratio=0.6)
        assert "s2" in penalties and penalties["s2"] < 1.0

    def test_apply_paper_penalty_reorders(self):
        from medscholar.feedback import apply_paper_penalty
        from medscholar.models import Paper

        class Hit:
            def __init__(self, pid, score):
                self.paper = Paper(title=f"p{pid}", source="pubmed", paper_id=pid)
                self.score = score

        hits = [Hit(1, 1.0), Hit(2, 0.9)]
        # 1 号被降权 0.5 → 应该排到后面
        reordered = apply_paper_penalty(hits, {1: 0.5})
        assert [h.paper.paper_id for h in reordered] == [2, 1]

    def test_apply_without_penalties_is_identity(self):
        from medscholar.feedback import apply_paper_penalty

        assert apply_paper_penalty([], {}) == []


# ================================================== 论文：要素与结构
class TestManuscriptBrief:
    def test_missing_required_fields(self):
        brief = ManuscriptBrief()
        missing = brief.missing()
        assert "研究目标" in missing and "研究设计" in missing

    def test_complete_brief_has_no_missing(self):
        brief = ManuscriptBrief(goal="目标", design="RCT", population="人群", outcomes="结局")
        assert brief.missing() == []

    def test_round_trip_dict(self):
        brief = ManuscriptBrief(title="T", goal="G", design="D")
        again = ManuscriptBrief.from_dict(brief.to_dict())
        assert again.title == "T" and again.goal == "G"

    def test_prompt_includes_data_block(self):
        brief = ManuscriptBrief(goal="G", data="组别,例数\nA,60")
        text = brief.text_for_prompt()
        assert "【研究目标】G" in text
        assert "【实验数据】" in text and "组别,例数" in text

    def test_outline_is_imrad(self):
        keys = [item["key"] for item in build_outline(ManuscriptBrief(goal="G"))]
        assert keys == ["abstract", "introduction", "methods", "results", "discussion", "conclusion"]
        assert [k for k, _ in IMRAD_SECTIONS][1:] == keys


# ============================ 论文：数字溯源（防编造数据的关键防线）
class TestNumberProvenance:
    def brief(self, **kw):
        base = dict(
            goal="评估加速rTMS疗效", design="RCT", population="卒中后抑郁患者",
            outcomes="HAMD 评分", statistics="t 检验",
            data="组别,例数,HAMD治疗后\n加速rTMS,60,9.8±3.2\n常规rTMS,60,14.6±4.1",
            results="差异有统计学意义（P<0.001，95%CI 2.1~7.5）",
        )
        base.update(kw)
        return ManuscriptBrief(**base)

    def test_numbers_from_user_data_pass(self):
        draft = "共纳入 60 例患者，治疗后 HAMD 为 9.8，对照组为 14.6，P<0.001。"
        result = check_number_provenance(draft, brief=self.brief())
        assert result["verdict"] == "pass", result["unverified"]
        assert result["unverified_count"] == 0

    def test_invented_number_is_flagged(self):
        draft = "共纳入 60 例患者，治疗后 HAMD 下降 4.2 分，P<0.001。"
        result = check_number_provenance(draft, brief=self.brief())
        assert result["unverified_count"] >= 1
        flagged = [u["value"] for u in result["unverified"]]
        assert "4.2" in flagged, flagged
        assert result["verdict"] in {"warn", "fail"}

    def test_numbers_from_literature_text_allowed(self):
        draft = "既往研究报道有效率为 55%。"
        result = check_number_provenance(
            draft, brief=self.brief(), literature_text="某研究有效率为 55%。"
        )
        assert result["unverified_count"] == 0

    def test_percentage_matches_bare_number(self):
        """数据显示 12，正文写 12% 也应算命中。"""
        brief = self.brief(data="反应率\n60/120=12")
        result = check_number_provenance("反应率为 12%。", brief=brief)
        assert result["unverified_count"] == 0

    def test_report_includes_sentence_for_manual_review(self):
        draft = "本研究共招募 9999 例受试者。"
        result = check_number_provenance(draft, brief=self.brief())
        assert result["unverified"]
        assert "9999" in result["unverified"][0]["sentence"]

    def test_no_numbers_verdict(self):
        result = check_number_provenance("本研究未测量相关指标。", brief=self.brief())
        assert result["verdict"] == "no_numbers"

    def test_warn_vs_fail_threshold(self):
        """少量可疑数字给 warn，大量给 fail —— 让用户能区分严重程度。"""
        many_bad = " ".join(f"{900+i}.5" for i in range(20))
        result = check_number_provenance(many_bad, brief=self.brief())
        assert result["verdict"] == "fail"

    def test_note_tells_user_not_to_submit_blindly(self):
        result = check_number_provenance("共 777 例。", brief=self.brief())
        assert "学术不端" in result["note"]


# ==================================================== 论文：持久化
class TestManuscriptStorage:
    def test_save_and_read(self, db):
        mid = save_manuscript(
            title="我的论文", brief={"goal": "G"}, draft="# 正文", checks={"verdict": "pass"}, db=db
        )
        assert mid > 0
        record = get_manuscript(mid, db=db)
        assert record["title"] == "我的论文"
        assert record["brief"]["goal"] == "G"
        assert record["checks"]["verdict"] == "pass"

    def test_update_existing(self, db):
        mid = save_manuscript(title="初版", brief={}, draft="a", db=db)
        save_manuscript(title="改后", brief={}, draft="b", manuscript_id=mid, db=db)
        record = get_manuscript(mid, db=db)
        assert record["title"] == "改后" and record["draft"] == "b"
        assert len(list_manuscripts(db=db)) == 1

    def test_broken_json_does_not_raise(self, db):
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO manuscripts(title, brief, draft) VALUES (?,?,?)",
                ("坏数据", "{not json", "x"),
            )
        record = list_manuscripts(db=db)[0]
        assert record["title"] == "坏数据"

    def test_missing_returns_none(self, db):
        assert get_manuscript(99999, db=db) is None


# ==================================================== 论文：生成流程
class TestDraftManuscript:
    async def test_missing_elements_raise_actionable_error(self, db):
        from medscholar.config import AppConfig
        from medscholar.manuscript import draft_manuscript

        with pytest.raises(ValueError, match="还缺少必要的论文要素"):
            await draft_manuscript(ManuscriptBrief(goal=""), config=AppConfig())

    async def test_generates_all_sections_with_fake_llm(self, monkeypatch, db):
        from medscholar import manuscript as ms
        from medscholar.config import AppConfig

        class FakeClient:
            async def start(self):
                return None

            async def chat(self, messages, **kwargs):
                return "本节正文，共 60 例，P<0.001。"

            async def stream(self, messages, **kwargs):
                yield "本节正文，共 60 例，P<0.001。"

        monkeypatch.setattr(ms, "get_llm", lambda *a, **k: FakeClient())

        brief = ManuscriptBrief(
            goal="评估疗效", design="RCT", population="患者", outcomes="HAMD",
            data="例数 60", results="P<0.001",
        )
        result = await ms.draft_manuscript(brief, config=AppConfig())
        assert set(result["sections"]) == {
            "abstract", "introduction", "methods", "results", "discussion", "conclusion"
        }
        assert result["errors"] == []
        text = ms.assemble(result["sections"], result["titles"], result["order"])
        assert "## 摘要" in text and "## 方法" in text

    async def test_section_failure_is_isolated(self, monkeypatch):
        from medscholar import manuscript as ms
        from medscholar.config import AppConfig
        from medscholar.llm.client import LLMError

        calls = {"n": 0}

        class FlakyClient:
            async def start(self):
                return None

            async def chat(self, messages, **kwargs):
                calls["n"] += 1
                if calls["n"] == 1:
                    raise LLMError("模型抽风")
                return "正常内容"

            async def stream(self, messages, **kwargs):
                yield "x"

        monkeypatch.setattr(ms, "get_llm", lambda *a, **k: FlakyClient())

        brief = ManuscriptBrief(goal="G", design="D", population="P", outcomes="O")
        result = await ms.draft_manuscript(brief, config=AppConfig())
        assert result["errors"], "第一节的失败必须被记录"
        assert any(result["sections"].values()), "其余章节仍应产出"

    async def test_memory_text_is_injected(self, monkeypatch):
        """纠错记忆必须真的进入提示词 —— 否则"学习"就是假的。"""
        from medscholar import manuscript as ms
        from medscholar.config import AppConfig

        seen: list[str] = []

        class SpyClient:
            async def start(self):
                return None

            async def chat(self, messages, **kwargs):
                seen.append(messages[0]["content"])
                return "ok"

            async def stream(self, messages, **kwargs):
                yield "ok"

        monkeypatch.setattr(ms, "get_llm", lambda *a, **k: SpyClient())

        brief = ManuscriptBrief(goal="G", design="D", population="P", outcomes="O")
        await ms.draft_manuscript(
            brief, memory_text="以下是用户此前指出过的错误：不得把无效说成有效",
            config=AppConfig(),
        )
        assert seen and "不得把无效说成有效" in seen[0]
