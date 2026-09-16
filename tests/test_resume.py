"""断点续跑：阶段快照必须落库，续跑必须只重跑没做完的阶段。

用户痛点：检索 + 撰写动辄几十分钟，一次断线就全部重来。
所以每个阶段结束后都要留下快照，续跑时直接复用。
"""

from __future__ import annotations

import pytest

from medscholar.agent.graph import _phase_index
from medscholar.agent.runtime import AgentRuntime
from medscholar.agent.state import (
    AgentState,
    CritiqueResult,
    PlanQuery,
    ResearchPlan,
    ReviewResult,
)
from medscholar.config import AppConfig
from medscholar.db.connect import Database
from medscholar.db.repo import (
    get_run_step,
    list_run_steps,
    save_run_step,
)


def make_runtime(tmp_path, name: str = "rs.db") -> tuple[AgentRuntime, Database]:
    config = AppConfig(data_dir=str(tmp_path), offline=True)
    config.agent.require_approval = False
    db = Database(tmp_path / name, config=config)
    return AgentRuntime(config=config, db=db), db


class TestPhaseIndex:
    def test_ordering(self):
        assert _phase_index("") == -1
        assert _phase_index("plan") == 0
        assert _phase_index("execute") == 1
        assert _phase_index("reflect") == 2
        assert _phase_index("synthesize") == 3
        assert _phase_index("review") == 4

    def test_unknown_phase(self):
        assert _phase_index("计划") == -1


class TestRunStepStorage:
    def test_save_and_read_back(self, tmp_path):
        _, db = make_runtime(tmp_path)
        save_run_step("r1", "plan", {"plan": {"topic_zh": "rTMS"}}, db=db)
        got = get_run_step("r1", "plan", db=db)
        assert got is not None
        assert got["data"]["plan"]["topic_zh"] == "rTMS"

    def test_upsert_overwrites_same_phase(self, tmp_path):
        _, db = make_runtime(tmp_path)
        save_run_step("r1", "plan", {"n": 1}, db=db)
        save_run_step("r1", "plan", {"n": 2}, db=db)
        steps = list_run_steps("r1", db=db)
        assert len(steps) == 1
        assert steps[0]["data"]["n"] == 2

    def test_steps_are_ordered_by_pipeline(self, tmp_path):
        _, db = make_runtime(tmp_path)
        for phase in ("review", "plan", "reflect", "execute"):
            save_run_step("r1", phase, {}, db=db)
        assert [s["phase"] for s in list_run_steps("r1", db=db)] == [
            "plan",
            "execute",
            "reflect",
            "review",
        ]

    def test_broken_payload_does_not_raise(self, tmp_path):
        _, db = make_runtime(tmp_path)
        with db.transaction() as conn:
            conn.execute(
                "INSERT INTO run_steps(run_id, phase, payload) VALUES (?,?,?)",
                ("r1", "plan", "{not json"),
            )
        steps = list_run_steps("r1", db=db)
        assert steps[0]["data"] == {}


class TestCheckpointDuringRun:
    async def test_offline_run_writes_checkpoints(self, tmp_path):
        """一次完整运行结束后，走过的阶段都应当留下快照。

        离线模式检索不到文献，工作流会在执行阶段后按设计提前结束，
        所以这里只断言"跑过的阶段都有快照"，并核对最后一个快照与最终阶段一致。
        """
        runtime, db = make_runtime(tmp_path, "cp.db")
        handle = await runtime.start(topic="测试课题", offline=True, require_approval=False)
        async for _ in runtime.stream(handle.run_id, timeout=60):
            pass

        steps = list_run_steps(handle.run_id, db=db)
        phases = [s["phase"] for s in steps]
        assert "plan" in phases, "规划阶段必须有快照"
        assert "execute" in phases, "执行阶段必须有快照"
        assert phases[0] == "plan"
        assert phases[-1] == "execute", "离线无文献时最后一个完成的阶段就是执行"

    async def test_checkpoints_are_written_progressively(self, tmp_path):
        """快照要逐阶段写入，不能等运行结束才一次性写。"""
        runtime, db = make_runtime(tmp_path, "cp3.db")
        handle = await runtime.start(topic="测试课题", offline=True, require_approval=False)

        seen_early = False
        async for event in runtime.stream(handle.run_id, timeout=60):
            if event.type == "phase" and event.data.get("phase") == "execute":
                # 执行阶段刚开始，规划快照就应该已经在库里了
                if get_run_step(handle.run_id, "plan", db=db) is not None:
                    seen_early = True
            if event.type == "done":
                break
        assert seen_early, "规划阶段的快照应当在执行阶段开始前就已落库"

    async def test_plan_snapshot_contains_reusable_plan(self, tmp_path):
        runtime, db = make_runtime(tmp_path, "cp2.db")
        handle = await runtime.start(topic="测试课题", offline=True, require_approval=False)
        async for _ in runtime.stream(handle.run_id, timeout=60):
            pass
        step = get_run_step(handle.run_id, "plan", db=db)
        assert step is not None
        plan = step["data"]["plan"]
        assert plan["topic_zh"], "快照里的方案必须能还原出课题"
        assert plan["outline"], "快照里的方案必须带大纲"


class TestResume:
    async def test_resume_rejects_unknown_run(self, tmp_path):
        runtime, _ = make_runtime(tmp_path, "rr1.db")
        with pytest.raises(ValueError, match="不存在"):
            await runtime.resume("nope")

    async def test_resume_rejects_run_without_snapshots(self, tmp_path):
        from medscholar.db.repo import upsert_run

        runtime, db = make_runtime(tmp_path, "rr2.db")
        upsert_run("legacy", topic="旧课题", phase="plan", status="interrupted", db=db)
        with pytest.raises(ValueError, match="阶段快照"):
            await runtime.resume("legacy")

    async def test_resume_rejects_finished_run(self, tmp_path):
        from medscholar.db.repo import upsert_run

        runtime, db = make_runtime(tmp_path, "rr3.db")
        upsert_run("done1", topic="课题", phase="done", status="done", db=db)
        save_run_step("done1", "review", {}, db=db)
        with pytest.raises(ValueError, match="已经结束"):
            await runtime.resume("done1")

    async def test_resume_rejects_placeholder_topic(self, tmp_path):
        from medscholar.db.repo import upsert_run

        runtime, db = make_runtime(tmp_path, "rr4.db")
        topic = "例如：加速rTMS治疗卒中后抑郁的疗效与安全性"
        upsert_run("ph1", topic=topic, phase="plan", status="interrupted", db=db)
        save_run_step("ph1", "plan", {"plan": {"topic_zh": topic}}, db=db)
        with pytest.raises(ValueError, match="示例提示"):
            await runtime.resume("ph1")

    async def test_resume_reuses_plan_and_papers(self, tmp_path):
        """核心：已经规划 + 已检索的运行，续跑时不得重新规划、重新检索。"""
        from medscholar.db.repo import upsert_run

        runtime, db = make_runtime(tmp_path, "rr5.db")
        plan = ResearchPlan(
            topic_zh="加速rTMS治疗卒中后抑郁",
            topic_en="accelerated rTMS",
            queries=[PlanQuery(query="rTMS AND depression", sources=["pubmed"])],
            outline=[],
        ).to_dict()
        save_run_step("keep1", "plan", {"plan": plan}, db=db)
        save_run_step("keep1", "execute", {"plan": plan, "paper_ids": []}, db=db)
        upsert_run(
            "keep1", topic="加速rTMS治疗卒中后抑郁", phase="execute",
            status="interrupted", papers=0, db=db,
        )

        handle = await runtime.resume("keep1")
        assert handle.state.resumed_from == "execute"
        assert handle.state.plan is not None
        assert handle.state.plan.topic_zh == "加速rTMS治疗卒中后抑郁"
        # 复用同一个 run_id，产物与快照都挂在它上面
        assert handle.run_id == "keep1"

    async def test_resume_restores_papers_from_ids(self, tmp_path):
        """文献不存全文，只存 id，续跑时按 id 回库取，引用编号不会错位。"""
        from medscholar.db.repo import insert_paper, upsert_run
        from medscholar.models import Paper

        runtime, db = make_runtime(tmp_path, "rr6.db")
        ids = []
        for i in range(3):
            pid, _ = insert_paper(
                Paper(title=f"论文 {i}", source="pubmed", abstract=f"摘要 {i}"), db=db
            )
            ids.append(pid)

        plan = ResearchPlan(topic_zh="课题", queries=[]).to_dict()
        save_run_step("keep2", "plan", {"plan": plan}, db=db)
        save_run_step("keep2", "execute", {"plan": plan, "paper_ids": ids}, db=db)
        upsert_run("keep2", topic="课题", phase="execute", status="interrupted", db=db)

        handle = await runtime.resume("keep2")
        assert [p.paper_id for p in handle.state.papers] == ids
        assert [p.title for p in handle.state.papers] == ["论文 0", "论文 1", "论文 2"]
        assert handle.state.citation_map[1].paper_id == ids[0]

    async def test_resume_restores_critique_and_draft(self, tmp_path):
        from medscholar.db.repo import upsert_run

        runtime, db = make_runtime(tmp_path, "rr7.db")
        critique = CritiqueResult(
            assessments=[], evidence_quality="中", gaps=["缺少 RCT"], suggestions=["补检索"]
        ).to_dict()
        save_run_step("keep3", "plan", {"plan": ResearchPlan(topic_zh="课题").to_dict()}, db=db)
        save_run_step("keep3", "reflect", {"critique": critique}, db=db)
        save_run_step("keep3", "synthesize", {"draft": "# 已写好的草稿\n正文"}, db=db)
        upsert_run("keep3", topic="课题", phase="synthesize", status="interrupted", db=db)

        handle = await runtime.resume("keep3")
        assert handle.state.resumed_from == "synthesize"
        assert handle.state.critique is not None
        assert handle.state.critique.gaps == ["缺少 RCT"]
        assert handle.state.draft.startswith("# 已写好的草稿")


class TestResultRoundTrip:
    """快照要能完整往返，否则续跑会丢内容。"""

    def test_critique_round_trip(self):
        original = CritiqueResult(
            assessments=[],
            evidence_quality="高",
            gaps=["g1", "g2"],
            suggestions=["s1"],
            used_llm=True,
        )
        restored = CritiqueResult.from_dict(original.to_dict())
        assert restored.evidence_quality == "高"
        assert restored.gaps == ["g1", "g2"]
        assert restored.suggestions == ["s1"]
        assert restored.used_llm is True

    def test_review_round_trip(self):
        original = ReviewResult(
            verdict="revise",
            score=6.5,
            issues=[{"severity": "high", "detail": "引用越界"}],
            strengths=["结构清晰"],
            invalid_citations=[99],
        )
        restored = ReviewResult.from_dict(original.to_dict())
        assert restored.verdict == "revise"
        assert restored.score == 6.5
        assert restored.issues[0]["detail"] == "引用越界"
        assert restored.invalid_citations == [99]

    def test_garbage_payloads_are_safe(self):
        assert CritiqueResult.from_dict("不是字典").evidence_quality == "中"
        assert ReviewResult.from_dict(None).verdict == "pass"
        assert ReviewResult.from_dict({"score": "abc"}).score == 0.0


class TestGraphPhaseSkipping:
    """resumed_from 必须让 graph 真的跳过已经做完的阶段。"""

    async def test_run_skips_plan_when_restored(self, tmp_path, monkeypatch):
        """续跑时绝不能再调一次 LLM 重新规划。"""
        from medscholar.agent.graph import ResearchGraph

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "g1.db", config=config)
        graph = ResearchGraph(config=config, db=db)

        called = {"plan": 0}

        async def boom(state, *, emit=None, feedback=""):
            called["plan"] += 1
            raise AssertionError("续跑时不应重新规划")

        monkeypatch.setattr(graph, "plan", boom)

        state = AgentState(topic="课题", offline=True, resumed_from="plan")
        state.plan = ResearchPlan(
            topic_zh="复用来的课题",
            queries=[PlanQuery(query="preset query")],
            outline=[],
        )
        events: list[tuple[str, dict]] = []

        async def emit(kind: str, data: dict) -> None:
            events.append((kind, data))

        await graph.run(state, emit=emit)
        assert called["plan"] == 0, "规划节点被重复调用了"
        assert state.plan.topic_zh == "复用来的课题"

    async def test_run_skips_execute_when_papers_restored(self, tmp_path, monkeypatch):
        """续跑时不得重新联网检索已入库的文献。"""
        from medscholar.agent.graph import ResearchGraph
        from medscholar.models import Paper

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        db = Database(tmp_path / "g2.db", config=config)
        graph = ResearchGraph(config=config, db=db)

        called = {"execute": 0}

        async def boom(state, *, emit=None):
            called["execute"] += 1
            raise AssertionError("续跑时不应重新检索")

        monkeypatch.setattr(graph, "execute", boom)

        state = AgentState(topic="课题", offline=True, resumed_from="execute")
        state.plan = ResearchPlan(topic_zh="课题")
        state.papers = [Paper(title="已入库论文", source="pubmed")]

        async def emit(kind: str, data: dict) -> None:
            return None

        await graph.run(state, emit=emit)
        assert called["execute"] == 0, "执行节点被重复调用了"
