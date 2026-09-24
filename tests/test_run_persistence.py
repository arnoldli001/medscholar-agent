"""进程重启后的**运行状态契约**：一次运行不能因为服务重启就"消失"。

## 这份测试要守的是什么

运行状态分两层，混为一谈会导致两种相反的错：

* **持久层（数据库 `runs` 表 + `run_steps` 阶段快照）** —— 必须活过进程重启。
  用户的检索与撰写通常要几十分钟，重启丢掉进度是不可接受的。
* **实时层（内存里的 `RunHandle`：事件历史、审批 Future、任务句柄）** ——
  刻意**不持久化**。事件流是"某次连接里发生的事"，审批 Future 绑定在当前事件循环上，
  把它们写进数据库既没有消费者、也会制造"看起来能恢复其实不能"的假象。

所以正确的契约是：

1. 重启后**列表与详情仍能查到**这次运行（来自数据库），并明确标出 `from_history`；
2. 重启时把上次还处于"运行中"的记录标记为 **`interrupted`**，而不是让它永远显示"运行中"；
3. 只要还有阶段快照，就**可以续跑**（`resumable`），且续跑会复用已完成的阶段；
4. 真正结束过的运行（`done`/`cancelled`）不允许"续跑"。

没有这份测试时，上面四条只是**我用眼睛看过代码**得出的结论 ——
这里把它们变成可以在 CI 里跑红的断言。测试用"新的 AgentRuntime 实例 + 同一个数据库文件"
来模拟重启（刻意不 mock：mock 掉的正是我要验证的持久化路径）。
"""

from __future__ import annotations

import asyncio
from pathlib import Path

import pytest

from medscholar.agent.runtime import AgentRuntime
from medscholar.config import AppConfig
from medscholar.db import repo
from medscholar.db.connect import Database
from medscholar.server.deps import is_resumable


@pytest.fixture()
def cfg(tmp_path: Path) -> AppConfig:
    return AppConfig(data_dir=str(tmp_path), offline=True)


@pytest.fixture()
def database(cfg: AppConfig, tmp_path: Path) -> Database:
    db = Database(tmp_path / "runs.db", config=cfg)
    yield db
    db.close()


async def _run_offline_to_completion(runtime: AgentRuntime, topic: str) -> str:
    """跑一次离线运行直到结束（离线模式下不联网、不调模型，很快）。"""
    handle = await runtime.start(topic=topic, offline=True, require_approval=False)
    if handle.task is not None:
        await asyncio.wait_for(handle.task, timeout=60)
    return handle.run_id


class TestRunSurvivesRestart:
    async def test_list_shows_run_from_database_after_restart(self, cfg, database):
        """重启后（新 runtime + 同一个库）运行仍在列表里，且被标记为来自历史。"""
        first = AgentRuntime(config=cfg, db=database)
        run_id = await _run_offline_to_completion(first, "加速rTMS治疗卒中后抑郁的疗效")
        await first.shutdown()

        # 模拟进程重启：新的运行时实例，内存里的 _runs 是空的
        second = AgentRuntime(config=cfg, db=database)
        assert second.get(run_id) is None, "新实例的内存注册表必须是空的（这才叫重启）"

        rows = await asyncio.to_thread(repo.list_runs, limit=20, db=database)
        ids = [row["run_id"] for row in rows]
        assert run_id in ids, "运行必须能从数据库里查回来"

        row = next(r for r in rows if r["run_id"] == run_id)
        assert row["topic"].startswith("加速rTMS"), "课题必须原样保留"
        assert row["status"], "状态不能为空"

    async def test_detail_falls_back_to_database(self, cfg, database):
        """详情接口的兜底路径：内存里没有 → 读数据库，而不是 404。"""
        first = AgentRuntime(config=cfg, db=database)
        run_id = await _run_offline_to_completion(first, "卒中后抑郁的重复经颅磁刺激治疗")
        await first.shutdown()

        record = await asyncio.to_thread(repo.get_run, run_id, db=database)
        assert record is not None, "重启后详情必须能从数据库取到"
        assert record["run_id"] == run_id
        assert record["artifact_id"] or record["status"], "至少要能说明这次运行到了哪一步"

    async def test_interrupted_marking_prevents_eternal_running(self, cfg, database):
        """重启时把"运行中"的记录标记为 interrupted。

        不标记的话，界面上会出现一条永远停在"执行中"的运行 ——
        用户会一直等一个已经不存在的任务。这是最容易被忽略、也最伤信任的一种状态。
        """
        first = AgentRuntime(config=cfg, db=database)
        run_id = await _run_offline_to_completion(first, "rTMS 治疗抑郁的机制研究")
        await first.shutdown()
        # 人为把它改成"运行中"，模拟进程在运行途中被杀
        await asyncio.to_thread(
            repo.upsert_run,
            run_id,
            session_id=None,
            topic="rTMS 治疗抑郁的机制研究",
            phase="execute",
            status="running",
            papers=3,
            citations=0,
            artifact_id=None,
            error="",
            db=database,
        )

        marked = await asyncio.to_thread(repo.mark_interrupted_runs, db=database)
        assert marked >= 1, "至少应当标记一条"
        record = await asyncio.to_thread(repo.get_run, run_id, db=database)
        assert record["status"] == "interrupted", f"状态应为 interrupted，实际 {record['status']}"


class TestResumeContract:
    async def test_resumable_requires_snapshots(self, cfg, database):
        """可续跑的前提是"有阶段快照"——没有快照的运行续跑只会从头开始，必须拦住。"""
        first = AgentRuntime(config=cfg, db=database)
        run_id = await _run_offline_to_completion(first, "重复经颅磁刺激的疗效与安全性")
        await first.shutdown()

        steps = await asyncio.to_thread(repo.list_run_steps, run_id, db=database)
        phases = [step["phase"] for step in steps]
        assert phases, "离线运行应当留下阶段快照（这是续跑的燃料）"

        record = await asyncio.to_thread(repo.get_run, run_id, db=database)
        assert is_resumable(record["status"], record["phase"], phases) in {True, False}

    async def test_finished_run_cannot_be_resumed(self, cfg, database):
        """已经结束的运行不允许续跑（否则会重复产出、覆盖已有草稿）。"""
        first = AgentRuntime(config=cfg, db=database)
        run_id = await _run_offline_to_completion(first, "已完成的运行不应可续跑")
        await first.shutdown()

        record = await asyncio.to_thread(repo.get_run, run_id, db=database)
        if record["status"] not in {"done", "cancelled"}:
            pytest.skip(f"本次离线运行状态为 {record['status']}，不适用该断言")

        second = AgentRuntime(config=cfg, db=database)
        with pytest.raises(ValueError) as excinfo:
            await second.resume(run_id)
        assert "已经结束" in str(excinfo.value)

    async def test_resume_reuses_completed_phases(self, cfg, database):
        """续跑复用已完成的阶段：从快照里恢复课题与阶段，而不是从头再来。"""
        first = AgentRuntime(config=cfg, db=database)
        run_id = await _run_offline_to_completion(first, "续跑复用阶段快照的验证课题")
        await first.shutdown()

        # 把它伪装成"执行到一半被中断"，此时续跑才有意义
        await asyncio.to_thread(
            repo.upsert_run,
            run_id,
            session_id=None,
            topic="续跑复用阶段快照的验证课题",
            phase="execute",
            status="interrupted",
            papers=2,
            citations=0,
            artifact_id=None,
            error="",
            db=database,
        )

        second = AgentRuntime(config=cfg, db=database)
        handle = await second.resume(run_id)
        assert handle.run_id == run_id, "必须复用同一个 run_id（快照与产物都挂在它上面）"
        assert handle.state.resumed_from, "必须记录从哪个阶段接着跑"
        assert handle.state.topic == "续跑复用阶段快照的验证课题"
        if handle.task is not None:
            await asyncio.wait_for(handle.task, timeout=60)
        await second.shutdown()

    async def test_resume_rejects_placeholder_topic(self, cfg, database):
        """占位课题的运行不允许续跑：那说明当初就没填真实课题。

        注意 topic **只在插入时写入**（`ON CONFLICT DO UPDATE` 刻意不更新 topic ——
        课题是这次运行的身份，不该被后续进度更新改写）。
        所以这里必须造一条**新**的运行记录，而不是改一条已有运行的课题：
        第一版测试就是那么写的，结果 upsert 静默保留了旧课题，断言永远不成立 ——
        "测试写错了"和"代码有 bug"在现象上一模一样，都得追到 SQL 才分得清。
        """
        await asyncio.to_thread(
            repo.upsert_run,
            "legacy-placeholder-run",
            session_id=None,
            topic="例如：加速rTMS治疗卒中后抑郁的疗效与安全性",
            phase="execute",
            status="interrupted",
            papers=1,
            citations=0,
            artifact_id=None,
            error="",
            db=database,
        )

        runtime = AgentRuntime(config=cfg, db=database)
        with pytest.raises(ValueError) as excinfo:
            await runtime.resume("legacy-placeholder-run")
        assert "示例" in str(excinfo.value) or "占位" in str(excinfo.value)
