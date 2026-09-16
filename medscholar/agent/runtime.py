"""Agent 运行时：后台运行、事件流与人机审批。

为什么需要这一层：需求里的 **Plan → 用户审批** 是一个真正的暂停点。
如果直接在 HTTP 请求里 await，客户端一断线整个任务就没了。因此这里

* 把工作流跑在**后台 task** 里；
* 所有事件按顺序追加到 ``history``，并用一个 ``asyncio.Event`` 唤醒订阅者
  —— 事件列表是**唯一事实来源**，因此断线重连可以完整补播，也不会丢事件；
* 审批用一个 ``Future`` 表示，由 ``POST /api/agent/approve/{run_id}`` 兑现。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from typing import Any, AsyncIterator, Sequence

from ..config import AppConfig, get_config
from ..db.connect import Database, get_db
from ..db.repo import add_message
from ..db.repo import create_session as db_create_session
from ..db.repo import get_papers_by_ids as db_get_papers_by_ids
from ..db.repo import get_run as db_get_run
from ..db.repo import list_run_steps as db_list_run_steps
from ..db.repo import save_run_step as db_save_run_step
from ..db.repo import upsert_run as db_upsert_run
from .graph import ResearchGraph, _phase_index
from .state import (
    AgentEvent,
    AgentState,
    CritiqueResult,
    Phase,
    ResearchPlan,
    ReviewResult,
    coerce_plan_payload,
    new_run_id,
)

logger = logging.getLogger(__name__)

__all__ = ["RunHandle", "AgentRuntime", "get_runtime"]

#: 终止事件类型
_TERMINAL = {"done"}

#: 已完成运行在内存中的保留时长（秒）
_RETENTION_SECONDS = 3600

#: 事件轮询间隔（秒）与 SSE 心跳间隔（秒）
_POLL_SECONDS = 0.2
_HEARTBEAT_SECONDS = 15.0

#: 典型的界面占位提示前缀。真实的研究课题不会以这些词开头，
#: 而一旦把占位提示当成课题，就会跑一次十几分钟、结果毫无意义的研究流程。
_PLACEHOLDER_PREFIXES = (
    "例如：", "例如:", "示例：", "示例:", "比如：", "比如:", "如：", "如:",
    "请输入", "请在此输入", "在此输入", "在这里输入", "输入研究课题", "输入课题",
)


def normalize_review_chars(
    min_chars: int | None, max_chars: int | None, agent: Any
) -> tuple[int, int]:
    """把界面传来的字数范围夹到合理区间，并保证 min < max。

    兜底规则（不信任前端）：范围缺失或为 0 时用配置默认值；上下限颠倒时交换；
    越界时夹到 800~40000 字——比 800 还短写不出综述，比 4 万字一次本地
    生成不现实（8B 本地模型约 45 tok/s，4 万字要跑很久）。

    **总是返回具体的有效区间**，调用方可以直接使用。
    """
    low = int(min_chars) if min_chars else 0
    high = int(max_chars) if max_chars else 0
    if low <= 0:
        low = int(getattr(agent, "review_min_chars", 0) or 0)
    if high <= 0:
        high = int(getattr(agent, "review_max_chars", 0) or 0)
    if low <= 0 and high <= 0:
        low, high = 4000, 8000
    if low <= 0:
        low = int(high * 0.6)
    if high <= 0:
        high = int(low * 1.6)
    if low > high:
        low, high = high, low
    # 先把 high 夹进上限，再让 low 留在 high 之下，避免两者互相顶出边界
    high = max(1000, min(high, 40000))
    low = max(800, min(low, high - 200))
    return low, high


def looks_like_placeholder(topic: str) -> bool:
    """判断课题是否其实是界面上的占位提示文本。

    实测出现过课题为「例如：加速rTMS治疗卒中后抑郁的疗效与安全性」——
    正是前端输入框的 placeholder 原文（可能来自浏览器 autofill 或粘贴）。
    前端已做防御；这里再做一道，因为 CLI 与 MCP 也走同一个入口。

    >>> looks_like_placeholder("例如：rTMS 治疗抑郁")
    True
    >>> looks_like_placeholder("加速rTMS治疗卒中后抑郁的疗效与安全性")
    False
    """
    text = (topic or "").strip()
    if not text:
        return True
    return any(text.startswith(prefix) for prefix in _PLACEHOLDER_PREFIXES)


@dataclass
class RunHandle:
    """一次 Agent 运行的运行时句柄。"""

    run_id: str
    state: AgentState
    history: list[AgentEvent] = field(default_factory=list)
    approval: asyncio.Future | None = None
    task: asyncio.Task | None = None
    status: str = "running"      # running | awaiting_approval | done | cancelled | error
    created_at: float = field(default_factory=time.time)
    closed: bool = False

    @property
    def phase(self) -> str:
        return self.state.phase.value

    def push(self, event: AgentEvent) -> None:
        """追加事件（唯一写入点；订阅方按 ``history`` 轮询消费）。"""
        self.history.append(event)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "topic": self.state.topic,
            "phase": self.phase,
            "phase_label": self.state.phase.label,
            "status": self.status,
            "papers": len(self.state.papers),
            "elapsed_ms": self.state.elapsed_ms,
            "created_at": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.created_at)),
            "errors": list(self.state.errors),
        }


class AgentRuntime:
    """管理所有 Agent 运行实例。"""

    def __init__(self, *, config: AppConfig | None = None, db: Database | None = None) -> None:
        self.config = config or get_config()
        self.db = db or get_db()
        self._runs: dict[str, RunHandle] = {}

    # ---------------------------------------------------------------- 启动
    async def start(
        self,
        *,
        topic: str,
        sources: Sequence[str] | None = None,
        session_id: int | None = None,
        project_id: int | None = None,
        citation_style: str = "gb7714",
        require_approval: bool | None = None,
        offline: bool | None = None,
        new_session: bool = True,
        review_min_chars: int | None = None,
        review_max_chars: int | None = None,
    ) -> RunHandle:
        """创建并启动一次运行。"""
        topic = (topic or "").strip()
        if not topic:
            raise ValueError("课题不能为空")
        if looks_like_placeholder(topic):
            raise ValueError(
                f"课题「{topic[:40]}」看起来是界面上的示例提示，而不是真实的研究课题。"
                "请填写具体的研究问题后重试（例如：加速rTMS治疗卒中后抑郁的疗效与安全性）。"
            )

        need_approval = (
            self.config.agent.require_approval if require_approval is None else require_approval
        )
        is_offline = self.config.offline if offline is None else offline
        want_min, want_max = normalize_review_chars(
            review_min_chars, review_max_chars, self.config.agent
        )

        if session_id is None and new_session:
            title = topic[:40] + ("…" if len(topic) > 40 else "")
            session_id = db_create_session(title=title, topic=topic, db=self.db)

        state = AgentState(
            topic=topic,
            run_id=new_run_id(),
            session_id=session_id,
            project_id=project_id,
            sources=list(sources or []),
            offline=is_offline,
            citation_style=citation_style,
            review_min_chars=want_min,
            review_max_chars=want_max,
        )

        handle = RunHandle(run_id=state.run_id, state=state)
        if need_approval:
            handle.approval = asyncio.get_running_loop().create_future()

        # 每次运行可有自己的审批/离线开关，因此复制一份配置
        run_config = self.config.model_copy(
            update={
                "offline": is_offline,
                "agent": self.config.agent.model_copy(
                    update={"require_approval": need_approval}
                ),
            }
        )

        if session_id is not None:
            add_message(session_id, "user", topic, meta={"run_id": state.run_id}, db=self.db)

        handle.task = asyncio.create_task(
            self._execute(handle, run_config), name=f"medscholar-run-{state.run_id}"
        )

        self._runs[state.run_id] = handle
        self._prune()
        # 落库：服务重启后仍能查到"这次运行存在过、走到哪一步"
        try:
            db_upsert_run(
                state.run_id,
                session_id=session_id,
                topic=topic,
                phase=state.phase.value,
                status=handle.status,
                db=self.db,
            )
        except Exception as exc:  # pragma: no cover - 记录失败不应影响运行
            logger.debug("运行落库失败：%s", exc)
        logger.info("启动 Agent 运行 %s：%s", state.run_id, topic[:60])
        return handle

    async def resume(self, run_id: str) -> RunHandle:
        """从已保存的阶段快照继续一次被中断的运行。

        只会重跑"没做完"的阶段：规划/检索/评估/撰写/审查中已经完成的直接复用。
        这也是用户最需要的：检索和撰写很贵，不能因为一次连接断开就全部重来。
        """
        row = await asyncio.to_thread(db_get_run, run_id, db=self.db)
        if not row:
            raise ValueError(f"运行 {run_id} 不存在，无法继续。")
        if row.get("status") in {"done", "cancelled"}:
            raise ValueError(f"运行 {run_id} 已经结束（{row.get('status')}），无需继续。")

        topic = str(row.get("topic") or "").strip()
        if not topic:
            raise ValueError(f"运行 {run_id} 没有记录课题，无法继续。")
        if looks_like_placeholder(topic):
            raise ValueError(f"运行 {run_id} 的课题是界面示例提示，请重新发起研究。")

        steps = await asyncio.to_thread(db_list_run_steps, run_id, db=self.db)
        if not steps:
            raise ValueError(
                f"运行 {run_id} 没有可用的阶段快照（可能是旧版本产生的运行），请重新发起研究。"
            )

        done_phase = str(steps[-1]["phase"])
        merged: dict[str, Any] = {}
        for step in steps:
            data = step.get("data") or {}
            if isinstance(data, dict):
                merged.update(data)

        session_id = row.get("session_id")
        state = AgentState(
            topic=topic,
            run_id=run_id,  # 复用同一个 run_id，快照与产物都挂在它上面
            session_id=session_id,
            sources=[],
            offline=self.config.offline,
            citation_style=str(row.get("citation_style") or "gb7714"),
            resumed_from=done_phase,
        )

        plan_payload = merged.get("plan")
        if isinstance(plan_payload, dict) and plan_payload:
            state.plan = ResearchPlan.from_dict(coerce_plan_payload(plan_payload))

        paper_ids = [int(i) for i in (merged.get("paper_ids") or []) if str(i).isdigit()]
        if paper_ids:
            papers = await asyncio.to_thread(
                db_get_papers_by_ids, paper_ids, db=self.db
            )
            # 保持与快照一致的顺序，引用编号才不会错位
            state.papers = [papers[i] for i in paper_ids if i in papers]

        critique_payload = merged.get("critique")
        if isinstance(critique_payload, dict) and critique_payload:
            state.critique = CritiqueResult.from_dict(critique_payload)

        if isinstance(merged.get("draft"), str):
            state.draft = merged["draft"]

        review_payload = merged.get("review")
        if isinstance(review_payload, dict) and review_payload:
            state.review = ReviewResult.from_dict(review_payload)

        artifact_id = merged.get("artifact_id") or row.get("artifact_id")
        if artifact_id:
            state.artifact_id = int(artifact_id)

        for message in (merged.get("errors") or [])[:5]:
            state.add_error(str(message))

        if state.papers:
            state.citation_map = {
                index: paper for index, paper in enumerate(state.papers, start=1)
            }
            state.paper_ids = [p.paper_id or 0 for p in state.papers]

        need_approval = (
            self.config.agent.require_approval
            and _phase_index(done_phase) <= _phase_index("plan")
        )
        handle = RunHandle(run_id=run_id, state=state)
        if need_approval:
            handle.approval = asyncio.get_running_loop().create_future()

        run_config = self.config.model_copy(
            update={
                "offline": state.offline,
                "agent": self.config.agent.model_copy(
                    update={"require_approval": need_approval}
                ),
            }
        )

        handle.task = asyncio.create_task(
            self._execute(handle, run_config), name=f"medscholar-resume-{run_id}"
        )
        self._runs[run_id] = handle
        self._prune()
        await self._persist(handle)
        logger.info(
            "继续运行 %s：从「%s」之后接着跑（已有 %d 篇文献）",
            run_id,
            done_phase,
            len(state.papers),
        )
        return handle

    async def _execute(self, handle: RunHandle, run_config: AppConfig) -> None:
        """后台执行工作流，把事件推入历史。"""

        async def emit(event_type: str, data: dict[str, Any]) -> None:
            # 阶段快照只落库，不进事件流：里面可能有上万字的草稿
            if event_type == "step":
                await self._save_step(handle, data)
                return
            if event_type == "awaiting_approval":
                handle.status = "awaiting_approval"
            handle.push(AgentEvent(type=event_type, data=data))
            # 每个阶段切换都同步落库，服务重启后可追溯中断位置
            if event_type == "phase":
                await self._persist(handle)
            elif event_type in {"artifact", "done"}:
                await self._persist(handle, data)

        async def approval() -> tuple[str, str]:
            assert handle.approval is not None
            try:
                decision, feedback = await handle.approval
            except asyncio.CancelledError:
                return "cancel", ""
            handle.status = "running"
            return decision, feedback

        graph = ResearchGraph(config=run_config, db=self.db)
        try:
            await graph.run(
                handle.state,
                emit=emit,
                approval=approval if handle.approval is not None else None,
            )
            if handle.status != "cancelled":
                handle.status = "error" if handle.state.phase == Phase.ERROR else "done"
        except asyncio.CancelledError:
            handle.status = "cancelled"
            handle.state.phase = Phase.CANCELLED
            handle.push(
                AgentEvent(
                    type="done",
                    data={
                        "run_id": handle.run_id,
                        "phase": "cancelled",
                        "summary": handle.state.summary(),
                    },
                )
            )
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("运行 %s 异常终止", handle.run_id)
            handle.status = "error"
            handle.state.add_error(f"{type(exc).__name__}: {exc}")
            handle.push(AgentEvent(type="error", data={"message": str(exc)}))
            handle.push(
                AgentEvent(
                    type="done",
                    data={
                        "run_id": handle.run_id,
                        "phase": "error",
                        "summary": handle.state.summary(),
                    },
                )
            )
        finally:
            await graph.close()
            handle.closed = True
            await self._persist(handle, None, final=True)

    async def _save_step(self, handle: RunHandle, data: dict[str, Any]) -> None:
        """把阶段快照写库（失败不影响运行）。"""
        phase = str(data.get("phase") or "")
        snapshot = data.get("snapshot")
        if not phase or not isinstance(snapshot, dict):
            return
        try:
            await asyncio.to_thread(
                db_save_run_step, handle.run_id, phase, snapshot, db=self.db
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("阶段快照落库失败（%s）：%s", phase, exc)

    async def _persist(
        self,
        handle: RunHandle,
        payload: dict[str, Any] | None = None,
        *,
        final: bool = False,
    ) -> None:
        """把运行进度写入数据库（失败不影响运行本身）。"""
        state = handle.state
        artifact_id = None
        if payload and payload.get("artifact_id"):
            artifact_id = payload["artifact_id"]
        elif state.artifact_id:
            artifact_id = state.artifact_id
        try:
            await asyncio.to_thread(
                db_upsert_run,
                handle.run_id,
                session_id=state.session_id,
                topic=state.topic,
                phase=state.phase.value,
                status=handle.status,
                papers=len(state.papers),
                citations=len(state.citation_map),
                artifact_id=artifact_id,
                error="; ".join(state.errors[:3]),
                db=self.db,
            )
        except Exception as exc:  # pragma: no cover
            logger.debug("运行落库失败（%s）：%s", "final" if final else "progress", exc)

    # ---------------------------------------------------------------- 审批
    async def approve(
        self, run_id: str, decision: str = "approve", feedback: str = ""
    ) -> bool:
        """兑现审批 Future。返回是否成功（运行不存在或已结束则 False）。"""
        handle = self._runs.get(run_id)
        if handle is None or handle.approval is None or handle.approval.done():
            return False
        decision = (decision or "approve").strip().lower()
        if decision not in {"approve", "revise", "cancel"}:
            decision = "approve"
        handle.approval.set_result((decision, feedback or ""))
        handle.status = "running"
        return True

    async def cancel(self, run_id: str) -> bool:
        """取消运行（包括尚未审批的）。"""
        handle = self._runs.get(run_id)
        if handle is None:
            return False
        if handle.approval is not None and not handle.approval.done():
            handle.approval.set_result(("cancel", ""))
        if handle.task is not None and not handle.task.done():
            handle.task.cancel()
        handle.status = "cancelled"
        handle.state.phase = Phase.CANCELLED
        return True

    # ---------------------------------------------------------------- 查询
    def get(self, run_id: str) -> RunHandle | None:
        return self._runs.get(run_id)

    def list_runs(self, *, limit: int = 30) -> list[dict[str, Any]]:
        handles = sorted(self._runs.values(), key=lambda h: -h.created_at)[:limit]
        return [h.to_dict() for h in handles]

    async def stream(
        self, run_id: str, *, timeout: float = 3600.0
    ) -> AsyncIterator[AgentEvent]:
        """产出事件流：先补播历史，再实时跟随，直到 ``done``。

        实现说明：这里刻意**不用** ``asyncio.Event`` 做唤醒 —— Event/Future
        都会绑定到创建它的那个事件循环，一旦运行时被跨循环访问（测试里同时
        跑 ASGITransport 与 uvicorn、或未来接入多进程）就会静默死锁。
        改为按 200ms 轮询历史列表，代价可以忽略，但彻底消除了这类隐患。
        """
        handle = self._runs.get(run_id)
        if handle is None:
            yield AgentEvent(type="error", data={"message": f"运行不存在：{run_id}"})
            yield AgentEvent(type="done", data={"run_id": run_id, "phase": "error"})
            return

        delivered = 0
        deadline = time.monotonic() + timeout
        last_output = time.monotonic()

        while True:
            # 1) 消费所有已产生的事件
            if delivered < len(handle.history):
                event = handle.history[delivered]
                delivered += 1
                last_output = time.monotonic()
                yield event
                if event.type in _TERMINAL:
                    return
                continue

            # 2) 运行结束且事件已发完
            if handle.closed:
                yield AgentEvent(
                    type="done",
                    data={
                        "run_id": run_id,
                        "phase": handle.phase,
                        "summary": handle.state.summary(),
                    },
                )
                return

            if time.monotonic() >= deadline:
                yield AgentEvent(type="error", data={"message": "事件流超时"})
                yield AgentEvent(type="done", data={"run_id": run_id, "phase": handle.phase})
                return

            # 3) 等待新事件；长时间无输出时发心跳，避免代理掐断连接
            await asyncio.sleep(_POLL_SECONDS)
            if time.monotonic() - last_output >= _HEARTBEAT_SECONDS:
                last_output = time.monotonic()
                yield AgentEvent(type="ping", data={})

    # ---------------------------------------------------------------- 维护
    def _prune(self) -> None:
        now = time.time()
        stale = [
            run_id
            for run_id, handle in self._runs.items()
            if handle.closed and now - handle.created_at > _RETENTION_SECONDS
        ]
        for run_id in stale:
            self._runs.pop(run_id, None)

    async def shutdown(self) -> None:
        """取消全部运行（进程退出时调用）。"""
        for handle in list(self._runs.values()):
            if handle.task is not None and not handle.task.done():
                handle.task.cancel()
        await asyncio.sleep(0)


# --------------------------------------------------------------- 全局单例
_RUNTIME: AgentRuntime | None = None


def get_runtime(config: AppConfig | None = None, db: Database | None = None) -> AgentRuntime:
    """获取全局运行时单例。"""
    global _RUNTIME
    if _RUNTIME is None:
        _RUNTIME = AgentRuntime(config=config, db=db)
    return _RUNTIME
