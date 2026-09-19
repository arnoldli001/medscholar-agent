"""Agent 工作流路由：``tags=["Agent"]``（运行 / SSE / 审批 / 续跑 / 历史）+ ``tags=["产物"]``。

运行本身由 ``medscholar.agent.runtime`` 的**进程级单例**驱动（见 deps.get_runtime），
路由只负责把它暴露成 HTTP：启动、推流、审批、取消、查历史、按阶段快照续跑。

历史记录为什么重要：服务重启会清空内存中的运行，而界面仍在等一个永远不会到来的
草稿；有了数据库里的运行记录与阶段快照，前端才能明确显示"这次运行在综合阶段被中断"
并给出「继续」按钮。
"""

from __future__ import annotations

import asyncio
import json
import logging
from typing import Any, AsyncIterator

from fastapi import APIRouter, HTTPException, Query, Request
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field

from ...agent.state import Phase
from ...cite import detect_style
from ...db import repo
from ..deps import get_db, get_runtime, is_resumable, sse_headers

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================ 请求体模型
class AgentRunRequest(BaseModel):
    topic: str = Field(min_length=1)
    sources: list[str] | None = None
    session_id: int | None = None
    project_id: int | None = None
    citation_style: str = "gb7714"
    require_approval: bool = True
    offline: bool | None = None
    #: 综述正文目标字数范围（中文字符，不含参考文献）。留空则用 config 默认值。
    review_min_chars: int | None = Field(default=None, ge=0, le=60000)
    review_max_chars: int | None = Field(default=None, ge=0, le=60000)


class ApprovalRequest(BaseModel):
    decision: str = "approve"
    feedback: str = ""


# ============================================================ 运行生命周期
@router.post("/api/agent/run", tags=["Agent"])
async def agent_run(req: AgentRunRequest) -> dict[str, Any]:
    try:
        handle = await get_runtime().start(
            topic=req.topic,
            sources=req.sources,
            session_id=req.session_id,
            project_id=req.project_id,
            citation_style=detect_style(req.citation_style),
            require_approval=req.require_approval,
            offline=req.offline,
            review_min_chars=req.review_min_chars,
            review_max_chars=req.review_max_chars,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {"run_id": handle.run_id, "session_id": handle.state.session_id}


@router.get("/api/agent/stream/{run_id}", tags=["Agent"])
async def agent_stream(run_id: str, request: Request) -> StreamingResponse:
    async def generator() -> AsyncIterator[bytes]:
        try:
            async for event in get_runtime().stream(run_id):
                if await request.is_disconnected():
                    break
                payload = json.dumps(event.data, ensure_ascii=False, default=str)
                yield f"event: {event.type}\ndata: {payload}\n\n".encode("utf-8")
        except asyncio.CancelledError:  # 客户端断开
            raise
        except Exception as exc:  # pragma: no cover
            logger.exception("SSE 推送异常")
            data = json.dumps({"message": str(exc)}, ensure_ascii=False)
            yield f"event: error\ndata: {data}\n\n".encode("utf-8")
            yield b"event: done\ndata: {}\n\n"

    return StreamingResponse(
        generator(),
        media_type="text/event-stream",
        headers=sse_headers(),
    )


@router.post("/api/agent/approve/{run_id}", tags=["Agent"])
async def agent_approve(run_id: str, req: ApprovalRequest) -> dict[str, Any]:
    ok = await get_runtime().approve(run_id, req.decision, req.feedback)
    if not ok:
        raise HTTPException(
            status_code=409, detail="该运行不存在、已完成或已经审批过"
        )
    return {"ok": True, "decision": req.decision}


@router.post("/api/agent/cancel/{run_id}", tags=["Agent"])
async def agent_cancel(run_id: str) -> dict[str, Any]:
    return {"ok": await get_runtime().cancel(run_id)}


@router.get("/api/agent/runs", tags=["Agent"])
async def agent_runs(limit: int = Query(20, ge=1, le=200)) -> dict[str, Any]:
    """运行列表：内存中的实时运行 + 数据库里的历史记录（含被中断的）。

    历史记录很重要：服务重启会清空内存中的运行，而用户界面仍在等一个
    永远不会到来的草稿。有了历史，前端就能明确显示"这次运行在综合阶段被中断"。
    """
    live = {r["run_id"]: r for r in get_runtime().list_runs(limit=limit)}
    db = get_db()
    history = await asyncio.to_thread(repo.list_runs, limit=limit, db=db)
    steps_map = await asyncio.to_thread(
        repo.run_step_phases, [row["run_id"] for row in history], db=db
    )

    merged: list[dict[str, Any]] = []
    for row in history:
        run_id = row["run_id"]
        if run_id in live:
            item = dict(live.pop(run_id))
            item["artifact_title"] = row.get("artifact_title")
            merged.append(item)
        else:
            merged.append(
                {
                    "run_id": run_id,
                    "topic": row["topic"],
                    "phase": row["phase"],
                    "phase_label": Phase(row["phase"]).label
                    if row["phase"] in {p.value for p in Phase}
                    else row["phase"],
                    "status": row["status"],
                    "papers": row["papers"],
                    "citations": row["citations"] or 0,
                    "artifact_id": row["artifact_id"],
                    "artifact_title": row.get("artifact_title"),
                    "created_at": row["created_at"],
                    "updated_at": row["updated_at"],
                    "errors": [row["error"]] if row["error"] else [],
                    "interrupted": row["status"] == "interrupted",
                    "from_history": True,
                    "done_phases": steps_map.get(run_id, []),
                    "resumable": is_resumable(
                        row["status"], row["phase"], steps_map.get(run_id, [])
                    ),
                }
            )
    # 内存里有、数据库里还没有的（极少见）也补上
    merged.extend(live.values())
    return {"runs": merged[:limit]}


@router.post("/api/agent/resume/{run_id}", tags=["Agent"])
async def agent_resume(run_id: str) -> dict[str, Any]:
    """从阶段快照继续一次被中断的运行。

    只重跑没做完的阶段：已经检索入库的文献、已写好的草稿都会直接复用，
    不必因为一次断线就从头再来（检索 + 撰写通常要几十分钟）。
    """
    try:
        handle = await get_runtime().resume(run_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "run_id": handle.run_id,
        "session_id": handle.state.session_id,
        "resumed_from": handle.state.resumed_from,
        "papers": len(handle.state.papers),
    }


@router.get("/api/agent/steps/{run_id}", tags=["Agent"])
async def agent_steps(run_id: str) -> dict[str, Any]:
    """列出一次运行各阶段的快照摘要（供界面展示"已完成了哪些内容"）。"""
    steps = await asyncio.to_thread(repo.list_run_steps, run_id, db=get_db())
    out: list[dict[str, Any]] = []
    for step in steps:
        data = step.get("data") or {}
        plan = data.get("plan") if isinstance(data.get("plan"), dict) else {}
        draft = data.get("draft") if isinstance(data.get("draft"), str) else ""
        out.append(
            {
                "phase": step["phase"],
                "created_at": step["created_at"],
                "queries": len(plan.get("queries") or []),
                "outline": len(plan.get("outline") or []),
                "papers": len(data.get("paper_ids") or []),
                "has_critique": bool(data.get("critique")),
                "draft_chars": len(draft),
                "artifact_id": data.get("artifact_id"),
            }
        )
    return {"run_id": run_id, "steps": out}


@router.get("/api/agent/latest", tags=["Agent"])
async def agent_latest() -> dict[str, Any]:
    """最近一次可续跑或已完成的运行 + 其阶段摘要，供页面加载时恢复现场。"""
    db = get_db()
    runs = await asyncio.to_thread(repo.list_runs, limit=5, db=db)
    if not runs:
        return {"run": None, "steps": []}
    steps_map = await asyncio.to_thread(
        repo.run_step_phases, [r["run_id"] for r in runs], db=db
    )
    chosen = next(
        (
            r
            for r in runs
            if is_resumable(r["status"], r["phase"], steps_map.get(r["run_id"], []))
        ),
        runs[0],
    )
    steps = await asyncio.to_thread(repo.list_run_steps, chosen["run_id"], db=db)
    summary = [
        {
            "phase": s["phase"],
            "created_at": s["created_at"],
            "papers": len((s.get("data") or {}).get("paper_ids") or []),
            "draft_chars": len((s.get("data") or {}).get("draft") or ""),
            "has_critique": bool((s.get("data") or {}).get("critique")),
        }
        for s in steps
    ]
    return {
        "run": {
            "run_id": chosen["run_id"],
            "topic": chosen["topic"],
            "phase": chosen["phase"],
            "status": chosen["status"],
            "papers": chosen["papers"],
            "artifact_id": chosen["artifact_id"],
            "artifact_title": chosen.get("artifact_title"),
            "updated_at": chosen["updated_at"],
            "done_phases": steps_map.get(chosen["run_id"], []),
            "resumable": is_resumable(
                chosen["status"],
                chosen["phase"],
                steps_map.get(chosen["run_id"], []),
            ),
        },
        "steps": summary,
    }


@router.get("/api/agent/runs/{run_id}", tags=["Agent"])
async def agent_run_detail(run_id: str) -> dict[str, Any]:
    handle = get_runtime().get(run_id)
    if handle is not None:
        return {**handle.to_dict(), "summary": handle.state.summary(), "errors": handle.state.errors}
    # 内存里没有（服务重启过）→ 回退到数据库记录，让前端能说明发生了什么
    record = await asyncio.to_thread(repo.get_run, run_id, db=get_db())
    if record is None:
        raise HTTPException(status_code=404, detail=f"运行不存在：{run_id}")
    return {
        "run_id": record["run_id"],
        "topic": record["topic"],
        "phase": record["phase"],
        "status": record["status"],
        "papers": record["papers"],
        "artifact_id": record["artifact_id"],
        "created_at": record["created_at"],
        "errors": [record["error"]] if record["error"] else [],
        "interrupted": record["status"] == "interrupted",
        "from_history": True,
    }


# ============================================================ 产物
@router.get("/api/artifacts", tags=["产物"])
async def artifacts_list(
    session_id: int | None = None, limit: int = Query(50, ge=1, le=500)
) -> dict[str, Any]:
    return {"items": await asyncio.to_thread(repo.list_artifacts, session_id=session_id, limit=limit)}


@router.get("/api/artifacts/{artifact_id}", tags=["产物"])
async def artifact_detail(artifact_id: int) -> dict[str, Any]:
    artifact = await asyncio.to_thread(repo.get_artifact, artifact_id)
    if artifact is None:
        raise HTTPException(status_code=404, detail=f"未找到产物 id={artifact_id}")
    return artifact


@router.delete("/api/artifacts/{artifact_id}", tags=["产物"])
async def artifact_delete(artifact_id: int) -> dict[str, Any]:
    return {"ok": await asyncio.to_thread(repo.delete_artifact, artifact_id)}
