"""基础路由：健康探测、脱敏配置、统计、数据源与首页/favicon（页面路由不进 OpenAPI）。

/api/health 绝不能阻塞：立即返回上次探测结果（可能标记"检测中"），后台任务异步刷新。
"""

from __future__ import annotations

import asyncio
import time
from typing import Any

from fastapi import APIRouter
from fastapi.responses import FileResponse, JSONResponse, Response

from ... import __version__
from ...api import ALL_SOURCES
from ...embedding.pipeline import embedding_status
from ...llm.client import get_llm
from ..deps import WEB_DIR, get_config, get_db, get_registry, render_index

router = APIRouter()

# ------------------------------------------------------------------ 进程级探测状态
# 语义为"进程启动至今"：多次装配应用也共享同一份探测缓存，避免重复烧一次推理
probe_cache: dict[str, Any] = {"llm": None, "embedding": None, "at": 0.0}
probe_task: dict[str, "asyncio.Task[None] | None"] = {"task": None}
started_at = time.time()


async def _run_probes() -> None:
    cfg = get_config()
    try:
        llm_ok, llm_message = await get_llm(cfg).health()
    except Exception as exc:  # noqa: BLE001 - 探测失败只影响展示
        cfg = get_config()
        llm_ok, llm_message = False, f"{type(exc).__name__}: {exc}"
    try:
        embedding = await embedding_status(cfg)
    except Exception as exc:  # noqa: BLE001
        embedding = {
            "ok": False,
            "provider": cfg.embedding.provider,
            "model": cfg.embedding.model,
            "dim": cfg.embedding.dim,
            "message": str(exc),
        }
    probe_cache["llm"] = {
        "ok": llm_ok,
        "message": llm_message,
        "provider": cfg.llm.provider,
        "model": cfg.llm.model,
    }
    probe_cache["embedding"] = embedding
    probe_cache["at"] = time.time()


def _schedule_probe(force: bool) -> None:
    """按需在后台刷新探测结果（同一时刻最多一个探测任务）。"""
    task = probe_task["task"]
    if task is not None and not task.done():
        return
    if not force and probe_cache["at"] and time.time() - probe_cache["at"] < 120.0:
        return
    probe_task["task"] = asyncio.create_task(_run_probes(), name="health-probe")


# ------------------------------------------------------------------ 健康与配置
@router.get("/api/ping", tags=["基础"])
async def ping() -> dict[str, Any]:
    """极轻量的连通性探测：不读数据库、不碰模型，用于前端判断后端是否在线。"""
    return {
        "ok": True,
        "version": __version__,
        "uptime_s": round(time.time() - started_at, 1),
    }


@router.get("/api/health", tags=["基础"])
async def health(probe: bool = False) -> dict[str, Any]:
    cfg = get_config()
    db = get_db()
    _schedule_probe(force=probe)

    llm_cached = probe_cache["llm"]
    embed_cached = probe_cache["embedding"]
    return {
        "ok": True,
        "version": __version__,
        "offline": cfg.offline,
        "uptime_s": round(time.time() - started_at, 1),
        # ok 为 None 表示"正在后台检测"，前端据此显示"检测中"而非"不可用"
        "llm": llm_cached
        or {
            "ok": None,
            "provider": cfg.llm.provider,
            "model": cfg.llm.model,
            "message": "正在后台检测模型可用性…",
        },
        "embedding": embed_cached
        or {
            "ok": None,
            "provider": cfg.embedding.provider,
            "model": cfg.embedding.model,
            "dim": cfg.embedding.dim,
            "message": "正在后台检测嵌入模型…",
        },
        "database": await asyncio.to_thread(db.stats),
        "sources": get_registry(cfg).describe(),
    }


@router.get("/api/config", tags=["基础"])
async def config_view() -> dict[str, Any]:
    return get_config().public_dict()


@router.get("/api/stats", tags=["基础"])
async def stats() -> dict[str, Any]:
    return get_db().stats()


@router.get("/api/sources", tags=["基础"])
async def sources() -> dict[str, Any]:
    cfg = get_config()
    return {"items": get_registry(cfg).describe(), "all": list(ALL_SOURCES)}


@router.get("/api/metrics", tags=["基础"])
async def metrics() -> dict[str, Any]:
    """运行指标：LLM 用量成本、熔断状态、注入扫描等聚合数字。

    只暴露聚合数字，不含文献内容或提示词正文（指标接口常被贴到 issue/群里，不能泄漏用户数据）。
    """
    from ...llm.transport import breaker_stats
    from ...platform.cache import cache_stats
    from ...platform.observability import LEDGER
    from ...retrieval import injection_scan_stats

    return {
        "llm": LEDGER.summary(),
        "llm_recent": [item.to_dict() for item in LEDGER.recent(10)],
        "breakers": breaker_stats(),
        "caches": cache_stats(),
        "injection": injection_scan_stats(),
        "uptime_s": round(time.time() - started_at, 1),
    }


@router.get("/api/metrics/schema", tags=["基础"])
async def schema_status() -> dict[str, Any]:
    """数据库 schema 版本与待执行迁移（只读）；单独路径因需读 DB 元表，比内存指标重。"""
    from ...db.migrate import migration_status

    db = get_db()

    def _status() -> dict[str, Any]:
        try:
            return migration_status(db.conn)
        except Exception as exc:  # noqa: BLE001 - 状态查询失败不应影响服务
            return {"available": False, "reason": f"{type(exc).__name__}: {exc}"}

    return await asyncio.to_thread(_status)


# ------------------------------------------------------------------ 首页与图标
# 页面路由不进 OpenAPI；首页经 render_index 注入 mtime 版本号并强制 no-store
@router.get("/", include_in_schema=False)
async def index() -> Any:
    index_file = WEB_DIR / "index.html"
    if not index_file.is_file():
        return JSONResponse(
            status_code=503,
            content={
                "detail": "前端文件缺失：medscholar/web/index.html 不存在。",
                "api_docs": "/docs",
            },
        )
    html = await asyncio.to_thread(render_index, index_file)
    return Response(
        content=html,
        media_type="text/html; charset=utf-8",
        headers={"Cache-Control": "no-store, must-revalidate"},
    )


@router.get("/favicon.ico", include_in_schema=False)
async def favicon() -> Any:
    icon = WEB_DIR / "favicon.ico"
    if icon.is_file():
        return FileResponse(icon)
    return JSONResponse(status_code=204, content=None)
