"""引用、论文写作与学习闭环路由。

* ``tags=["引用"]``：引用样式清单、按样式格式化引用、导出文件；
* ``tags=["论文写作"]``：IMRaD 初稿生成与数字溯源校验、稿件列表/详情/复检；
* ``tags=["学习闭环"]``：赞/踩/质疑的反馈记录，以及"纠正立刻变成记忆"的可见进度。

三组放在一起是因为它们共用同一条链路：引用格式化既服务于写作，也服务于导出；
而用户的纠错记忆会在下次写作时被带进提示词（不需要重新训练模型）。
"""

from __future__ import annotations

import asyncio
import logging
from typing import Any

from fastapi import APIRouter, HTTPException, Query
from pydantic import BaseModel, Field

from ...cite import STYLE_LABELS, STYLES, detect_style
from ...db import repo
from ...db.repositories.search import search_log_summary
from ...export.exporters import EXPORT_FORMATS, export_csv, export_json
from ...prisma import build_prisma_flow, prisma_checklist_status, render_prisma_text
from ..deps import get_db

logger = logging.getLogger(__name__)

router = APIRouter()


# ============================================================ 请求体模型
class FeedbackRequest(BaseModel):
    """用户对 AI 输出的反馈 / 质疑。"""

    target_type: str = "message"      # message / artifact / paper / plan / manuscript / search_result
    target_id: str = ""
    run_id: str = ""
    session_id: int | None = None
    verdict: str = "up"               # up / down / challenge
    category: str = ""
    comment: str = ""
    #: 用户给出的正确版本 —— 有它才能变成"纠错记忆"与训练偏好对
    corrected_text: str = ""
    quoted_text: str = ""
    topic: str = ""


class ManuscriptRequest(BaseModel):
    """论文生成请求：用户的目标 / 方法 / 实验数据。"""

    brief: dict[str, Any] = Field(default_factory=dict)
    session_id: int | None = None
    #: 复用的文献材料（默认自动取本地知识库中与本课题最相关的若干篇）
    use_library: bool = True
    top_k: int = 12
    manuscript_id: int | None = None
    save: bool = True


class CiteRequest(BaseModel):
    paper_ids: list[int]
    style: str = "gb7714"
    format: str = "list"


class ExportRequest(BaseModel):
    paper_ids: list[int] = Field(default_factory=list)
    format: str = "bibtex"
    name: str = "medscholar"


# ============================================================ 用户反馈与学习闭环
@router.post("/api/feedback", tags=["学习闭环"])
async def submit_feedback(req: FeedbackRequest) -> dict[str, Any]:
    """记录赞/踩/质疑。

    质疑若带上了「正确说法」，会立刻成为后续生成的纠错记忆
    （见 feedback.memories_as_prompt），不需要重新训练模型。
    """
    from ...feedback import FeedbackEntry, record_feedback

    entry = FeedbackEntry(
        target_type=req.target_type,
        target_id=req.target_id,
        run_id=req.run_id,
        session_id=req.session_id,
        verdict=req.verdict,
        category=req.category,
        comment=req.comment,
        corrected_text=req.corrected_text,
        quoted_text=req.quoted_text,
        topic=req.topic,
    )
    feedback_id = await asyncio.to_thread(record_feedback, entry, db=get_db())
    became_memory = bool(
        req.verdict == "challenge" and req.corrected_text.strip()
    )
    return {
        "ok": True,
        "feedback_id": feedback_id,
        "became_memory": became_memory,
        "message": (
            "已记住这条纠正，后续同类生成会参考它（立即生效）。"
            if became_memory
            else "已记录你的反馈。"
        ),
    }


@router.get("/api/feedback", tags=["学习闭环"])
async def list_feedback(
    run_id: str = "",
    session_id: int | None = None,
    limit: int = Query(50, ge=1, le=500),
) -> dict[str, Any]:
    from ...feedback import list_feedback as _list

    items = await asyncio.to_thread(
        _list, run_id=run_id, session_id=session_id, limit=limit, db=get_db()
    )
    return {"count": len(items), "items": [i.to_dict() for i in items]}


@router.get("/api/feedback/summary", tags=["学习闭环"])
async def feedback_summary() -> dict[str, Any]:
    """学习闭环的可见进度：多少条反馈、多少条已成为生效记忆。"""
    from ...feedback import feedback_summary as _summary

    return await asyncio.to_thread(_summary, db=get_db())


@router.get("/api/feedback/export", tags=["学习闭环"])
async def feedback_export(limit: int = Query(1000, ge=1, le=10000)) -> dict[str, Any]:
    """导出 DPO 风格偏好对（把真实使用数据变成训练数据）。"""
    from ...feedback import export_preference_pairs

    pairs = await asyncio.to_thread(export_preference_pairs, limit=limit, db=get_db())
    return {"count": len(pairs), "pairs": pairs}


# ============================================================ 论文（IMRaD）
@router.post("/api/manuscript/draft", tags=["论文写作"])
async def manuscript_draft(req: ManuscriptRequest) -> dict[str, Any]:
    """基于用户的目标/方法/实验数据生成论文初稿，并做数字溯源校验。"""
    from ...manuscript import (
        ManuscriptBrief,
        assemble,
        check_number_provenance,
        draft_manuscript,
        save_manuscript,
    )

    brief = ManuscriptBrief.from_dict(req.brief)
    missing = brief.missing()
    if missing:
        raise HTTPException(
            status_code=400,
            detail="还缺少必要的论文要素：" + "、".join(missing) + "，请补齐后再生成。",
        )

    literature_text = ""
    if req.use_library:
        try:
            from ...retrieval import build_context_digest, search_knowledge_base

            hits = await search_knowledge_base(
                " ".join(filter(None, [brief.title, brief.goal, brief.outcomes]))
                or brief.title
                or brief.goal,
                top_k=req.top_k,
            )
            entries = [
                (index, hit.paper) for index, hit in enumerate(hits, start=1)
            ]
            if entries:
                literature_text = build_context_digest(entries, max_abstract=600)
        except Exception as exc:  # 文献材料拿不到不该阻断写作
            logger.warning("检索本地文献材料失败：%s", exc)

    # 纠错记忆：把用户此前明确指出过的错误带进提示词
    from ...feedback import memories_as_prompt

    memory_text = await asyncio.to_thread(
        memories_as_prompt, brief.goal or brief.title, db=get_db()
    )

    try:
        result = await draft_manuscript(
            brief, literature_text=literature_text, memory_text=memory_text
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("论文生成失败")
        raise HTTPException(status_code=500, detail=f"论文生成失败：{exc}") from exc

    draft = assemble(result["sections"], result["titles"], result["order"])
    checks = check_number_provenance(
        draft, brief=brief, literature_text=literature_text
    )
    checks["sections"] = result["order"]
    checks["generation_errors"] = result["errors"]
    checks["used_literature"] = bool(literature_text)
    checks["used_memories"] = bool(memory_text)

    manuscript_id = None
    if req.save:
        manuscript_id = await asyncio.to_thread(
            save_manuscript,
            title=brief.title or brief.goal[:50],
            brief=brief.to_dict(),
            draft=draft,
            checks=checks,
            session_id=req.session_id,
            manuscript_id=req.manuscript_id,
            db=get_db(),
        )

    return {
        "ok": True,
        "manuscript_id": manuscript_id,
        "draft": draft,
        "sections": result["sections"],
        "order": result["order"],
        "checks": checks,
    }


@router.get("/api/manuscripts", tags=["论文写作"])
async def manuscripts_list(
    session_id: int | None = None, limit: int = Query(20, ge=1, le=200)
) -> dict[str, Any]:
    from ...manuscript import list_manuscripts

    items = await asyncio.to_thread(
        list_manuscripts, session_id=session_id, limit=limit, db=get_db()
    )
    return {"count": len(items), "items": items}


@router.get("/api/manuscripts/{manuscript_id}", tags=["论文写作"])
async def manuscript_detail(manuscript_id: int) -> dict[str, Any]:
    from ...manuscript import get_manuscript

    record = await asyncio.to_thread(get_manuscript, manuscript_id, db=get_db())
    if record is None:
        raise HTTPException(status_code=404, detail=f"未找到稿件 id={manuscript_id}")
    return record


@router.get("/api/manuscript/recheck/{manuscript_id}", tags=["论文写作"])
async def manuscript_recheck(manuscript_id: int) -> dict[str, Any]:
    """重新做数字溯源校验（用户改过数据后可以再查一次）。"""
    from ...manuscript import ManuscriptBrief, check_number_provenance, get_manuscript

    record = await asyncio.to_thread(get_manuscript, manuscript_id, db=get_db())
    if record is None:
        raise HTTPException(status_code=404, detail=f"未找到稿件 id={manuscript_id}")
    brief = ManuscriptBrief.from_dict(record.get("brief") or {})
    checks = check_number_provenance(record.get("draft") or "", brief=brief)
    return {"manuscript_id": manuscript_id, "checks": checks}


# ============================================================ 引用与导出
@router.get("/api/cite/styles", tags=["引用"])
async def cite_styles() -> dict[str, Any]:
    return {"styles": [{"key": key, "label": STYLE_LABELS[key]} for key in STYLES]}


@router.post("/api/cite", tags=["引用"])
async def cite(req: CiteRequest) -> dict[str, Any]:
    from ...cite import format_records, format_reference_list

    if not req.paper_ids:
        raise HTTPException(status_code=400, detail="paper_ids 不能为空")
    mapping = await asyncio.to_thread(repo.get_papers_by_ids, req.paper_ids)
    papers = [mapping[i] for i in req.paper_ids if i in mapping]
    if not papers:
        raise HTTPException(status_code=404, detail="指定的文献不存在于本地库")

    style = detect_style(req.style)
    if req.format == "bibtex" or style == "bibtex":
        content = format_records(papers, "bibtex")
    elif req.format == "ris" or style == "ris":
        content = format_records(papers, "ris")
    else:
        content = format_reference_list(papers, style)
    return {
        "style": style,
        "label": STYLE_LABELS.get(style, style),
        "content": content,
        "count": len(papers),
    }


@router.post("/api/export", tags=["引用"])
async def export(req: ExportRequest) -> dict[str, Any]:
    from ...cite import format_records
    from ...export.exporters import write_export

    fmt = str(req.format).lower()
    mapping = await asyncio.to_thread(repo.get_papers_by_ids, req.paper_ids)
    papers = [mapping[i] for i in req.paper_ids if i in mapping]

    if fmt in {"bibtex", "ris"}:
        content = format_records(papers, fmt)
    elif fmt == "csv":
        content = export_csv(papers)
    elif fmt == "json":
        content = export_json(papers)
    elif fmt in {"apa7", "vancouver", "gb7714", "chicago"}:
        content = format_records(papers, fmt)
    else:
        raise HTTPException(
            status_code=400,
            detail=f"不支持的导出格式：{req.format}（可选：{', '.join(EXPORT_FORMATS)}）",
        )

    path = await asyncio.to_thread(
        write_export, content, name=req.name, fmt=fmt
    )
    return {
        "path": str(path),
        "format": fmt,
        "bytes": len(content.encode("utf-8")),
        "count": len(papers),
    }


# ============================================================ PRISMA
@router.get("/api/prisma/flow", tags=["论文写作"])
async def prisma_flow(
    since: str = Query("", description="只统计该时间之后的检索（ISO 文本，可留空）"),
    included: int = Query(0, ge=0, description="最终纳入研究数（研究者的判断，需手工填）"),
    excluded_screening: int = Query(0, ge=0, description="题目/摘要阶段排除数"),
    not_retrieved: int = Query(0, ge=0, description="未获取到全文的报告数"),
) -> dict[str, Any]:
    """按 PRISMA 2020 口径汇总本次检索流程的数字，并给出自检提示。

    系统评价/Meta 分析投稿时必须附 PRISMA 流程图，而研究者现在的做法是拿 Excel 手工数：
    检索记录在一处、去重结果在另一处、全文评估在第三处，数一遍半小时，改一次检索式还得重数。

    能自动算的：各数据库识别到的记录数（来自检索日志）、重复移除数。
    必须人工填的：「题目/摘要排除」「未获取到全文」「最终纳入」——
    这些是研究者的学术判断，工具只负责把数字做成自洽的、可复现的，
    并在数字对不上时提前拦住，而不是生成一张看起来很漂亮但会被审稿人质疑的图。
    """
    summary = await asyncio.to_thread(search_log_summary, since=since or None, db=get_db())
    flow = build_prisma_flow(
        identified=summary["by_source"],
        # 说明：日志里 result_count 是"该源返回条数"，new_count 是"新入库条数"，
        # 两者之差包含了"重复"与"被合并富化"两种情况。这与 PRISMA 的
        # "duplicates removed" 语义略有差异（后者只算重复），
        # 是日志里能拿到的最接近口径；需要严格口径时请按提示手工填入。
        duplicates_removed=max(0, summary["total_results"] - summary["total_new"]),
        excluded_at_screening=excluded_screening,
        not_retrieved=not_retrieved,
        included=included,
    )
    return {
        "flow": flow.to_dict(),
        "text": render_prisma_text(flow),
        "warnings": flow.warnings(),
        "checklist": prisma_checklist_status(flow),
        "search_summary": summary,
        "note": (
            "「题目/摘要排除」「未获取到全文」「最终纳入」需要研究者自己判断后填入；"
            "本接口只保证数字自洽与可复现。"
        ),
    }
