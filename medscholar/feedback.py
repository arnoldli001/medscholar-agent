"""反馈、质疑与学习闭环——回答"用户点的那一下踩、写的那条纠错，到底改变了后面的什么行为"：

1. 纠错记忆（即时生效，无需训练）：带正确说法的质疑作为 few-shot 记忆注入后续
   同主题的写作/评审提示词，减少同类错误复发；
2. 偏好对导出（离线生效）：``up``/``down`` 与 ``corrected_text`` 构成
   ``(prompt, chosen, rejected)`` 三元组导出 DPO/RLHF JSONL；本模块只生成合规训练集，
   运行时不做梯度更新（本地 8B 量化模型上做在线 RL 不现实，也不假装能做）；
3. 来源与文献重加权（即时生效）：被反复质疑的文献/长期被否决的数据源在后续检索降权。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .constants import (
    FEEDBACK_LIST_LIMIT,
    FIELD_COMMENT_MAX,
    FIELD_CORRECTED_MAX,
    FIELD_QUOTED_MAX,
    FIELD_RUN_ID_MAX,
    FIELD_TARGET_ID_MAX,
    FIELD_TARGET_TYPE_MAX,
    FIELD_TOPIC_MAX,
    MEMORY_CANDIDATE_FACTOR,
    MEMORY_COMMENT_MAX,
    MEMORY_CORRECTED_MAX,
    MEMORY_FEWSHOT_LIMIT,
    MEMORY_QUOTED_MAX,
    MEMORY_TOPIC_KEYWORD,
    PREFERENCE_EXPORT_LIMIT,
)
from .db.connect import Database
from .db.repo import _db  # 复用统一的数据库句柄解析

logger = logging.getLogger(__name__)

__all__ = [
    "FeedbackEntry",
    "record_feedback",
    "list_feedback",
    "feedback_summary",
    "correction_memories",
    "export_preference_pairs",
    "paper_penalty",
    "source_penalty",
    "FEEDBACK_CATEGORIES",
]

#: 质疑/纠错的分类（前端下拉选项与此保持一致）
FEEDBACK_CATEGORIES: dict[str, str] = {
    "citation": "引用错误（编号、出处对不上）",
    "fact": "事实错误（结论被说反、数据不准）",
    "omission": "遗漏重要文献或关键结论",
    "overreach": "过度推断（证据不支持该结论）",
    "structure": "结构或逻辑问题",
    "style": "文风与表述问题",
    "data": "与我提供的数据不一致",
    "other": "其它",
}
VERDICTS = ("up", "down", "challenge")


@dataclass(slots=True)
class FeedbackEntry:
    """一条反馈。"""

    target_type: str = "message"
    target_id: str = ""
    run_id: str = ""
    session_id: int | None = None
    verdict: str = "up"
    category: str = ""
    comment: str = ""
    corrected_text: str = ""
    quoted_text: str = ""
    topic: str = ""
    id: int | None = None
    created_at: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "target_type": self.target_type,
            "target_id": self.target_id,
            "run_id": self.run_id,
            "session_id": self.session_id,
            "verdict": self.verdict,
            "category": self.category,
            "category_label": FEEDBACK_CATEGORIES.get(self.category, self.category),
            "comment": self.comment,
            "corrected_text": self.corrected_text,
            "quoted_text": self.quoted_text,
            "topic": self.topic,
            "created_at": self.created_at,
        }


def record_feedback(
    entry: FeedbackEntry, *, db: Database | None = None
) -> int:
    """写入一条反馈，返回 id。"""
    database = _db(db)
    verdict = entry.verdict if entry.verdict in VERDICTS else "up"
    category = entry.category if entry.category in FEEDBACK_CATEGORIES else ""
    with database.transaction() as conn:
        cursor = conn.execute(
            "INSERT INTO feedback(target_type, target_id, run_id, session_id, verdict, "
            "                    category, comment, corrected_text, quoted_text, topic) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                (entry.target_type or "message")[:FIELD_TARGET_TYPE_MAX],
                str(entry.target_id or "")[:FIELD_TARGET_ID_MAX],
                str(entry.run_id or "")[:FIELD_RUN_ID_MAX],
                entry.session_id,
                verdict,
                category,
                (entry.comment or "")[:FIELD_COMMENT_MAX],
                (entry.corrected_text or "")[:FIELD_CORRECTED_MAX],
                (entry.quoted_text or "")[:FIELD_QUOTED_MAX],
                (entry.topic or "")[:FIELD_TOPIC_MAX],
            ),
        )
        return int(cursor.lastrowid or 0)


def list_feedback(
    *,
    run_id: str = "",
    session_id: int | None = None,
    target_type: str = "",
    limit: int = FEEDBACK_LIST_LIMIT,
    db: Database | None = None,
) -> list[FeedbackEntry]:
    """按条件列出反馈（最近的在前）。"""
    where: list[str] = []
    params: list[Any] = []
    if run_id:
        where.append("run_id = ?")
        params.append(run_id)
    if session_id is not None:
        where.append("session_id = ?")
        params.append(session_id)
    if target_type:
        where.append("target_type = ?")
        params.append(target_type)
    sql = "SELECT * FROM feedback"
    if where:
        sql += " WHERE " + " AND ".join(where)
    sql += " ORDER BY id DESC LIMIT ?"
    params.append(int(limit))

    rows = _db(db).query(sql, tuple(params))
    out: list[FeedbackEntry] = []
    for row in rows:
        out.append(
            FeedbackEntry(
                id=int(row["id"]),
                target_type=row["target_type"] or "",
                target_id=row["target_id"] or "",
                run_id=row["run_id"] or "",
                session_id=row["session_id"],
                verdict=row["verdict"] or "",
                category=row["category"] or "",
                comment=row["comment"] or "",
                corrected_text=row["corrected_text"] or "",
                quoted_text=row["quoted_text"] or "",
                topic=row["topic"] or "",
                created_at=row["created_at"] or "",
            )
        )
    return out


def feedback_summary(*, db: Database | None = None, limit: int = 5000) -> dict[str, Any]:
    """汇总反馈，供前端展示"系统的学习进度"。"""
    rows = _db(db).query(
        "SELECT verdict, category, COUNT(*) AS n FROM feedback "
        "GROUP BY verdict, category ORDER BY n DESC LIMIT ?",
        (int(limit),),
    )
    total = 0
    by_verdict: dict[str, int] = {}
    by_category: dict[str, int] = {}
    for row in rows:
        count = int(row["n"] or 0)
        total += count
        by_verdict[row["verdict"] or "up"] = by_verdict.get(row["verdict"] or "up", 0) + count
        key = row["category"] or ""
        if key:
            by_category[key] = by_category.get(key, 0) + count

    memories = correction_memories(db=db, limit=1000)
    return {
        "total": total,
        "up": by_verdict.get("up", 0),
        "down": by_verdict.get("down", 0),
        "challenge": by_verdict.get("challenge", 0),
        "by_category": [
            {"category": key, "label": FEEDBACK_CATEGORIES.get(key, key), "count": value}
            for key, value in sorted(by_category.items(), key=lambda kv: -kv[1])
        ],
        # 即时生效的那部分：已沉淀为生成时记忆的纠错条数
        "active_memories": len(memories),
        "categories": [
            {"key": key, "label": label} for key, label in FEEDBACK_CATEGORIES.items()
        ],
    }


# ============================================================ 1) 纠错记忆
def correction_memories(
    *,
    topic: str = "",
    limit: int = MEMORY_FEWSHOT_LIMIT,
    db: Database | None = None,
) -> list[dict[str, str]]:
    """取出可作为 few-shot 记忆的纠错（即时生效的闭环）。

    优先同主题，再补少量跨主题（通用性错误同样值得记）；
    只取给出了正确说法的质疑——光说"错了"没有可复用信息。
    """
    sql = (
        "SELECT topic, category, comment, corrected_text, quoted_text "
        "FROM feedback WHERE verdict = 'challenge' AND corrected_text != '' "
    )
    params: list[Any] = []
    order = " ORDER BY id DESC LIMIT ?"
    rows = _db(db).query(sql + order, (*params, int(limit) * MEMORY_CANDIDATE_FACTOR))

    same_topic: list[dict[str, str]] = []
    others: list[dict[str, str]] = []
    needle = (topic or "").strip()[:MEMORY_TOPIC_KEYWORD]
    for row in rows:
        item = {
            "topic": row["topic"] or "",
            "category": FEEDBACK_CATEGORIES.get(row["category"] or "", row["category"] or ""),
            "comment": row["comment"] or "",
            "corrected_text": row["corrected_text"] or "",
            "quoted_text": row["quoted_text"] or "",
        }
        if needle and needle in (row["topic"] or ""):
            same_topic.append(item)
        else:
            others.append(item)

    picked = same_topic[:limit]
    if len(picked) < limit:
        picked.extend(others[: limit - len(picked)])
    return picked


def memories_as_prompt(
    topic: str = "", *, limit: int = MEMORY_FEWSHOT_LIMIT, db: Database | None = None
) -> str:
    """把纠错记忆渲染成可直接拼进提示词的文本；无记忆返回空串，
    调用方据此决定是否插入，避免无谓的提示词膨胀。
    """
    items = correction_memories(topic=topic, limit=limit, db=db)
    if not items:
        return ""
    lines = [
        "以下是**用户此前明确指出过的错误**，本次生成必须避免重犯：",
    ]
    for index, item in enumerate(items, start=1):
        lines.append(f"{index}. 问题类型：{item['category'] or '未分类'}")
        if item["quoted_text"]:
            lines.append(f"   错误原文：「{item['quoted_text'][:MEMORY_QUOTED_MAX]}」")
        if item["comment"]:
            lines.append(f"   用户说明：{item['comment'][:MEMORY_COMMENT_MAX]}")
        lines.append(f"   正确说法：{item['corrected_text'][:MEMORY_CORRECTED_MAX]}")
    return "\n".join(lines)


# ======================================================= 2) 偏好对导出
def export_preference_pairs(
    *,
    limit: int = PREFERENCE_EXPORT_LIMIT,
    db: Database | None = None,
) -> list[dict[str, Any]]:
    """导出 DPO 风格偏好对。

    一条 ``down``/``challenge`` 反馈配上"正确版本"即构成
    ``(prompt, chosen, rejected)``：chosen 是用户给的正确说法，rejected 是模型原输出。
    """
    rows = _db(db).query(
        "SELECT * FROM feedback WHERE verdict IN ('down', 'challenge') "
        "AND (corrected_text != '' OR quoted_text != '') "
        "ORDER BY id DESC LIMIT ?",
        (int(limit),),
    )
    pairs: list[dict[str, Any]] = []
    for row in rows:
        rejected = (row["quoted_text"] or "").strip()
        chosen = (row["corrected_text"] or "").strip()
        if not rejected or not chosen:
            continue
        pairs.append(
            {
                "prompt": (
                    f"课题：{row['topic']}\n"
                    f"用户指出的问题类型：{FEEDBACK_CATEGORIES.get(row['category'] or '', row['category'] or '未分类')}\n"
                    f"用户说明：{row['comment']}"
                ).strip(),
                "chosen": chosen,
                "rejected": rejected,
                "meta": {
                    "feedback_id": row["id"],
                    "run_id": row["run_id"],
                    "target_type": row["target_type"],
                    "category": row["category"],
                    "created_at": row["created_at"],
                },
            }
        )
    return pairs


def export_jsonl(path: str, *, limit: int = 1000, db: Database | None = None) -> int:
    """把偏好对写成 JSONL（可直接用于 DPO/RLHF 训练流水线）。"""
    pairs = export_preference_pairs(limit=limit, db=db)
    with open(path, "w", encoding="utf-8") as handle:
        for pair in pairs:
            handle.write(json.dumps(pair, ensure_ascii=False) + "\n")
    return len(pairs)


# ================================================ 3) 文献 / 来源重加权
def paper_penalty(*, db: Database | None = None, max_penalty: float = 0.5) -> dict[int, float]:
    """返回 ``{paper_id: 惩罚系数}``（1.0 表示不惩罚），用于检索后重排。

    被明确指出事实/引用/过度推断问题且带纠错的文献下次降权，避免同一个坑踩第二次。
    纯"踩"不计入——口味不同不该把有价值的文献永久压下去。
    """
    rows = _db(db).query(
        "SELECT target_id, COUNT(*) AS n FROM feedback "
        "WHERE verdict = 'challenge' AND category IN ('fact', 'citation', 'overreach') "
        "AND corrected_text != '' AND target_type IN ('paper', 'artifact', 'message') "
        "GROUP BY target_id",
    )
    out: dict[int, float] = {}
    for row in rows:
        raw = str(row["target_id"] or "")
        if not raw.isdigit():
            continue
        count = int(row["n"] or 0)
        # 每被质疑一次降 10%，最多降到 max_penalty
        out[int(raw)] = max(max_penalty, 1.0 - 0.1 * count)
    return out


def source_penalty(*, db: Database | None = None, min_ratio: float = 0.6, min_samples: int = 5):
    """返回 ``{source: 系数}``：哪些数据源被反复否决。

    样本数低于 ``min_samples`` 不作判断，否则一两次差评就会永久改变排序。
    """
    rows = _db(db).query(
        "SELECT target_id, verdict FROM feedback WHERE target_type = 'search_result' LIMIT 5000"
    )
    stats: dict[str, list[int]] = {}
    for row in rows:
        key = str(row["target_id"] or "").strip().lower()
        if not key:
            continue
        stats.setdefault(key, [0, 0])
        stats[key][0] += 1
        if row["verdict"] in {"down", "challenge"}:
            stats[key][1] += 1

    out: dict[str, float] = {}
    for source, (total, bad) in stats.items():
        if total < min_samples:
            continue
        good_ratio = 1.0 - bad / total
        if good_ratio < min_ratio:
            out[source] = max(0.5, good_ratio)
    return out


def apply_paper_penalty(
    scored: Sequence[Any], penalties: Mapping[int, float]
) -> list[Any]:
    """按惩罚系数调整检索分数并重排（分数越高越相关）。"""
    if not penalties:
        return list(scored)
    adjusted: list[tuple[float, Any]] = []
    for item in scored:
        paper = getattr(item, "paper", None)
        paper_id = getattr(paper, "paper_id", None) if paper is not None else None
        factor = penalties.get(int(paper_id), 1.0) if paper_id else 1.0
        base = float(getattr(item, "score", 0.0) or 0.0)
        adjusted.append((base * factor, item))
    adjusted.sort(key=lambda pair: -pair[0])
    return [item for _, item in adjusted]
