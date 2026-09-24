"""基于用户实验数据生成论文初稿（IMRaD）。

与"综述草稿"的区别：综述是基于文献的总结；这里是基于用户自己的
实验数据写论文。因此最大的风险不是文风，而是编造数据。

本模块的核心防线是 :func:`check_number_provenance`：
正文里出现的每一个数字，都必须能在「用户提供的数据/统计结果」或
「本地知识库的文献证据」里找到出处，否则标记为无法溯源。
这条检查让"AI 帮你写论文"从不可控变成可审计 —— 医学论文里编一个
P 值就是学术不端，必须由程序兜住，而不是靠提示词祈祷。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from .config import AppConfig, get_config
from .constants import LLM_MAX_TOKENS_MANUSCRIPT, LLM_TEMPERATURE_MANUSCRIPT
from .db.connect import Database
from .db.repo import _db
from .llm.client import LLMError, get_llm

logger = logging.getLogger(__name__)

__all__ = [
    "ManuscriptBrief",
    "NumberCheck",
    "check_number_provenance",
    "build_outline",
    "draft_manuscript",
    "save_manuscript",
    "get_manuscript",
    "list_manuscripts",
    "IMRAD_SECTIONS",
]

#: IMRaD 结构（医学论文的标准骨架）
IMRAD_SECTIONS: tuple[tuple[str, str], ...] = (
    ("title", "标题"),
    ("abstract", "摘要"),
    ("introduction", "引言"),
    ("methods", "方法"),
    ("results", "结果"),
    ("discussion", "讨论"),
    ("conclusion", "结论"),
)

#: 数字识别：整数、小数、百分比、P 值、置信区间、样本量等
_NUMBER_RE = re.compile(
    r"(?<![\w.])"
    r"(\d+(?:\.\d+)?\s*(?:%|‰)?)"
    r"(?![\w])"
)
#: 无意义的数字（年份、编号、章节号）不算"数据"
_IGNORABLE = {"1", "2", "3", "4", "5", "0"}


@dataclass(slots=True)
class NumberCheck:
    """一个数字的溯源结果。"""

    value: str
    count: int = 1
    in_user_data: bool = False
    in_literature: bool = False
    sentence: str = ""

    @property
    def ok(self) -> bool:
        return self.in_user_data or self.in_literature

    def to_dict(self) -> dict[str, Any]:
        return {
            "value": self.value,
            "count": self.count,
            "in_user_data": self.in_user_data,
            "in_literature": self.in_literature,
            "ok": self.ok,
            "sentence": self.sentence[:200],
        }


@dataclass(slots=True)
class ManuscriptBrief:
    """用户填写的论文要素。"""

    title: str = ""
    goal: str = ""            # 研究目标 / 要回答的问题
    hypothesis: str = ""
    design: str = ""          # 研究设计（RCT / 队列 / 病例对照…）
    population: str = ""      # 对象与纳排标准
    intervention: str = ""    # 干预与对照
    outcomes: str = ""        # 主要/次要结局
    statistics: str = ""      # 统计方法
    data: str = ""            # 实验数据（表格/CSV/文本）
    results: str = ""         # 已知统计结果（P 值、CI 等）
    limitations: str = ""     # 已知局限
    journal: str = ""         # 目标期刊
    language: str = "zh"
    extra: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "title": self.title, "goal": self.goal, "hypothesis": self.hypothesis,
            "design": self.design, "population": self.population,
            "intervention": self.intervention, "outcomes": self.outcomes,
            "statistics": self.statistics, "data": self.data, "results": self.results,
            "limitations": self.limitations, "journal": self.journal,
            "language": self.language, "extra": self.extra,
        }

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any] | None) -> "ManuscriptBrief":
        raw = raw or {}
        fields = {k: str(raw.get(k) or "") for k in cls.__dataclass_fields__}
        return cls(**fields)

    def missing(self) -> list[str]:
        """还缺哪些关键要素（用于给用户明确提示，而不是默默生成垃圾）。"""
        required = {
            "goal": "研究目标", "design": "研究设计",
            "population": "研究对象", "outcomes": "结局指标",
        }
        return [label for key, label in required.items() if not getattr(self, key).strip()]

    def text_for_prompt(self) -> str:
        blocks: list[str] = []
        labels = {
            "title": "拟定标题", "goal": "研究目标", "hypothesis": "研究假设",
            "design": "研究设计", "population": "研究对象与纳排标准",
            "intervention": "干预与对照", "outcomes": "结局指标",
            "statistics": "统计方法", "results": "已知统计结果",
            "limitations": "已知局限", "journal": "目标期刊", "extra": "补充说明",
        }
        for key, label in labels.items():
            value = getattr(self, key, "").strip()
            if value:
                blocks.append(f"【{label}】{value}")
        if self.data.strip():
            blocks.append(f"【实验数据】\n{self.data.strip()}")
        return "\n".join(blocks)


def _extract_numbers(text: str) -> dict[str, str]:
    """抽出文本里的数字 → 首次出现的句子（便于人工复核）。"""
    found: dict[str, str] = {}
    for match in _NUMBER_RE.finditer(text or ""):
        value = match.group(1).strip()
        if value in found:
            continue
        start = max(0, (text or "").rfind("。", 0, match.start()) + 1)
        end = (text or "").find("。", match.end())
        sentence = (text or "")[start : end if end > 0 else match.end() + 60]
        found[value] = sentence.strip()
    return found


def _number_variants(value: str) -> set[str]:
    """一个数字的等价写法（``12%`` / ``12`` / ``12.0``）。"""
    variants = {value}
    bare = value.replace("%", "").replace("‰", "").strip()
    variants.add(bare)
    try:
        number = float(bare)
        variants.add(f"{number:g}")
        variants.add(f"{number:.1f}")
        variants.add(f"{number:.2f}")
        if number == int(number):
            variants.add(str(int(number)))
    except ValueError:
        pass
    return {v for v in variants if v}


def check_number_provenance(
    draft: str,
    *,
    brief: ManuscriptBrief,
    literature_text: str = "",
) -> dict[str, Any]:
    """校验正文里每个数字的出处。

    Returns:
        ``{"total": n, "ok": n, "unverified": [NumberCheck...], "verdict": ...}``
    """
    allowed_source = " ".join([brief.text_for_prompt(), brief.results, brief.data])
    allowed_literature = literature_text or ""

    checks: list[NumberCheck] = []
    for value, sentence in _extract_numbers(draft).items():
        variants = _number_variants(value)
        in_user = any(v in allowed_source for v in variants)
        in_lit = any(v in allowed_literature for v in variants)
        checks.append(
            NumberCheck(
                value=value,
                in_user_data=in_user,
                in_literature=in_lit,
                sentence=sentence,
            )
        )

    unverified = [c for c in checks if not c.ok and c.value not in _IGNORABLE]
    ok_count = len(checks) - len(unverified)
    if not checks:
        verdict = "no_numbers"
    elif not unverified:
        verdict = "pass"
    elif len(unverified) <= max(2, len(checks) // 10):
        verdict = "warn"
    else:
        verdict = "fail"

    return {
        "total": len(checks),
        "ok": ok_count,
        "unverified_count": len(unverified),
        "unverified": [c.to_dict() for c in unverified[:30]],
        "verdict": verdict,
        "note": (
            "正文中的每个数字都已能在你提供的数据或本地文献中找到出处。"
            if verdict == "pass"
            else "有数字无法在你提供的数据/文献中找到出处 —— 请务必逐条核对，"
            "**不要直接投稿**。医学论文里编造数据属于学术不端。"
        ),
    }


def build_outline(brief: ManuscriptBrief) -> list[dict[str, str]]:
    """按 IMRaD 组织写作任务；每节都明确"这一节只能用哪些材料"。"""
    outline: list[dict[str, str]] = []
    outline.append({
        "key": "abstract",
        "title": "摘要",
        "hint": "结构化摘要（背景/方法/结果/结论），200~300 字，只用已提供的数据。",
    })
    outline.append({
        "key": "introduction",
        "title": "引言",
        "hint": "从研究背景写到本研究的空缺与目标；引用本地知识库中的文献编号 [n]。",
    })
    outline.append({
        "key": "methods",
        "title": "方法",
        "hint": "研究设计、对象与纳排标准、干预与对照、结局定义、统计方法。只写用户提供的信息。",
    })
    outline.append({
        "key": "results",
        "title": "结果",
        "hint": "严格按用户提供的数据与统计结果陈述；**每个数字都必须来自用户数据**，不得推算或补全。",
    })
    outline.append({
        "key": "discussion",
        "title": "讨论",
        "hint": "先讲主要发现，再与本地文献比较（引用 [n]），最后写局限。不得夸大因果。",
    })
    outline.append({
        "key": "conclusion",
        "title": "结论",
        "hint": "回应研究目标，措辞与证据强度匹配（避免'证实''治愈'等强断言）。",
    })
    return outline


_SYSTEM = (
    "你是一位资深医学论文写作指导专家，帮助研究者把**他们自己的**实验数据"
    "整理成规范的学术论文。\n\n"
    "不可违背的规则：\n"
    "1. 绝对不得编造、推测或「补全」任何数据。正文里出现的每个数字都必须"
    "来自用户提供的数据或统计结果；没有的数据就写「本研究未测量」或留待补充。\n"
    "2. 不得把相关性表述为因果；不得使用「证实」「治愈」「突破性」等超出证据的措辞。\n"
    "3. 引用一律使用方括号编号 [n]，且编号必须来自给定文献材料；没有依据就不写引用。\n"
    "4. 输出规范的学术中文（除非指明英文），不使用口语与主观表达。\n"
)


async def draft_manuscript(
    brief: ManuscriptBrief,
    *,
    literature_text: str = "",
    memory_text: str = "",
    sections: Sequence[str] | None = None,
    on_token: Any = None,
    config: AppConfig | None = None,
) -> dict[str, Any]:
    """逐节生成论文初稿，返回 ``{sections: {...}, order: [...], errors: [...]}``。"""
    cfg = config or get_config()
    missing = brief.missing()
    if missing:
        raise ValueError(
            "还缺少必要的论文要素：" + "、".join(missing)
            + "。请补齐后再生成 —— 缺这些信息写出来的只能是空话。"
        )

    outline = build_outline(brief)
    if sections:
        wanted = {s.strip().lower() for s in sections}
        outline = [item for item in outline if item["key"] in wanted] or outline

    client = get_llm(cfg)
    await client.start()

    produced: dict[str, str] = {}
    errors: list[str] = []
    for item in outline:
        prompt_parts = [
            brief.text_for_prompt(),
        ]
        if literature_text:
            prompt_parts.append(f"【可引用的本地文献材料】\n{literature_text}")
        if memory_text:
            prompt_parts.append(memory_text)
        prompt_parts.append(
            f"\n请撰写论文的「{item['title']}」部分。要求：{item['hint']}\n"
            "直接输出该部分正文，不要重复标题，不要输出解释。"
        )
        prompt = "\n\n".join(prompt_parts)
        body = ""
        try:
            if on_token is not None:
                async for chunk in client.stream(
                    [{"role": "user", "content": prompt}],
                    system=_SYSTEM,
                    temperature=LLM_TEMPERATURE_MANUSCRIPT,
                    max_tokens=LLM_MAX_TOKENS_MANUSCRIPT,
                ):
                    body += chunk
                    await on_token(chunk)
            else:
                body = await client.chat(
                    [{"role": "user", "content": prompt}],
                    system=_SYSTEM,
                    temperature=LLM_TEMPERATURE_MANUSCRIPT,
                    max_tokens=LLM_MAX_TOKENS_MANUSCRIPT,
                )
        except LLMError as exc:
            errors.append(f"「{item['title']}」生成失败：{exc}")
            body = ""
        produced[item["key"]] = body.strip()

    return {
        "sections": produced,
        "order": [item["key"] for item in outline],
        "titles": {item["key"]: item["title"] for item in outline},
        "errors": errors,
    }


def assemble(draft: Mapping[str, str], titles: Mapping[str, str], order: Sequence[str]) -> str:
    """把各节拼成 Markdown 全文。"""
    parts: list[str] = []
    for key in order:
        body = (draft.get(key) or "").strip()
        if not body:
            continue
        if key == "abstract":
            parts.append("## 摘要\n\n" + body)
        else:
            parts.append(f"## {titles.get(key, key)}\n\n{body}")
    return "\n\n".join(parts)


# ---------------------------------------------------------------- 持久化
def save_manuscript(
    *,
    title: str,
    brief: Mapping[str, Any],
    draft: str,
    checks: Mapping[str, Any] | None = None,
    session_id: int | None = None,
    manuscript_id: int | None = None,
    db: Database | None = None,
) -> int:
    """保存或更新稿件，返回 id。"""
    import json as _json

    database = _db(db)
    brief_json = _json.dumps(dict(brief), ensure_ascii=False)
    checks_json = _json.dumps(dict(checks or {}), ensure_ascii=False)
    with database.transaction() as conn:
        if manuscript_id:
            conn.execute(
                "UPDATE manuscripts SET title = ?, brief = ?, draft = ?, checks = ?, "
                "  updated_at = datetime('now') WHERE id = ?",
                (title[:500], brief_json, draft, checks_json, int(manuscript_id)),
            )
            return int(manuscript_id)
        cursor = conn.execute(
            "INSERT INTO manuscripts(session_id, title, brief, draft, checks) "
            "VALUES (?, ?, ?, ?, ?)",
            (session_id, title[:500], brief_json, draft, checks_json),
        )
        return int(cursor.lastrowid or 0)


def get_manuscript(manuscript_id: int, *, db: Database | None = None) -> dict[str, Any] | None:
    import json as _json

    row = _db(db).query_one("SELECT * FROM manuscripts WHERE id = ?", (int(manuscript_id),))
    if not row:
        return None
    out = dict(row)
    for key in ("brief", "checks"):
        try:
            out[key] = _json.loads(out.get(key) or "{}")
        except (TypeError, ValueError):
            out[key] = {}
    return out


def list_manuscripts(
    *, session_id: int | None = None, limit: int = 20, db: Database | None = None
) -> list[dict[str, Any]]:
    database = _db(db)
    if session_id is not None:
        rows = database.query(
            "SELECT id, session_id, title, status, created_at, updated_at, "
            "       length(draft) AS chars FROM manuscripts "
            "WHERE session_id = ? ORDER BY id DESC LIMIT ?",
            (int(session_id), int(limit)),
        )
    else:
        rows = database.query(
            "SELECT id, session_id, title, status, created_at, updated_at, "
            "       length(draft) AS chars FROM manuscripts "
            "ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
    return [dict(row) for row in rows]
