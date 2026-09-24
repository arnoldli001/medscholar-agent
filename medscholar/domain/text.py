"""文本归一化与中日韩（CJK）分词辅助。

SQLite FTS5 的 ``unicode61`` 分词器把连续汉字当成单个 token，
中文摘要（尤其 CNKI 来源）会被关键词检索漏掉。

做法是在索引写入和查询构造两侧都插入汉字之间的空格，
使 unicode61 退化为单字索引，再用 FTS5 的短语查询（``"加 速 治 疗"``）
还原子串匹配语义。中英文共用同一张 FTS5 表，无需分词依赖。
"""

from __future__ import annotations

import hashlib
import re
import unicodedata

__all__ = [
    "CJK_RANGES",
    "is_cjk",
    "has_cjk",
    "segment_cjk",
    "normalize_text",
    "clean_abstract",
    "fts_quote",
    "build_match_query",
    "normalize_doi",
    "normalize_title_key",
    "title_fingerprint",
    "estimate_tokens",
    "to_family_first",
    "truncate",
]

#: 需要逐字切分的字符区间：CJK 统一表意文字、扩展 A、假名、谚文、全角标点周边
CJK_RANGES: tuple[tuple[int, int], ...] = (
    (0x3040, 0x30FF),   # 日文假名
    (0x3400, 0x4DBF),   # CJK 扩展 A
    (0x4E00, 0x9FFF),   # CJK 基本区
    (0xF900, 0xFAFF),   # CJK 兼容表意文字
    (0xAC00, 0xD7AF),   # 谚文音节
    (0x20000, 0x2FA1F),  # CJK 扩展 B~F
)

_CJK_RE = re.compile(
    "[" + "".join(f"{chr(lo)}-{chr(hi)}" for lo, hi in CJK_RANGES) + "]"
)
_WS_RE = re.compile(r"\s+")
_CTRL_RE = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
# JATS / PubMed 摘要里的结构化小标题，如 "<h4>BACKGROUND</h4>"
_JATS_TAG_RE = re.compile(r"<[^>]{1,80}>")


def is_cjk(ch: str) -> bool:
    """判断单个字符是否属于 CJK 区间。"""
    return bool(ch) and bool(_CJK_RE.match(ch))


def estimate_tokens(text: str | None) -> int:
    """粗略估算一段文本的 token 数，用于给上下文预算留位置。

    为什么不用真正的分词器：这里只需要"够准到能做决策"。实测依据——
    qwen3:8b 下 15,676 字（以英文摘要为主）的材料块占用 4,129 token，
    本函数给 4,479，误差约 8%，足以判断"提示词会不会把输出挤掉"。

    中日韩字符约 1 token/字；其余（英文、数字、标点）约 1 token/3.5 字。

    >>> estimate_tokens("")
    0
    >>> estimate_tokens("中文大约一字一token")
    11
    """
    body = text or ""
    if not body:
        return 0
    cjk = 0
    for ch in body:
        if _CJK_RE.match(ch) or "\u3000" <= ch <= "\u303f" or "\uff00" <= ch <= "\uffef":
            cjk += 1
    other = len(body) - cjk
    return int(cjk + other / 3.5) + 1


def has_cjk(text: str | None) -> bool:
    """文本中是否含汉字（用于判断是否需要走中文检索通路）。"""
    return any("\u4e00" <= ch <= "\u9fff" for ch in (text or ""))


def segment_cjk(text: str | None) -> str:
    """在汉字/假名/谚文字符两侧插入空格，供 FTS5 unicode61 逐字索引。

    >>> segment_cjk("加速rTMS治疗")
    '加 速 rTMS 治 疗'
    """
    if not text:
        return ""
    out: list[str] = []
    prev_cjk = False
    for ch in text:
        cur_cjk = is_cjk(ch)
        if cur_cjk:
            if out and not prev_cjk and not out[-1].isspace():
                out.append(" ")
            out.append(ch)
            out.append(" ")
        else:
            out.append(ch)
        prev_cjk = cur_cjk
    return _WS_RE.sub(" ", "".join(out)).strip()


def normalize_text(text: str | None) -> str:
    """Unicode 归一化 + 控制字符清理 + 空白压缩。"""
    if not text:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = _CTRL_RE.sub(" ", text)
    return _WS_RE.sub(" ", text).strip()


def clean_abstract(abstract: str | None) -> str:
    """清理摘要：剥离 JATS/HTML 标签、解码常见实体、压缩空白。"""
    if not abstract:
        return ""
    text = str(abstract)
    text = _JATS_TAG_RE.sub(" ", text)
    for entity, repl in (
        ("&lt;", "<"), ("&gt;", ">"), ("&amp;", "&"),
        ("&quot;", '"'), ("&apos;", "'"), ("&#x27;", "'"), ("&nbsp;", " "),
    ):
        text = text.replace(entity, repl)
    return normalize_text(text)


def fts_quote(token: str) -> str:
    """把单个词元安全地包成 FTS5 字符串字面量（内部双引号翻倍）。"""
    return '"' + str(token).replace('"', '""') + '"'


def build_match_query(
    query: str,
    *,
    mode: str = "and",
    cjk: str = "phrase",
    prefix: bool = False,
) -> str:
    """把自然语言检索词转换为 FTS5 MATCH 表达式。

    * 英文/数字词元按原样保留，可选加 ``*`` 前缀通配；
    * 中文被切分为单字后，按 ``cjk`` 策略合并：

      ``phrase``（默认，高精度）
          相邻单字合成一个短语，等价于连续子串匹配。
          ``"治 疗 卒 中 后 抑 郁"`` 不会命中「治疗脑卒中后抑郁」。
      ``bigram``（高召回）
          相邻单字两两组成重叠二元组再取 AND：``"治疗" AND "疗卒" AND …``。
          同样对上述文本仍然失败，但对「卒中后抑郁」这类常见改写更宽容。

    实际检索时 :func:`medscholar.db.repo.search_fts` 会按
    ``phrase → bigram → or`` 逐级放宽，兼顾精度与召回，因此这里保持单一策略、
    由调用方决定用哪一级。

    >>> build_match_query("加速rTMS治疗卒中后抑郁")
    '"加 速" AND "rTMS" AND "治 疗 卒 中 后 抑 郁"'
    >>> build_match_query("卒中后抑郁", cjk="bigram")
    '("卒 中" AND "中 后" AND "后 抑" AND "抑 郁")'
    """
    if not query or not query.strip():
        return ""
    segmented = segment_cjk(normalize_text(query))
    joiner = " OR " if str(mode).lower() == "or" else " AND "
    parts: list[str] = []
    cjk_run: list[str] = []

    def flush() -> None:
        if not cjk_run:
            return
        if cjk != "bigram" or len(cjk_run) == 1:
            parts.append(fts_quote(" ".join(cjk_run)))
        else:
            # 二元组必须写成「两个单字token组成的短语」（"治 疗"），
            # 而不是一个双字 token（"治疗"）—— 因为索引侧的 segment_cjk
            # 把汉字切成了单字，双字 token 在索引里根本不存在，永远匹配不到。
            bigrams: list[str] = []
            for index in range(len(cjk_run) - 1):
                bigram = " ".join(cjk_run[index : index + 2])
                if bigram not in bigrams:
                    bigrams.append(bigram)
            if len(bigrams) == 1:
                parts.append(fts_quote(bigrams[0]))
            else:
                parts.append("(" + joiner.join(fts_quote(b) for b in bigrams) + ")")
        cjk_run.clear()

    for token in segmented.split(" "):
        if not token:
            continue
        if len(token) == 1 and is_cjk(token):
            cjk_run.append(token)
            continue
        flush()
        token = token.strip('"')
        if not token:
            continue
        if prefix and token.isascii() and token.isalnum() and len(token) >= 3:
            parts.append(fts_quote(token) + "*")
        else:
            parts.append(fts_quote(token))
    flush()

    if not parts:
        return ""
    return joiner.join(parts)


def normalize_doi(doi: str | None) -> str | None:
    """归一化 DOI：去前缀、统一小写，用于跨库去重。"""
    if not doi:
        return None
    value = str(doi).strip().lower()
    for prefix in ("https://doi.org/", "http://doi.org/", "https://dx.doi.org/", "doi:"):
        if value.startswith(prefix):
            value = value[len(prefix):]
    value = value.strip().strip(".")
    if not value or "/" not in value:
        return None
    return value


def normalize_title_key(title: str | None) -> str:
    """标题归一化：小写、仅保留字母数字与汉字，用于标题级去重。"""
    if not title:
        return ""
    text = unicodedata.normalize("NFKC", str(title)).lower()
    return "".join(ch for ch in text if ch.isalnum() or is_cjk(ch))


def title_fingerprint(title: str | None) -> str:
    """标题指纹（sha1 前 16 位），作为无 DOI/PMID 文献的去重键。"""
    key = normalize_title_key(title)
    if not key:
        return ""
    return hashlib.sha1(key.encode("utf-8")).hexdigest()[:16]


def truncate(text: str | None, limit: int, suffix: str = "…") -> str:
    """按字符数截断，避免把句子切在半个代理对上。"""
    if not text:
        return ""
    text = str(text)
    return text if len(text) <= limit else text[: max(0, limit - len(suffix))] + suffix


#: 西方姓氏前缀（van / von / de …），出现这些前缀说明已是「姓 名」顺序
_NAME_PARTICLES = frozenset(
    {
        "van", "von", "de", "del", "della", "der", "den", "la", "le", "du",
        "di", "da", "dos", "bin", "ibn", "ter", "ten", "op", "st",
    }
)


def to_family_first(name: str | None) -> str:
    """把西文姓名统一成「姓 名」顺序。

    各数据源的姓名顺序并不一致（PubMed/Crossref 是「姓 名」，
    OpenAlex/Semantic Scholar/arXiv 是「名 姓」），而引用格式化必须能可靠地
    提取姓氏，因此在这里统一。

    >>> to_family_first("Wei Zhang")
    'Zhang Wei'
    >>> to_family_first("Zhang, Wei")
    'Zhang Wei'
    >>> to_family_first("Zhang Wei")
    'Zhang Wei'
    >>> to_family_first("王伟")
    '王伟'
    """
    name = normalize_text(name)
    if not name:
        return ""
    if has_cjk(name):
        return name
    if "," in name:
        family, _, given = name.partition(",")
        return f"{family.strip()} {given.strip()}".strip()
    tokens = name.split()
    if len(tokens) < 2:
        return name
    if tokens[0].lower() in _NAME_PARTICLES:
        return name  # 已是「姓 名」（如 "van der Berg Jan"），不翻转
    return " ".join([tokens[-1], *tokens[:-1]])
