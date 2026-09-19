"""引用的**支持性**校验：``[n]`` 是否真的支持那句话。

与已有的"引用存在性校验"（`formatter.validate_citations`：编号是否指向真实文献）
的区别：存在性只保证"引用对得上号"，**不保证引用内容支持该论断**。
"结论被说反了"这类错误能通过存在性校验，但它是医学写作里最严重的错误之一。

## 分层设计（这是本模块最重要的约定）

蕴含判断（entailment）本质上不是正则能做的事，所以这里**不假装**一步到位，
而是分两层，报告中**永远分开呈现**：

* **Tier 0 —— 确定性规则**（无模型、离线、CI 可跑）
  只抓**高置信度、可复核**的问题：
  1. ``existence``   编号不存在（兜底，正常已被剔除）
  2. ``numbers``     论断里的数字在**被引文献里找不到**（编造数据的强信号）
  3. ``direction``   论断说"显著有效/优于"，而被引文献明确写"无显著差异"
  4. ``overclaim``   用了"证实/治愈/完全"这类超出证据强度的措辞
  5. ``grounding``   论断与被引文献的实词重合度极低（可能引错了文献）
  规则的取向是**宁可漏报、不可误报** —— 误报会让人不再信任这份报告。

* **Tier 1 —— LLM 裁判**（可选，需要模型）
  逐条给 ``supported / partial / unsupported / contradicted`` + 理由 + 证据片段。
  已知偏差（必须写进报告，不能藏起来）：
  - 偏好流畅文本，容易被措辞说服；
  - 对"部分支持"的判定不稳定；
  - 单次判定有噪声，因此支持**自一致性检查**（跑两次，不一致的标为 uncertain）。

**Tier 0 的通过不等于论断正确**，只说明"没触发已知的高置信度问题模式"。
报告里必须把这句写出来，否则使用者会把"没报警"当成"已核实"。
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

logger = logging.getLogger(__name__)

__all__ = [
    "Claim",
    "ClaimVerdict",
    "FaithfulnessReport",
    "extract_claims",
    "check_claim_rules",
    "verify_claims_llm",
    "analyse_draft",
    "VERDICTS",
    "SEVERITY_ORDER",
]

#: 判定取值（按严重程度从轻到重）
#:
#: ``weakly_supported`` 是实测逼出来的一个档：中文综述引英文文献时，
#: Tier 0 只能靠语言无关信号（数字/术语/缩写）核对，语义蕴含无法确认。
#: 把它报成 ``supported`` 会让人误以为"已核实"，报成 ``unverifiable`` 又会让
#: 整个工具在主要场景下全是"无法核实"。单列一档最诚实。
VERDICTS = (
    "supported",
    "weakly_supported",
    "unsupported",
    "overclaim",
    "contradicted",
    "unverifiable",
)
SEVERITY_ORDER = {
    "supported": 0,
    "weakly_supported": 0,   # 不是问题，但证据强度弱
    "unverifiable": 1,
    "unsupported": 1,
    "overclaim": 2,
    "contradicted": 3,
}

#: 正文里的引用标记（与 writer._CITATION_RE 保持一致）
_CITATION_RE = re.compile(r"[\[【]\s*(\d{1,3}(?:\s*[,，\-–]\s*\d{1,3})*)\s*[\]】]")
#: 参考文献条目：行首就是 [n] 并跟空格
_REF_LINE_RE = re.compile(r"^\s*[\[【]\d{1,3}[\]】]\s+\S")
_REF_HEADING_RE = re.compile(r"^#{1,4}\s*(参考文献|References|REFERENCES|文献列表)\s*$")
#: 句子切分：中英文句末标点（中文无空格，所以要单独处理）
_SENTENCE_RE = re.compile(r"[^。！？!?；;\n]+[。！？!?；;]?")
#: 数字（排除引用编号后的正文数字）；含小数、百分比、P 值
_NUMBER_RE = re.compile(r"(?<![\w.\[])(\d+(?:\.\d+)?\s*(?:%|‰)?)(?![\w\]])")
_IGNORABLE_NUMBERS = {"0", "1", "2", "3", "4", "5"}

#: 论断侧"强主张"措辞（出现即需要证据强度支撑）
_STRONG_CLAIM_CUES = (
    "证实", "证明", "治愈", "根治", "完全", "彻底", "显著优于", "明显优于", "优于",
    "必然", "毫无疑问", "突破性", "首次证明", "确凿",
    "prove", "proves", "proven", "cure", "cures", "breakthrough", "conclusively",
    "significantly better", "superior to", "definitively",
)
#: 被引文献里的"无效/阴性结果"表述 —— 与上面的强主张同时出现即为矛盾信号
_NULL_RESULT_CUES = (
    "无显著差异", "无统计学意义", "未见显著", "没有显著", "差异无统计学意义",
    "未达到统计学意义", "不优于", "未优于", "与安慰剂相当", "无差异",
    "no significant difference", "no significant", "not significant",
    "did not differ", "no difference", "failed to show", "no benefit",
    "not superior", "comparable to placebo", "no significant improvement",
)
#: 证据强度偏弱的载体（论断若很强气，而来源是这些，需要提醒）
_WEAK_EVIDENCE_CUES = (
    "protocol", "study protocol", "研究方案", "preprint", "预印本", "bioRxiv", "medRxiv",
    "case report", "病例报告", "pilot study", "初步", "会议摘要", "abstract only",
    "研究计划", "尚未", "正在进行",
)
#: 实词过滤（中文虚词与英文停用词）
_STOPWORDS = frozenset(
    """的 了 和 与 及 或 在 是 为 对 中 上 下 有 无 不 也 都 而 但 并 等 这 那 其 之 从 到
    本 该 这些 那些 我们 研究 结果 显示 表明 提示 提示了 可能 可以 需 要 应 该 通过 进行
    the a an and or of in on at to for with by is are was were be been this that these those
    we our study results result show shows showed suggest suggests may can could should
    """.split()
)
#: 中文虚字：用于剔除「纯虚词组成的 bigram」，避免"的的""在与"这类噪声进入重合度计算
_CJK_FUNCTION_CHARS = frozenset("的了和与及或在是为对中上下有无不也都而但并等这那其之从到本该")
_TOKEN_RE = re.compile(r"[A-Za-z][A-Za-z\-]{2,}|[\u4e00-\u9fff]{2,}")


@dataclass(slots=True)
class Claim:
    """一条带引用的论断。"""

    text: str
    citations: list[int] = field(default_factory=list)
    section: str = ""
    #: 在正文中的字符偏移，便于前端定位
    offset: int = 0

    def to_dict(self) -> dict[str, Any]:
        return {
            "text": self.text,
            "citations": list(self.citations),
            "section": self.section,
            "offset": self.offset,
        }


@dataclass(slots=True)
class ClaimVerdict:
    """一条论断的校验结果。"""

    claim: Claim
    verdict: str = "supported"
    #: 命中的规则/裁判给出的问题
    problems: list[dict[str, Any]] = field(default_factory=list)
    #: tier0 / tier1 / tier0+tier1
    tier: str = "tier0"
    #: LLM 裁判给出的证据片段（tier1）
    evidence: str = ""
    reason: str = ""
    #: 自一致性检查时两次判定不一致
    uncertain: bool = False

    @property
    def severity(self) -> int:
        return SEVERITY_ORDER.get(self.verdict, 1)

    def to_dict(self) -> dict[str, Any]:
        return {
            "claim": self.claim.to_dict(),
            "verdict": self.verdict,
            "tier": self.tier,
            "problems": list(self.problems),
            "evidence": self.evidence,
            "reason": self.reason,
            "uncertain": self.uncertain,
        }


@dataclass(slots=True)
class FaithfulnessReport:
    """整篇草稿的支持性报告。"""

    claims: int = 0
    #: 逐条结果
    details: list[ClaimVerdict] = field(default_factory=list)
    #: 按判定统计
    by_verdict: dict[str, int] = field(default_factory=dict)
    #: 按规则类型统计（tier0）
    by_rule: dict[str, int] = field(default_factory=dict)
    tier1_ran: bool = False
    tier1_model: str = ""
    #: 覆盖面：有引用的句子数 / 全部句子数
    sentences: int = 0
    cited_sentences: int = 0
    notes: list[str] = field(default_factory=list)

    @property
    def supported_rate(self) -> float | None:
        """tier0 未报警的比例。**这不是"正确率"**，见 notes。"""
        if not self.claims:
            return None
        ok = self.by_verdict.get("supported", 0)
        return ok / self.claims

    def to_dict(self) -> dict[str, Any]:
        return {
            "claims": self.claims,
            "sentences": self.sentences,
            "cited_sentences": self.cited_sentences,
            "coverage": (self.cited_sentences / self.sentences) if self.sentences else None,
            "supported_rate": self.supported_rate,
            "by_verdict": dict(self.by_verdict),
            "by_rule": dict(self.by_rule),
            "tier1_ran": self.tier1_ran,
            "tier1_model": self.tier1_model,
            "notes": list(self.notes),
            "details": [d.to_dict() for d in self.details],
        }


# ============================================================ 论断抽取
def extract_claims(draft: str, *, max_claim_chars: int = 400) -> list[Claim]:
    """把草稿切成"带引用的句子"，并跳过参考文献列表。

    两个容易出错的点：

    * **参考文献列表必须排除**：每条以 ``[n]`` 开头，会被误当成带引用的论断，
      导致报告里出现一堆"数据无法溯源"的假警报；
    * **中文没有空格**，不能按空格切句，要按中英文句末标点切。
    """
    claims: list[Claim] = []
    section = ""
    in_references = False
    offset = 0

    for raw_line in (draft or "").splitlines():
        line = raw_line.rstrip()
        stripped = line.strip()
        if _REF_HEADING_RE.match(stripped):
            in_references = True
            section = "参考文献"
            offset += len(raw_line) + 1
            continue
        if stripped.startswith("#"):
            in_references = False
            section = stripped.lstrip("# ").strip()
            offset += len(raw_line) + 1
            continue
        if in_references or _REF_LINE_RE.match(line):
            offset += len(raw_line) + 1
            continue

        for match in _SENTENCE_RE.finditer(line):
            sentence = match.group(0).strip()
            base = offset + match.start()
            if len(sentence) < 8:
                continue
            citations = _parse_citations(sentence)
            if citations:
                claims.append(
                    Claim(
                        text=sentence[:max_claim_chars],
                        citations=citations,
                        section=section,
                        offset=base,
                    )
                )
        offset += len(raw_line) + 1
    return claims


def _parse_citations(text: str) -> list[int]:
    """抽取句子里的引用编号（支持 ``[1]`` / ``[1,2]`` / ``[1-3]`` / ``【1】``）。"""
    found: list[int] = []
    for match in _CITATION_RE.finditer(text or ""):
        for token in re.split(r"[,，]", match.group(1)):
            token = token.strip()
            if not token:
                continue
            range_match = re.match(r"^(\d+)\s*[-–]\s*(\d+)$", token)
            if range_match:
                low, high = int(range_match.group(1)), int(range_match.group(2))
                if 0 < low <= high <= low + 20:
                    found.extend(range(low, high + 1))
                continue
            if token.isdigit():
                found.append(int(token))
    return sorted(set(found))


def count_sentences(draft: str) -> tuple[int, int]:
    """返回 ``(全部句子数, 带引用的句子数)`` —— 用于报告覆盖面。

    覆盖面很重要：如果只有 5% 的句子带引用，那么"支持率 100%"毫无意义。
    """
    total = 0
    cited = 0
    in_references = False
    for line in (draft or "").splitlines():
        if _REF_HEADING_RE.match(line.strip()):
            in_references = True
            continue
        if in_references or _REF_LINE_RE.match(line):
            continue
        for match in _SENTENCE_RE.finditer(line):
            sentence = match.group(0).strip()
            if len(sentence) < 8:
                continue
            total += 1
            if _parse_citations(sentence):
                cited += 1
    return total, cited


# ==================================================== Tier 0：确定性规则
def _content_words(text: str) -> set[str]:
    """抽实词（两种字符体系的并集），用于"是否有共同主题"这类弱判断。

    注意：**不要在这里重复一遍分词逻辑** —— 之前就是两处实现分头维护，
    修好了 `_lexical_profile` 却漏了这里，导致同一段文本在两处产出不同的词。
    现在统一走 `_lexical_profile`，只保留一个事实来源。
    """
    profile = _lexical_profile(text)
    return profile["latin"] | profile["cjk"]


def _is_cjk_token(token: str) -> bool:
    return bool(token) and "\u4e00" <= token[0] <= "\u9fff"


def _lexical_profile(text: str) -> dict[str, set[str]]:
    """按**字符体系**分别抽出实词：``{"latin": {...}, "cjk": {...}}``。

    为什么要分开：论断与被引文献常常**跨语言**（中文综述引英文文献是常态）。
    把中文 bigram 与英文单词塞进同一个集合算重合度是"拿橘子和苹果比"——
    比值天然很低，于是每一条跨语言引用都会被误报为"可能引错了文献"。
    实测确实如此：修好 CJK 分词后，全部中文论断都被误报。

    正确做法是**同类比同类**：英文对英文、中文 bigram 对中文 bigram；
    某一侧在对方语言里没有对应物时，这一类**不做判断**（abstain），
    由调用方按"无法核实"处理，而不是指控它有问题。
    """
    groups: dict[str, set[str]] = {"latin": set(), "cjk": set()}
    for token in _TOKEN_RE.findall((text or "").lower()):
        if token in _STOPWORDS:
            continue
        if _is_cjk_token(token):
            if len(token) <= 3:
                groups["cjk"].add(token)
                continue
            for index in range(len(token) - 1):
                gram = token[index : index + 2]
                # 以虚字**开头**的 bigram 不是实词（汉语复合词几乎不会以「的/了/与/而/等」起头）。
                # 只判断"两字皆虚"是不够的：那样「的方」「的结」会被当成实词留下。
                if gram[0] in _CJK_FUNCTION_CHARS:
                    continue
                if all(ch in _CJK_FUNCTION_CHARS for ch in gram):
                    continue
                groups["cjk"].add(gram)
        elif len(token) >= 3:
            groups["latin"].add(token)
    return groups


def _overlap_by_script(
    claim_bare: str, source_text: str, *, min_overlap: float
) -> tuple[float | None, set[str], list[str], bool]:
    """按字符体系算重合度。

    Returns:
        ``(最优重合率 或 None, 重合词, 无法判断的字符体系, 是否只靠语言无关信号)``。
        ``None`` 表示**没有任何可比较的字符体系** —— 此时必须 abstain。

    两类字符体系的门槛**不一样**，这是实测调出来的：

    * 中文（cjk）要求至少 3 个 bigram，否则比例没有统计意义；
    * 拉丁（latin）只要求 1 个 token —— 因为中文医学文本里的拉丁词几乎都是
      **术语与缩写**（rTMS / PSD / HAMD / PEDro），它们**不随语言变化**，
      是最可靠的语言无关证据。

    这个差异很关键：最初对两类统一要求 ≥3 个词，结果中文综述引英文文献时
    （中文论断里通常只有 1~2 个拉丁缩写）被判为"无法比较"，
    **实测你库里那份真实综述 21 条论断有 17 条因此落进 unverifiable** ——
    校验器对最主要的真实场景完全失效。放开拉丁侧门槛后才有判断力。
    """
    claim_groups = _lexical_profile(claim_bare)
    source_groups = _lexical_profile(source_text)

    best: float | None = None
    best_overlap: set[str] = set()
    unjudgeable: list[str] = []
    cjk_unjudgeable = False
    latin_matched = False

    for script in ("latin", "cjk"):
        claim_words = claim_groups.get(script) or set()
        source_words = source_groups.get(script) or set()
        minimum = 1 if script == "latin" else 3
        if len(claim_words) < minimum:
            # 这一侧本身词太少，比例没有意义（但要区分"没写这种文字"与"写了但太少"）
            if script == "cjk" and len(claim_words) < minimum and source_words:
                cjk_unjudgeable = True
            continue
        if not source_words:
            unjudgeable.append(script)
            if script == "cjk":
                cjk_unjudgeable = True
            continue
        overlap = claim_words & source_words
        ratio = len(overlap) / len(claim_words)
        if script == "latin" and overlap:
            latin_matched = True
        if best is None or ratio > best:
            best = ratio
            best_overlap = overlap

    # 「只靠语言无关信号」：中文这一侧无法比较（或被跳过），但拉丁术语/缩写对上了。
    # 这种情况**不能**说无法核实，但证据强度确实弱于同语言比对，需要如实标注。
    script_independent_only = bool(best is not None and latin_matched and cjk_unjudgeable)
    return best, best_overlap, unjudgeable, script_independent_only


def _number_variants(value: str) -> set[str]:
    variants = {value}
    bare = value.replace("%", "").replace("‰", "").strip()
    variants.add(bare)
    try:
        number = float(bare)
        variants.update({f"{number:g}", f"{number:.1f}", f"{number:.2f}"})
        if number == int(number):
            variants.add(str(int(number)))
    except ValueError:
        pass
    return {v for v in variants if v}


def _strip_citations(text: str) -> str:
    return _CITATION_RE.sub(" ", text or "")


def check_claim_rules(
    claim: Claim,
    sources: Mapping[int, str],
    *,
    source_meta: Mapping[int, Mapping[str, Any]] | None = None,
    valid_ids: Iterable[int] | None = None,
    min_overlap: float = 0.34,
) -> list[dict[str, Any]]:
    """Tier 0：对一条论断跑确定性规则，返回问题列表（空表示未发现问题）。

    Args:
        sources: ``引用编号 → 可用文本``（摘要或全文）。
        valid_ids: **确实存在的**引用编号集合。与 ``sources`` 分开是必要的：
            "编号不存在"（越界引用）与"编号存在但拿不到文本"（无法核实）
            是两回事，处理方式也不同。``None`` 时退化为"sources 里有键才算存在"。

    取向是**宁可漏报、不可误报**：任何一条都可能被人拿去做判断，
    误报会让人不再信任这份报告。因此每条规则都要求多个条件同时成立。
    """
    problems: list[dict[str, Any]] = []
    text = claim.text
    bare = _strip_citations(text)
    available = {cid: (sources.get(cid) or "") for cid in claim.citations}
    combined_source = "\n".join(v for v in available.values() if v)

    # --- 规则 1：编号不存在（越界引用；正常已被上游剔除，这里兜底）
    known = set(valid_ids) if valid_ids is not None else set(sources)
    missing = [cid for cid in claim.citations if cid not in known]
    if missing:
        problems.append({
            "rule": "existence",
            "severity": "high",
            "detail": f"引用了不存在的编号 {missing}",
            "suggestion": "修正或删除该引用",
        })

    # --- 规则 1b：编号存在但拿不到任何文本 → 无法核实（不是"不支持"）
    if not combined_source.strip() and not missing:
        problems.append({
            "rule": "no_source_text",
            "severity": "medium",
            "detail": "被引文献没有可用的摘要或全文，无法核对",
            "suggestion": "补充全文（OA 来源或图书馆跳转）后再核对",
        })
        return problems

    # --- 规则 2：论断里的数字必须在被引文献里找得到
    source_lower = combined_source.lower()
    unverified: list[str] = []
    for match in _NUMBER_RE.finditer(bare):
        value = match.group(1).strip()
        if value in _IGNORABLE_NUMBERS:
            continue
        # 年份不算"数据"
        if re.fullmatch(r"(19|20)\d{2}", value):
            continue
        if not any(v in combined_source for v in _number_variants(value)):
            unverified.append(value)
    if unverified:
        problems.append({
            "rule": "numbers",
            "severity": "high",
            "detail": f"数字 {unverified[:6]} 在被引文献中找不到出处",
            "suggestion": "核对数据来源；医学论文里编造数据属于学术不端",
        })

    # --- 规则 3：强主张 vs 被引文献的阴性结论（矛盾信号）
    strong = [cue for cue in _STRONG_CLAIM_CUES if cue in text.lower() or cue in text]
    null_cues = [cue for cue in _NULL_RESULT_CUES if cue in source_lower]
    if strong and null_cues:
        # 要求论断与被引文献在实词上有重合，降低"引错文献导致的假矛盾"
        overlap = _content_words(bare) & _content_words(combined_source)
        if overlap:
            problems.append({
                "rule": "direction",
                "severity": "high",
                "detail": (
                    f"论断使用了强主张（{'、'.join(strong[:3])}），"
                    f"而被引文献出现阴性结果表述（{'、'.join(null_cues[:2])}）"
                    f"；共同主题词：{sorted(overlap)[:5]}"
                ),
                "suggestion": "核对原文结论方向 —— 这是最严重的一类引用错误，务必逐字复核",
            })

    # --- 规则 4：超出证据强度的措辞
    if strong:
        meta = (source_meta or {}).get(claim.citations[0], {}) if claim.citations else {}
        weak = [cue for cue in _WEAK_EVIDENCE_CUES if cue in source_lower]
        pub_type = str(meta.get("publication_type") or "").lower()
        if weak or pub_type in {"preprint", "protocol", "case-report"}:
            problems.append({
                "rule": "overclaim",
                "severity": "medium",
                "detail": (
                    f"论断措辞较强（{'、'.join(strong[:3])}），"
                    f"但被引文献的证据载体较弱"
                    + (f"（{weak[0]}）" if weak else f"（{pub_type}）")
                ),
                "suggestion": "改用与证据强度相称的表述（如「提示」「相关」而非「证实」「治愈」）",
            })

    # --- 规则 5：实词重合度过低（可能引错了文献）
    #     按字符体系分开比较；没有可比较的一侧时**必须 abstain**，
    #     不能因为"中文论断 vs 英文文献"就指控它引错了文献。
    ratio, overlap, unjudgeable, script_only = _overlap_by_script(
        bare, combined_source, min_overlap=min_overlap
    )
    if ratio is not None and ratio < min_overlap:
        problems.append({
            "rule": "grounding",
            "severity": "medium",
            "detail": (
                f"论断与被引文献的实词重合度仅 {ratio:.0%}"
                f"（重合：{sorted(overlap)[:5] or '无'}）"
            ),
            "suggestion": "确认引用编号对应的是正确的文献；纯语义改写也可能造成低重合",
        })
    elif ratio is None and unjudgeable:
        problems.append({
            "rule": "cross_lingual",
            "severity": "low",
            "detail": (
                "论断与被引文献不在同一字符体系（"
                + "、".join(unjudgeable)
                + "），且没有共同的语言无关信号（数字/术语/缩写），无法核对"
            ),
            "suggestion": "跨语言引用的支持性需要 Tier 1（LLM 裁判或 NLI 模型）判断",
        })

    if script_only:
        # 只靠术语/缩写对上（如论断里的 rTMS 出现在英文文献里）——能过，但证据弱，
        # 如实标注出来，报告里会统计有多少条属于这种情况。
        problems.append({
            "rule": "script_independent_only",
            "severity": "low",
            "detail": (
                "跨语言引用：仅凭语言无关信号（数字/术语/缩写 "
                + "、".join(sorted(overlap and list(overlap) or [])[:4])
                + "）判定相关，未做语义蕴含核对"
            ),
            "suggestion": "要确认语义是否真被支持，请开 Tier 1",
        })

    return problems


def _verdict_from_problems(problems: Sequence[Mapping[str, Any]]) -> str:
    """按最严重的问题决定判定。

    **仲裁顺序是明确的、有意的**（并在文档与标注集里都写出来）：
    ``existence/numbers`` > ``direction`` > ``overclaim`` > ``grounding``
    > ``unverifiable`` > ``weakly_supported`` > ``supported``。

    为什么 numbers 优先于 direction：编造数字是最硬、最容易复核、后果最严重的问题；
    而"方向矛盾"有赖于线索词匹配，置信度略低一档。
    多条规则同时命中时，取**最可执行**的那一条，而不是把判定搅在一起。
    """
    rules = {p.get("rule") for p in problems}
    if "existence" in rules or "numbers" in rules:
        return "unsupported"
    if "direction" in rules:
        return "contradicted"
    if "overclaim" in rules:
        return "overclaim"
    if "grounding" in rules:
        return "unsupported"
    if "no_source_text" in rules or "cross_lingual" in rules:
        # 「拿不到文本」与「跨语言且无任何语言无关信号」都属于**无法核实**，
        # 不应该被报成"不支持"（那会冤枉一篇可能完全没问题的引用）。
        return "unverifiable"
    if "script_independent_only" in rules:
        # 有信号、能过，但只靠语言无关信号（数字/术语/缩写）——
        # 单列一档，避免被误读成"已核实语义支持"
        return "weakly_supported"
    return "supported"


# ==================================================== Tier 1：LLM 裁判
_JUDGE_SYSTEM = (
    "你是医学文献核查专家。给定一句带引用的论断，以及**该引用所指文献**的摘要/正文片段，"
    "判断该文献是否支持这句话。\n\n"
    "判定必须从下列四选一：\n"
    "- supported：文献明确支持该论断；\n"
    "- partial：部分支持，或证据弱于论断的表述强度；\n"
    "- unsupported：文献中找不到支持该论断的内容；\n"
    "- contradicted：文献结论与该论断**方向相反**。\n\n"
    "严格要求：\n"
    "1. 只依据给出的文献片段判断，不要用你自己的先验知识；\n"
    "2. 不要因为论断写得流畅就判为 supported；\n"
    "3. 必须引用文献片段中的**原文词句**作为证据；找不到就填空并在理由里说明；\n"
    "4. 只输出 JSON：{\"verdict\": \"...\", \"evidence\": \"原文片段\", \"reason\": \"一句话理由\"}"
)


def _build_judge_prompt(claim: Claim, sources: Mapping[int, str], *, max_source_chars: int = 2200) -> str:
    blocks = []
    for cid in claim.citations:
        text = (sources.get(cid) or "").strip()
        blocks.append(f"【文献 [{cid}]】\n{text[:max_source_chars] or '（无可用内容）'}")
    return (
        "待核查的论断（来自一篇综述草稿）：\n"
        f"「{_strip_citations(claim.text).strip()}」\n\n"
        "该论断引用的文献内容：\n" + "\n\n".join(blocks)
    )


async def verify_claims_llm(
    claims: Sequence[Claim],
    sources: Mapping[int, str],
    *,
    config: Any = None,
    max_claims: int = 40,
    self_consistency: bool = False,
) -> dict[int, ClaimVerdict]:
    """Tier 1：用 LLM 逐条核查。返回 ``{claim 下标: ClaimVerdict}``。

    ``self_consistency=True`` 时每条问两次（不同温度），两次判定不一致的标为
    ``uncertain`` —— 这是对"裁判本身有噪声"的诚实处理，而不是假装它稳定。
    """
    from ..config import get_config
    from ..llm.client import LLMError, get_llm

    cfg = config or get_config()
    client = get_llm(cfg)
    await client.start()

    out: dict[int, ClaimVerdict] = {}
    for index, claim in enumerate(claims[:max_claims]):
        prompt = _build_judge_prompt(claim, sources)
        try:
            payload = await client.chat_json(
                [{"role": "user", "content": prompt}],
                system=_JUDGE_SYSTEM,
                temperature=0.1,
                max_tokens=400,
                retries=1,
            )
        except LLMError as exc:
            logger.warning("第 %d 条论断核查失败：%s", index + 1, exc)
            continue
        if not isinstance(payload, dict):
            continue

        verdict = str(payload.get("verdict") or "").strip().lower()
        if verdict not in {"supported", "partial", "unsupported", "contradicted"}:
            verdict = "unsupported"
        result = ClaimVerdict(
            claim=claim,
            verdict={"partial": "overclaim"}.get(verdict, verdict),
            tier="tier1",
            evidence=str(payload.get("evidence") or "")[:500],
            reason=str(payload.get("reason") or "")[:300],
        )

        if self_consistency:
            try:
                again = await client.chat_json(
                    [{"role": "user", "content": prompt}],
                    system=_JUDGE_SYSTEM,
                    temperature=0.7,
                    max_tokens=400,
                    retries=0,
                )
                second = str((again or {}).get("verdict") or "").strip().lower()
                if second and second != verdict:
                    result.uncertain = True
                    result.reason = (
                        f"两次判定不一致（{verdict} vs {second}）—— 需要人工复核。" + result.reason
                    )
            except LLMError:
                pass
        out[index] = result
    return out


# ============================================================ 整体分析
def analyse_draft(
    draft: str,
    sources: Mapping[int, str],
    *,
    source_meta: Mapping[int, Mapping[str, Any]] | None = None,
    valid_ids: Iterable[int] | None = None,
    llm_verdicts: Mapping[int, ClaimVerdict] | None = None,
    tier1_model: str = "",
) -> FaithfulnessReport:
    """对整篇草稿做支持性分析（Tier 0 必跑；Tier 1 结果可选传入）。

    ``valid_ids`` 应传 ``citation_map`` 的键 —— 这样"越界引用"与"拿不到全文"
    才能被正确区分。
    """
    claims = extract_claims(draft)
    total, cited = count_sentences(draft)

    report = FaithfulnessReport(
        claims=len(claims),
        sentences=total,
        cited_sentences=cited,
        tier1_ran=bool(llm_verdicts),
        tier1_model=tier1_model,
    )

    by_verdict: dict[str, int] = {}
    by_rule: dict[str, int] = {}
    for index, claim in enumerate(claims):
        problems = check_claim_rules(
            claim, sources, source_meta=source_meta, valid_ids=valid_ids
        )
        verdict = ClaimVerdict(
            claim=claim,
            verdict=_verdict_from_problems(problems),
            problems=problems,
            tier="tier0",
        )

        # Tier 1 结果与 Tier 0 取**更严重**的一方，并保留两者的信息
        llm = (llm_verdicts or {}).get(index)
        if llm is not None:
            verdict.tier = "tier0+tier1"
            verdict.evidence = llm.evidence
            verdict.reason = llm.reason
            verdict.uncertain = llm.uncertain
            if llm.severity > verdict.severity:
                verdict.verdict = llm.verdict
            verdict.problems = list(verdict.problems) + [{
                "rule": "llm_judge",
                "severity": "high" if llm.verdict in {"contradicted", "unsupported"} else "medium",
                "detail": f"LLM 裁判判定：{llm.verdict}｜{llm.reason}",
                "suggestion": "结合证据片段人工复核",
            }]

        by_verdict[verdict.verdict] = by_verdict.get(verdict.verdict, 0) + 1
        for problem in verdict.problems:
            rule = str(problem.get("rule") or "unknown")
            by_rule[rule] = by_rule.get(rule, 0) + 1
        report.details.append(verdict)

    report.by_verdict = by_verdict
    report.by_rule = by_rule

    # 覆盖面与解释性说明 —— 必须写进报告，避免被过度解读
    if total:
        coverage = cited / total
        if coverage < 0.25:
            report.notes.append(
                f"⚠️ 只有 {coverage:.0%} 的句子带引用（{cited}/{total}），"
                "支持率的分母很小，不要据此认为整篇已被核实。"
            )
    report.notes.append(
        "Tier 0 规则只覆盖**已知的高置信度问题模式**；"
        "未报警 ≠ 已核实。要做真正的蕴含判断请开 Tier 1（LLM 裁判）或接入 NLI 模型。"
    )
    # 跨语言论断是本项目最主要的真实场景（中文综述引英文文献）。
    # Tier 0 在那种情况下只能靠数字/术语/缩写这类语言无关信号，必须如实说明覆盖率。
    weak = by_rule.get("script_independent_only", 0)
    abstained = by_rule.get("cross_lingual", 0)
    if weak or abstained:
        report.notes.append(
            f"跨语言引用：{weak} 条仅凭语言无关信号（数字/术语/缩写）核对通过，"
            f"{abstained} 条因无任何共同信号而**无法核实**。"
            "这类论断的语义支持性需要 Tier 1（`--llm`）才能确认 —— "
            "对中文综述引英文文献的场景，建议开启。"
        )
    if report.tier1_ran:
        report.notes.append(
            "Tier 1 由 LLM 判定，已知偏差：偏好流畅文本、对「部分支持」不稳定。"
            "建议对 contradicted / unsupported 逐条人工复核。"
        )
    return report
