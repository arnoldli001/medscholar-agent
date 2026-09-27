"""引用支持性校验 —— Tier 0 确定性规则（纯函数、零 IO、离线可用）。

`[n]` 是否真的支持被标注的那句话？区别于只查编号存在性的
`formatter.validate_citations`——**存在不代表支持，结论说反是最严重的错误之一**。

五类规则（每条都要求多条件同时成立，取向宁可漏报不可误报）：

- `existence` 编号不存在（越界引用，正常已被上游剔除，这里兜底）
- `numbers`   论断里的数字在被引文献中找不到出处（编造数据强信号）
- `direction` 强主张措辞撞上被引文献的阴性结论
- `overclaim` 措辞超出证据强度（如把相关写成导致）
- `grounding` 与所引文献的实词重合度过低

**Tier 0 未报警 ≠ 已核实**，此声明必须随报告一起展示（见 :class:FaithfulnessReport.notes）。

Tier 1（LLM 逐条裁判）需要调模型，属于基础设施层，见
:mod:medscholar.infrastructure.faithfulness_judge；其结果通过
:func:nalyse_draft 的 `llm_verdicts` 参数合并进来，与 Tier 0 取更严重的一方。

分层说明：本模块只依赖标准库，因此可以住在 `domain/`，被生成链路
（`agent` 属 application 层）合法调用——这正是它原先住在 `eval/` 时做不到的事。
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
    "analyse_draft",
    "VERDICTS",
    "SEVERITY_ORDER",
    "count_sentences",
]

#: 判定取值（按严重程度从轻到重）。weakly_supported 单列一档：跨语言引用只能靠
#: 数字/术语/缩写等语言无关信号核对，报 supported 会被误读为已核实，报 unverifiable
#: 又会让中文综述引英文文献这一主要场景全部"无法核实"。
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
        """tier0 未报警的比例。注意这不是"正确率"，见 notes。"""
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
    """切出带引用的句子。两个坑：参考文献条目以 [n] 开头会被误当论断，必须排除；
    中文无空格，按中英文句末标点切句。"""
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
    """返回 (全部句子数, 带引用的句子数)，用于报告引用覆盖面（覆盖面太低则支持率无意义）。"""
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
    """抽两种字符体系实词的并集；统一走 _lexical_profile，避免分词逻辑多处分叉。"""
    profile = _lexical_profile(text)
    return profile["latin"] | profile["cjk"]


def _is_cjk_token(token: str) -> bool:
    return bool(token) and "\u4e00" <= token[0] <= "\u9fff"


def _lexical_profile(text: str) -> dict[str, set[str]]:
    """按字符体系分抽实词 {"latin", "cjk"}：跨语言引用是常态，中英混在一个集合算
    重合度会把所有中文论断误报为引错文献。只做同类比对，一侧无可比物时 abstain。"""
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
                # 以虚字开头的 bigram 即判虚词；只判"两字皆虚"会漏掉「的方」这类噪声。
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
    """按字符体系算重合度，返回 (最优重合率或 None, 重合词, 无法判断的体系,
    是否只靠语言无关信号)；None 表示无可比体系，必须 abstain。

    门槛实测：cjk 至少 3 个 bigram 否则比例无意义；latin 只要 1 个 token——
    中文医学文本里的拉丁词几乎都是跨语言不变的术语缩写（rTMS/HAMD），是最可靠信号。
    两侧统一 ≥3 会让中文综述引英文文献的主要场景大面积落进 unverifiable。
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

    # 「只靠语言无关信号」：中文侧无法比较但拉丁术语对上了，能过但证据弱于同语言比对。
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
    """Tier 0：对一条论断跑确定性规则，返回问题列表（空=未发现问题）。

    valid_ids 与 sources 必须分开：编号不存在（越界引用）与编号存在但拿不到文本
    （无法核实）是两回事。取向宁可漏报不可误报，每条规则都要求多条件同时成立。
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

    # --- 规则 5：实词重合度过低（可能引错文献）；按字符体系分比，无可比一侧时 abstain。
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
        # 仅凭术语/缩写对上，能过但证据弱，单列规则供报告统计。
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
    """按仲裁顺序取最严重问题：existence/numbers > direction > overclaim > grounding
    > unverifiable > weakly_supported > supported。numbers 先于 direction：编造数字
    最硬可复核，方向矛盾仅靠线索词匹配置信度低一档。"""
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
        # 拿不到文本 / 跨语言且无共同信号 → 无法核实，不能报"不支持"冤枉好引用。
        return "unverifiable"
    if "script_independent_only" in rules:
        # 有信号能过但只靠语言无关信号，单列一档避免被误读为"语义已核实"。
        return "weakly_supported"
    return "supported"


def analyse_draft(
    draft: str,
    sources: Mapping[int, str],
    *,
    source_meta: Mapping[int, Mapping[str, Any]] | None = None,
    valid_ids: Iterable[int] | None = None,
    llm_verdicts: Mapping[int, ClaimVerdict] | None = None,
    tier1_model: str = "",
) -> FaithfulnessReport:
    """整篇草稿支持性分析（Tier 0 必跑，Tier 1 结果可选传入）。

    valid_ids 应传 citation_map 的键，以区分越界引用与拿不到全文。
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

        # Tier 1 结果与 Tier 0 取更严重的一方，并保留两者的信息
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
    # 中文综述引英文文献是主要场景，Tier 0 只能靠语言无关信号，必须如实说明覆盖率。
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
