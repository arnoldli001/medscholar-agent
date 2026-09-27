"""引用支持性校验 —— Tier 1（LLM 逐条裁判）+ Tier 0 兼容外壳。

**本模块的定位**

Tier 0 的确定性规则已下沉到 :mod:`medscholar.domain.faithfulness`（纯函数、零 IO）。
原先整个文件住在 ``eval/``，而架构校验器规定「运行时不得依赖 eval」——于是生成链路
（``agent`` 属 application 层）想调用这组规则就会违规。这正是它长期只活在 CLI、
没接进 review 阶段的原因。搬到 ``domain/`` 之后这条路才通
（见 ``medscholar/agent/graph.py`` 的 ``_tier0_faithfulness``）。

留在本模块的是 **Tier 1**：需要 ``config`` 与 ``llm.client`` 的 LLM 裁判。
它目前只被评测脚本（``scripts/eval_faithfulness.py --llm``）与自评估使用，
不需要进生成链路，因此留在 ``eval/`` 是合适的——eval 层允许依赖任意层。

同时按项目惯例重导出 Tier 0 的全部公开名（含被测试/脚本引用的私有名），
保持 ``medscholar.eval.faithfulness.xxx`` 这条既有路径可用。
新代码请按需直接 import 对应层：

- 纯规则 → :mod:`medscholar.domain.faithfulness`
- LLM 裁判 → 本模块 :func:`verify_claims_llm`

**Tier 1 的设计取舍**（已知边界，写在这里备查）

- 判定不在词表内一律降级为 ``unsupported`` 而不是抛错：裁判输出的健壮性优先，
  代价是"模型乱答"与"证据不足"在外层无法区分。
- ``chat_json`` 失败时该条论断被跳过：调用方看到的是"这条没有 Tier 1 结论"，
  而不是"这条被判定为不支持"——两者语义不同，不能混。
- ``self_consistency=True`` 用不同温度问两次，不一致标 ``uncertain``：
  用温度差换一点独立性，代价是双倍 token。
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Sequence

from ..constants import (
    JUDGE_EVIDENCE_MAX,
    JUDGE_MAX_CLAIMS,
    JUDGE_REASON_MAX,
    LLM_MAX_TOKENS_JUDGE,
    LLM_TEMPERATURE_JUDGE,
    LLM_TEMPERATURE_JUDGE_SC,
)
from ..domain.faithfulness import Claim, ClaimVerdict, _strip_citations

# Tier 0 兼容重导出：``medscholar.eval.faithfulness.X`` 这条路径继续可用。
from ..domain.faithfulness import *  # noqa: F401,F403
from ..domain.faithfulness import __all__ as _TIER0_ALL  # noqa: F401
from ..domain.faithfulness import (  # noqa: F401
    _content_words,
    _lexical_profile,
    _overlap_by_script,
    _parse_citations,
    _verdict_from_problems,
    count_sentences,
)

logger = logging.getLogger(__name__)

#: 公开面 = Tier 0 全部 + Tier 1 入口（与搬迁前的 ``__all__`` 语义一致）。
__all__ = [*_TIER0_ALL, "verify_claims_llm"]


# ===========================================================================
# Tier 1：LLM 逐条裁判
# ===========================================================================

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
    max_claims: int = JUDGE_MAX_CLAIMS,
    self_consistency: bool = False,
) -> dict[int, ClaimVerdict]:
    """Tier 1：LLM 逐条核查，返回 {claim 下标: ClaimVerdict}。

    self_consistency=True 时用不同温度问两次，不一致标 uncertain。
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
                temperature=LLM_TEMPERATURE_JUDGE,
                max_tokens=LLM_MAX_TOKENS_JUDGE,
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
            evidence=str(payload.get("evidence") or "")[:JUDGE_EVIDENCE_MAX],
            reason=str(payload.get("reason") or "")[:JUDGE_REASON_MAX],
        )

        if self_consistency:
            try:
                again = await client.chat_json(
                    [{"role": "user", "content": prompt}],
                    system=_JUDGE_SYSTEM,
                    temperature=LLM_TEMPERATURE_JUDGE_SC,
                    max_tokens=LLM_MAX_TOKENS_JUDGE,
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
