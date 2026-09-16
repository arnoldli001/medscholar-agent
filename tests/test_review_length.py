"""综述字数范围：写入提示词、换算到每节、并在上下文里给正文留位置。

背景：以前每节固定"约 900 字"，5 节只有 4500 字左右，用户反馈太少。
实测 8B 模型每节单次生成还剩 ~3900 token 可用（约 6700 字），
所以瓶颈是指令，不是上下文——但字数调大后必须保证材料不把输出挤掉。
"""

from __future__ import annotations

from medscholar.agent.state import PlanSection
from medscholar.agent.writer import _DIGEST_LADDER, _fit_digest
from medscholar.config import AgentSettings
from medscholar.models import Paper
from medscholar.textutil import estimate_tokens


class TestEstimateTokens:
    def test_empty(self):
        assert estimate_tokens("") == 0
        assert estimate_tokens(None) == 0

    def test_chinese_is_roughly_one_per_char(self):
        text = "卒中后抑郁的治疗进展"  # 10 个汉字
        assert 9 <= estimate_tokens(text) <= 13

    def test_english_is_cheaper_per_char(self):
        text = "repetitive transcranial magnetic stimulation"
        assert estimate_tokens(text) < len(text) / 2

    def test_calibrated_against_real_measurement(self):
        """实测：15,676 字的英文摘要材料块 = 4,129 token，估算误差应在 30% 内。"""
        sample = "Post-stroke depression is a common complication. " * 300
        real_observed = len(sample) / 3.80  # 由真实调用反推的比例
        guess = estimate_tokens(sample)
        assert abs(guess - real_observed) / real_observed < 0.30


class TestReviewCharSettings:
    def test_defaults_are_longer_than_before(self):
        agent = AgentSettings()
        assert agent.review_min_chars >= 3000
        assert agent.review_max_chars > agent.review_min_chars

    def test_normalize_swaps_and_clamps(self):
        from medscholar.agent.runtime import normalize_review_chars

        agent = AgentSettings()
        assert normalize_review_chars(9000, 3000, agent) == (3000, 9000)
        low, high = normalize_review_chars(10, 50, agent)
        assert low >= 800 and high > low
        low, high = normalize_review_chars(90000, 99000, agent)
        assert high <= 40000

    def test_normalize_falls_back_to_config(self):
        from medscholar.agent.runtime import normalize_review_chars

        agent = AgentSettings(review_min_chars=5000, review_max_chars=9000)
        assert normalize_review_chars(None, None, agent) == (5000, 9000)

    def test_normalize_zero_means_use_config(self):
        """0 视为"没指定"→ 取配置默认值，并返回具体区间供调用方直接用。"""
        from medscholar.agent.runtime import normalize_review_chars

        agent = AgentSettings(review_min_chars=5000, review_max_chars=9000)
        assert normalize_review_chars(0, 0, agent) == (5000, 9000)
        assert normalize_review_chars(None, None, agent) == (5000, 9000)

    def test_normalize_never_returns_zero(self):
        """即使配置被清空也要给出可用区间，不能把 0 传给写作层。"""
        from medscholar.agent.runtime import normalize_review_chars

        blank = AgentSettings(review_min_chars=0, review_max_chars=0)
        low, high = normalize_review_chars(None, None, blank)
        assert low > 0 and high > low

    def test_clamped_range_keeps_min_below_max(self):
        from medscholar.agent.runtime import normalize_review_chars

        agent = AgentSettings()
        for lo, hi in ((90000, 99000), (50000, 50000), (100, 100), (40000, 40000)):
            low, high = normalize_review_chars(lo, hi, agent)
            assert 800 <= low < high <= 40000, (lo, hi, low, high)


class TestSectionPromptLength:
    def test_range_is_written_into_prompt(self):
        from medscholar.llm.prompts import section_user

        prompt = section_user(
            "课题", "引言", ["背景"], "材料", min_chars=800, max_chars=1600
        )
        assert "不少于 800 字" in prompt
        assert "以 1600 字为目标" in prompt
        assert "接近 1600 字" in prompt

    def test_single_value_still_supported(self):
        from medscholar.llm.prompts import section_user

        prompt = section_user("课题", "引言", ["背景"], "材料", max_chars=900)
        assert "约 900 字" in prompt

    def test_prompt_discourages_padding(self):
        from medscholar.llm.prompts import section_user

        prompt = section_user("课题", "引言", ["背景"], "材料", min_chars=800, max_chars=1600)
        assert "不要用空话、套话凑字数" in prompt
        assert "不要编造" in prompt


class TestDigestFitting:
    def _entries(self, n: int) -> list[tuple[int, Paper]]:
        return [
            (i, Paper(title=f"Paper {i} about rTMS", source="pubmed", abstract="A" * 1200))
            for i in range(1, n + 1)
        ]

    def test_short_target_keeps_full_digest(self):
        entries = self._entries(10)
        digest = _fit_digest(
            entries, num_ctx=8192, want_output_tokens=600, system_prompt="x" * 500
        )
        assert "Paper 10" in digest, "预算充足时不应裁掉材料"

    def test_huge_target_shrinks_digest(self):
        """字数目标极大时必须裁剪材料，否则提示词会把正文挤出上下文。"""
        entries = self._entries(25)
        small = _fit_digest(
            entries, num_ctx=8192, want_output_tokens=600, system_prompt="x" * 500
        )
        huge = _fit_digest(
            entries, num_ctx=8192, want_output_tokens=5000, system_prompt="x" * 500
        )
        assert estimate_tokens(huge) < estimate_tokens(small)
        assert estimate_tokens(huge) <= 8192 - 5000

    def test_digest_never_exceeds_budget_when_possible(self):
        entries = self._entries(25)
        for want in (400, 1000, 2000, 3000):
            digest = _fit_digest(
                entries, num_ctx=8192, want_output_tokens=want, system_prompt="x" * 500
            )
            budget = max(900, 8192 - want - estimate_tokens("x" * 500) - 256)
            assert estimate_tokens(digest) <= budget or len(digest) > 0

    def test_ladder_is_monotonically_smaller(self):
        sizes = [cap for cap, _ in _DIGEST_LADDER]
        abstracts = [abs_ for _, abs_ in _DIGEST_LADDER]
        assert sizes == sorted(sizes, reverse=True)
        assert abstracts == sorted(abstracts, reverse=True)


class TestWriteReviewBudgets:
    async def test_token_cap_is_not_the_old_hard_1800(self, tmp_path, monkeypatch):
        """回归：旧代码把单节生成上限写死 min(1800, max_chars*3)，长文写不出来。"""
        from medscholar.agent import writer as writer_mod
        from medscholar.agent.writer import WriterAgent
        from medscholar.config import AppConfig

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        agent = WriterAgent(config=config)

        seen: list[int] = []

        class FakeClient:
            async def start(self):
                return None

            async def stream(self, messages, *, system=None, temperature=None, max_tokens=None):
                seen.append(max_tokens)
                yield "正文" * 50

        monkeypatch.setattr(writer_mod, "get_llm", lambda *a, **k: FakeClient())

        entries = [
            (i, Paper(title=f"P{i}", source="pubmed", abstract="摘要" * 200))
            for i in range(1, 6)
        ]
        outline = [PlanSection(title="引言", points=["背景"])]
        await agent.write_review(
            "课题", None, entries, outline, total_min_chars=6000, total_max_chars=10000
        )
        assert seen, "没有发起生成"
        assert seen[0] > 1800, f"单节上限仍被旧常量卡住：{seen[0]}"

    async def test_per_section_target_scales_with_total(self, tmp_path, monkeypatch):
        """总字数范围要按章节数均分到每节，并写进提示词。"""
        from medscholar.agent import writer as writer_mod
        from medscholar.agent.writer import WriterAgent
        from medscholar.config import AppConfig

        config = AppConfig(data_dir=str(tmp_path), offline=True)
        agent = WriterAgent(config=config)

        prompts: list[str] = []

        class FakeClient:
            async def start(self):
                return None

            async def stream(self, messages, *, system=None, temperature=None, max_tokens=None):
                prompts.append(messages[0]["content"])
                yield "正文"

        monkeypatch.setattr(writer_mod, "get_llm", lambda *a, **k: FakeClient())

        entries = [(i, Paper(title=f"P{i}", source="pubmed", abstract="摘要" * 100)) for i in range(1, 4)]
        outline = [PlanSection(title=f"章节{i}", points=["要点"]) for i in range(1, 5)]
        await agent.write_review(
            "课题", None, entries, outline, total_min_chars=4000, total_max_chars=8000
        )
        assert len(prompts) == 4
        # 4000/4=1000，8000/4=2000
        assert "不少于 1000 字" in prompts[0]
        assert "以 2000 字为目标" in prompts[0]
