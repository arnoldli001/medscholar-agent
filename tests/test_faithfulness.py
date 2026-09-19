"""引用支持性校验（claim-level faithfulness）与校验器自评估。

这里有两层测试，缺一不可：

1. **规则本身**（抽取、各条规则、判定仲裁）—— 用手工构造的明确案例钉死；
2. **校验器的自评估**（在人工标注集上的 precision/recall）—— 一个从没被衡量过的
   校验器，可信度和"没有校验器"差不多。而且这里的自评估**真的抓到过 bug**：
   CJK 分词错误导致中文规则静默失效、以及跨语言误报。
"""

from __future__ import annotations

import pytest

from medscholar.eval.faithfulness import (
    Claim,
    analyse_draft,
    check_claim_rules,
    count_sentences,
    extract_claims,
    _content_words,
    _lexical_profile,
    _overlap_by_script,
    _parse_citations,
    _verdict_from_problems,
)
from medscholar.eval.faithfulness_eval import (
    LABELS_PATH,
    evaluate_rules,
    load_labeled_cases,
    render_faithfulness_report,
)


# ============================================================ 论断抽取
class TestCitationParsing:
    def test_single_and_multi(self):
        assert _parse_citations("结论如此 [1]。") == [1]
        assert _parse_citations("多个来源 [1,2,3]。") == [1, 2, 3]

    def test_ranges(self):
        assert _parse_citations("综述 [1-3]。") == [1, 2, 3]
        assert _parse_citations("范围 [2–4]。") == [2, 3, 4]

    def test_chinese_brackets(self):
        assert _parse_citations("中文引用【5】。") == [5]

    def test_dedupes_and_sorts(self):
        assert _parse_citations("重复 [3,1,3]。") == [1, 3]

    def test_no_citation(self):
        assert _parse_citations("这句话没有引用。") == []

    def test_absurd_range_is_ignored(self):
        # 防止 [1-9999] 这类噪声把引用编号撑爆
        assert _parse_citations("异常 [1-9999]。") == []


class TestExtractClaims:
    def test_extracts_only_cited_sentences(self):
        claims = extract_claims(
            "这是没有引用的一句。这是有引用的一句 [1]。另一句也没有引用。"
        )
        assert len(claims) == 1
        assert claims[0].citations == [1]

    def test_reference_list_is_excluded(self):
        """参考文献列表以 [n] 开头，必须排除，否则会产生一堆假警报。"""
        draft = (
            "## 引言\n有一个论断 [1]。\n\n"
            "## 参考文献\n[1] 张三. 某研究. 期刊, 2021.\n[2] 李四. 另一研究. 期刊, 2020.\n"
        )
        claims = extract_claims(draft)
        assert len(claims) == 1
        assert "张三" not in claims[0].text

    def test_english_references_heading(self):
        draft = "We found an effect [1].\n\n## References\n[1] Smith J. A study. J, 2020.\n"
        assert len(extract_claims(draft)) == 1

    def test_section_tracking(self):
        draft = "## 方法\n我们纳入了 100 例患者 [2]。\n"
        claims = extract_claims(draft)
        assert claims[0].section == "方法"

    def test_short_sentences_ignored(self):
        assert extract_claims("短 [1]。") == []

    def test_chinese_sentence_splitting(self):
        """中文没有空格，必须按句末标点切 —— 按空格切会把整段当成一句。"""
        draft = "第一句有引用 [1]。第二句也有引用 [2]；第三句没有引用。"
        claims = extract_claims(draft)
        assert len(claims) == 2
        assert claims[0].citations == [1]
        assert claims[1].citations == [2]

    def test_empty_draft(self):
        assert extract_claims("") == []

    def test_count_sentences_reports_coverage(self):
        total, cited = count_sentences(
            "没有引用的一句。有引用的一句 [1]。还有一句没引用。"
        )
        assert total == 3
        assert cited == 1


# ============================================== CJK 分词（回归：真踩过的坑）
class TestCjkTokenization:
    def test_chinese_run_is_bigrammed_not_one_token(self):
        """回归：`[\\u4e00-\\u9fff]{2,}` 会把整句汉字变成一个 token，
        导致中文的重合度恒为 0、中文规则**静默失效**（实测中文案例全部漏报）。
        这个 bug 是被人工标注集抓出来的。"""
        words = _content_words("该疗法可完全治愈卒中后抑郁")
        assert len(words) > 1, f"整句被当成了一个 token：{words}"
        assert "卒中" in words or "抑郁" in words

    def test_shared_bigrams_detected(self):
        claim = _content_words("卒中后抑郁的治疗进展")
        source = _content_words("卒中后抑郁影响约 30% 的幸存者")
        assert claim & source, "共享的汉字词应当能被检出"

    def test_function_char_bigrams_filtered(self):
        """以虚字开头的 bigram 不是实词（否则「的方」「的结」会被当成实词留下）。"""
        words = _content_words("的方法与的结果")
        assert "的方" not in words and "的结" not in words and "与的" not in words
        assert "方法" in words and "结果" in words

    def test_profile_splits_by_script(self):
        profile = _lexical_profile("rTMS 可改善卒中后抑郁")
        assert "rtms" in profile["latin"]
        assert profile["cjk"], "中文部分应当产出 bigram"

    def test_english_tokens_kept(self):
        profile = _lexical_profile("accelerated rTMS for depression")
        assert "accelerated" in profile["latin"]
        assert "depression" in profile["latin"]


# ============================================ 跨语言：必须 abstain 而非指控
class TestCrossLingualAbstention:
    def test_pure_cross_script_has_no_signal(self):
        ratio, _, unjudgeable, script_only = _overlap_by_script(
            "针灸可改善胰岛素抵抗", "This review covers coil design.", min_overlap=0.34
        )
        assert ratio is None
        assert unjudgeable, "应当指出哪一侧无法比较"
        assert script_only is False

    def test_shared_technical_token_gives_signal(self):
        """共享 rTMS 这类术语时，英文这一侧仍然可比 —— 这是跨语言场景下
        Tier 0 唯一可用的语言无关信号。"""
        ratio, overlap, _, script_only = _overlap_by_script(
            "rTMS improves outcomes", "Accelerated rTMS improves outcomes in stroke",
            min_overlap=0.3,
        )
        assert ratio is not None and ratio >= 0.3
        assert "rtms" in overlap
        # 两边都是拉丁，不算"只靠语言无关信号"
        assert script_only is False

    def test_single_acronym_in_chinese_claim_is_enough(self):
        """**实测驱动的关键修正**：中文论断里通常只有 1~2 个拉丁缩写，
        按原设计（两侧统一要求 ≥3 个词）会被判为无法比较 ——
        结果真实综述里 21 条论断有 17 条落进 unverifiable，工具对主场景失效。
        放开拉丁侧门槛后才有判断力。"""
        ratio, overlap, _, script_only = _overlap_by_script(
            "加速rTMS可缩短起效时间",
            "Accelerated rTMS shortened the time to onset in post-stroke depression.",
            min_overlap=0.34,
        )
        assert ratio == 1.0, "单个术语命中就应当被视为有信号"
        assert "rtms" in overlap
        assert script_only is True, "但必须标注为弱证据（只靠语言无关信号）"

    def test_acronym_mismatch_is_caught(self):
        """语言无关信号不只是"能过"，它也能抓错。"""
        ratio, overlap, _, _ = _overlap_by_script(
            "加速rTMS可改善卒中后抑郁",
            "This paper reports a physiotherapy dosing trial after knee replacement.",
            min_overlap=0.34,
        )
        assert ratio == 0.0
        assert not overlap

    def test_cross_lingual_claim_abstains_rather_than_accuses(self):
        """这是**最重要的一条**：中文综述引英文文献是常态，
        词面代理在这里没有判断力，必须 abstain（unverifiable），
        绝不能被报成"不支持/引错了文献"。"""
        problems = check_claim_rules(
            Claim(text="针灸可改善胰岛素抵抗 [1]。", citations=[1]),
            {1: "This review covers transcranial magnetic stimulation coil design."},
        )
        rules = {p["rule"] for p in problems}
        assert "cross_lingual" in rules
        assert "grounding" not in rules, "跨语言时不应触发低重合指控"
        assert _verdict_from_problems(problems) == "unverifiable"

    def test_same_language_low_overlap_still_flagged(self):
        problems = check_claim_rules(
            Claim(text="针灸可以降低血压并改善胰岛素抵抗 [1]。", citations=[1]),
            {1: "本文综述了经颅磁刺激线圈设计与电场建模方法。"},
        )
        assert "grounding" in {p["rule"] for p in problems}
        assert _verdict_from_problems(problems) == "unsupported"


# ================================================== 各条 Tier 0 规则
class TestRuleExistence:
    def test_missing_id_flagged(self):
        problems = check_claim_rules(
            Claim(text="该结论有支持 [99]。", citations=[99]),
            {1: "无关内容"},
            valid_ids=[1, 2],
        )
        assert "existence" in {p["rule"] for p in problems}
        assert _verdict_from_problems(problems) == "unsupported"

    def test_valid_ids_distinguishes_missing_text_from_missing_citation(self):
        """「编号不存在」与「编号存在但拿不到全文」后果完全不同，必须分开。"""
        no_text = check_claim_rules(
            Claim(text="该疗法可降低复发率 [25]。", citations=[25]),
            {25: ""},
            valid_ids=[25],
        )
        rules = {p["rule"] for p in no_text}
        assert "no_source_text" in rules
        assert "existence" not in rules
        assert _verdict_from_problems(no_text) == "unverifiable"


class TestRuleNumbers:
    def test_fabricated_number_flagged(self):
        problems = check_claim_rules(
            Claim(text="本研究纳入 9999 例患者 [1]。", citations=[1]),
            {1: "本研究纳入 120 例患者。"},
        )
        assert "numbers" in {p["rule"] for p in problems}
        assert _verdict_from_problems(problems) == "unsupported"

    def test_matching_number_passes(self):
        problems = check_claim_rules(
            Claim(text="HAMD 评分下降 12.5 分（P<0.001）[1]。", citations=[1]),
            {1: "HAMD scores decreased by 12.5 points (P<0.001)."},
        )
        assert "numbers" not in {p["rule"] for p in problems}

    def test_percentage_matches_bare_number(self):
        problems = check_claim_rules(
            Claim(text="发生率为 12% [1]。", citations=[1]),
            {1: "Adverse events occurred in 12 of 100 sessions."},
        )
        assert "numbers" not in {p["rule"] for p in problems}

    def test_year_is_not_treated_as_data(self):
        problems = check_claim_rules(
            Claim(text="该指南于 2019 年更新 [1]。", citations=[1]),
            {1: "Guidelines on therapeutic use were updated."},
        )
        assert "numbers" not in {p["rule"] for p in problems}

    def test_small_integers_ignored(self):
        problems = check_claim_rules(
            Claim(text="共 2 组，每组 1 例 [1]。", citations=[1]),
            {1: "Two groups were compared in a case series."},
        )
        assert "numbers" not in {p["rule"] for p in problems}


class TestRuleDirection:
    def test_strong_claim_vs_null_result_is_contradiction(self):
        problems = check_claim_rules(
            Claim(text="加速rTMS显著优于常规rTMS [1]。", citations=[1]),
            {1: "加速rTMS组与常规rTMS组的 HAMD 下降幅度无显著差异。"},
        )
        assert "direction" in {p["rule"] for p in problems}
        assert _verdict_from_problems(problems) == "contradicted"

    def test_english_null_result(self):
        problems = check_claim_rules(
            Claim(text="The treatment was superior to placebo [1].", citations=[1]),
            {1: "The intervention failed to show benefit over placebo."},
        )
        assert _verdict_from_problems(problems) == "contradicted"

    def test_strong_claim_with_supporting_source_not_flagged(self):
        """**关键负例**：强主张 + 强证据（RCT）不应被判为矛盾或过度主张。"""
        problems = check_claim_rules(
            Claim(
                text="This randomized controlled trial proves accelerated rTMS significantly improves HAMD scores [1].",
                citations=[1],
            ),
            {1: "In this randomized controlled trial, accelerated rTMS significantly improved HAMD scores compared with sham (P=0.003)."},
        )
        rules = {p["rule"] for p in problems}
        assert "direction" not in rules
        assert "overclaim" not in rules

    def test_requires_shared_topic_to_avoid_false_contradiction(self):
        """强主张 + 阴性表述，但两者毫无共同主题 → 可能是引错文献，不判矛盾。"""
        problems = check_claim_rules(
            Claim(text="针灸治愈失眠 [1]。", citations=[1]),
            {1: "The vaccine trial found no significant difference in mortality."},
        )
        assert "direction" not in {p["rule"] for p in problems}


class TestRuleOverclaim:
    def test_strong_wording_with_weak_carrier(self):
        problems = check_claim_rules(
            Claim(text="本方案可彻底根治卒中后抑郁 [1]。", citations=[1]),
            {1: "一项治疗卒中后抑郁的研究方案，正在招募。"},
        )
        assert "overclaim" in {p["rule"] for p in problems}

    def test_publication_type_marks_weak_carrier(self):
        problems = check_claim_rules(
            Claim(text="This definitively proves the mechanism [1].", citations=[1]),
            {1: "Mechanism of action was explored in healthy volunteers."},
            source_meta={1: {"publication_type": "preprint"}},
        )
        assert "overclaim" in {p["rule"] for p in problems}

    def test_mild_wording_not_flagged(self):
        problems = check_claim_rules(
            Claim(text="两组安全性相似 [1]。", citations=[1]),
            {1: "The safety profile of the two groups was similar."},
        )
        assert "overclaim" not in {p["rule"] for p in problems}


class TestVerdictArbitration:
    def test_arbitration_priority(self):
        """仲裁顺序是有意的设计决定：
        numbers/existence > direction > overclaim > grounding > unverifiable > weakly_supported。"""
        assert _verdict_from_problems([{"rule": "numbers"}, {"rule": "overclaim"}]) == "unsupported"
        assert _verdict_from_problems([{"rule": "direction"}, {"rule": "overclaim"}]) == "contradicted"
        assert _verdict_from_problems([{"rule": "overclaim"}]) == "overclaim"
        assert _verdict_from_problems([{"rule": "grounding"}]) == "unsupported"
        assert _verdict_from_problems([{"rule": "cross_lingual"}]) == "unverifiable"
        assert _verdict_from_problems([{"rule": "script_independent_only"}]) == "weakly_supported"
        assert _verdict_from_problems([]) == "supported"

    def test_missing_text_is_not_reported_as_unsupported(self):
        assert _verdict_from_problems([{"rule": "no_source_text"}]) == "unverifiable"


# ==================================================== 整篇分析
class TestAnalyseDraft:
    def draft(self) -> str:
        return (
            "## 引言\n"
            "卒中后抑郁是常见并发症，发生率约 30% [1]。\n"
            "加速rTMS显著优于常规rTMS，可完全治愈卒中后抑郁 [2]。\n"
            "\n## 结果\n"
            "治疗后 HAMD 评分下降 12.5 分（P<0.001）[3]。\n"
            "本研究纳入 9999 例患者 [1]。\n"
            "\n## 参考文献\n[1] 张三. 某研究. 2021.\n"
        )

    def sources(self) -> dict[int, str]:
        return {
            1: "卒中后抑郁影响约 30% 的卒中幸存者。一项随机试验比较了加速rTMS与常规rTMS，两组 HAMD 下降幅度无显著差异。",
            2: "一项随机试验比较了加速rTMS与常规rTMS，两组 HAMD 下降幅度无显著差异。",
            3: "治疗后 HAMD 评分下降 12.5 分（P<0.001）。",
        }

    def test_end_to_end_report(self):
        report = analyse_draft(self.draft(), self.sources(), valid_ids=[1, 2, 3])
        assert report.claims == 4
        assert report.by_verdict.get("contradicted", 0) >= 1
        assert report.by_verdict.get("unsupported", 0) >= 1
        assert report.by_verdict.get("supported", 0) >= 1

    def test_reports_coverage_and_disclaimers(self):
        report = analyse_draft(self.draft(), self.sources(), valid_ids=[1, 2, 3])
        assert report.sentences >= report.cited_sentences
        notes = " ".join(report.notes)
        assert "未报警 ≠ 已核实" in notes, "必须写明 Tier 0 的边界，避免被当成已核实"

    def test_low_coverage_warning(self):
        """覆盖率低于 25% 时必须提示 —— 否则"支持率 100%"会被误读成整篇已核实。"""
        # 6 句长句都没有引用，只有第 7 句带引用（≈14%）
        draft = (
            "这一句完全没有引用标记。这一句也没有引用标记。"
            "这一句同样没有引用标记。这一句依然没有引用标记。"
            "这一句还是没有引用标记。这一句仍然没有引用标记。"
            "只有这一句带有引用 [1]。"
        )
        report = analyse_draft(draft, {1: "some source"}, valid_ids=[1])
        assert report.cited_sentences == 1
        assert report.sentences >= 6
        assert any("只有" in n and "带引用" in n for n in report.notes), report.notes

    def test_serializes_to_json(self):
        import json

        report = analyse_draft(self.draft(), self.sources(), valid_ids=[1, 2, 3])
        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["claims"] == 4
        assert "by_rule" in payload

    def test_details_include_evidence_for_review(self):
        report = analyse_draft(self.draft(), self.sources(), valid_ids=[1, 2, 3])
        flagged = [d for d in report.details if d.verdict != "supported"]
        assert flagged
        for item in flagged:
            assert item.problems, "被标记的论断必须给出具体问题"


# ====================================== 校验器自评估（evaluate the evaluator）
class TestRuleEvalOnLabels:
    @pytest.fixture(scope="class")
    def report(self):
        return evaluate_rules()

    def test_dataset_loads_and_is_well_formed(self):
        cases = load_labeled_cases()
        assert len(cases) >= 25
        ids = [c.id for c in cases]
        assert len(ids) == len(set(ids)), "案例 id 不能重复"
        allowed = {
            "supported", "weakly_supported", "unsupported",
            "overclaim", "contradicted", "unverifiable",
        }
        for case in cases:
            assert case.gold in allowed, f"{case.id} 的 gold 取值非法：{case.gold}"
            assert case.claim_text and case.citations

    def test_weakly_supported_is_its_own_class(self):
        """跨语言但共享术语的论断单列为一档 —— 不能被混进 supported，
        否则"21 条 supported"会被误读成"语义已核实"。"""
        from medscholar.eval.faithfulness import _verdict_from_problems

        assert _verdict_from_problems([{"rule": "script_independent_only"}]) == "weakly_supported"
        # 但它不是"问题"，二分类里应算作没发现问题
        from medscholar.eval.faithfulness_eval import _PROBLEM_VERDICTS

        assert "weakly_supported" not in _PROBLEM_VERDICTS

    def test_dataset_file_is_where_expected(self):
        assert LABELS_PATH.is_file()

    def test_recall_on_real_problems_is_perfect(self, report):
        """**这是最重要的一条指标**：标注集里所有真问题都必须被抓到。

        漏报的代价远大于误报 —— 一个"结论被说反了"没被发现，可能直接进论文。
        """
        assert report.binary["fn"] == 0, f"有漏报：{report.binary}"
        assert report.binary["recall"] == 1.0

    def test_precision_is_acceptable(self, report):
        """允许少量误报（语义改写导致的词面低重合是代理指标的固有盲区），
        但必须保持在可接受水平，否则报告会被忽略。"""
        assert report.binary["precision"] >= 0.85, report.binary

    def test_all_expected_rules_fire(self, report):
        assert report.rule_total > 0
        assert report.rule_hit == report.rule_total, report.failures

    def test_contradiction_class_is_fully_detected(self, report):
        """方向矛盾是后果最严重的一类错误（比编造数字更隐蔽），必须全抓。"""
        assert report.per_class["contradicted"]["recall"] == 1.0
        assert report.per_class["contradicted"]["precision"] == 1.0

    def test_abstention_class_works(self, report):
        assert report.per_class["unverifiable"]["recall"] == 1.0

    def test_known_blind_spots_are_documented_not_hidden(self, report):
        """语义改写的误报必须在报告里被明确列为已知盲区，而不是悄悄略过。"""
        notes = " ".join(report.notes)
        assert "盲区" in notes or "误报" in notes
        blind = [f for f in report.failures if f["gold"] == "supported"]
        assert blind, "标注集应当保留语义改写的盲区案例，用于持续度量误报率"

    def test_report_serializes(self, report):
        import json

        payload = json.loads(json.dumps(report.to_dict(), ensure_ascii=False))
        assert payload["total"] >= 25
        assert "confusion" in payload and "per_class" in payload


class TestRenderReport:
    def test_renders_sections(self):
        report = analyse_draft(
            "有一个强论断，可完全治愈该病 [1]。",
            {1: "该研究未发现显著差异。"},
            valid_ids=[1],
        )
        text = render_faithfulness_report(report)
        assert "引用支持性核查" in text
        assert "判定分布" in text
        assert "需要核对的论断" in text
        assert "说明" in text

    def test_renders_with_no_claims(self):
        text = render_faithfulness_report(analyse_draft("没有引用的草稿。", {}))
        assert "0 条" in text
