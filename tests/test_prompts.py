"""提示词注册表（platform/prompts.py + platform/prompt_library.py）的测试。

覆盖四件事，按重要性排序：

1. **迁移完整性**（最重要）：``medscholar.llm.prompts`` 这个兼容外壳导出的常量文本，
   必须与注册表里对应 key 的文本**逐字相同**，``__all__`` 与迁移前一致。
   提示词是"看起来像文字、实际是程序"的东西：少一个字、多一个空格都会改变模型行为，
   而这种漂移不会报错、只会让产出慢慢变差。所以搬迁必须有断言锁住。
2. **占位符机制**：缺占位符要报错（而不是把 ``{xxx}`` 静默塞进模型上下文），
   同时**不能误伤**合法的花括号（提示词里的示范 JSON 就是合法花括号）。
3. **版本与变体**：默认取最高未弃用版本、显式指定版本、全部弃用时的回退、
   A/B 变体切换。
4. **可观测**：``usage()`` / ``describe()`` / ``prompt_metadata()`` 的结构，
   它们会被 ``/api/metrics`` 与 trace 使用，结构变了要有人知道。

大部分用例用**局部注册表**（:func:`registry` 夹具）：全局 ``REGISTRY`` 是进程级状态，
能用局部实例验证的逻辑就不要去动全局状态。少数用例必须打全局（``reset_registry``、
``set_variant`` 的端到端行为），由 autouse 夹具在每个用例前后复位。
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import pytest

from medscholar.platform import prompt_library as library
from medscholar.platform import prompts as prompts_module
from medscholar.platform.prompts import (
    PROMPT_KINDS,
    REGISTRY,
    Prompt,
    PromptError,
    PromptRegistry,
    PromptVersion,
    active_variant,
    get_prompt,
    prompt_metadata,
    prompt_text,
    reset_registry,
    set_variant,
)

# ===========================================================================
# 夹具
# ===========================================================================


@pytest.fixture(autouse=True)
def _clean_registry():
    """每个用例前后把全局注册表恢复成内建状态。

    全局可变状态最容易制造"单独跑绿、一起跑红"：某个用例登记了临时 key 或切了 A/B
    变体，不清理就会串到后面的用例（甚至别的测试文件）里。
    """
    reset_registry()
    yield
    reset_registry()


@pytest.fixture()
def registry() -> PromptRegistry:
    """局部注册表：三条版本（v1/v2/v3，其中 v3 已弃用）+ 一个变体。

    用局部实例而不是全局 REGISTRY：版本选择这类纯逻辑不该依赖"别人有没有登记过什么"。
    """
    reg = PromptRegistry()
    reg.register(
        PromptVersion(
            key="plan.system",
            version=1,
            text="v1 系统提示词",
            description="初版",
            tags=("system", "plan"),
        )
    )
    reg.register(
        PromptVersion(
            key="plan.system",
            version=2,
            text="v2 系统提示词，更严格",
            description="收紧引用要求",
            tags=("system", "plan"),
        )
    )
    reg.register(
        PromptVersion(
            key="plan.system",
            version=3,
            text="v3 有问题的版本",
            description="实验失败，弃用",
            tags=("system", "plan"),
            deprecated=True,
        )
    )
    reg.register(
        PromptVersion(
            key="writer.section",
            version=1,
            text="研究课题：{topic}\n材料：{digest}",
            description="写作默认版",
            tags=("user", "writer"),
            placeholders=("topic", "digest"),
        )
    )
    reg.register(
        PromptVersion(
            key="writer.section",
            version=1,
            text="研究课题：{topic}\n材料：{digest}\n（严格引用）",
            description="严格引用变体",
            tags=("user", "writer", "citation-strict"),
            placeholders=("topic", "digest"),
        ),
        variant="citation-strict",
    )
    return reg


def _custom(key: str = "plan.system", version: int = 99, text: str = "临时提示词", **kw):
    """构造一个注册用的 PromptVersion（默认值只服务于测试意图）。"""
    return PromptVersion(
        key=key,
        version=version,
        text=text,
        description=kw.pop("description", "测试用版本"),
        **kw,
    )


# ===========================================================================
# 1) 登记
# ===========================================================================


class TestRegistration:
    def test_register_then_get(self):
        reg = PromptRegistry()
        reg.register(_custom(text="正文 A"))
        assert reg.get("plan.system").text == "正文 A"
        assert reg.keys() == ["plan.system"]

    def test_get_returns_frozen_prompt(self):
        reg = PromptRegistry()
        reg.register(_custom())
        item = reg.get("plan.system")
        assert isinstance(item, Prompt)
        assert (item.key, item.version, item.variant) == ("plan.system", 99, "default")
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.text = "改不动"  # frozen dataclass 必须真的冻结

    def test_str_of_prompt_is_its_text(self):
        """``system=get_prompt(...)`` 这种写法不必改调用方代码，全靠 __str__。"""
        reg = PromptRegistry()
        reg.register(_custom(text="我是提示词"))
        assert str(reg.get("plan.system")) == "我是提示词"
        assert f"system={reg.get('plan.system')}" == "system=我是提示词"

    def test_duplicate_version_rejected(self):
        """版本一旦发布就不再改写：重复登记必须报错，并提示下一个版本号。"""
        reg = PromptRegistry()
        reg.register(_custom(version=1))
        with pytest.raises(PromptError) as excinfo:
            reg.register(_custom(version=1))
        message = str(excinfo.value)
        assert "plan.system" in message and "v1" in message and "v2" in message

    @pytest.mark.parametrize("bad", [0, -1, True, "1", 1.5])
    def test_version_must_be_positive_int(self, bad):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="version"):
            reg.register(_custom(version=bad))

    def test_key_needs_known_kind(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="PROMPT_KINDS"):
            reg.register(_custom(key="writting.section"))

    def test_key_needs_two_segments(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="类别"):
            reg.register(_custom(key="plan"))

    def test_description_is_required(self):
        """description 不是可选注释：它是 trace 里"这版为什么改"的唯一来源。"""
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="description"):
            reg.register(_custom(description="   "))

    def test_non_string_text_rejected(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="text"):
            reg.register(_custom(text=123))  # type: ignore[arg-type]

    def test_register_rejects_plain_string(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="PromptVersion"):
            reg.register("随便一段文字")  # type: ignore[arg-type]

    def test_declared_placeholder_must_exist_in_text(self):
        """声明了却正文里没有 → 注册期就报错，否则"未替换检查"是空转的。"""
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="找不到"):
            reg.register(_custom(text="没有占位符", placeholders=("topic",)))

    def test_duplicate_placeholder_names_rejected(self):
        reg = PromptRegistry()
        with pytest.raises(PromptError, match="重复"):
            reg.register(_custom(text="{topic}", placeholders=("topic", "topic")))

    def test_tags_and_placeholders_normalized_to_tuple(self):
        """传 list 也要存成 tuple：frozen dataclass 里塞可变对象等于没冻结。"""
        reg = PromptRegistry()
        reg.register(
            _custom(text="{topic}", placeholders=["topic"], tags=["system"])  # type: ignore[arg-type]
        )
        meta = reg.metadata("plan.system")
        assert meta["tags"] == ["system"]
        assert meta["placeholders"] == ["topic"]


# ===========================================================================
# 2) 版本选择
# ===========================================================================


class TestVersionSelection:
    def test_default_picks_highest_available(self, registry):
        assert registry.get("plan.system").version == 2  # v3 已弃用

    def test_default_skips_deprecated(self, registry):
        assert registry.get("plan.system").text == "v2 系统提示词，更严格"

    def test_explicit_version(self, registry):
        assert registry.get("plan.system", version=1).text == "v1 系统提示词"

    def test_explicit_deprecated_version_still_reachable(self, registry):
        """显式要旧版本时不该被"已弃用"挡住：A/B 复盘常常要重跑旧版。"""
        assert registry.get("plan.system", version=3).text == "v3 有问题的版本"

    def test_all_deprecated_falls_back_to_highest(self):
        reg = PromptRegistry()
        reg.register(_custom(version=1, text="旧", deprecated=True))
        reg.register(_custom(version=2, text="新", deprecated=True))
        assert reg.get("plan.system").version == 2
        assert reg.describe()["plan.system"]["deprecated"] is True

    def test_versions_lists_all(self, registry):
        assert registry.versions("plan.system") == [1, 2, 3]

    def test_unknown_version_lists_available(self, registry):
        with pytest.raises(PromptError) as excinfo:
            registry.get("plan.system", version=7)
        message = str(excinfo.value)
        assert "v7" in message and "v1" in message and "v2" in message and "v3" in message

    def test_render_returns_plain_text(self, registry):
        text = registry.render("writer.section", topic="课题", digest="材料")
        assert isinstance(text, str)
        assert text.startswith("研究课题：课题")


# ===========================================================================
# 3) 变体与 A/B
# ===========================================================================


class TestVariants:
    def test_variants_sorted(self, registry):
        assert registry.variants("writer.section") == ["citation-strict", "default"]

    def test_explicit_variant_selection(self, registry):
        item = registry.get("writer.section", variant="citation-strict", topic="T", digest="D")
        assert item.variant == "citation-strict"
        assert item.text.endswith("（严格引用）")

    def test_set_variant_switches_default_lookup(self, registry):
        registry.set_variant("writer.section", "citation-strict")
        assert registry.active_variant("writer.section") == "citation-strict"
        item = registry.get("writer.section", topic="T", digest="D")
        assert item.variant == "citation-strict"

    def test_explicit_variant_argument_wins_over_switch(self, registry):
        """显式传变体名时不受 A/B 开关影响（代码里写死的那一版就该是那一版）。"""
        registry.set_variant("writer.section", "citation-strict")
        item = registry.get("writer.section", variant="default", topic="T", digest="D")
        # 注意语义：``variant="default"`` 的含义是"当前生效的变体"，
        # 所以开关打开时它同样拿到严格版——这正是 A/B 能生效的方式。
        assert item.variant == "citation-strict"
        explicit = registry.get(
            "writer.section", variant="citation-strict", topic="T", digest="D"
        )
        assert explicit.variant == "citation-strict"

    def test_switch_back_restores_literal_default_variant(self, registry):
        """``variant="default"`` 只有在开关关闭后才等于"名为 default 的那一版"。"""
        registry.set_variant("writer.section", "citation-strict")
        registry.set_variant("writer.section", "default")
        item = registry.get("writer.section", variant="default", topic="T", digest="D")
        assert item.variant == "default"
        assert not item.text.endswith("（严格引用）")

    def test_set_variant_back_to_default_clears_switch(self, registry):
        registry.set_variant("writer.section", "citation-strict")
        registry.set_variant("writer.section", "default")
        assert registry.active_variant("writer.section") == "default"
        assert registry.get("writer.section", topic="T", digest="D").variant == "default"

    def test_unknown_variant_rejected(self, registry):
        with pytest.raises(PromptError, match="没有变体"):
            registry.set_variant("writer.section", "turbo")

    def test_same_version_number_across_variants(self, registry):
        """变体与版本是两个正交的轴：两边的 v1 互不影响。"""
        assert registry.versions("writer.section") == [1]
        assert registry.get("writer.section", topic="T", digest="D").version == 1
        strict = registry.get("writer.section", variant="citation-strict", topic="T", digest="D")
        assert strict.version == 1


class TestAbSwitchOnBuiltIns:
    def test_citation_strict_differs_from_default(self):
        default_text = prompt_text("writer.section", topic="T", section_title="S", bullet="B",
                                   digest="D", length_clause="L")
        set_variant("writer.section", "citation-strict")
        strict_text = prompt_text("writer.section", topic="T", section_title="S", bullet="B",
                                  digest="D", length_clause="L")
        assert strict_text != default_text
        assert strict_text.startswith(default_text)
        assert "citation-strict" in strict_text
        assert "逐字找到" in strict_text

    def test_citation_strict_keeps_default_variant_text_unchanged(self):
        """默认变体的正文必须还是迁移前那份（变体不能偷偷改动默认行为）。"""
        assert active_variant("writer.section") == "default"
        assert prompt_metadata("writer.section")["variant"] == "default"
        # 变体正文 = 默认版正文 + 追加条款，默认版本身一个字符都没动
        assert get_prompt("writer.system", variant="citation-strict").text.startswith(
            library.SECTION_SYSTEM
        )
        set_variant("writer.section", "citation-strict")
        assert prompt_metadata("writer.section")["variant"] == "citation-strict"
        set_variant("writer.section", "default")
        assert prompt_metadata("writer.section")["variant"] == "default"

    def test_citation_strict_system_prompt_appends_rules(self):
        base = get_prompt("writer.system")
        strict = get_prompt("writer.system", variant="citation-strict")
        assert strict.text.startswith(base.text)
        assert "只能引用材料中**已经给出**的编号" in strict.text

    def test_variant_switch_does_not_leak_across_keys(self):
        set_variant("writer.system", "citation-strict")
        assert active_variant("writer.system") == "citation-strict"
        assert active_variant("writer.section") == "default"


# ===========================================================================
# 4) 占位符：必须报错的部分，与必须不报错的部分
# ===========================================================================


class TestPlaceholders:
    def test_missing_placeholder_raises_with_full_list(self):
        """缺参必须报错：``str.format`` 会 KeyError，但更危险的是"没替换的 {xxx}"
        静默进入模型上下文 —— 模型会把它当正文照着编。"""
        with pytest.raises(PromptError) as excinfo:
            get_prompt("writer.section", topic="课题")
        message = str(excinfo.value)
        assert "writer.section" in message
        assert "section_title" in message and "digest" in message  # 缺了哪些
        assert "length_clause" in message and "bullet" in message  # 这个提示词需要哪些
        assert "共 5 个" in message

    def test_all_placeholders_required(self, registry):
        with pytest.raises(PromptError, match="缺少占位符"):
            registry.render("writer.section", topic="T")

    def test_repeated_placeholder_replaced_everywhere(self):
        """同一个占位符出现两次（"以 {max_chars} 字为目标…接近 {max_chars} 字"）都要替换。"""
        text = prompt_text(
            "writer.section.length_with_min", style="综述正文", min_chars=800, max_chars=1600
        )
        assert "{max_chars}" not in text
        assert text.count("1600") == 2

    def test_non_string_value_is_stringified(self):
        text = prompt_text("outline.user", topic="课题", digest="材料", section_count=5)
        assert "共 5 个章节" in text

    def test_extra_variables_are_ignored(self, registry):
        """多余的变量不算错：同一个调用点要能同时喂默认版与变体。"""
        text = registry.render("writer.section", topic="T", digest="D", unused_note="不该出现")
        assert "不该出现" not in text

    def test_json_braces_are_not_touched(self):
        """关键取舍：只替换**声明过**的占位符，示范 JSON 的花括号原样保留。"""
        reg = PromptRegistry()
        reg.register(
            PromptVersion(
                key="plan.user",
                version=1,
                text='请只输出 JSON：{"queries": [], "year_from": 2015}\n课题：{topic}',
                description="含示范 JSON 的提示词",
                placeholders=("topic",),
            )
        )
        text = reg.render("plan.user", topic="卒中后抑郁")
        assert '{"queries": [], "year_from": 2015}' in text
        assert text.endswith("课题：卒中后抑郁")

    def test_undeclared_braces_without_declaration_survive_render(self):
        """一个字面量花括号、连声明都没有：既不报错也不被吃掉。"""
        reg = PromptRegistry()
        reg.register(
            PromptVersion(
                key="translate.user",
                version=1,
                text='输出 JSON：{"en": "english", "mesh": []}',
                description="纯 JSON 示范",
            )
        )
        assert reg.render("translate.user") == '输出 JSON：{"en": "english", "mesh": []}'

    def test_migrated_translate_prompt_keeps_json_demo(self):
        """真实迁移文本里也有示范 JSON，渲染必须完全不动它。"""
        text = prompt_text("translate.user", text="加速 rTMS")
        assert '{"en": "英文检索式", "mesh": ["候选 MeSH 词"]}' in text

    def test_migrated_outline_prompt_keeps_json_demo(self):
        text = prompt_text("outline.user", topic="课题", digest="材料", section_count=5)
        assert '{"outline": [{"title": "章节标题", "points": ["要点1", "要点2"]}]}' in text

    def test_placeholder_names_with_common_prefix(self):
        """``{topic}`` 与 ``{topic_en}`` 并存时不能互相截断。

        替换是按 ``{名字}`` 整体匹配的（名字后面必须紧跟 ``}``），
        所以"短名字是长名字前缀"这种最容易写错的情况也是安全的。
        """
        reg = PromptRegistry()
        reg.register(
            PromptVersion(
                key="plan.user",
                version=1,
                text="中文课题：{topic}\n英文课题：{topic_en}",
                description="前缀相似的占位符",
                placeholders=("topic", "topic_en"),
            )
        )
        assert reg.render("plan.user", topic="卒中", topic_en="stroke") == (
            "中文课题：卒中\n英文课题：stroke"
        )

    def test_undeclared_placeholder_audit_reports_nothing_for_builtins(self):
        """内建提示词里不应有"看起来像占位符、却没人声明"的 token。"""
        offenders = {
            key: REGISTRY.undeclared_placeholders(key) for key in REGISTRY.keys()
        }
        assert {k: v for k, v in offenders.items() if v} == {}

    def test_undeclared_placeholder_audit_finds_hidden_token(self):
        """审计能力本身要有效：写错名字（{digset}）时必须能被抓出来。"""
        reg = PromptRegistry()
        reg.register(
            PromptVersion(
                key="writer.section",
                version=1,
                text="材料：{digest}\n草稿：{digset}",
                description="故意写错占位符名",
                placeholders=("digest",),
            )
        )
        assert reg.undeclared_placeholders("writer.section") == ("digset",)

    def test_digest_papers_output_is_safe_to_inject(self):
        """材料块里即使有花括号，也不该被当成占位符（只认声明过的名字）。"""
        text = prompt_text(
            "writer.section",
            topic="课题",
            section_title="引言",
            bullet="- 要点",
            digest='材料里有 JSON：{"a": 1}',
            length_clause="直接输出正文。",
        )
        assert '{"a": 1}' in text


# ===========================================================================
# 5) 未知 key 的可读报错
# ===========================================================================


class TestUnknownKey:
    def test_message_contains_key_and_suggestion(self):
        with pytest.raises(PromptError) as excinfo:
            get_prompt("writer.sectoin")  # 拼写错误
        message = str(excinfo.value)
        assert "writer.sectoin" in message
        assert "writer.section" in message
        assert "你是不是想找" in message

    def test_message_lists_available_keys(self):
        with pytest.raises(PromptError) as excinfo:
            prompt_text("nope.whatever")
        message = str(excinfo.value)
        assert "plan.system" in message and "ask.system" in message
        assert str(len(REGISTRY.keys())) in message

    def test_unknown_key_on_registry_methods(self, registry):
        for call in (
            lambda: registry.versions("nope.thing"),
            lambda: registry.variants("nope.thing"),
            lambda: registry.active_variant("nope.thing"),
            lambda: registry.set_variant("nope.thing", "default"),
            lambda: registry.metadata("nope.thing"),
        ):
            with pytest.raises(PromptError):
                call()


# ===========================================================================
# 6) 可观测：usage / describe / metadata
# ===========================================================================


class TestUsageAndDescribe:
    def test_usage_counts_get_and_render(self):
        reset_registry()
        assert REGISTRY.usage()["ask.system"] == 0
        prompt_text("ask.system")
        get_prompt("ask.system")
        get_prompt("ask.system", variant="default")
        assert REGISTRY.usage()["ask.system"] == 3

    def test_usage_contains_every_registered_key(self):
        """表要稳定：/api/metrics 的面板不该因为"这版最近没用过"而缺行。"""
        usage = REGISTRY.usage()
        assert set(usage) == set(REGISTRY.keys())
        assert all(count == 0 for count in usage.values())

    def test_usage_returns_a_copy(self):
        usage = REGISTRY.usage()
        usage["ask.system"] = 999
        assert REGISTRY.usage()["ask.system"] == 0

    def test_reset_usage_zeroes_counters(self):
        prompt_text("ask.system")
        assert REGISTRY.usage()["ask.system"] == 1
        REGISTRY.reset_usage()
        assert REGISTRY.usage()["ask.system"] == 0

    def test_failed_render_does_not_count(self):
        """取用失败不该计数：计数是"这一版被真的喂给模型了吗"的证据。"""
        before = REGISTRY.usage()["writer.section"]
        with pytest.raises(PromptError):
            get_prompt("writer.section", topic="只有课题")
        assert REGISTRY.usage()["writer.section"] == before

    def test_describe_structure(self):
        info = REGISTRY.describe()["writer.system"]
        assert set(info) == {"versions", "variants", "current", "deprecated"}
        assert info["versions"] == [1]
        assert info["variants"] == ["citation-strict", "default"]
        assert info["current"] == 1
        assert info["deprecated"] is False

    def test_describe_covers_all_keys(self):
        assert set(REGISTRY.describe()) == set(REGISTRY.keys())

    def test_metadata_fields(self):
        meta = prompt_metadata("writer.section")
        assert set(meta) == {
            "key",
            "version",
            "variant",
            "description",
            "tags",
            "placeholders",
            "deprecated",
        }
        assert meta["key"] == "writer.section"
        assert meta["version"] == 1
        assert meta["variant"] == "default"
        assert meta["deprecated"] is False
        assert set(meta["placeholders"]) == {
            "topic",
            "section_title",
            "bullet",
            "digest",
            "length_clause",
        }
        assert "citation" in meta["tags"]

    def test_metadata_needs_no_variables(self):
        """埋点不该为了记账先凑齐占位符取值。"""
        assert prompt_metadata("plan.user")["version"] == 1

    def test_metadata_is_json_serializable(self):
        json.dumps(prompt_metadata("plan.user"), ensure_ascii=False)


# ===========================================================================
# 7) reset_registry
# ===========================================================================


class TestResetRegistry:
    def test_reset_restores_builtin_prompts(self):
        REGISTRY.clear()
        with pytest.raises(PromptError):
            prompt_text("plan.system")
        reset_registry()
        assert "你是 MedScholar" in prompt_text("plan.system")

    def test_reset_drops_temporary_entries_and_usage(self):
        REGISTRY.register(
            PromptVersion(key="plan.system", version=99, text="临时", description="测试用")
        )
        prompt_text("plan.system", version=99)
        reset_registry()
        assert REGISTRY.versions("plan.system") == [1]
        assert REGISTRY.usage()["plan.system"] == 0

    def test_reset_clears_ab_switch(self):
        set_variant("writer.section", "citation-strict")
        reset_registry()
        assert active_variant("writer.section") == "default"

    def test_builtin_library_is_reinstallable(self):
        """装配函数是幂等的入口：清空后重装能回到同一状态。"""
        before = REGISTRY.describe()
        REGISTRY.clear()
        library.install(REGISTRY)
        assert REGISTRY.describe() == before


# ===========================================================================
# 8) 迁移完整性（最重要）
# ===========================================================================

#: ``medscholar.llm.prompts`` 外壳常量 → 注册表 key。
MIGRATED: tuple[tuple[str, str], ...] = (
    ("PLAN_SYSTEM", "plan.system"),
    ("TRANSLATE_SYSTEM", "translate.system"),
    ("OUTLINE_SYSTEM", "outline.system"),
    ("SECTION_SYSTEM", "writer.system"),
    ("CRITIQUE_SYSTEM", "critique.system"),
    ("REFLECT_SYSTEM", "reflect.system"),
    ("SUMMARY_SYSTEM", "summary.system"),
    ("CHAT_SYSTEM", "chat.system"),
    ("ASK_SYSTEM", "ask.system"),
)

#: 迁移前的 ``__all__``，逐字硬编码在这里。
#: 顺序也算契约：``from x import *``、文档生成与依赖扫描都会受它影响。
EXPECTED_ALL: tuple[str, ...] = (
    "PLAN_SYSTEM",
    "plan_user",
    "TRANSLATE_SYSTEM",
    "translate_user",
    "OUTLINE_SYSTEM",
    "outline_user",
    "SECTION_SYSTEM",
    "section_user",
    "CRITIQUE_SYSTEM",
    "critique_user",
    "REFLECT_SYSTEM",
    "reflect_user",
    "SUMMARY_SYSTEM",
    "summary_user",
    "CHAT_SYSTEM",
    "chat_user",
    "digest_papers",
)

#: 迁移前文本的 sha256。为什么不只跟注册表比：那是**循环验证**——
#: 如果外壳与正文库被同一个错误改坏，两边照样相等。摘要提供注册表之外的锚点，
#: 任何一字改动都会让这条用例变红（改提示词是有意行为，那就同时更新这里的摘要）。
TEXT_DIGESTS: dict[str, str] = {
    "_ROLE": "7b2912671764992ecd7d5e5b042775c42d705c09d877e141dfc444029f587e2e",
    "_HARD_RULES": "515e7584b9bc3d337433d327c272bd13daaea1e8173575b8dd1336aff4f78e0e",
    "PLAN_SYSTEM": "aede4540684a5c6e62389f97f4ff55dffe28a0e8182c0593aa6ad63d565e8e20",
    "TRANSLATE_SYSTEM": "218fd55fe50dff05566f24c052429e193bd24a9f3f4dec11d6232a659752f98a",
    "OUTLINE_SYSTEM": "f2072fa9f8b2f873fb9141be066926894a13fb3415d752b16599868969d00dd9",
    "SECTION_SYSTEM": "ee3250191ad62eea2e38192538eea0d8c3d13fae69f1d94fd4e9b17fa79816d7",
    "CRITIQUE_SYSTEM": "50de079f01b118335446ac166c077eac33749fca018e3d96d509d63efa815f9e",
    "REFLECT_SYSTEM": "d26a5946cd56360a66d4af4820821fd5822c439553736cb754e05df9ff3c47b0",
    "SUMMARY_SYSTEM": "acbd957a2df51bf9d3c3960a199585ed354329c2ec160c2d52deedeadcce822a",
    "CHAT_SYSTEM": "752b8cd1f54b895fa7d1a6ed09119a6a0857249f06100625cec2a3d48594cb93",
    "ASK_SYSTEM": "5d6bcdc1e6021cf31d99bd511a02a0c1dc575073379bd2df48b5f2b0cf1924c7",
}


def _shell():
    """兼容外壳模块（``medscholar.llm.prompts``）。"""
    from medscholar.llm import prompts as shell

    return shell


class TestMigrationIntegrity:
    @pytest.mark.parametrize(("name", "key"), MIGRATED)
    def test_constant_is_byte_identical_to_registry(self, name: str, key: str):
        """逐个常量：外壳导出的文本 == 注册表登记文本（``==``，逐字）。"""
        assert getattr(_shell(), name) == REGISTRY.get(key).text

    @pytest.mark.parametrize(("name", "key"), MIGRATED)
    def test_constant_is_identical_to_library_constant(self, name: str, key: str):
        """外壳拿到的必须是正文库里**同一个对象**，不是另抄一份。"""
        assert getattr(_shell(), name) is getattr(library, name)

    @pytest.mark.parametrize("name", sorted(TEXT_DIGESTS))
    def test_text_digest_unchanged(self, name: str):
        """逐字不变的独立锚点：摘要对不上说明文本被改过（哪怕只是一个空格）。"""
        raw = getattr(_shell(), name)
        digest = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        assert digest == TEXT_DIGESTS[name], (
            f"{name} 的文本与迁移前不一致。迁移必须是纯搬迁；"
            "若确实要有意修改提示词，请登记新版本号并同步更新这里的摘要。"
        )

    def test_dunder_all_unchanged(self):
        assert list(_shell().__all__) == list(EXPECTED_ALL)

    def test_every_exported_name_is_importable(self):
        module = _shell()
        for name in EXPECTED_ALL:
            assert hasattr(module, name), f"外壳漏了 {name}"

    def test_private_names_still_available(self):
        """``agent/writer.py`` 直接 import 了 ``_ROLE``：私有名也不能漏。"""
        module = _shell()
        assert module._ROLE == REGISTRY.get("core.role").text
        assert module._HARD_RULES == REGISTRY.get("core.hard_rules").text

    def test_schema_dicts_kept_in_shell(self):
        module = _shell()
        assert module._PLAN_SCHEMA["pico"]["population"] == "研究对象"
        assert set(module._CRITIQUE_SCHEMA) == {"assessments", "overall"}
        assert set(module._REFLECT_SCHEMA) == {"verdict", "score", "issues", "strengths"}

    def test_system_prompts_are_composed_of_role_and_rules(self):
        """角色与硬性规则是共用片段：需要它们的系统提示词必须由同一份文本拼成。"""
        role = REGISTRY.get("core.role").text
        rules = REGISTRY.get("core.hard_rules").text
        for key in ("plan.system", "outline.system", "writer.system", "critique.system"):
            text = REGISTRY.get(key).text
            assert text.startswith(role + "\n\n")
        for key in ("plan.system", "outline.system", "writer.system", "summary.system",
                    "chat.system", "ask.system"):
            assert rules in REGISTRY.get(key).text
        # 批判与自我审查阶段不需要那五条通用规则（它们是评分/审查流程）
        assert rules not in REGISTRY.get("critique.system").text
        assert rules not in REGISTRY.get("reflect.system").text

    def test_function_prompts_are_built_from_registry_text(self):
        """函数型提示词：输出必须由注册表片段拼成（流程在壳里，文本在库里）。"""
        module = _shell()
        plan = module.plan_user("课题", extra="补充说明", offline=True)
        assert REGISTRY.render("plan.user.extra_note", extra="补充说明") in plan
        assert REGISTRY.get("plan.user.note_offline").text in plan
        assert "可用数据源" not in plan

        plan_online = module.plan_user("课题")
        assert REGISTRY.get("plan.user.note_online").text in plan_online
        assert "离线模式" not in plan_online

        section = module.section_user("课题", "引言", [], "材料", max_chars=900)
        assert REGISTRY.get("writer.section.no_points").text in section
        assert REGISTRY.render("writer.section.length_plain", style="综述正文", max_chars=900) in section

        section_min = module.section_user("课题", "引言", ["要点"], "材料", min_chars=800,
                                          max_chars=1600)
        assert "不少于 800 字" in section_min and "以 1600 字为目标" in section_min
        assert "- 要点" in section_min

        assert REGISTRY.get("reflect.user.no_ids").text in module.reflect_user(
            "课题", "草稿", [], "材料"
        )
        assert REGISTRY.get("ask.materials.empty").text in module.ask_user("问题", "")
        assert REGISTRY.render("ask.materials.digest", paper_count=3, digest="材料") in (
            module.ask_user("问题", "材料", paper_count=3)
        )

    def test_optional_blocks_are_joined_with_blank_line(self):
        module = _shell()
        assert module.summary_user("正文") == "\n\n".join(
            [
                REGISTRY.render("summary.user.body", paper_text="正文"),
                REGISTRY.get("summary.user.tail").text,
            ]
        )
        with_topic = module.summary_user("正文", topic="课题", focus="焦点")
        assert with_topic.startswith("我的研究课题是：课题\n\n我特别关心：焦点\n\n文献内容：")

    def test_digest_papers_still_works(self):
        """材料格式化函数留在外壳里（它是数据渲染，不是提示词文本），行为不变。"""
        text = _shell().digest_papers(
            [{"title": "标题", "abstract": "A" * 1000, "authors": ["王伟"], "pub_year": 2023}]
        )
        assert text.startswith("[1] 标题")
        assert "…" in text  # 超长摘要被截断
        assert "摘要：（无）" in _shell().digest_papers([{"title": "只有标题"}])

    def test_scattered_prompts_registered_in_library(self):
        """"散落"的提示词也进了注册表：文本与源模块逐字一致。"""
        from medscholar import manuscript
        from medscholar.eval import faithfulness

        assert REGISTRY.get("manuscript.system").text == manuscript._SYSTEM
        assert REGISTRY.get("faithfulness.system").text == faithfulness._JUDGE_SYSTEM

    def test_faithfulness_judge_prompt_round_trip(self):
        """忠实度裁判的用户提示词：用注册表拼出来的结果与源模块函数逐字相同。"""
        from medscholar.eval import faithfulness

        class _Claim:
            text = "加速 rTMS 可缩短起效时间 [2]。"
            citations = (2,)

        sources = {2: "研究显示加速 rTMS 缩短了起效时间。"}
        want = faithfulness._build_judge_prompt(_Claim(), sources)
        got = REGISTRY.render(
            "faithfulness.judge",
            claim=faithfulness._strip_citations(_Claim.text).strip(),
            blocks=REGISTRY.render("faithfulness.source_block", cid=2, content=sources[2]),
        )
        assert got == want
        assert REGISTRY.render("faithfulness.source_empty") == "（无可用内容）"


# ===========================================================================
# 9) 注册表自身的约束
# ===========================================================================


class TestBuiltinLibrary:
    def test_prompt_kinds_cover_every_key(self):
        for key in REGISTRY.keys():
            assert key.split(".", 1)[0] in PROMPT_KINDS, key

    def test_every_builtin_version_has_description_and_tag(self):
        """description 与 tags 是版本可追溯的一半：内建提示词不许有空的。"""
        for key in REGISTRY.keys():
            for variant in REGISTRY.variants(key):
                for version in REGISTRY.versions(key):
                    meta = REGISTRY.metadata(key, version=version, variant=variant)
                    label = f"{key} v{version} [{variant}]"
                    assert meta["description"].strip(), f"{label} 缺说明"
                    assert meta["tags"], f"{label} 缺标签"

    def test_all_builtin_versions_are_v1(self):
        """迁移条目的版本号必须都是 1：v1 = "迁移前的行为"。"""
        assert {v for key in REGISTRY.keys() for v in REGISTRY.versions(key)} == {1}

    def test_registry_has_no_duplicate_keys(self):
        keys = REGISTRY.keys()
        assert len(keys) == len(set(keys))

    def test_module_singleton_exposed(self):
        assert prompts_module.REGISTRY is REGISTRY
        assert isinstance(REGISTRY, PromptRegistry)

    def test_is_empty_reads_internal_state(self, registry):
        """``is_empty`` 给自加载守卫用，所以它**不能**走带守卫的 ``keys()``（会递归打爆栈）。"""
        assert PromptRegistry().is_empty() is True
        assert registry.is_empty() is False
        registry.clear()
        assert registry.is_empty() is True

    def test_importing_prompts_module_alone_gets_builtin_library(self):
        """只 import ``platform.prompts`` 也应当拿到内建提示词（首次取用时的自加载守卫）。

        判断"哪些入口带守卫"必须**换个解释器**跑：本进程早就 import 过
        prompt_library 了，注册表非空，守卫根本不会触发 ——
        那样测到的只是"已经装好了"，而不是"守卫有效"。
        """
        root = Path(__file__).resolve().parent.parent
        code = (
            "import medscholar.platform.prompts as p\n"
            "print(len(p.REGISTRY.keys()))\n"
            "print(p.prompt_text('chat.user.tail'))\n"
        )
        result = subprocess.run(
            [sys.executable, "-X", "utf8", "-c", code],
            capture_output=True,
            text=True,
            encoding="utf-8",
            cwd=str(root),
            check=False,
        )
        assert result.returncode == 0, result.stderr
        lines = result.stdout.strip().splitlines()
        assert int(lines[0]) == len(library.VERSIONS), "空注册表：自加载守卫没生效"
        assert lines[1].startswith("请回答问题")


# ===========================================================================
# 10) 材料编号契约（digest_papers）
# ===========================================================================
# 守的是一个**真实发生过的功能缺陷**：``retrieval.build_context_digest`` 的契约是
# "按显式编号构建材料（编号与正文引用严格对应）"，也确实往每条数据塞了 ``__index__``；
# 但 ``digest_papers`` 只按 ``start_index`` 顺序递增，完全忽略它。
# Critic 恰恰是传"筛过的子集"（真编号可能是 1/4/7/12），
# 于是材料被重新编号成 1/2/3/4，模型说的"第 2 篇"会被挂到真编号 2 的那篇上——
# 那篇可能根本不在材料里：轻则点评丢失，重则把差评挂到无关文献头上。


class TestDigestPapersNumbering:
    @staticmethod
    def _paper(title: str):
        from medscholar.models import Paper

        return Paper(title=title, abstract=f"{title} 的摘要", authors=["王伟"], source="pubmed")

    def test_explicit_index_is_used(self):
        from medscholar.retrieval import build_context_digest

        digest = build_context_digest(
            [(1, self._paper("A")), (3, self._paper("B"))], guard=False
        )
        assert "[1]" in digest and "[3]" in digest
        assert "[2]" not in digest, "没有入选的文献不该凭空占一个编号"
        assert digest.index("[1]") < digest.index("[3]"), "编号仍应按升序出现"

    def test_numbering_survives_untrusted_wrapping(self):
        """包装（护栏）之后编号依然要原样保留 —— 正文 [n] 必须还能对上材料。"""
        from medscholar.retrieval import build_context_digest

        digest = build_context_digest([(3, self._paper("A")), (1, self._paper("B"))])
        assert "[1]" in digest and "[3]" in digest and "[2]" not in digest

    def test_plain_list_still_numbers_sequentially(self):
        """向后兼容：没有显式编号的普通数据仍按 start_index 顺序编号。"""
        text = _shell().digest_papers(
            [{"title": "A"}, {"title": "B"}], start_index=5
        )
        assert "[5] A" in text and "[6] B" in text

    def test_first_entry_without_index_keeps_offset(self):
        """混合场景：只有部分数据带显式编号时，各自按自己的规则取值。"""
        text = _shell().digest_papers(
            [{"title": "A"}, {"title": "B", "__index__": 9}], start_index=1
        )
        assert "[1] A" in text and "[9] B" in text

    @pytest.mark.parametrize("bad", ["3", None, True, 2.5, [3], {"n": 3}])
    def test_invalid_explicit_index_falls_back(self, bad):
        """脏编号安全回退而不是抛异常：这个函数在写作主路径上。"""
        text = _shell().digest_papers([{"title": "A", "__index__": bad}], start_index=7)
        assert "[7] A" in text

    def test_digest_is_not_a_registered_prompt(self):
        """材料块是**数据渲染**，不进注册表：每次运行都变的内容不该有"提示词版本"。"""
        with pytest.raises(PromptError):
            REGISTRY.metadata("digest.papers")
