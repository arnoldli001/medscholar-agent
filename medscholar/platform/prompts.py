"""带版本号的提示词注册表（platform 层，只依赖标准库）。

## 为什么提示词需要版本号

本项目原先所有提示词都是 ``medscholar/llm/prompts.py`` 里的模块级字符串常量，
看着"简单直接"，实际有三个绕不开的问题：

1. **不可追溯**：模型输出变差时，没有任何记录能回答"这次运行用的是哪一版提示词"——
   提示词被就地改写，历史只留在 git 里，而 git 的提交信息不会进入 trace 与日志；
2. **不可比较**：想做 A/B（"更严格的引用要求" vs "更简洁"）只能改代码再跑一遍，
   两版提示词无法在同一进程里共存，更做不到"只切换一次调用"；
3. **散落**：部分提示词写在各 agent 模块的内联字符串里，没有单一入口，
   改一处忘一处。

版本号的成本很低（一个整数 + 一句中文说明），收益是：日志、trace、``/api/metrics``
里都能带上 ``key@version[variant]``，出问题时能**定位到具体那一版**，
而不是"大概是上周三改的那句话"。

## 为什么不用模板引擎（jinja2 等）

提示词是**要被逐字审阅的东西**：一句"只输出 JSON，不要解释"删掉半句，模型行为就会
漂移，而漂移看不见。模板引擎会把"提示词到底长什么样"拆成继承、include、过滤器、
自动转义等间接层——审阅时得在脑子里先渲染两层，静态阅读能力直接下降。
本模块只用 ``str.replace`` 语义的**受限替换**（见下），任何一版提示词都能一眼读到底。

## 占位符：只处理"声明过的名字"（本模块最重要的取舍）

渲染不能简单用 :meth:`str.format`：

* 提示词里**合法地**存在花括号——例如示范 JSON 输出的 ``{"queries": [...]}``、
  ``{"outline": [{"title": ...}]}``。``format`` 会把它们当占位符，轻则报 KeyError，
  重则被替换成奇怪内容；
* 反过来，提示词里多写一个没被替换的 ``{xxx}`` 会**静默进入模型上下文**——
  模型会把它当正文照着编，这类问题在产物里极难发现（看起来像模型"自由发挥"）。

同时满足这两点的做法只有一种：**只对 ``placeholders`` 里显式声明过的名字做替换与
检查**，其余花括号一律当普通字符。于是：

* 声明了却没传值 → 抛 :class:`PromptError`（列出缺了哪些、这个提示词要哪些）；
* 声明了但正文里根本不存在这个 ``{name}`` → **注册时**就报错（声明与实际不符会让
  "未替换检查"变成空转）；
* 没声明的花括号（JSON 示例）→ 完全不动，也不误报。

配套的 :meth:`PromptRegistry.undeclared_placeholders` 用来审计"看起来像占位符、
却没人声明"的 token（只识别 ``{标识符}`` 形态，JSON 示例天然不会命中）。

## 与 prompt_library 的分工（以及为什么用 loader 而不是直接 import）

本模块是**机制**：注册、选版、渲染、统计，不含任何具体提示词文本；
具体文本在 :mod:`medscholar.platform.prompt_library` 里登记。
装配方向是 ``prompt_library → prompts``，本模块**不能**反向 import 它——
两个模块互相 import 会形成模块级环，被 ``scripts/check_arch.py`` 的循环依赖检查
抓出来（那不是形式主义：环一旦形成，import 顺序就会决定行为）。
因此这里提供 :func:`register_library_loader` 注册"装配函数"，
:func:`reset_registry` 靠它把注册表重建回刚 import 完的状态。
"""

from __future__ import annotations

import difflib
import re
from dataclasses import dataclass
from typing import Any, Callable

__all__ = [
    "PROMPT_KINDS",
    "PromptError",
    "PromptVersion",
    "Prompt",
    "PromptRegistry",
    "REGISTRY",
    "register_library_loader",
    "get_prompt",
    "prompt_text",
    "prompt_metadata",
    "active_variant",
    "set_variant",
    "reset_registry",
]

#: 提示词类别的**受控词表**，取 key 的第一段（``writer.section`` 的类别是 ``writer``）。
#:
#: 为什么要受控：key 是唯一标识，一旦出现 ``writer.section`` / ``writing.section``
#: 这种同义并存，注册表就退化成一堆字符串，A/B 与统计都无从下手。
#: 新增类别时先在这里登记（``register`` 会拒绝词表外的类别），
#: 这样"提示词一共有哪几类"永远是一眼可见的。
PROMPT_KINDS: tuple[str, ...] = (
    "core",  # 角色设定、硬性规则等被多处复用的片段
    "system",  # 通用系统提示词（保留给尚未归类的场景）
    "plan",  # 检索规划
    "translate",  # 术语翻译
    "outline",  # 大纲规划
    "critique",  # 文献批判
    "reflect",  # 自我审查
    "writer",  # 综述/章节撰写
    "ask",  # 知识库问答
    "chat",  # 对话
    "summary",  # 单篇速读
    "manuscript",  # 论文初稿
    "faithfulness",  # 引用忠实度核查
)

#: ``{name}`` 形态的占位符 token。
#:
#: 刻意只匹配**标识符**（字母/下划线开头）：JSON 示例里的 ``{"queries": []}``
#: 花括号后面紧跟的是引号，因此永远不会被当成占位符。这条正则同时服务于
#: 注册期校验与 :meth:`PromptRegistry.undeclared_placeholders`。
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PromptError(ValueError):
    """提示词注册表的使用错误（未知 key、缺占位符、重复版本……）。

    继承 :class:`ValueError` 而不是自定义基类：调用方（HTTP 路由、CLI、agent）
    已经习惯把 ``ValueError`` 当"输入不对"处理，多一层自定义异常层次只会增加导入。
    消息一律用中文写清楚"哪里错了、怎么修"——这类错误最终是给深夜排障的人看的。
    """


@dataclass(frozen=True)
class PromptVersion:
    """提示词的**一个不可变版本**。

    版本一旦登记就不再改写：要改提示词就登记新版本号。这是"可追溯"的前提——
    如果 v2 可以覆盖 v1，那么历史 trace 里的 ``writer.section@1`` 就再也对不上
    当时的文本了。

    :param key: 唯一标识，形如 ``writer.section``（类别见 :data:`PROMPT_KINDS`）。
    :param version: 版本号，从 1 开始递增。
    :param text: 提示词正文，可含 ``{name}`` 占位符。
    :param description: 中文说明：**这一版改了什么、为什么改**。
        它不是注释而是数据——会被 :func:`prompt_metadata` 带进 trace 与 ``/api/metrics``。
    :param tags: 标签，例如 ``("system", "citation-strict")``，便于按主题筛选。
    :param placeholders: 声明本版需要的占位符名（不带花括号）。
        声明是"必须传值"的承诺，也是"只替换这些"的边界。
    :param deprecated: 已弃用标记。取用时不指定版本会优先挑**最高未弃用**版本；
        全部弃用时退回最高版本，并在 :meth:`PromptRegistry.describe` 里标出来。
    """

    key: str
    version: int
    text: str
    description: str
    tags: tuple[str, ...] = ()
    placeholders: tuple[str, ...] = ()
    deprecated: bool = False


@dataclass(frozen=True)
class Prompt:
    """一次取用的结果：**已渲染好的文本** + 它的版本坐标。

    ``text`` 已经是最终文本，因此 ``str(prompt)`` 就能直接当字符串用
    （``system=get_prompt("writer.system")`` 这种写法不必改调用方代码）。
    ``version`` / ``variant`` 是给 trace 用的坐标：模型输出变差时，
    先看这两个值就知道当时用的是哪一版。
    """

    key: str
    text: str
    version: int
    variant: str = "default"

    def __str__(self) -> str:
        return self.text


def _kind_of(key: str) -> str:
    """取 key 的类别（第一段）。"""
    return key.split(".", 1)[0]


def _ensure_library_loaded() -> None:
    """首次取用时确保提示词库已完成注册。

    为什么需要这个守卫：注册方向是 ``prompt_library → prompts``（为了避免循环 import），
    所以**只**导入本模块时注册表是空的。而空注册表**不会报错** ——
    它只会让 :meth:`describe` 返回 ``{}``、:meth:`metadata` 抛"未知 key"，
    表现为"指标面板显示 0 条提示词"，看起来像没人调用。这种静默空状态最难定位，
    所以在所有查询入口都加一道自加载：谁先被导入都能拿到完整注册表。

    两个实现细节都是踩出来的：

    * 判空必须读**内部状态**（:meth:`PromptRegistry.is_empty`），不能调 ``keys()`` ——
      而 ``keys()`` 自己就带守卫，于是守卫调守卫，直接把栈打爆
      （实测 `RecursionError: maximum recursion depth exceeded`）。
    * 用 ``_LOADING`` 标志做重入保护：加载过程中若又有人问注册表，
      必须立刻返回而不是再触发一次 import。
    """
    global _LIBRARY_LOADING
    if _LIBRARY_LOADING or not REGISTRY.is_empty():
        return
    _LIBRARY_LOADING = True
    try:
        import importlib

        importlib.import_module("medscholar.platform.prompt_library")
    finally:
        _LIBRARY_LOADING = False


#: 自加载重入保护（见 :func:`_ensure_library_loaded`）
_LIBRARY_LOADING = False


class PromptRegistry:
    """提示词注册表：登记、选版、渲染与用量统计。

    数据结构是一棵朴素的三层字典：``key → variant → version → PromptVersion``。
    变体与版本是两个**正交**的轴——同一条 ``writer.section`` 可以同时存在
    "default 变体 v1"、"default 变体 v2（更严格）"和 "citation-strict 变体 v1"，
    不会互相覆盖。A/B 只需要切变体，不必复制整条 key。
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, dict[int, PromptVersion]]] = {}
        #: A/B 开关：key → 当前生效变体（未设置的 key 不在这里，等价于 "default"）
        self._active: dict[str, str] = {}
        #: 进程内取用计数（每个已登记 key 一条，未取用过记 0）
        self._usage: dict[str, int] = {}

    # ------------------------------------------------------------- 登记
    def register(self, prompt: PromptVersion, *, variant: str = "default") -> None:
        """登记一个提示词版本。参数不合法时抛 :class:`PromptError`。

        校验刻意做得啰嗦：提示词注册表的失效方式不是崩溃，而是**静默**走偏
        （少传一个占位符、声明写错名字、同义 key 并存），所以错误必须在登记期就炸出来，
        而不是等模型产出一篇格式崩坏的文章。
        """
        if not isinstance(prompt, PromptVersion):
            raise PromptError(
                f"register() 只接受 PromptVersion，收到 {type(prompt).__name__}；"
                "提示词正文请用 PromptVersion(key=..., version=..., text=..., description=...)"
            )
        key = prompt.key
        version = prompt.version
        text = prompt.text

        if not isinstance(key, str) or not key.strip():
            raise PromptError("提示词 key 不能为空")
        if "." not in key:
            raise PromptError(
                f"提示词 key 必须写成「<类别>.<用途>」两段以上，例如 writer.section；收到 {key!r}"
            )
        kind = _kind_of(key)
        if kind not in PROMPT_KINDS:
            raise PromptError(
                f"未知的提示词类别 {kind!r}（来自 key={key!r}）。"
                f"已登记的类别见 PROMPT_KINDS：{'、'.join(PROMPT_KINDS)}。"
                "新类别请先在 PROMPT_KINDS 里登记，否则同义 key 会悄悄并存。"
            )
        if isinstance(version, bool) or not isinstance(version, int) or version < 1:
            raise PromptError(f"{key} 的 version 必须是 >= 1 的整数，收到 {version!r}")
        if not isinstance(text, str):
            raise PromptError(f"{key} v{version} 的 text 必须是 str，收到 {type(text).__name__}")
        if not isinstance(prompt.description, str) or not prompt.description.strip():
            raise PromptError(
                f"{key} v{version} 缺少 description。"
                "描述是版本可追溯的一半：请写清「这一版改了什么、为什么改」（中文一句话即可）。"
            )
        if not isinstance(variant, str) or not variant.strip():
            raise PromptError(f"{key} v{version} 的变体名不能为空")

        placeholders = tuple(prompt.placeholders)
        seen: set[str] = set()
        for name in placeholders:
            if not isinstance(name, str) or not name.strip():
                raise PromptError(f"{key} v{version} 的 placeholders 里有非法名字：{name!r}")
            if name in seen:
                raise PromptError(f"{key} v{version} 的 placeholders 里有重复项：{name}")
            seen.add(name)
            if "{" + name + "}" not in text:
                raise PromptError(
                    f"{key} v{version} 声明了占位符 {name}，但正文里找不到 {{{name}}}。"
                    "声明与实际不符会让调用方白传一个变量，也会让「未替换检查」形同虚设；"
                    "要么补上这个占位符，要么把它从 placeholders 里删掉。"
                )

        item = PromptVersion(
            key=key,
            version=version,
            text=text,
            description=prompt.description,
            tags=tuple(prompt.tags),
            placeholders=placeholders,
            deprecated=bool(prompt.deprecated),
        )
        bucket = self._items.setdefault(key, {}).setdefault(variant, {})
        if version in bucket:
            raise PromptError(
                f"{key} 的变体 {variant!r} 已经登记过 v{version}。"
                "版本一旦发布就不再改写：要改提示词请登记 v"
                f"{max(bucket) + 1}，这样历史 trace 里的旧坐标仍然对得上当时的文本。"
            )
        bucket[version] = item
        self._usage.setdefault(key, 0)

    # ------------------------------------------------------------- 查询
    def is_empty(self) -> bool:
        """注册表是否为空（**不带自加载守卫**，供守卫自己判空用，避免递归）。"""
        return not self._items

    def keys(self) -> list[str]:
        """全部已登记的 key（字典序，便于稳定输出与 diff）。"""
        _ensure_library_loaded()
        return sorted(self._items)

    def versions(self, key: str) -> list[int]:
        """该 key 的版本号（**所有变体的并集**，去重升序）。"""
        _ensure_library_loaded()
        variants = self._entry(key)
        return sorted({version for bucket in variants.values() for version in bucket})

    def variants(self, key: str) -> list[str]:
        """该 key 已登记的变体名（字典序）。"""
        return sorted(self._entry(key))

    def active_variant(self, key: str) -> str:
        """当前生效的变体名（没切过就是 ``"default"``）。"""
        self._entry(key)
        return self._active.get(key, "default")

    def set_variant(self, key: str, variant: str) -> None:
        """进程内切换某条提示词的变体（A/B 开关）。

        传 ``"default"`` 表示**取消** A/B，回到默认变体（不是切换到名为 default 的旁路），
        这样开关不会在 :meth:`describe` / :meth:`usage` 里留下痕迹。
        变体不存在时直接报错而不是静默退回默认——静默退回会让 A/B 实验结果无法解释。
        """
        known = self._entry(key)
        if variant not in known:
            raise PromptError(
                f"{key} 没有变体 {variant!r}；已登记的变体：{'、'.join(sorted(known))}。"
                "先用 register(..., variant=...) 登记该变体再切换。"
            )
        if variant == "default":
            self._active.pop(key, None)
        else:
            self._active[key] = variant

    def get(
        self,
        key: str,
        *,
        version: int | None = None,
        variant: str = "default",
        **variables: Any,
    ) -> Prompt:
        """取用并渲染提示词。

        :param version: ``None`` 表示"该 key 的最高**未弃用**版本"；全部弃用时
            退回最高版本（并在 :meth:`describe` 里标出 deprecated），
            这样线上不会因为一次弃用标记就突然取不到提示词。
        :param variant: ``"default"``（默认值）表示**当前生效的变体**——
            没切过就是名为 default 的那一版，切过（:meth:`set_variant`）就是切过去的那一版。
            这正是 A/B 开关能生效的方式：调用点不必知道自己被做了实验。
            传**具体变体名**（例如 ``"citation-strict"``）则是显式指定，与开关无关；
            想显式回到默认版，先 :meth:`set_variant` 回 ``"default"``。
        :param variables: 占位符取值。**多余的变量会被忽略**——一个调用点
            （例如 writer 写章节）要能同时喂默认版与 citation-strict 变体，
            而两者声明的占位符未必完全一致。

        未知 key / 未知版本 / 缺占位符都会抛 :class:`PromptError`，
        错误消息里带 key 名、可用 key 的建议与缺失清单。
        """
        _ensure_library_loaded()
        resolved_variant, item = self._resolve(key, version=version, variant=variant)
        text = self._substitute(item, variables)
        self._usage[item.key] = self._usage.get(item.key, 0) + 1
        return Prompt(key=item.key, text=text, version=item.version, variant=resolved_variant)

    def render(
        self,
        key: str,
        *,
        version: int | None = None,
        variant: str = "default",
        **variables: Any,
    ) -> str:
        """同 :meth:`get`，但只返回文本（最常用：直接拿字符串拼进 messages）。"""
        _ensure_library_loaded()
        return self.get(key, version=version, variant=variant, **variables).text

    def metadata(
        self, key: str, *, version: int | None = None, variant: str = "default"
    ) -> dict[str, Any]:
        """该 key 当前取用坐标的元信息（供 trace / ``/api/metrics`` 记录）。

        不渲染，因此**不需要占位符取值**——埋点代码不必先凑齐变量才能记账。
        ``tags`` / ``placeholders`` 用 list 而不是 tuple：这份字典会直接进 JSON，
        list 与 dataclass 里的 tuple 在序列化上等价，但读起来更符合 JSON 的形状。
        """
        _ensure_library_loaded()
        resolved_variant, item = self._resolve(key, version=version, variant=variant)
        return {
            "key": item.key,
            "version": item.version,
            "variant": resolved_variant,
            "description": item.description,
            "tags": list(item.tags),
            "placeholders": list(item.placeholders),
            "deprecated": item.deprecated,
        }

    def undeclared_placeholders(
        self, key: str, *, version: int | None = None, variant: str = "default"
    ) -> tuple[str, ...]:
        """审计：正文里"看起来像占位符、却没有声明"的 token。

        只识别 ``{标识符}`` 形态，所以 JSON 示例（``{"queries": []}``）不会命中。
        返回值非空不代表一定是 bug（可能是刻意示范 JSON 片段），
        但它正是"漏替换的 ``{xxx}`` 静默进入模型上下文"的唯一入口，
        值得在测试里对全部 key 断言一遍。
        """
        _, item = self._resolve(key, version=version, variant=variant)
        declared = set(item.placeholders)
        found = dict.fromkeys(_PLACEHOLDER_RE.findall(item.text))
        return tuple(name for name in found if name not in declared)

    def describe(self) -> dict[str, dict[str, Any]]:
        """全局概览：``{key: {versions, variants, current, deprecated}}``。

        ``current`` 与 :meth:`get` 的选择规则完全一致（最高未弃用→否则最高），
        ``deprecated`` 表示"当前变体下已无可用版本，只能取到弃用版本"。
        """
        _ensure_library_loaded()
        out: dict[str, dict[str, Any]] = {}
        for key in self.keys():
            variant = self._active.get(key, "default")
            if variant not in self._items[key]:  # 变体被清掉后的防御性回退
                variant = "default"
            bucket = self._items[key][variant]
            usable = [v for v in sorted(bucket) if not bucket[v].deprecated]
            out[key] = {
                "versions": self.versions(key),
                "variants": self.variants(key),
                "current": max(usable) if usable else max(bucket),
                "deprecated": not usable,
            }
        return out

    def usage(self) -> dict[str, int]:
        """进程内取用计数（:meth:`get` 每成功一次 +1，:meth:`render` 也走 get 同样 +1）。

        包含**所有**已登记的 key（没用过记 0）：``/api/metrics`` 需要一张稳定的表，
        缺行会让面板上的曲线"忽隐忽现"。计数是进程内的，重启即归零——
        它不是审计账本，只是"这一版提示词最近有没有被真的用到"。
        """
        return {key: self._usage.get(key, 0) for key in self.keys()}

    def reset_usage(self) -> None:
        """把计数清零（测试与长驻进程的统计窗口切换用）。"""
        for key in self._usage:
            self._usage[key] = 0

    def clear(self) -> None:
        """清空全部内容（含 A/B 开关与计数）。主要给 :func:`reset_registry` 与测试用。"""
        self._items.clear()
        self._active.clear()
        self._usage.clear()

    # ------------------------------------------------------------- 内部
    def _entry(self, key: str) -> dict[str, dict[int, PromptVersion]]:
        """按 key 取变体表；未知 key 抛出带**相似 key 建议**的错误。"""
        try:
            return self._items[key]
        except KeyError:
            raise self._unknown_key(key) from None

    def _unknown_key(self, key: str) -> PromptError:
        """构造可读的"未知 key"错误：包含 key 名、相似建议与可用 key 清单。

        拼写错误（``writer.sectoin``）是最常见的调用事故，difflib 的建议比
        "KeyError: 'writer.sectoin'" 有用得多；列全量 key 则省掉一次 grep。
        """
        available = self.keys()
        close = difflib.get_close_matches(str(key), available, n=3, cutoff=0.5)
        hint = f"你是不是想找：{'、'.join(close)}？" if close else ""
        listing = "、".join(available) if available else "（注册表为空）"
        return PromptError(
            f"未知的提示词 key：{key!r}。{hint}"
            f"当前可用 key 共 {len(available)} 个：{listing}"
        )

    def _resolve(
        self, key: str, *, version: int | None, variant: str
    ) -> tuple[str, PromptVersion]:
        """把 (key, version, variant) 解析成一次具体的 (变体名, 版本对象)。"""
        variants = self._entry(key)
        if variant == "default":
            # 只有"默认值"才会被 A/B 开关改写：显式传变体名的调用点不受影响
            variant = self._active.get(key, "default")
        if variant not in variants:
            raise PromptError(
                f"{key} 没有变体 {variant!r}；已登记的变体：{'、'.join(sorted(variants))}"
            )
        bucket = variants[variant]
        if version is None:
            usable = [v for v in sorted(bucket) if not bucket[v].deprecated]
            # 全部弃用时退回最高版本：宁可用旧提示词，也不要在运行时突然取不到
            picked = max(usable) if usable else max(bucket)
        elif isinstance(version, bool) or not isinstance(version, int):
            raise PromptError(f"{key} 的 version 必须是整数，收到 {version!r}")
        elif version not in bucket:
            raise PromptError(
                f"{key} 的变体 {variant!r} 没有 v{version}；"
                f"已登记的版本：{'、'.join(f'v{v}' for v in sorted(bucket))}"
            )
        else:
            picked = version
        return variant, bucket[picked]

    @staticmethod
    def _substitute(item: PromptVersion, variables: dict[str, Any]) -> str:
        """受限替换 + 未替换检查（见模块 docstring 的"占位符"一节）。"""
        if not item.placeholders:
            return item.text
        missing = [name for name in item.placeholders if name not in variables]
        if missing:
            raise PromptError(
                f"渲染 {item.key} v{item.version} 缺少占位符：{'、'.join(missing)}。"
                f"这个提示词需要：{'、'.join(item.placeholders)}"
                f"（共 {len(item.placeholders)} 个）。"
                "缺参会把 {...} 静默留在提示词里被模型当成正文，所以这里直接报错，"
                "而不是用空串兜底。"
            )
        pattern = re.compile(
            "|".join(r"\{" + re.escape(name) + r"\}" for name in item.placeholders)
        )
        used: set[str] = set()

        def _pick(match: re.Match[str]) -> str:
            name = match.group(0)[1:-1]
            used.add(name)
            value = variables[name]
            return value if isinstance(value, str) else str(value)

        rendered = pattern.sub(_pick, item.text)
        # 双保险一：声明过的占位符必须都被替换过（登记期已校验正文存在，这里防"绕过登记"）
        unused = [name for name in item.placeholders if name not in used]
        if unused:
            raise PromptError(
                f"渲染 {item.key} v{item.version} 后，占位符 {'、'.join(unused)} 一次都没被替换："
                "正文与 placeholders 声明不一致，请检查登记内容是否被改写。"
            )
        # 双保险二：渲染结果里不应再有声明过的占位符残留。
        # 替换值与替换都做完了还残留，只可能是"某个变量的取值里正好含 {名字}"，
        # 消息里点明这一点，省掉一轮排查。
        leftover = [name for name in item.placeholders if "{" + name + "}" in rendered]
        if leftover:
            raise PromptError(
                f"渲染 {item.key} v{item.version} 后仍有未替换的占位符："
                f"{'、'.join('{' + n + '}' for n in leftover)}。"
                "如果不是变量取值里恰好包含这种文本，就说明模板里写了两次而只替换了部分。"
            )
        return rendered


#: 全局注册表。内建提示词由 :mod:`medscholar.platform.prompt_library` 在 import 时装配
#: （见模块 docstring：本模块不能反向 import 它，否则成环）。
REGISTRY = PromptRegistry()

#: 内建提示词的装配函数。注册进来是为了让 :func:`reset_registry` 能把注册表
#: 重建回"刚 import 完"的状态，而不是清空后取不到任何提示词。
_LIBRARY_LOADERS: list[Callable[[PromptRegistry], None]] = []


def register_library_loader(loader: Callable[[PromptRegistry], None]) -> None:
    """注册一个"往注册表里装内建提示词"的函数（供 prompt_library 调用）。

    用回调而不是 import：prompt_library 依赖本模块的类定义，本模块若再 import 它，
    模块级依赖就成环了。回调让依赖保持单向，代价是"装配时机"从 import 变成了显式动作
    （表现在 :func:`reset_registry` 里）。
    重复注册同一个函数对象是幂等的（模块被重新 import 时不会装两遍）。
    """
    if not callable(loader):
        raise PromptError(f"装配函数必须可调用，收到 {type(loader).__name__}")
    if not any(existing is loader for existing in _LIBRARY_LOADERS):
        _LIBRARY_LOADERS.append(loader)


def reset_registry() -> None:
    """把全局注册表恢复成"刚 import 完 prompt_library"的状态（测试用）。

    顺序很重要：先清空再重装，因此测试里临时登记的 key、切过的 A/B 变体与计数
    都会被抹掉，不会串到下一个用例——全局可变状态最容易制造"单独跑绿、一起跑红"。
    """
    REGISTRY.clear()
    for loader in _LIBRARY_LOADERS:
        loader(REGISTRY)


def get_prompt(
    key: str,
    *,
    version: int | None = None,
    variant: str = "default",
    **variables: Any,
) -> Prompt:
    """从全局注册表取用提示词（见 :meth:`PromptRegistry.get`）。"""
    return REGISTRY.get(key, version=version, variant=variant, **variables)


def prompt_text(key: str, **variables: Any) -> str:
    """最常用的入口：直接拿渲染好的提示词文本。

    签名刻意保持"只有变量"：调用点读起来就是 ``prompt_text("chat.user.tail")``，
    不需要知道版本与变体的存在——版本选择是运维手段，不该污染业务代码。
    """
    return REGISTRY.render(key, **variables)


def prompt_metadata(key: str) -> dict[str, Any]:
    """当前取用坐标的元信息（见 :meth:`PromptRegistry.metadata`），用于 trace/指标。"""
    return REGISTRY.metadata(key)


def active_variant(key: str) -> str:
    """当前生效的变体名（见 :meth:`PromptRegistry.active_variant`）。"""
    return REGISTRY.active_variant(key)


def set_variant(key: str, variant: str) -> None:
    """进程内切换 A/B 变体（见 :meth:`PromptRegistry.set_variant`）。"""
    REGISTRY.set_variant(key, variant)
