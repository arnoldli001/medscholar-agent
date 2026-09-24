"""带版本号的提示词注册表（platform 层，只依赖标准库）。

版本化让日志、trace、``/api/metrics`` 都能带上 ``key@version[variant]``，
回答"这次跑的是哪一版"，并支持两版同进程共存的 A/B；成本只是一个整数加一句中文说明。
不用模板引擎（jinja2 等）：提示词要逐字审阅，继承/include/过滤器会让静态阅读
先脑内渲染两层。占位符用受限替换而不是 :meth:`str.format`：正文里合法存在示范 JSON
的花括号（``{"queries": [...]}``），而漏替换的 ``{xxx}`` 会静默进模型上下文被当正文照编：

* 声明了却没传值 → 抛 :class:`PromptError`；
* 声明了但正文里没有该 ``{name}`` → 注册时就报错；
* 没声明的花括号（JSON 示例）→ 完全不动，也不误报。

:meth:`PromptRegistry.undeclared_placeholders` 审计"像占位符却没人声明"的 token。
与 prompt_library 分工：本模块是机制（注册/选版/渲染/统计），文本在该库登记；
装配方向 ``prompt_library → prompts``，反向 import 成环（check_arch 会拦），
故用 :func:`register_library_loader` 回调装配，:func:`reset_registry` 靠它重建。
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

#: key 第一段类别的受控词表（``writer.section`` → ``writer``）。同义并存
#: （writer/writing）会让注册表退化成字符串堆；新增类别先在此登记，register 拒绝词表外类别。
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

#: ``{标识符}`` 占位符 token（字母/下划线开头）：JSON 示例里花括号后紧跟引号，永不命中。
#: 同一条正则服务注册期校验与 :meth:`PromptRegistry.undeclared_placeholders`。
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")


class PromptError(ValueError):
    """提示词注册表的使用错误（未知 key、缺占位符、重复版本等），继承 :class:`ValueError`。

    不另立异常层次（调用方已习惯按 ValueError 处理"输入不对"）；
    消息一律用中文写清"哪里错了、怎么修"——最终是给深夜排障的人看的。
    """


@dataclass(frozen=True)
class PromptVersion:
    """提示词的一个不可变版本。登记后不再改写，要改就登新版本号——
    v2 若能覆盖 v1，历史 trace 里的 ``writer.section@1`` 就对不上当时的文本。

    ``description`` 是数据不是注释，会被 :func:`prompt_metadata` 带进 trace 与 metrics；
    ``placeholders`` 既是"必须传值"的承诺，也是"只替换这些"的边界；
    ``deprecated`` 取用时优先选最高未弃用版本，全弃用则退回最高版本并在 describe 标出。
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
    """一次取用的结果：已渲染好的文本 + 版本坐标。

    ``str(prompt)`` 即最终文本，既有调用点无需改动；``version``/``variant``
    供 trace 使用，输出变差时先看坐标即可知道当时跑的是哪一版。
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

    注册方向是 ``prompt_library → prompts``，只导入本模块时注册表为空；而空注册表
    不报错，只表现为面板 0 条提示词/未知 key，最难定位，故所有查询入口加自加载。
    两个坑：判空必须读 :meth:`PromptRegistry.is_empty` 不能调 ``keys()``——
    守卫调守卫会递归打爆栈（RecursionError）；``_LIBRARY_LOADING`` 做重入保护，
    加载中再被问必须立刻返回，不再触发 import。
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

    三层字典 ``key → variant → version → PromptVersion``。变体与版本是正交的轴：
    default 的 v1/v2 与 citation-strict 的 v1 可共存而不互相覆盖，A/B 只切变体不必复制 key。
    """

    def __init__(self) -> None:
        self._items: dict[str, dict[str, dict[int, PromptVersion]]] = {}
        #: A/B 开关：key → 当前生效变体（未设置的 key 不在这里，等价于 "default"）
        self._active: dict[str, str] = {}
        #: 进程内取用计数（每个已登记 key 一条，未取用过记 0）
        self._usage: dict[str, int] = {}

    # ------------------------------------------------------------- 登记
    def register(self, prompt: PromptVersion, *, variant: str = "default") -> None:
        """登记一个提示词版本，参数不合法抛 :class:`PromptError`。

        校验刻意严格：注册表的失效方式不是崩溃而是静默走偏（漏占位符、声明名写错、
        同义 key 并存），错误必须在登记期炸出来，而不是等模型产出格式崩坏的文章。
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
        """注册表是否为空（不带自加载守卫，供守卫自己判空用，避免递归）。"""
        return not self._items

    def keys(self) -> list[str]:
        """全部已登记的 key（字典序，便于稳定输出与 diff）。"""
        _ensure_library_loaded()
        return sorted(self._items)

    def versions(self, key: str) -> list[int]:
        """该 key 的版本号（所有变体的并集，去重升序）。"""
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

        传 ``"default"`` 取消 A/B，开关不在 describe/usage 留痕；变体不存在直接报错
        而不是静默退回默认，否则 A/B 实验结果无法解释。
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
        """取用并渲染提示词；未知 key / 未知版本 / 缺占位符抛 :class:`PromptError`。

        ``version=None`` 选最高未弃用版本，全弃用时退回最高版本（避免一次弃用标记
        让线上突然取不到）。``variant="default"``（默认）表示跟随 :meth:`set_variant`
        的 A/B 开关，调用点无需感知；传具体变体名则是显式指定，与开关无关。
        多余的 variables 一律忽略：同一调用点可能要喂占位符不完全一致的两个变体。
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
        """当前取用坐标的元信息（供 trace / ``/api/metrics``）。

        不渲染，埋点代码不必先凑齐占位符变量；tags/placeholders 用 list 输出以贴合 JSON 形状。
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

        只识别 ``{标识符}``，JSON 示例不会命中。返回值非空不一定是 bug（可能刻意示范 JSON），
        但它是漏替换的 ``{xxx}`` 静默进入模型上下文的唯一入口，建议测试对全部 key 断言。
        """
        _, item = self._resolve(key, version=version, variant=variant)
        declared = set(item.placeholders)
        found = dict.fromkeys(_PLACEHOLDER_RE.findall(item.text))
        return tuple(name for name in found if name not in declared)

    def describe(self) -> dict[str, dict[str, Any]]:
        """全局概览：``{key: {versions, variants, current, deprecated}}``。

        ``current`` 的选择规则与 :meth:`get` 一致（最高未弃用→否则最高）；
        ``deprecated`` 表示当前变体下已无未弃用版本、只能取到弃用版本。
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
        """进程内取用计数（:meth:`get`/:meth:`render` 每成功一次 +1）。

        包含所有已登记 key（没用过记 0），保证 ``/api/metrics`` 曲线稳定不缺行；
        计数重启即归零，它不是审计账本，只反映这一版最近有没有被真的用到。
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
        """按 key 取变体表；未知 key 抛出带相似 key 建议的错误。"""
        try:
            return self._items[key]
        except KeyError:
            raise self._unknown_key(key) from None

    def _unknown_key(self, key: str) -> PromptError:
        """构造可读的"未知 key"错误。

        拼写错误（``writer.sectoin``）是最常见调用事故，difflib 的相似建议
        （cutoff 0.5，最多 3 个）比裸 KeyError 有用；并附全量 key 清单省一次 grep。
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
        # 第一道检查：声明过的占位符必须都被替换过（登记期已校验正文存在，这里防"绕过登记"）
        unused = [name for name in item.placeholders if name not in used]
        if unused:
            raise PromptError(
                f"渲染 {item.key} v{item.version} 后，占位符 {'、'.join(unused)} 一次都没被替换："
                "正文与 placeholders 声明不一致，请检查登记内容是否被改写。"
            )
        # 第二道检查：替换后仍残留声明过的占位符，只可能是变量取值本身含 {名字}。
        leftover = [name for name in item.placeholders if "{" + name + "}" in rendered]
        if leftover:
            raise PromptError(
                f"渲染 {item.key} v{item.version} 后仍有未替换的占位符："
                f"{'、'.join('{' + n + '}' for n in leftover)}。"
                "如果不是变量取值里恰好包含这种文本，就说明模板里写了两次而只替换了部分。"
            )
        return rendered


#: 全局注册表。内建提示词由 prompt_library 在 import 时装配（本模块不能反向 import，否则成环）。
REGISTRY = PromptRegistry()

#: 内建装配函数清单，供 :func:`reset_registry` 清空后重装回"刚 import 完"状态。
_LIBRARY_LOADERS: list[Callable[[PromptRegistry], None]] = []


def register_library_loader(loader: Callable[[PromptRegistry], None]) -> None:
    """注册一个"往注册表里装内建提示词"的函数（供 prompt_library 调用）。

    用回调而不是 import，保持模块依赖单向不成环；重复注册同一函数对象幂等
    （模块被重新 import 时不会装两遍）。
    """
    if not callable(loader):
        raise PromptError(f"装配函数必须可调用，收到 {type(loader).__name__}")
    if not any(existing is loader for existing in _LIBRARY_LOADERS):
        _LIBRARY_LOADERS.append(loader)


def reset_registry() -> None:
    """恢复成"刚 import 完 prompt_library"的状态（测试用）。

    顺序必须是先清空再重装：临时 key、切过的 A/B 变体与计数都会被抹掉，
    不串到下一个用例——全局可变状态最容易制造"单独跑绿、一起跑红"。
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
    """最常用入口：直接拿渲染好的文本。

    签名刻意只有变量：版本选择是运维手段，调用点不必知道版本/变体的存在。
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
