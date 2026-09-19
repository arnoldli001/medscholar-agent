"""提示词注册表的**自加载**行为。

这条测试守的是一个"静默空状态"：注册方向是 `prompt_library → prompts`
（避免循环 import），所以只 import `platform.prompts` 时注册表是空的。
空注册表**不会报错**，只会让 `describe()` 返回 `{}`、`metadata()` 抛"未知 key" ——
在指标面板上看起来就是"没有提示词"，跟"没人调用"长得一模一样。

修法是在查询入口加一道 `_ensure_library_loaded()` 守卫。写这个守卫时踩了一个坑，
所以这里两个方向都要断言：
1. 裸导入本模块后，注册表**必须**已经能用（自加载生效）；
2. 守卫不能自己调自己 —— 第一版用 `REGISTRY.keys()` 判空，而 `keys()` 也带守卫，
   直接把栈打爆（`RecursionError`）。所以额外断言重复调用是幂等的。
"""

from __future__ import annotations

import importlib

import pytest


def test_bare_import_populates_registry():
    """只导入 platform.prompts（不导入 prompt_library）时，注册表也必须可用。"""
    module = importlib.import_module("medscholar.platform.prompts")
    keys = module.REGISTRY.keys()
    assert keys, "裸导入后注册表为空 —— 自加载守卫失效了"
    assert len(keys) >= 10, f"提示词数量异常：{len(keys)}"


def test_describe_and_metadata_work_without_explicit_library_import():
    module = importlib.import_module("medscholar.platform.prompts")
    described = module.REGISTRY.describe()
    assert described, "describe() 返回空 —— 指标面板会显示 0 条提示词"
    key = sorted(described)[0]
    meta = module.prompt_metadata(key)
    assert meta["key"] == key
    assert isinstance(meta["version"], int)


def test_guard_is_idempotent_and_does_not_recurse():
    """重复调用守卫不能递归、不能改变已有注册内容。"""
    module = importlib.import_module("medscholar.platform.prompts")
    first = module.REGISTRY.keys()
    for _ in range(5):
        module._ensure_library_loaded()
    assert module.REGISTRY.keys() == first


def test_is_empty_reads_internal_state_without_loading():
    """`is_empty()` 只读内部状态、不触发加载 —— 守卫用它判空，否则会递归。

    顺带验证"新实例不会被全局注册表污染"：自加载只填全局 :data:`REGISTRY`，
    不会往调用方自己 new 出来的实例里塞东西（多实例语义要清晰）。
    """
    from medscholar.platform.prompts import REGISTRY, PromptRegistry

    fresh = PromptRegistry()
    for _ in range(50):  # 重复调用：如果判空走了带守卫的 keys()，这里会 RecursionError
        assert fresh.is_empty() is True
    assert fresh.is_empty() is True, "新实例必须是空的（is_empty 只读内部状态）"
    assert not REGISTRY.is_empty(), "全局注册表应已被自加载填好"
    assert fresh.is_empty(), "自加载不能把内容塞进调用方自己的实例"


@pytest.mark.parametrize("key", ["writer.system", "plan.system"])
def test_known_keys_are_retrievable(key: str):
    """常用提示词必须能被取到（迁移完整性由 tests/test_prompts.py 逐字断言）。"""
    from medscholar.platform.prompts import REGISTRY

    assert key in REGISTRY.keys()
    assert REGISTRY.describe()[key]["current"] >= 1
