"""兼容外壳测试：锁定"搬迁后旧 import 路径仍然可用"。

## 为什么需要这个测试

做了一次渐进式重构：把领域模块搬到 ``medscholar/domain/``、
把配置搬到 ``medscholar/platform/``，但**在旧路径留下只做重导出的外壳**，
因为 ``medscholar.models`` / ``medscholar.config`` 这些路径是公开契约
（CLI、HTTP、MCP、评测脚本、文档、用户二次脚本都在用）。

外壳的风险是**静默漏名字**。这不是假设：
第一版外壳只写了 ``from X import *``，而 ``import *`` 只带出 ``__all__`` 里的名字，
于是 ``from medscholar.config import _deep_merge``（私有名）直接 ImportError，
**20 个测试模块连收集都失败**。所以外壳必须有测试锁住，而不是靠记性。

这里断言的不只是"能导入"，而是**身份一致**：
``getattr(外壳, 名字) is getattr(真实模块, 名字)``。
用 ``is`` 而不是 ``==`` 是因为：如果是两份独立定义，
``isinstance()`` 判断会莫名其妙地失败，是最难查的一类 bug。
"""

from __future__ import annotations

import importlib

import pytest

#: (外壳模块, 真实模块, 外部确实引用到的非 __all__ 名字)
#: 第三列是 ``grep -rn "from medscholar.X import _"`` 得到的真实使用点，
#: 不是猜的 —— 每一条都对应某处会 ImportError 的代码。
SHIMS: tuple[tuple[str, str, tuple[str, ...]], ...] = (
    ("medscholar.textutil", "medscholar.domain.text", ("_WS_RE", "_CJK_RE")),
    ("medscholar.models", "medscholar.domain.models", ()),
    ("medscholar.dedupe", "medscholar.domain.dedupe", ("_richness",)),
    ("medscholar.query", "medscholar.domain.query", ("_tokenize", "_strip_quotes")),
    (
        "medscholar.config",
        "medscholar.platform.config",
        (
            "LLMSettings",
            "EmbeddingSettings",
            "RetrievalSettings",
            "AgentSettings",
            "SourcesSettings",
            "_looks_like_ollama_model",
            "_env_overrides",
            "_deep_merge",
            "_load_dotenv",
        ),
    ),
    ("medscholar.cite", "medscholar.domain.citation", ()),
    ("medscholar.cite.styles", "medscholar.domain.citation.styles", ()),
)


@pytest.mark.parametrize(("shim_name", "real_name", "extras"), SHIMS)
def test_shim_importable(shim_name: str, real_name: str, extras: tuple[str, ...]) -> None:
    """外壳必须能导入，且指向的模块名与预期一致。"""
    shim = importlib.import_module(shim_name)
    assert shim is not None
    assert real_name in str(shim.__doc__ or ""), (
        f"{shim_name} 的 docstring 里应指向 {real_name}，方便后来者知道该 import 哪里"
    )


@pytest.mark.parametrize(("shim_name", "real_name", "extras"), SHIMS)
def test_shim_all_matches_real(shim_name: str, real_name: str, extras: tuple[str, ...]) -> None:
    """``__all__`` 必须与真实模块完全一致（不能多、不能少、顺序也要一致）。"""
    shim = importlib.import_module(shim_name)
    real = importlib.import_module(real_name)
    assert list(shim.__all__) == list(real.__all__)


@pytest.mark.parametrize(("shim_name", "real_name", "extras"), SHIMS)
def test_shim_reexports_identical_objects(
    shim_name: str, real_name: str, extras: tuple[str, ...]
) -> None:
    """逐个断言身份一致：`is` 而不是 `==`，避免出现两份定义。"""
    shim = importlib.import_module(shim_name)
    real = importlib.import_module(real_name)
    for name in [*real.__all__, *extras]:
        assert hasattr(shim, name), f"{shim_name} 漏了 {name}（会直接 ImportError）"
        assert getattr(shim, name) is getattr(real, name), (
            f"{shim_name}.{name} 与 {real_name}.{name} 不是同一个对象"
        )


def test_shim_does_not_reexport_mutable_module_state() -> None:
    """模块级可变状态（``_CONFIG``）故意不重导出。

    重导出拿到的是**导入那一刻的引用快照**：真实模块里 ``global _CONFIG`` 的
    重新绑定不会传播到外壳，读外壳只会永远看到 ``None`` —— 这种"看起来能用、
    实际是过期值"的接口比没有更危险。
    """
    import medscholar.config as shim

    assert not hasattr(shim, "_CONFIG")
    import medscholar.platform.config as real

    assert hasattr(real, "_CONFIG")
