"""进程内缓存：TTL、LRU 淘汰、命中率统计，以及**为什么只缓存这些**。

这些测试全部用**注入的假时钟**驱动过期，不出现真实 ``sleep``：
用真实时间测 TTL 会让测试变慢，而慢测试的下场一定是被跳过或被无脑重跑。
"""

from __future__ import annotations

import pytest

from medscholar.platform.cache import (
    TTLCache,
    cache_registry,
    cache_stats,
    cached,
    clear_all_caches,
)


class FakeClock:
    """可手动推进的单调时钟。"""

    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class TestTTLCache:
    def test_set_get_roundtrip(self):
        cache = TTLCache[str, int]("t", ttl=60, clock=FakeClock())
        cache.set("a", 1)
        assert cache.get("a") == 1

    def test_missing_key_returns_default(self):
        cache = TTLCache[str, int]("t", ttl=60, clock=FakeClock())
        assert cache.get("nope") is None
        assert cache.get("nope", 42) == 42

    def test_entry_expires_after_ttl(self):
        clock = FakeClock()
        cache = TTLCache[str, int]("t", ttl=10, clock=clock)
        cache.set("a", 1)
        clock.advance(9.9)
        assert cache.get("a") == 1, "还没到期就必须命中"
        clock.advance(0.2)
        assert cache.get("a") is None, "过期后必须失效（缓存永不过期 = 把旧值永久固化）"

    def test_none_ttl_never_expires(self):
        clock = FakeClock()
        cache = TTLCache[str, int]("t", ttl=None, clock=clock)
        cache.set("a", 1)
        clock.advance(10**9)
        assert cache.get("a") == 1

    def test_lru_eviction_respects_maxsize(self):
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock(), maxsize=2)
        cache.set("a", 1)
        cache.set("b", 2)
        cache.get("a")  # a 变成最近使用
        cache.set("c", 3)  # 淘汰 b
        assert cache.get("b") is None
        assert cache.get("a") == 1 and cache.get("c") == 3

    def test_eviction_counted(self):
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock(), maxsize=1)
        cache.set("a", 1)
        cache.set("b", 2)
        assert cache.stats.evictions == 1

    def test_hit_rate_is_measured(self):
        """没有命中率的缓存无法回答"它到底有没有用"。"""
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock())
        cache.set("a", 1)
        cache.get("a")
        cache.get("miss1")
        cache.get("miss2")
        assert cache.stats.hits == 1
        assert cache.stats.misses == 2
        assert cache.stats.hit_rate == pytest.approx(1 / 3)

    def test_expiration_counted_separately_from_miss(self):
        clock = FakeClock()
        cache = TTLCache[str, int]("t", ttl=1, clock=clock)
        cache.set("a", 1)
        clock.advance(2)
        cache.get("a")
        assert cache.stats.expirations == 1
        assert cache.stats.misses == 1

    def test_invalidate(self):
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock())
        cache.set("a", 1)
        assert cache.invalidate("a") is True
        assert cache.invalidate("a") is False
        assert cache.get("a") is None

    def test_invalidate_prefix(self):
        """导入新文献后要能按前缀清掉相关缓存，而不是等自然过期。"""
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock())
        cache.set("search:rTMS", 1)
        cache.set("search:depression", 2)
        cache.set("other", 3)
        assert cache.invalidate_prefix("search:") == 2
        assert cache.get("other") == 3

    def test_clear(self):
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock())
        cache.set("a", 1)
        cache.clear()
        assert len(cache) == 0
        assert cache.get("a") is None

    def test_contains_uses_ttl(self):
        clock = FakeClock()
        cache = TTLCache[str, int]("t", ttl=1, clock=clock)
        cache.set("a", 1)
        assert "a" in cache
        clock.advance(2)
        assert "a" not in cache

    def test_get_or_set_calls_factory_once(self):
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock())
        calls: list[int] = []

        def factory() -> int:
            calls.append(1)
            return 7

        assert cache.get_or_set("a", factory) == 7
        assert cache.get_or_set("a", factory) == 7
        assert len(calls) == 1, "第二次必须命中缓存"

    def test_get_or_set_does_not_cache_none(self):
        """``None`` 表示"这次没算出来"，缓存它会把一次失败固化成一个小时的空结果。"""
        cache = TTLCache[str, int]("t", ttl=None, clock=FakeClock())
        calls: list[int] = []

        def factory() -> None:
            calls.append(1)
            return None

        cache.get_or_set("a", factory)
        cache.get_or_set("a", factory)
        assert len(calls) == 2

    def test_to_dict_shape(self):
        cache = TTLCache[str, int]("t", ttl=30, clock=FakeClock(), maxsize=8)
        cache.set("a", 1)
        payload = cache.to_dict()
        assert payload["size"] == 1
        assert payload["ttl"] == 30
        assert payload["maxsize"] == 8
        assert "hit_rate" in payload


class TestRegistry:
    def setup_method(self):
        clear_all_caches()

    def test_same_name_returns_same_instance(self):
        first = cache_registry("demo", ttl=10)
        second = cache_registry("demo", ttl=10)
        assert first is second, "同名缓存必须是同一个实例，否则清理与统计都会漏"

    def test_stats_lists_registered_caches(self):
        cache = cache_registry("demo", ttl=10)
        cache.set("k", "v")
        cache.get("k")
        stats = cache_stats()
        assert "demo" in stats
        assert stats["demo"]["hits"] == 1

    def test_clear_all(self):
        cache_registry("a", ttl=10).set("k", 1)
        cache_registry("b", ttl=10).set("k", 1)
        clear_all_caches()
        assert len(cache_registry("a", ttl=10)) == 0


class TestCachedDecorator:
    def setup_method(self):
        clear_all_caches()

    def test_caches_by_explicit_key(self):
        calls: list[str] = []

        @cached("demo-decorator", key=lambda text: text, ttl=None)
        def expensive(text: str) -> str:
            calls.append(text)
            return f"结果:{text}"

        assert expensive("rTMS") == "结果:rTMS"
        assert expensive("rTMS") == "结果:rTMS"
        assert expensive("抑郁") == "结果:抑郁"
        assert calls == ["rTMS", "抑郁"], "同 key 只应计算一次"

    def test_key_function_sees_all_arguments(self):
        seen: list[tuple] = []

        @cached("demo-key", key=lambda a, *, flag=False: (a, flag), ttl=None)
        def fn(a: int, *, flag: bool = False) -> str:
            seen.append((a, flag))
            return f"{a}-{flag}"

        fn(1, flag=True)
        fn(1, flag=False)
        fn(1, flag=True)
        assert seen == [(1, True), (1, False)], "不同关键字参数必须分开缓存"

    def test_wrapper_keeps_name_and_doc(self):
        @cached("demo-meta", key=lambda: "k", ttl=None)
        def documented() -> int:
            """文档字符串在包装后仍要可见。"""
            return 1

        assert documented.__name__ == "documented"
        assert documented.__doc__ == "文档字符串在包装后仍要可见。"
