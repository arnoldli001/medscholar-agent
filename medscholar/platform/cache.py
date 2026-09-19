"""进程内缓存：TTL + 容量上限 + 命中率统计。

## 为什么自己写而不是用 functools.lru_cache

``lru_cache`` 缺三样在生产里必须有的东西：

1. **TTL**：缓存永不过期就等于把错误的旧值永久固化。检索类缓存尤其如此 ——
   用户刚导入一篇新文献，如果查询向量或检索结果被永久缓存，他会看到"我明明加了文献却搜不到"。
2. **显式的失效钩子**：导入/删除文献后必须能按前缀清掉相关缓存，而不是等它自然过期。
3. **可观测性**：没有命中率就无法回答"这个缓存到底有没有用"。
   一个命中率 0.02 的缓存只是在消耗内存和引入 bug 面。

## 缓存什么、不缓存什么（重要取舍）

* **缓存**：查询向量（同样的检索词反复出现，且嵌入是一次网络/模型调用）、
  数据集统计这类"纯函数且昂贵"的结果。
* **不缓存**：LLM 生成结果。表面上很诱人，实际上有三个问题 ——
  ①用户期望"重新生成"能得到不同结果；②按温度采样本来就非确定；
  ③缓存命中会让"用哪次的结果"变得不可追溯，而本项目的核心卖点是**可审计**。
  宁可慢一点，也不要让"这段文字是从哪来的"说不清。

## 线程安全

用 ``threading.Lock`` 而不是 ``asyncio.Lock``：缓存会在同步与异步两条路径上被用到
（同步的 CLI、异步的 HTTP），asyncio 锁会把同步调用点逼成 await，
而且**漏一个 await 就是静默失效**。锁内只做纯内存操作，不做 IO、不 await。
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from typing import Any, Callable, Generic, Iterator, TypeVar

__all__ = ["CacheStats", "TTLCache", "cache_registry", "cached", "clear_all_caches"]

K = TypeVar("K")
V = TypeVar("V")


@dataclass
class CacheStats:
    """缓存命中统计。``hit_rate`` 是唯一值得盯的数字。"""

    hits: int = 0
    misses: int = 0
    evictions: int = 0
    expirations: int = 0
    sets: int = 0

    @property
    def lookups(self) -> int:
        return self.hits + self.misses

    @property
    def hit_rate(self) -> float:
        return self.hits / self.lookups if self.lookups else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "hits": self.hits,
            "misses": self.misses,
            "evictions": self.evictions,
            "expirations": self.expirations,
            "sets": self.sets,
            "lookups": self.lookups,
            "hit_rate": round(self.hit_rate, 4),
            "size": None,  # 由 TTLCache.to_dict 填充
        }


class TTLCache(Generic[K, V]):
    """带 TTL 与 LRU 淘汰的进程内缓存。

    Args:
        name: 缓存名（出现在统计与失效日志里）。
        ttl: 生存时间（秒）。``None`` 表示不过期（仅用于"手动失效"的纯函数缓存）。
        maxsize: 最大条目数，超出按 LRU 淘汰。
        clock: 时间源，可注入以便测试瞬时过期（**不要用真实 sleep 测 TTL**：
            那会让测试变慢，慢测试的下场是被跳过）。
    """

    def __init__(
        self,
        name: str,
        *,
        ttl: float | None = 300.0,
        maxsize: int = 512,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.ttl = ttl
        self.maxsize = max(1, maxsize)
        self._clock = clock
        self._data: OrderedDict[K, tuple[float, V]] = OrderedDict()
        self._lock = threading.Lock()
        self.stats = CacheStats()

    # ---------------------------------------------------------------- 基本操作
    def get(self, key: K, default: V | None = None) -> V | None:
        with self._lock:
            item = self._data.get(key)
            if item is None:
                self.stats.misses += 1
                return default
            expires_at, value = item
            if self.ttl is not None and expires_at <= self._clock():
                del self._data[key]
                self.stats.expirations += 1
                self.stats.misses += 1
                return default
            self._data.move_to_end(key)  # LRU：命中即刷新新鲜度
            self.stats.hits += 1
            return value

    def set(self, key: K, value: V) -> None:
        with self._lock:
            expires_at = float("inf") if self.ttl is None else self._clock() + self.ttl
            if key in self._data:
                self._data.move_to_end(key)
            self._data[key] = (expires_at, value)
            self.stats.sets += 1
            while len(self._data) > self.maxsize:
                self._data.popitem(last=False)
                self.stats.evictions += 1

    def get_or_set(self, key: K, factory: Callable[[], V]) -> V:
        """命中就返回；未命中才调用 ``factory``。

        注意：``factory`` 在锁**外**执行 —— 它可能是网络调用或模型推理，
        持锁执行会把整个缓存变成串行瓶颈（这是缓存实现里最常见的性能事故）。
        代价是并发未命中时可能重复计算一次，这个取舍是刻意的：
        重复算一次的成本，远小于所有请求排队等一个慢调用。
        """
        hit = self.get(key)
        if hit is not None:
            return hit
        value = factory()
        if value is not None:
            self.set(key, value)
        return value

    def invalidate(self, key: K) -> bool:
        with self._lock:
            return self._data.pop(key, None) is not None

    def invalidate_prefix(self, prefix: str) -> int:
        """按 key 前缀失效（例如导入文献后清掉所有 ``search:`` 开头的缓存）。"""
        with self._lock:
            doomed = [k for k in self._data if isinstance(k, str) and k.startswith(prefix)]
            for key in doomed:
                del self._data[key]
            return len(doomed)

    def clear(self) -> None:
        with self._lock:
            self._data.clear()

    def __len__(self) -> int:
        with self._lock:
            return len(self._data)

    def __contains__(self, key: object) -> bool:
        return self.get(key) is not None  # type: ignore[arg-type]

    def __iter__(self) -> Iterator[K]:
        with self._lock:
            return iter(list(self._data.keys()))

    def to_dict(self) -> dict[str, Any]:
        payload = self.stats.to_dict()
        payload["size"] = len(self)
        payload["ttl"] = self.ttl
        payload["maxsize"] = self.maxsize
        return payload


# ---------------------------------------------------------------------- 注册表
_CACHES: dict[str, TTLCache[Any, Any]] = {}
_REGISTRY_LOCK = threading.Lock()


def cache_registry(
    name: str, *, ttl: float | None = 300.0, maxsize: int = 512
) -> TTLCache[Any, Any]:
    """按名字取（或建）一个全局缓存。

    用注册表而不是让每个模块自己 new 一个：这样 ``clear_all_caches()``
    与 ``/api/metrics`` 能一次看全、一次清干净。
    同名缓存必须参数一致，否则会拿到先注册的那个（调用点分散时这是最容易踩的坑，
    所以这里用「先到先得 + 文档写明」而不是静默覆盖）。
    """
    with _REGISTRY_LOCK:
        cache = _CACHES.get(name)
        if cache is None:
            cache = TTLCache(name, ttl=ttl, maxsize=maxsize)
            _CACHES[name] = cache
        return cache


def cached(
    name: str,
    key: Callable[..., Any],
    *,
    ttl: float | None = 300.0,
    maxsize: int = 512,
) -> Callable[[Callable[..., V]], Callable[..., V]]:
    """把函数结果按 ``key(*args, **kwargs)`` 缓存起来的装饰器。

    ``key`` 必须是**纯函数**，且返回可哈希的值。显式要求调用方给出 key 函数，
    而不是自动用参数元组：很多参数（数据库连接、配置对象）不可哈希或语义上不该参与 key，
    自动推导会在运行时抛 `unhashable type`，或者更糟 —— 生成一个永远不命中的 key
    让缓存静默失效。
    """

    def decorator(func: Callable[..., V]) -> Callable[..., V]:
        cache = cache_registry(name, ttl=ttl, maxsize=maxsize)

        def wrapper(*args: Any, **kwargs: Any) -> V:
            cache_key = key(*args, **kwargs)
            return cache.get_or_set(cache_key, lambda: func(*args, **kwargs))

        wrapper.__name__ = getattr(func, "__name__", "wrapped")
        wrapper.__doc__ = func.__doc__
        wrapper.cache = cache  # type: ignore[attr-defined]
        return wrapper

    return decorator


def cache_stats() -> dict[str, dict[str, Any]]:
    """所有缓存的统计（供 /api/metrics）。"""
    with _REGISTRY_LOCK:
        return {name: cache.to_dict() for name, cache in _CACHES.items()}


def clear_all_caches() -> None:
    """清空所有缓存（配置变更、批量导入后调用）。"""
    with _REGISTRY_LOCK:
        for cache in _CACHES.values():
            cache.clear()
