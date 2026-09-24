"""韧性原语：指数退避重试、熔断、令牌桶限流、并发舱壁。

只依赖标准库，可被任意层复用。clock/sleep/rand 均可注入，测试不依赖墙钟。
"""

from __future__ import annotations

import asyncio
import random
import threading
import time
from contextlib import contextmanager
from enum import Enum
from typing import Any, Awaitable, Callable, Iterator, TypeVar

__all__ = [
    "expo_delay",
    "retry_async",
    "CircuitState",
    "CircuitOpenError",
    "CircuitBreaker",
    "TokenBucket",
    "Bulkhead",
]

T = TypeVar("T")


class _QuasiFloat(float):
    """可当方法调用的 float 子类：``bucket.tokens`` 与 ``bucket.tokens()`` 等价。"""

    __slots__ = ()

    def __call__(self) -> float:
        return float(self)

#: 指数位移上限：防止上游传入超大 attempt 时 2**attempt 溢出后才被 max_delay 截断。
_MAX_SHIFT = 32


def expo_delay(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: bool = True,
    rand: Callable[[], float] = random.random,
) -> float:
    """第 ``attempt`` 次重试前应等待的秒数（``attempt`` 从 0 开始）。

    退避序列 ``base_delay * 2**attempt`` 先被 ``max_delay`` 截断，再做全抖动
    （full jitter，``rand() * capped``）：全抖动把等待摊到 [0, capped] 全域，
    避免多个失败方的重试成簇到达触发限流。先截断后抖动，保证
    ``0 <= 返回值 <= max_delay``。
    """
    step = max(0, int(attempt))
    capped = min(float(max_delay), float(base_delay) * float(1 << min(step, _MAX_SHIFT)))
    if capped <= 0.0:
        # base_delay <= 0 是显式关闭退避（例如同步的本地调用），不算配置错误。
        return 0.0
    if not jitter:
        return capped
    # rand() 理论上属于 [0, 1)，这里再夹一次，防止被传入的假 rand 越界。
    return min(capped, max(0.0, float(rand()) * capped))


async def retry_async(
    fn: Callable[[], Awaitable[T]],
    *,
    attempts: int = 3,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: bool = True,
    retry_on: tuple[type[BaseException], ...] = (Exception,),
    give_up_on: tuple[type[BaseException], ...] = (),
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    rand: Callable[[], float] = random.random,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
    total_deadline: float | None = None,
) -> T:
    """重试 ``fn`` 直到成功、耗尽次数或命中放弃条件。

    * ``give_up_on`` 优先于 ``retry_on``（如 401/403 立即抛出，不再退避）。
    * ``CancelledError`` 无论是否配置都原样上抛：取消语义不是失败，吞掉会让停止链路失灵。
    * ``on_retry(attempt, exc, delay)`` 在每次休眠前回调，``attempt`` 从 0 开始。
    * 耗尽后抛最后一个原始异常，不包装，避免打断上层 ``except RateLimited`` 之类的分类。
    * ``total_deadline`` 是总耗时预算，剩余预算撑不下一次退避就立刻抛出。
    * ``attempts <= 0`` 按 1 次处理。
    """
    total = max(1, int(attempts))
    clock = time.monotonic
    started = clock()
    last_error: BaseException | None = None

    for attempt in range(total):
        try:
            return await fn()
        except asyncio.CancelledError:
            raise  # 取消语义：不走任何重试判定
        except BaseException as exc:  # noqa: BLE001 - 由 retry_on/give_up_on 决定是否重试
            last_error = exc
            if give_up_on and isinstance(exc, give_up_on):
                raise
            if not isinstance(exc, retry_on):
                raise
            if attempt + 1 >= total:
                break
            delay = expo_delay(
                attempt, base_delay=base_delay, max_delay=max_delay, jitter=jitter, rand=rand
            )
            if total_deadline is not None and (clock() - started) + delay > total_deadline:
                break  # 预算不够再退避一次：立刻失败，交上层降级
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay > 0:
                await sleep(delay)

    assert last_error is not None
    raise last_error


class CircuitState(str, Enum):
    """熔断器状态；继承 str 可直接进 JSON。"""

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """熔断打开时抛出，携带 ``name`` 与剩余冷却时间，供上层换源/降级。"""

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = max(0.0, float(retry_after))
        super().__init__(f"[{name}] 熔断已打开，约 {self.retry_after:.1f}s 后允许探测")


#: 熔断默认参数：连续失败 5 次打开，冷却 30 秒后半开探测
CIRCUIT_DEFAULT_FAILURE_THRESHOLD = 5
CIRCUIT_DEFAULT_RESET_SECONDS = 30.0


class CircuitBreaker:
    """按数据源计数的熔断器（CLOSED / OPEN / HALF_OPEN）。

    状态迁移：CLOSED 连续失败达 ``failure_threshold`` → OPEN（任一成功即清零，
    按"连续"而非累计计数）；OPEN 冷却期内一律拒绝，到期后首个 ``allow()``
    转 HALF_OPEN，仅放行 ``half_open_max_calls`` 个探测，避免恢复瞬间灌入积压流量；
    探测成功 → CLOSED 清零，失败 → 回 OPEN 并重置冷却。OPEN 态下在飞请求的失败
    不刷新冷却，否则死源会被无限"续杯"，永远无法探测。

    所有状态读写都在 threading.Lock 下，锁只护纯计算、不跨 await；
    ``clock`` 默认单调时钟且可注入，便于假时钟推进冷却测试。
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = CIRCUIT_DEFAULT_FAILURE_THRESHOLD,
        reset_timeout: float = CIRCUIT_DEFAULT_RESET_SECONDS,
        half_open_max_calls: int = 1,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.name = name
        self.failure_threshold = max(1, int(failure_threshold))
        self.reset_timeout = max(0.0, float(reset_timeout))
        self.half_open_max_calls = max(1, int(half_open_max_calls))
        self._clock = clock
        self._lock = threading.Lock()
        self._state = CircuitState.CLOSED
        self._failures = 0
        self._opened_at: float | None = None
        self._trips = 0
        self._half_open_calls = 0

    # -------------------------------------------------------------- 只读视图
    @property
    def state(self) -> CircuitState:
        """当前状态（只读；状态迁移只发生在 :meth:`allow` 与两个 ``on_*`` 里）。"""
        with self._lock:
            return self._state

    def stats(self) -> dict[str, Any]:
        """给状态接口/诊断面板用的快照（全部是可序列化的标量）。"""
        with self._lock:
            return {
                "name": self.name,
                "state": self._state.value,
                "failures": self._failures,
                "failure_threshold": self.failure_threshold,
                "reset_timeout": self.reset_timeout,
                "half_open_max_calls": self.half_open_max_calls,
                "half_open_calls": self._half_open_calls,
                "opened_at": self._opened_at,
                "trips": self._trips,
                "cooldown_remaining": self._remaining_locked(),
            }

    # ---------------------------------------------------------------- 放行判定
    def allow(self) -> bool:
        """是否放行一次调用。

        ``OPEN`` 且仍在冷却期内返回 ``False``；冷却到期后转入 ``HALF_OPEN``
        并放行最多 ``half_open_max_calls`` 个探测请求，其余返回 ``False``。
        """
        with self._lock:
            if self._state is CircuitState.CLOSED:
                return True
            if self._state is CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            # OPEN：只有冷却到期后的第一个调用者负责发起探测。
            if self._remaining_locked() > 0.0:
                return False
            self._state = CircuitState.HALF_OPEN
            self._half_open_calls = 1
            return True

    @property
    def retry_after(self) -> float:
        """距冷却结束还有多少秒（未打开时为 0）。"""
        with self._lock:
            return self._remaining_locked()

    # ------------------------------------------------------------ 结果回传
    def on_success(self) -> None:
        """成功：``CLOSED`` 下清零连续失败；``HALF_OPEN`` 下判定恢复并回到 ``CLOSED``。"""
        with self._lock:
            self._failures = 0
            self._state = CircuitState.CLOSED
            self._opened_at = None
            self._half_open_calls = 0

    def on_failure(self) -> None:
        """累加连续失败，达阈值转 OPEN。

        OPEN 态不重复计数、不刷新冷却：否则打开瞬间在飞请求的持续失败会
        不断重置计时，熔断器永远进不了 HALF_OPEN 探测。
        """
        with self._lock:
            if self._state is CircuitState.OPEN:
                return
            self._failures += 1
            if self._state is CircuitState.HALF_OPEN or self._failures >= self.failure_threshold:
                self._trip_locked()

    def reset(self) -> None:
        """人工复位（例如运维确认数据源已恢复，或配置热更新后重置统计）。"""
        with self._lock:
            self._state = CircuitState.CLOSED
            self._failures = 0
            self._opened_at = None
            self._half_open_calls = 0

    # ---------------------------------------------------------------- 上下文
    @contextmanager
    def guard(self) -> Iterator[None]:
        """同步用法：``with breaker.guard(): ...``。

        被拒绝时抛出 :class:`CircuitOpenError`（携带剩余冷却时间）；
        块内抛出的异常计入失败并原样向上抛，不吞不改。
        """
        if not self.allow():
            raise CircuitOpenError(self.name, self.retry_after)
        try:
            yield
        except BaseException as exc:
            self._record_exception(exc)
            raise
        else:
            self.on_success()

    async def __aenter__(self) -> None:
        if not self.allow():
            raise CircuitOpenError(self.name, self.retry_after)

    async def __aexit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        if exc is None:
            self.on_success()
        else:
            self._record_exception(exc)
        return False  # 异常继续向上抛，熔断器不改变业务异常语义

    # ---------------------------------------------------------------- 内部
    def _record_exception(self, exc: BaseException) -> None:
        """把异常记账为失败；``CancelledError`` 与 ``CircuitOpenError`` 都不算失败。

        前者是取消语义而非依赖故障；后者是熔断器自己的拒绝，再记一次会让
        失败计数被自己的拒绝刷爆。
        """
        if isinstance(exc, (asyncio.CancelledError, CircuitOpenError)):
            return
        self.on_failure()

    def _remaining_locked(self) -> float:
        """剩余冷却秒数。必须在持锁状态下调用。"""
        if self._state is not CircuitState.OPEN or self._opened_at is None:
            return 0.0
        return max(0.0, self._opened_at + self.reset_timeout - self._clock())

    def _trip_locked(self) -> None:
        """转 ``OPEN`` 并记录打开时刻。必须在持锁状态下调用。"""
        self._state = CircuitState.OPEN
        self._opened_at = self._clock()
        self._trips += 1
        self._half_open_calls = 0


class TokenBucket:
    """令牌桶限流器（同步，线程安全）。

    桶里攒下的额度允许一次性突发用掉（对应一次检索打 3~5 个请求），
    同时额度按 ``rate`` 匀速回填，约束长期平均速率不超过它（PubMed/OpenAlex
    等学术 API 的限流语义）。``capacity`` 默认等于 ``rate``，即最多攒 1 秒额度，
    避免故障恢复瞬间灌出大波流量。

    用 threading.Lock 而非 asyncio.Lock：同步客户端、线程池与事件循环可能
    共用同一个桶；:meth:`try_acquire` 非阻塞，异步等待走 :meth:`acquire`。
    """

    def __init__(
        self,
        rate: float,
        capacity: float | None = None,
        *,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate = max(1e-6, float(rate))
        self.capacity = float(capacity) if capacity is not None else self.rate
        if self.capacity <= 0.0:
            self.capacity = self.rate
        self._clock = clock
        self._lock = threading.Lock()
        self._tokens = self.capacity
        self._updated = self._clock()

    def _refill_locked(self) -> None:
        """按经过的时间回填令牌。必须在持锁状态下调用。"""
        now = self._clock()
        elapsed = now - self._updated
        if elapsed <= 0.0:
            # 时钟回退（假时钟或单调时钟异常）时只更新时间基准，不倒扣令牌。
            self._updated = now
            return
        self._tokens = min(self.capacity, self._tokens + elapsed * self.rate)
        self._updated = now

    def try_acquire(self, tokens: float = 1.0) -> bool:
        """非阻塞取令牌：够则扣减并返回 ``True``，不够立即返回 ``False``。

        调用方拿到 ``False`` 时应自行决定是等待、降级还是放弃数据源。
        """
        want = max(0.0, float(tokens))
        with self._lock:
            self._refill_locked()
            if self._tokens < want:
                return False
            self._tokens -= want
            return True

    @property
    def tokens(self) -> _QuasiFloat:
        """当前可用令牌数（会触发回填但不消耗令牌，供诊断面板）。

        返回 :class:`_QuasiFloat`，属性与 ``tokens()`` 调用等价。
        """
        with self._lock:
            self._refill_locked()
            return _QuasiFloat(self._tokens)

    async def acquire(
        self,
        tokens: float = 1.0,
        *,
        sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
        max_wait: float | None = None,
    ) -> bool:
        """等待直到取到令牌，或超过 ``max_wait`` 返回 ``False``（不消耗令牌）。

        循环等待而非单次计时：一次 sleep 醒来可能发现令牌被并发调用方抢走，需再等。
        ``sleep`` 可注入，测试用假 sleep + 假时钟即可瞬时完成。
        """
        want = max(0.0, float(tokens))
        waited = 0.0
        while True:
            with self._lock:
                self._refill_locked()
                if self._tokens >= want:
                    self._tokens -= want
                    return True
                # 缺多少就等多久：容量默认 <= rate，正常情况下一次回填就够。
                delay = (want - self._tokens) / self.rate
            if max_wait is not None and waited + delay > max_wait:
                return False
            if delay > 0:
                await sleep(delay)
                waited += delay


class Bulkhead:
    """并发舱壁：限制同时在飞的请求数，避免单个慢依赖吃光并发预算。

    ``peak`` 记录历史并发峰值，是断言"真的并发了"的可靠指标；
    不要用墙钟耗时断言并发（机器负载一变就偶发失败）。
    同步路径（:meth:`try_acquire` / :meth:`acquire`）用锁计数，供线程池里的
    阻塞任务使用；异步路径（``async with``）额外挂 ``asyncio.Semaphore``，
    让等待者在事件循环里让出控制权。信号量惰性创建：asyncio.Semaphore 会绑定
    创建时的事件循环，在 ``__init__``（可能处于导入期）构造会导致跨循环复用挂死。
    """

    def __init__(self, name: str, limit: int) -> None:
        self.name = name
        self.limit = max(1, int(limit))
        self._lock = threading.Lock()
        self._in_flight = 0
        self._peak = 0
        self._rejected = 0
        self._sem: asyncio.Semaphore | None = None

    # ---------------------------------------------------------------- 只读视图
    @property
    def peak(self) -> _QuasiFloat:
        """历史并发峰值；``peak`` 与 ``peak()`` 等价（见 :class:`_QuasiFloat`）。"""
        with self._lock:
            return _QuasiFloat(self._peak)

    def stats(self) -> dict[str, Any]:
        with self._lock:
            return {
                "name": self.name,
                "limit": self.limit,
                "in_flight": self._in_flight,
                "peak": self._peak,
                "rejected": self._rejected,
            }

    # ---------------------------------------------------------------- 同步路径
    def try_acquire(self) -> bool:
        """非阻塞占用一个舱位；已满返回 ``False`` 并计入 ``rejected``。"""
        with self._lock:
            if self._in_flight >= self.limit:
                self._rejected += 1
                return False
            self._in_flight += 1
            self._peak = max(self._peak, self._in_flight)
            return True

    def release(self) -> None:
        """释放一个舱位（与 :meth:`try_acquire` 或异步入口成对使用）。"""
        with self._lock:
            self._in_flight = max(0, self._in_flight - 1)

    def force_release(self) -> None:
        """把计数清零（仅用于故障恢复后的强行复位；正常路径请用 :meth:`release`）。"""
        with self._lock:
            self._in_flight = 0

    @contextmanager
    def acquire(self) -> Iterator[None]:
        """同步上下文管理器；舱位已满抛 :class:`RuntimeError`（同步路径不排队）。

        不静默放行（等于舱壁失效）也不静默跳过（调用方会误以为执行过）；
        需要降级语义请直接用 :meth:`try_acquire`。
        """
        if not self.try_acquire():
            raise RuntimeError(
                f"[{self.name}] 并发舱壁已满（limit={self.limit}），同步路径不排队"
            )
        try:
            yield
        finally:
            self.release()

    # ---------------------------------------------------------------- 异步路径
    async def __aenter__(self) -> None:
        # 先计数再等信号量，保证 in_flight 统计的是真正在干活的数量
        if self._sem is None:
            self._sem = asyncio.Semaphore(self.limit)
        await self._sem.acquire()
        if not self.try_acquire():
            # 理论上不会发生（信号量额度与 limit 同源），真发生了也绝不能泄漏信号量。
            self._sem.release()
            raise RuntimeError(f"[{self.name}] 舱位计数与信号量不一致")

    async def __aexit__(self, exc_type: object, exc: BaseException | None, tb: object) -> bool:
        self.release()
        if self._sem is not None:
            self._sem.release()
        return False
