"""韧性原语：指数退避重试、熔断、令牌桶限流、并发舱壁。

这一层是**最底层**（见 :mod:`medscholar.platform` 的依赖规则），只依赖标准库，
不导入 ``medscholar`` 的任何其他模块，因此任何层都可以安全复用它。

四条真实故障驱动了这里的四个原语：

1. **指数退避 + 全抖动**：9 个外部学术 API 里总有那么一两个在改版、限流或抖动。
   固定间隔重试会把多个并发调用方的重试**对齐**到同一时刻，形成重试风暴，
   把本来只是短暂 503 的数据源彻底按下去。
2. **异步重试**：一次检索要打 3~5 个请求，任何一次瞬时失败都不该让整条检索式失败，
   但 401/403 这类认证失败重试一万次也不会变好，只会白等。
3. **熔断**：项目里 ``_MAX_CONCURRENT_QUERIES = 3``，一个挂掉的数据源会让每次检索
   都等满 ``timeout``（默认 20~30s）。3 路并发里 1 路被一个死源吃住，
   整体吞吐直接掉 1/3；连等三轮就是分钟级卡顿。熔断让后续请求**立刻失败**，
   上层按"数据源不可用"降级，而不是陪着一起等。
4. **并发舱壁**：限制同时在飞的请求数，避免单个慢依赖把事件循环里的并发预算吃光。

**所有时间与休眠都可注入**（``clock`` / ``sleep`` / ``rand``）。这不是洁癖：
测试里断言"并发"和"退避"如果依赖墙钟和真实 ``sleep``，就只能靠调大等待时间来
降低偶发失败，结果是测试变慢且红灯不可信 —— 那等于训练人忽略红灯。
注入之后，测试用假时钟推进、用假 sleep 记录，既确定又瞬时。
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
    """既是浮点数、又能当方法调用的兼容返回值。

    只用于 :attr:`TokenBucket.tokens`：需求里写的是 ``tokens()``，
    但作为"当前额度"的自然读法又是属性。做成可调用浮点后
    ``bucket.tokens`` 与 ``bucket.tokens()`` 两种写法都返回同一个数值，
    集成方不必为一次签名歧义改代码（``float`` 的子类在序列化、
    比较、格式化上与普通浮点完全一致）。
    """

    __slots__ = ()

    def __call__(self) -> float:
        return float(self)

#: 指数上限。``attempt`` 由调用方传入，万一上游传了一个很大的值（例如把毫秒当次数），
#: ``2 ** attempt`` 会先炸成天文数字再被截断 —— 这里先按位移封顶，
#: 保证在 ``max_delay`` 截断之前不会出现溢出或长整数运算。
_MAX_SHIFT = 32


def expo_delay(
    attempt: int,
    base_delay: float = 0.5,
    max_delay: float = 8.0,
    jitter: bool = True,
    rand: Callable[[], float] = random.random,
) -> float:
    """第 ``attempt`` 次重试前应等待的秒数（``attempt`` 从 0 开始）。

    退避序列为 ``base_delay * 2**attempt``，并被 ``max_delay`` 截断。

    ``jitter=True`` 时使用 **全抖动**（full jitter）：``rand() * capped``，
    而不是"固定值 ± 一个小区间"。原因是多个客户端/协程一旦在相近时刻失败，
    固定抖动只把重试时刻打散了一个小窗口，下一轮它们仍然会成簇到达；
    全抖动把等待时间摊到 ``[0, capped]`` 全域，重试被彻底去同步化 ——
    这是 AWS 架构博客给出的实测结论，也是对付 API 限流最关键的一点。

    指数与截断只在抖动**之前**发生：先算上限再做随机，才能保证
    ``0 <= 返回值 <= max_delay`` 这个契约（先抖动再截断也能满足，但会让
    抖动分布在高位被压成一堆 ``max_delay``，退化成同步重试）。
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
    """重试 ``fn`` 直到成功、耗尽次数或触发放弃条件。

    * ``give_up_on`` **优先于** ``retry_on``：认证失败（401/403）、参数错误、
      以及 :class:`asyncio.CancelledError` 重试多少次都不会变好，
      立刻抛出比等 3 轮退避后拿到同一个错误更有价值。
    * ``CancelledError`` 无论是否写进 ``give_up_on`` 都**原样向上抛**：
      它是取消语义而不是失败。用户点"取消运行"时必须马上停 ——
      如果这里把它当普通异常吞掉重试，界面上的取消按钮就会看起来失灵，
      而这正是本模块最不能出的 bug（取消语义一旦被吞，整个停止链路失效）。
    * ``on_retry(attempt, exc, delay)`` 在每次**将要**休眠前回调，
      ``attempt`` 是刚失败的那一次（从 0 开始），便于把重试写进 trace/日志。
    * 重试耗尽后抛出**最后一个**原始异常，不包自定义异常：
      定位问题靠的是原始栈和错误类型，包装一层只会让上层的
      ``except RateLimited`` 之类的分类逻辑全部失效。
    * ``total_deadline`` 是总耗时预算（秒）。耗尽预算时不再休眠、立刻抛出最后异常 ——
      宁可把失败早点交给上层降级，也不要为了"再试一次"拖过整个检索的 SLA。
    * ``attempts <= 0`` 按 1 次处理：调用方传 0 的意思通常是"别重试"，
      而不是"什么都别做"。

    返回 ``fn`` 的返回值；失败时抛出 ``fn`` 抛出的最后一个异常。
    """
    total = max(1, int(attempts))
    clock = time.monotonic
    started = clock()
    last_error: BaseException | None = None

    for attempt in range(total):
        try:
            return await fn()
        except asyncio.CancelledError:
            # 取消不是失败：原样抛出，不走 retry_on / give_up_on 的判断。
            raise
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
                # 剩余预算撑不到下一次重试：现在失败，让上层决定是降级还是放弃。
                break
            if on_retry is not None:
                on_retry(attempt, exc, delay)
            if delay > 0:
                await sleep(delay)

    assert last_error is not None  # attempts >= 1 且走到这里必然至少失败过一次
    raise last_error


class CircuitState(str, Enum):
    """熔断器状态。

    继承 ``str`` 是为了能直接进 JSON（状态接口/日志不需要额外转换）。
    """

    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """熔断打开时抛出。带上 ``name`` 与剩余冷却时间，便于上层降级。

    上层典型用法：捕获后把该数据源标记为临时不可用，改写检索计划，
    而不是让整条检索式跟着一起超时。
    """

    def __init__(self, name: str, retry_after: float) -> None:
        self.name = name
        self.retry_after = max(0.0, float(retry_after))
        super().__init__(f"[{name}] 熔断已打开，约 {self.retry_after:.1f}s 后允许探测")


class CircuitBreaker:
    """按数据源计数的熔断器（CLOSED / OPEN / HALF_OPEN）。

    为什么必须有它：一个挂掉的外部 API 会让每次检索都等满超时。
    项目里 ``_MAX_CONCURRENT_QUERIES = 3``，3 路并发检索里只要有 1 路被死源吃住，
    整体吞吐就掉 1/3；连续几轮就是分钟级卡顿，而且用户完全看不出是哪一源的问题。
    熔断把"等 30s 再失败"变成"立刻失败"，让上层有机会换源、降级或提示。

    状态迁移：

    * ``CLOSED`` —— 正常放行；连续失败达到 ``failure_threshold`` 转 ``OPEN``；
      中间任何一次成功都把失败计数清零（"连续"失败，不是累计失败）。
    * ``OPEN`` —— 冷却期内一律拒绝（``allow() -> False``）；
      超过 ``reset_timeout`` 后由**第一次** ``allow()`` 转入 ``HALF_OPEN``。
    * ``HALF_OPEN`` —— 只放行 ``half_open_max_calls`` 个探测请求，
      其余仍然拒绝（避免恢复瞬间把积压流量一次性灌进去，把刚缓过来的源再打死）；
      探测成功 → ``CLOSED`` 并清零；探测失败 → 重新 ``OPEN`` 并**重置冷却计时**。

    **线程安全**：同一个 client 可能被多个线程/事件循环用到（例如检索线程池
    与后台预热任务共用注册表里的客户端），所有状态读写都在 ``threading.Lock`` 下完成。
    锁只保护纯计算，绝不跨越 ``await``，所以不会阻塞事件循环。

    时间来自注入的 ``clock``（默认单调时钟）：单调时钟不受系统时间调整影响，
    而且注入后测试可以用假时钟精确推进冷却，不需要真的等 30 秒。
    """

    def __init__(
        self,
        name: str,
        *,
        failure_threshold: int = 5,
        reset_timeout: float = 30.0,
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
        """失败：累加连续失败，达到阈值则转 ``OPEN`` 并记录打开时刻。

        已经是 ``OPEN`` 时**不重复计数、不刷新冷却时间**：
        打开瞬间可能还有若干在飞请求陆续失败，如果让它们重置计时，
        一个持续挂掉的源会被无限期地"续杯"，永远进不了 ``HALF_OPEN`` 探测。
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
        块内抛出的异常计入失败并**原样向上抛**，不吞不改。
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
        # 只判定不等待：熔断的意义就是不要被一个死依赖拖住。
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
        """把异常记账为失败；``CancelledError`` 与熔断拒绝本身都**不算**失败。

        * ``CancelledError`` 是取消语义（用户点了停止），不是依赖故障；
        * :class:`CircuitOpenError` 是熔断器自己抛的，把它再记一次会让
          "被拒绝"看起来像"依赖又坏了"，失败计数被自己的拒绝刷爆。
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

    为什么不用固定窗口计数：一次检索要打 3~5 个请求，固定窗口要么把这一批
    卡在窗口边界（等下一个窗口，白白空转），要么在窗口刚翻转时连放两批
    （瞬时 2 倍速率触发 429）。令牌桶同时满足两件事：

    * **允许突发** —— 桶里攒下的额度可以一次性用掉，正好对应"一次检索打 3 个请求"；
    * **约束长期平均速率** —— 额度按 ``rate`` 每秒匀速回填，长期速率不会超过它。

    这正是学术 API 的限流语义（PubMed 3 req/s、OpenAlex 10 req/s 之类）。
    ``capacity`` 默认等于 ``rate``：最多攒 1 秒的额度，突发上限可控，
    又不会因为桶太大而在故障恢复瞬间灌出一大波流量。

    同步实现（``threading.Lock``）而不是 ``asyncio.Lock``：
    同步客户端、线程池和事件循环都可能用到同一个桶，
    而 :meth:`try_acquire` 是非阻塞的 —— 异步调用方请用 :meth:`acquire`，
    它内部只做假 sleep，不占锁。
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
        """当前可用令牌数（回填后的精确值）。

        这是一个会推进内部时间基准的**观察**，但不会消耗令牌 —— 供诊断面板显示。
        返回值可当浮点用，也可当方法调用（``bucket.tokens()``），见 :class:`_QuasiFloat`。
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
        """等待直到取到令牌（或超过 ``max_wait``）。

        返回 ``True`` 表示已拿到；``False`` 表示 ``max_wait`` 内取不到，
        **没有消耗任何令牌** —— 由调用方决定是降级（少查一个源）还是放弃。

        ``sleep`` 可注入，因此测试里不需要真的等：假 sleep 记录等待秒数并推进假时钟。
        循环而不是单次计算等待时间，是为了容忍并发调用方在同一时刻抢令牌
        （一次 sleep 醒来发现被别人抢走，就再等下一轮）。
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
    """限制同时在飞的请求数，避免一个慢依赖吃光所有并发。

    ``peak`` 记录历史并发峰值 ---- 这是**唯一**可用来断言"真的并发了"的指标。
    用墙钟耗时断言并发是负资产：机器负载一高就偶发失败，
    最后大家学会的是重跑测试而不是查问题（本项目已经在
    ``tests/test_scout_concurrency.py`` 里确立了"断言峰值而不是耗时"的做法）。

    **同步与异步都能用**：

    * 同步路径（:meth:`try_acquire` / :meth:`acquire`）用锁保护计数，
      供线程池里的 PDF 解析、SQLite 写入这类阻塞任务使用；
    * 异步路径（``async with``）额外挂一个 ``asyncio.Semaphore``，
      让等待者在事件循环里让出控制权，而不是占着线程。
      信号量在**首次异步使用**时惰性创建：``asyncio.Semaphore`` 会绑定当时的
      事件循环，在 ``__init__`` 里创建会让"构造于导入期、使用于运行期"的对象
      报废（项目里 HTTP 客户端跨事件循环复用导致请求永久挂起的同类坑）。
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
        """历史并发峰值（同时占用舱位的最大数量）。

        返回可调用浮点，因此 ``bulkhead.peak`` 与 ``bulkhead.peak()`` 都可用。
        """
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
        """同步上下文管理器：``with bulkhead.acquire(): ...``。

        已满时抛 :class:`RuntimeError`：同步路径没法"等"，
        静默放行等于舱壁失效（超额就是超额），静默跳过又会让调用方以为跑过了。
        需要降级语义的调用方请直接用 :meth:`try_acquire` 自行判断。
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
        # 计数在这里就 +1，等待阶段只占用信号量 —— 这样 stats() 里的
        # in_flight 始终是"真正在干活"的数量，与同步路径语义一致。
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
