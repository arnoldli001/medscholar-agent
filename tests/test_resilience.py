"""``medscholar.platform.resilience`` 的确定性测试。

**这里不出现任何真实等待**：``time`` / ``asyncio`` 的休眠全部被替换成
"记录秒数 + 立刻返回 + 推进假时钟"的假实现，熔断器和令牌桶的 ``clock``
也换成可以按需推进的计数器。理由和 ``tests/test_scout_concurrency.py`` 一样：
用墙钟去断言退避与并发是负资产 —— 机器一忙就偶发失败，
最后大家学会的是重跑测试而不是排查问题。注入之后每条断言都是精确值。

断言并发度一律用 :attr:`Bulkhead.peak`，不看耗时。
"""

from __future__ import annotations

import asyncio
import threading
from unittest import mock

import pytest

from medscholar.platform.resilience import (
    Bulkhead,
    CircuitBreaker,
    CircuitOpenError,
    CircuitState,
    TokenBucket,
    expo_delay,
    retry_async,
)


class FakeClock:
    """可推进的假单调时钟。用单元素列表当可变计数器，省掉 nonlocal 样板。"""

    def __init__(self, start: float = 1000.0) -> None:
        self._now = [float(start)]

    def __call__(self) -> float:
        return self._now[0]

    def advance(self, seconds: float) -> float:
        self._now[0] += float(seconds)
        return self._now[0]

    def reset(self, start: float = 0.0) -> None:
        """把时钟拨到指定时刻（默认为 0）。"""
        self._now[0] = float(start)


class FakeSleep:
    """假休眠：记录每次被 sleep 的秒数，并同步推进假时钟，立即返回。"""

    def __init__(self, clock: FakeClock | None = None) -> None:
        self.calls: list[float] = []
        self._clock = clock

    async def __call__(self, seconds: float) -> None:
        self.calls.append(seconds)
        if self._clock is not None:
            self._clock.advance(seconds)

    @property
    def total(self) -> float:
        return sum(self.calls)


class Failing:
    """按给定次数失败后成功的可调用对象，用来精确断言调用次数。"""

    def __init__(self, errors: list[BaseException], result: str = "ok") -> None:
        self.errors = list(errors)
        self.result = result
        self.calls = 0

    async def __call__(self) -> str:
        self.calls += 1
        if self.errors:
            raise self.errors.pop(0)
        return self.result


class Unauthorized(Exception):
    """模拟 401：认证失败，重试没有意义。"""


class Flaky(Exception):
    """模拟可恢复的瞬时故障。"""


def fail_times(n: int, exc_type: type[BaseException] = Flaky) -> Failing:
    return Failing([exc_type(f"第 {i + 1} 次失败") for i in range(n)])


# =============================================================== 指数退避
class TestExpoDelay:
    def test_grows_exponentially_without_jitter(self):
        assert expo_delay(0, base_delay=0.5, max_delay=8.0, jitter=False) == pytest.approx(0.5)
        assert expo_delay(1, base_delay=0.5, max_delay=8.0, jitter=False) == pytest.approx(1.0)
        assert expo_delay(2, base_delay=0.5, max_delay=8.0, jitter=False) == pytest.approx(2.0)
        assert expo_delay(3, base_delay=0.5, max_delay=8.0, jitter=False) == pytest.approx(4.0)

    def test_capped_at_max_delay(self):
        assert expo_delay(10, base_delay=0.5, max_delay=8.0, jitter=False) == pytest.approx(8.0)
        # 极端 attempt 也不能溢出成天文数字
        assert expo_delay(9999, base_delay=0.5, max_delay=8.0, jitter=False) == pytest.approx(8.0)

    def test_cap_applies_even_when_base_exceeds_max(self):
        assert expo_delay(0, base_delay=30.0, max_delay=8.0, jitter=False) == pytest.approx(8.0)

    def test_negative_attempt_treated_as_zero(self):
        assert expo_delay(-5, base_delay=0.5, jitter=False) == pytest.approx(0.5)

    def test_zero_base_means_no_backoff(self):
        assert expo_delay(3, base_delay=0.0, jitter=False) == 0.0

    def test_full_jitter_scales_the_whole_window(self):
        """全抖动 = rand() * capped：rand=0 时归零，rand=1 时取满上限。"""
        assert expo_delay(2, base_delay=0.5, max_delay=8.0, jitter=True, rand=lambda: 0.0) == 0.0
        assert expo_delay(2, base_delay=0.5, max_delay=8.0, jitter=True, rand=lambda: 1.0) == pytest.approx(
            2.0
        )
        assert expo_delay(
            2, base_delay=0.5, max_delay=8.0, jitter=True, rand=lambda: 0.25
        ) == pytest.approx(0.5)

    def test_jitter_still_respects_cap(self):
        value = expo_delay(20, base_delay=0.5, max_delay=8.0, jitter=True, rand=lambda: 1.0)
        assert value == pytest.approx(8.0)

    def test_fake_rand_desynchronizes_two_clients(self):
        """同一个 attempt 上，不同随机源给出不同等待 -> 重试不再同步。"""
        first = expo_delay(3, base_delay=0.5, jitter=True, rand=lambda: 0.1)
        second = expo_delay(3, base_delay=0.5, jitter=True, rand=lambda: 0.9)
        assert first != second


# ================================================================= 异步重试
class TestRetryAsync:
    async def test_success_on_first_attempt_does_not_sleep(self):
        fn = fail_times(0)
        sleeper = FakeSleep()
        assert await retry_async(fn, attempts=3, sleep=sleeper) == "ok"
        assert fn.calls == 1
        assert sleeper.calls == []

    async def test_succeeds_after_retries_and_reports_attempt_count(self):
        fn = fail_times(2)
        sleeper = FakeSleep()
        result = await retry_async(
            fn, attempts=3, base_delay=0.5, max_delay=8.0, jitter=False, sleep=sleeper
        )
        assert result == "ok"
        assert fn.calls == 3, "两次失败 + 一次成功"
        assert sleeper.calls == [pytest.approx(0.5), pytest.approx(1.0)]

    async def test_jitter_is_applied_to_sleep_delays(self):
        fn = fail_times(1)
        sleeper = FakeSleep()
        await retry_async(fn, attempts=2, base_delay=0.5, jitter=True, rand=lambda: 0.5, sleep=sleeper)
        assert sleeper.calls == [pytest.approx(0.25)]

    async def test_exhausted_retries_raise_last_exception(self):
        first = Flaky("先坏")
        last = Flaky("最后坏")
        fn = Failing([first, last])
        sleeper = FakeSleep()
        with pytest.raises(Flaky) as info:
            await retry_async(fn, attempts=2, jitter=False, sleep=sleeper)
        assert info.value is last, "必须抛最后一个异常本身，不能包一层自定义异常"
        assert fn.calls == 2, "attempts=2 只能调用两次"

    async def test_attempts_zero_still_runs_once(self):
        fn = fail_times(0)
        assert await retry_async(fn, attempts=0, sleep=FakeSleep()) == "ok"
        assert fn.calls == 1

    async def test_attempts_zero_failure_propagates_immediately(self):
        fn = fail_times(5)
        with pytest.raises(Flaky):
            await retry_async(fn, attempts=0, sleep=FakeSleep())
        assert fn.calls == 1

    async def test_give_up_on_is_not_retried(self):
        fn = fail_times(3, Unauthorized)
        sleeper = FakeSleep()
        with pytest.raises(Unauthorized):
            await retry_async(
                fn, attempts=3, give_up_on=(Unauthorized,), jitter=False, sleep=sleeper
            )
        assert fn.calls == 1, "认证失败绝不该重试"
        assert sleeper.calls == []

    async def test_give_up_on_wins_over_retry_on(self):
        fn = fail_times(3, Unauthorized)
        with pytest.raises(Unauthorized):
            await retry_async(
                fn,
                attempts=5,
                retry_on=(Exception,),
                give_up_on=(Unauthorized,),
                sleep=FakeSleep(),
            )
        assert fn.calls == 1, "give_up_on 必须优先于 retry_on"

    async def test_unlisted_exception_type_is_raised_immediately(self):
        fn = fail_times(3, Unauthorized)
        with pytest.raises(Unauthorized):
            await retry_async(fn, attempts=5, retry_on=(Flaky,), sleep=FakeSleep())
        assert fn.calls == 1

    async def test_cancelled_error_propagates_without_retry(self):
        """取消语义：立刻向外抛，绝不吞掉重试（否则"取消运行"按钮看起来失灵）。"""
        fn = fail_times(3, asyncio.CancelledError)
        sleeper = FakeSleep()
        with pytest.raises(asyncio.CancelledError):
            await retry_async(fn, attempts=5, sleep=sleeper)
        assert fn.calls == 1
        assert sleeper.calls == []

    async def test_cancelled_error_wins_over_retry_on(self):
        """即使调用方把 BaseException 写进 retry_on，取消也必须直接抛出。"""

        async def cancel_once() -> str:
            raise asyncio.CancelledError

        with pytest.raises(asyncio.CancelledError):
            await retry_async(cancel_once, attempts=3, retry_on=(BaseException,), sleep=FakeSleep())

    async def test_on_retry_receives_attempt_exception_and_delay(self):
        fn = fail_times(2)
        sleeper = FakeSleep()
        seen: list[tuple[int, BaseException, float]] = []
        await retry_async(
            fn,
            attempts=3,
            base_delay=0.5,
            jitter=False,
            sleep=sleeper,
            on_retry=lambda attempt, exc, delay: seen.append((attempt, exc, delay)),
        )
        assert [item[0] for item in seen] == [0, 1], "attempt 从 0 开始且逐次递增"
        assert [type(item[1]) for item in seen] == [Flaky, Flaky]
        assert [item[2] for item in seen] == [pytest.approx(0.5), pytest.approx(1.0)]
        # 回调发生在休眠之前，且参数与实际休眠时长一致
        assert [item[2] for item in seen] == sleeper.calls

    async def test_on_retry_not_called_when_first_attempt_succeeds(self):
        seen: list[int] = []
        await retry_async(
            fail_times(0), attempts=3, sleep=FakeSleep(), on_retry=lambda a, e, d: seen.append(a)
        )
        assert seen == []

    async def test_total_deadline_stops_retrying(self):
        """退避会越过总预算时不再重试：宁可早点降级，也不拖过检索 SLA。

        ``retry_async`` 的总预算读的是模块里的 ``time.monotonic``（不是可注入的
        ``clock``），所以这里把它替换成假时钟 —— 同样的秒数、零真实等待。
        直接「真等 0.6 秒」也能测出来，但那是在用墙钟构造边界，
        机器一忙就会变成偶发红灯。
        """
        fn = fail_times(5)
        fake_now = FakeClock()
        sleeper = FakeSleep(fake_now)
        with mock.patch("time.monotonic", fake_now):
            with pytest.raises(Flaky):
                await retry_async(
                    fn,
                    attempts=5,
                    base_delay=0.5,
                    jitter=False,
                    sleep=sleeper,
                    total_deadline=0.6,
                )
        # 第一次退避 0.5s 用掉大部分预算，第二次退避 1.0s 必然越界。
        assert fn.calls == 2
        assert sleeper.calls == [pytest.approx(0.5)], "越预算时连休眠都不该发生"

    async def test_total_deadline_shorter_than_first_delay_prevents_the_first_retry(self):
        fn = fail_times(5)
        fake_now = FakeClock()
        sleeper = FakeSleep(fake_now)
        with mock.patch("time.monotonic", fake_now):
            with pytest.raises(Flaky):
                await retry_async(
                    fn,
                    attempts=5,
                    base_delay=0.5,
                    jitter=False,
                    sleep=sleeper,
                    total_deadline=0.1,
                )
        assert fn.calls == 1, "首次退避就超预算 -> 一次都不重试"
        assert sleeper.calls == []

    async def test_total_deadline_allows_retries_within_budget(self):
        fn = fail_times(2)
        fake_now = FakeClock()
        sleeper = FakeSleep(fake_now)
        with mock.patch("time.monotonic", fake_now):
            assert (
                await retry_async(
                    fn,
                    attempts=5,
                    base_delay=0.5,
                    jitter=False,
                    sleep=sleeper,
                    total_deadline=10.0,
                )
                == "ok"
            )
        assert fn.calls == 3
        assert sleeper.calls == [pytest.approx(0.5), pytest.approx(1.0)]
        assert sleeper.total == pytest.approx(1.5)

    async def test_no_deadline_never_blocks_retries(self):
        fn = fail_times(2)
        sleeper = FakeSleep()
        assert await retry_async(fn, attempts=5, jitter=False, sleep=sleeper) == "ok"
        assert fn.calls == 3


# =================================================================== 熔断器
class TestCircuitBreaker:
    def test_starts_closed(self):
        breaker = CircuitBreaker("pubmed", clock=FakeClock())
        assert breaker.state is CircuitState.CLOSED
        assert breaker.allow() is True

    def test_opens_after_threshold_failures(self):
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=3, reset_timeout=30.0, clock=clock)
        for _ in range(2):
            assert breaker.allow() is True
            breaker.on_failure()
        assert breaker.state is CircuitState.CLOSED, "未达阈值前必须继续放行"

        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN
        assert breaker.allow() is False, "冷却期内一律拒绝"

    def test_success_resets_consecutive_failure_count(self):
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=3, clock=clock)
        breaker.on_failure()
        breaker.on_failure()
        breaker.on_success()
        assert breaker.stats()["failures"] == 0
        breaker.on_failure()
        breaker.on_failure()
        assert breaker.state is CircuitState.CLOSED, "是连续失败计数，不是累计计数"

    def test_cooldown_then_half_open_allows_one_probe(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            "openalex", failure_threshold=1, reset_timeout=30.0, half_open_max_calls=1, clock=clock
        )
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN
        assert breaker.allow() is False

        clock.advance(29.9)
        assert breaker.allow() is False, "冷却未到期不能放行"

        clock.advance(0.1)
        assert breaker.allow() is True, "冷却到期后第一次 allow 转入 HALF_OPEN 并放行"
        assert breaker.state is CircuitState.HALF_OPEN
        assert breaker.allow() is False, "half_open_max_calls=1，第二个探测请求必须被拒"

    def test_multi_probe_budget_is_honored(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            "crossref", failure_threshold=1, reset_timeout=1.0, half_open_max_calls=2, clock=clock
        )
        breaker.on_failure()
        clock.advance(1.0)
        assert breaker.allow() is True
        assert breaker.allow() is True
        assert breaker.allow() is False

    def test_probe_success_closes_the_circuit(self):
        clock = FakeClock()
        breaker = CircuitBreaker("europepmc", failure_threshold=1, reset_timeout=5.0, clock=clock)
        breaker.on_failure()
        clock.advance(5.0)
        assert breaker.allow() is True
        breaker.on_success()
        assert breaker.state is CircuitState.CLOSED
        stats = breaker.stats()
        assert stats["failures"] == 0
        assert stats["opened_at"] is None
        assert breaker.allow() is True

    def test_probe_failure_reopens_and_restarts_cooldown(self):
        clock = FakeClock()
        breaker = CircuitBreaker("s2", failure_threshold=2, reset_timeout=10.0, clock=clock)
        breaker.on_failure()
        breaker.on_failure()
        clock.advance(10.0)
        assert breaker.allow() is True  # 进入 HALF_OPEN 探测
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN
        assert breaker.allow() is False, "探测失败后必须重新计时，不能立刻再放行"
        clock.advance(10.0)
        assert breaker.allow() is True

    def test_failures_while_open_do_not_extend_cooldown(self):
        """在飞请求的后续失败不能让冷却时间无限续杯。"""
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=1, reset_timeout=10.0, clock=clock)
        breaker.on_failure()
        opened_at = breaker.stats()["opened_at"]
        clock.advance(4.0)
        breaker.on_failure()
        assert breaker.stats()["opened_at"] == opened_at
        assert breaker.stats()["trips"] == 1
        clock.advance(6.0)
        assert breaker.allow() is True, "冷却本应在第 10 秒结束"

    def test_circuit_open_error_carries_name_and_retry_after(self):
        clock = FakeClock()
        clock.reset()
        breaker = CircuitBreaker("pubmed", failure_threshold=1, reset_timeout=30.0, clock=clock)
        breaker.on_failure()  # 打开时刻 = 0
        clock.advance(12.5)
        with pytest.raises(CircuitOpenError) as info:
            with breaker.guard():
                pytest.fail("熔断打开时块体不该执行")
        assert info.value.name == "pubmed"
        assert info.value.retry_after == pytest.approx(17.5)
        assert "pubmed" in str(info.value)
        assert info.value.retry_after == pytest.approx(breaker.retry_after)

    def test_guard_returns_immediately_when_open(self):
        breaker = CircuitBreaker("pubmed", failure_threshold=1, clock=FakeClock())
        breaker.on_failure()
        entered = False
        with pytest.raises(CircuitOpenError):
            with breaker.guard():
                entered = True
        assert entered is False, "被熔断时必须立刻失败，块体绝不能执行"

    def test_guard_records_failure_and_reraises_original(self):
        breaker = CircuitBreaker("pubmed", failure_threshold=1, clock=FakeClock())
        boom = ValueError("上游返回了畸形 JSON")
        with pytest.raises(ValueError) as info:
            with breaker.guard():
                raise boom
        assert info.value is boom, "熔断器不能改业务异常语义"
        assert breaker.state is CircuitState.OPEN
        assert breaker.stats()["trips"] == 1

    def test_guard_success_closes_and_counts_nothing(self):
        breaker = CircuitBreaker("pubmed", failure_threshold=3, clock=FakeClock())
        with breaker.guard():
            pass
        assert breaker.state is CircuitState.CLOSED
        assert breaker.stats()["failures"] == 0

    def test_guard_does_not_count_its_own_rejection_as_failure(self):
        """被拒绝不是依赖故障，否则失败计数会被自己的拒绝刷爆。"""
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=1, reset_timeout=10.0, clock=clock)
        breaker.on_failure()
        for _ in range(5):
            with pytest.raises(CircuitOpenError):
                with breaker.guard():
                    pytest.fail("熔断打开时块体不该执行")
        assert breaker.stats()["trips"] == 1
        clock.advance(10.0)
        assert breaker.allow() is True

    async def test_async_context_manager_records_failure(self):
        breaker = CircuitBreaker("pubmed", failure_threshold=1, clock=FakeClock())
        with pytest.raises(RuntimeError):
            async with breaker:
                raise RuntimeError("连接被重置")
        assert breaker.state is CircuitState.OPEN

    async def test_async_context_manager_success_and_rejection(self):
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=2, reset_timeout=5.0, clock=clock)
        async with breaker:
            pass
        assert breaker.state is CircuitState.CLOSED

        breaker.on_failure()
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN
        entered = False
        with pytest.raises(CircuitOpenError):
            async with breaker:
                entered = True
        assert entered is False

        clock.advance(5.0)
        async with breaker:
            pass
        assert breaker.state is CircuitState.CLOSED

    def test_trip_counter_accumulates_over_episodes(self):
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=1, reset_timeout=1.0, clock=clock)
        breaker.on_failure()
        clock.advance(1.0)
        breaker.allow()
        breaker.on_failure()
        assert breaker.stats()["trips"] == 2

    def test_reset_clears_everything(self):
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=1, reset_timeout=10.0, clock=clock)
        breaker.on_failure()
        breaker.reset()
        stats = breaker.stats()
        assert breaker.state is CircuitState.CLOSED
        assert stats["failures"] == 0
        assert stats["opened_at"] is None
        assert stats["cooldown_remaining"] == 0.0
        assert breaker.allow() is True

    def test_stats_exposes_state_name_and_thresholds(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            "openalex", failure_threshold=4, reset_timeout=20.0, half_open_max_calls=2, clock=clock
        )
        breaker.on_failure()
        stats = breaker.stats()
        assert stats["name"] == "openalex"
        assert stats["state"] == "closed", "stats 里是可以直接进 JSON 的字符串"
        assert stats["failures"] == 1
        assert stats["failure_threshold"] == 4
        assert stats["reset_timeout"] == 20.0
        assert stats["half_open_max_calls"] == 2
        assert stats["trips"] == 0
        assert stats["cooldown_remaining"] == 0.0

    def test_stats_reports_cooldown_remaining(self):
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=1, reset_timeout=30.0, clock=clock)
        breaker.on_failure()
        clock.advance(9.0)
        assert breaker.retry_after == pytest.approx(21.0)
        assert breaker.stats()["cooldown_remaining"] == pytest.approx(21.0)
        assert breaker.stats()["state"] == "open"

    def test_half_open_state_value_is_json_friendly(self):
        assert CircuitState.CLOSED == "closed"
        assert CircuitState.HALF_OPEN.value == "half_open"
        assert isinstance(CircuitState.OPEN, str)

    def test_threshold_at_least_one(self):
        breaker = CircuitBreaker("pubmed", failure_threshold=0, clock=FakeClock())
        breaker.on_failure()
        assert breaker.state is CircuitState.OPEN, "阈值 0 按 1 处理"

    def test_state_is_thread_safe_under_concurrent_failures(self):
        """同一个 client 可能被多个线程用到：计数不能丢。

        阈值设高，保证 200 次失败期间熔断器始终是 CLOSED ——
        这样才能验证"每一次失败都被记上"；一旦转 OPEN，后续失败按设计被忽略，
        计数就不再反映线程安全性了。
        """
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=10_000, clock=clock)

        def hammer() -> None:
            for _ in range(50):
                breaker.on_failure()

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert breaker.stats()["failures"] == 200, "锁保证计数无丢失"

    def test_threshold_trips_exactly_once_under_concurrent_failures(self):
        """4 个线程同时打阈值：只能熔断一次，冷却时间不能被重复刷新。"""
        clock = FakeClock()
        breaker = CircuitBreaker("pubmed", failure_threshold=50, reset_timeout=10.0, clock=clock)
        clock.advance(3.0)  # 让"打开时刻"落在非零位置，便于验证没有被刷新

        def hammer() -> None:
            for _ in range(25):
                breaker.on_failure()

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert breaker.state is CircuitState.OPEN
        assert breaker.stats()["trips"] == 1, "并发下也只能触发一次熔断"

    def test_concurrent_allow_calls_do_not_over_admit_half_open_probes(self):
        clock = FakeClock()
        breaker = CircuitBreaker(
            "pubmed", failure_threshold=1, reset_timeout=1.0, half_open_max_calls=3, clock=clock
        )
        breaker.on_failure()
        clock.advance(1.0)
        granted: list[bool] = []
        lock = threading.Lock()

        def hammer() -> None:
            for _ in range(20):
                got = breaker.allow()
                with lock:
                    granted.append(got)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sum(granted) == 3, "半开探测额度是全局的，不能每个线程各自放行"


# ================================================================= 令牌桶
class TestTokenBucket:
    def test_burst_uses_capacity_then_rejects(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=2.0, capacity=3.0, clock=clock)
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False, "突发额度用尽后必须立刻拒绝"

    def test_default_capacity_equals_rate(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=4.0, clock=clock)
        assert bucket.tokens == pytest.approx(4.0)
        for _ in range(4):
            assert bucket.try_acquire() is True
        assert bucket.try_acquire() is False

    def test_tokens_are_both_readable_and_callable(self):
        bucket = TokenBucket(rate=2.0, capacity=2.0, clock=FakeClock())
        assert bucket.tokens == pytest.approx(2.0)
        assert bucket.tokens() == pytest.approx(2.0)

    def test_refill_after_clock_advances(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=2.0, capacity=2.0, clock=clock)
        assert bucket.try_acquire(2.0) is True
        assert bucket.try_acquire() is False

        clock.advance(0.25)
        assert bucket.tokens == pytest.approx(0.5)
        assert bucket.try_acquire() is False, "不足一个令牌时仍然拒绝"

        clock.advance(0.25)
        assert bucket.try_acquire() is True

    def test_refill_never_exceeds_capacity(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=2.0, clock=clock)
        bucket.try_acquire(2.0)
        clock.advance(3600.0)
        assert bucket.tokens == pytest.approx(2.0), "桶容量是长期平均速率的上界"

    def test_fractional_tokens_supported(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=1.0, clock=clock)
        assert bucket.try_acquire(0.25) is True
        assert bucket.tokens == pytest.approx(0.75)
        assert bucket.try_acquire(1.0) is False

    async def test_acquire_returns_immediately_when_tokens_available(self):
        bucket = TokenBucket(rate=2.0, capacity=2.0, clock=FakeClock())
        sleeper = FakeSleep()
        assert await bucket.acquire(sleep=sleeper) is True
        assert sleeper.calls == [], "有令牌时不应休眠"

    async def test_acquire_sleeps_exactly_the_deficit(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=2.0, capacity=2.0, clock=clock)
        sleeper = FakeSleep(clock)
        assert bucket.try_acquire(2.0) is True

        assert await bucket.acquire(sleep=sleeper) is True
        assert sleeper.calls == [pytest.approx(0.5)], "缺 1 个令牌，速率 2/s -> 等 0.5s"
        assert bucket.tokens == pytest.approx(0.0)

    async def test_acquire_gives_up_past_max_wait_without_sleeping(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=1.0, clock=clock)
        sleeper = FakeSleep(clock)
        bucket.try_acquire()
        assert await bucket.acquire(sleep=sleeper, max_wait=0.5) is False
        assert sleeper.calls == [], "超预算时不应真的睡一觉再放弃"

    async def test_acquire_waits_when_max_wait_is_exactly_enough(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=1.0, clock=clock)
        sleeper = FakeSleep(clock)
        bucket.try_acquire()
        assert await bucket.acquire(sleep=sleeper, max_wait=1.0) is True
        assert sleeper.calls == [pytest.approx(1.0)]

    async def test_acquire_retries_when_token_stolen_by_another_caller(self):
        """等待醒来发现令牌被抢走 -> 再等一轮，而不是空手返回 True。"""
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=2.0, clock=clock)
        assert bucket.try_acquire(2.0) is True, "先清空，逼出等待路径"
        calls: list[float] = []

        async def stealing_sleep(seconds: float) -> None:
            calls.append(seconds)
            clock.advance(seconds)
            if len(calls) == 1:
                bucket.try_acquire()  # 模拟并发的另一个调用方抢走刚回填的令牌

        assert await bucket.acquire(tokens=2.0, sleep=stealing_sleep, max_wait=60.0) is True
        assert len(calls) == 2, "第一次醒来被抢走，必须再等一轮"
        # 第一次等满 2 个令牌（2s）；被抢走 1 个后只剩 1 个，第二轮只需再等 1s。
        assert calls == [pytest.approx(2.0), pytest.approx(1.0)]
        assert bucket.tokens == pytest.approx(0.0)

    async def test_acquire_with_no_wait_budget_and_empty_bucket(self):
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=1.0, clock=clock)
        bucket.try_acquire()
        assert await bucket.acquire(sleep=FakeSleep(clock), max_wait=0.0) is False

    def test_is_thread_safe_under_concurrent_acquires(self):
        """多线程抢令牌：成功次数必须恰好等于容量，不能超发。"""
        clock = FakeClock()
        bucket = TokenBucket(rate=1.0, capacity=50.0, clock=clock)
        granted: list[bool] = []
        lock = threading.Lock()

        def hammer() -> None:
            for _ in range(50):
                got = bucket.try_acquire()
                with lock:
                    granted.append(got)

        threads = [threading.Thread(target=hammer) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert sum(granted) == 50, "容量 50 就最多发 50 个令牌"


# =============================================================== 并发舱壁
class TestBulkhead:
    def test_try_acquire_rejects_over_limit(self):
        bulkhead = Bulkhead("pdf", limit=2)
        assert bulkhead.try_acquire() is True
        assert bulkhead.try_acquire() is True
        assert bulkhead.try_acquire() is False
        stats = bulkhead.stats()
        assert stats == {"name": "pdf", "limit": 2, "in_flight": 2, "peak": 2, "rejected": 1}

    def test_release_frees_a_slot(self):
        bulkhead = Bulkhead("pdf", limit=1)
        assert bulkhead.try_acquire() is True
        assert bulkhead.try_acquire() is False
        bulkhead.release()
        assert bulkhead.try_acquire() is True
        assert bulkhead.stats()["in_flight"] == 1

    def test_peak_tracks_concurrency_and_is_callable(self):
        bulkhead = Bulkhead("pdf", limit=4)
        for _ in range(3):
            bulkhead.try_acquire()
        assert bulkhead.peak == 3
        assert bulkhead.peak() == 3
        bulkhead.release()
        assert bulkhead.peak == 3, "峰值是历史值，不会随当前并发回落"

    def test_extra_release_does_not_corrupt_counters(self):
        bulkhead = Bulkhead("pdf", limit=1)
        bulkhead.release()
        assert bulkhead.stats()["in_flight"] == 0

    def test_sync_context_manager_raises_when_full(self):
        bulkhead = Bulkhead("pdf", limit=1)
        with bulkhead.acquire():
            assert bulkhead.stats()["in_flight"] == 1
            with pytest.raises(RuntimeError):
                with bulkhead.acquire():
                    pytest.fail("舱壁已满，块体不该执行")
        assert bulkhead.stats()["in_flight"] == 0

    def test_sync_context_manager_releases_on_exception(self):
        bulkhead = Bulkhead("pdf", limit=1)
        with pytest.raises(ValueError):
            with bulkhead.acquire():
                raise ValueError("解析失败")
        assert bulkhead.stats()["in_flight"] == 0

    def test_sync_peak_under_real_threads(self):
        """真并发只发生在多个线程/协程同时持有舱位时。"""
        bulkhead = Bulkhead("pdf", limit=4)
        ready = threading.Barrier(4)
        done = threading.Barrier(4)

        def worker() -> None:
            with bulkhead.acquire():
                ready.wait(timeout=5)
                done.wait(timeout=5)

        threads = [threading.Thread(target=worker) for _ in range(4)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert bulkhead.peak == 4, "4 个线程同时在舱内"
        assert bulkhead.stats()["in_flight"] == 0

    async def test_async_context_manager_tracks_peak(self):
        bulkhead = Bulkhead("search", limit=3)
        entered: list[int] = []

        async def worker() -> None:
            async with bulkhead:
                entered.append(bulkhead.stats()["in_flight"])
                await asyncio.sleep(0)
                await asyncio.sleep(0)

        await asyncio.gather(*[worker() for _ in range(3)])
        assert bulkhead.peak == 3, "用峰值断言真实并发度，而不是墙钟耗时"
        assert sorted(entered) == [1, 2, 3]
        assert bulkhead.stats()["in_flight"] == 0

    async def test_async_context_manager_releases_on_exception(self):
        bulkhead = Bulkhead("search", limit=1)
        with pytest.raises(RuntimeError):
            async with bulkhead:
                raise RuntimeError("上游 500")
        assert bulkhead.stats()["in_flight"] == 0

    async def test_async_waits_instead_of_rejecting(self):
        bulkhead = Bulkhead("search", limit=1)
        order: list[str] = []

        async def worker(tag: str) -> None:
            async with bulkhead:
                order.append(f"in-{tag}")
                await asyncio.sleep(0)
            order.append(f"out-{tag}")

        await asyncio.gather(worker("a"), worker("b"))
        assert order in (["in-a", "out-a", "in-b", "out-b"], ["in-b", "out-b", "in-a", "out-a"])
        assert bulkhead.peak == 1, "limit=1 时不可能出现两个同时在舱内"
        assert bulkhead.stats()["rejected"] == 0, "异步路径应该排队而不是拒绝"

    async def test_async_limit_is_enforced_under_contention(self):
        bulkhead = Bulkhead("search", limit=2)
        live: list[int] = []

        async def worker() -> None:
            async with bulkhead:
                live.append(bulkhead.stats()["in_flight"])
                for _ in range(3):
                    await asyncio.sleep(0)

        await asyncio.gather(*[worker() for _ in range(6)])
        assert bulkhead.peak == 2
        assert max(live) <= 2
        assert bulkhead.stats()["in_flight"] == 0
