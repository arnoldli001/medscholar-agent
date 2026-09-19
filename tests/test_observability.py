"""可观测性模块测试：成本账本、失败分类、trace/span 树、contextvars 并发隔离。

**全部离线、全部确定性**：不联网、不调 LLM、不用 ``time.sleep`` 制造耗时。
耗时只做弱断言（``>= 0``）——项目里已经明确写过"用墙钟断言并发是负资产"：
CI 机器一抖动就红，维护者最后只能把它删掉，等于没测。
并发隔离用 ``asyncio.gather`` + 结构断言（各自的 trace_id / 各自的孩子节点）来验证，
那才是真正要保证的性质。
"""

from __future__ import annotations

import asyncio
import json
import threading
import time
from pathlib import Path

import pytest

from medscholar.platform import observability as ob
from medscholar.platform.observability import (
    FAILURE_KINDS,
    LEDGER,
    PRICES,
    LLMUsage,
    Span,
    TraceRecorder,
    UsageLedger,
    classify_failure,
    create_trace,
    current_span,
    current_trace,
    estimate_cost,
    run_in_trace,
    set_current_trace,
)


@pytest.fixture(autouse=True)
def _clean_globals():
    """清理两个进程级全局对象：模块单例账本与 contextvars 绑定。

    共享单例是设计的一部分（没有依赖注入的地方要能直接用），代价就是测试之间会互相
    污染：某个用例记了 3 条，另一个用例断言 ``calls == 0`` 就会随机失败。
    这里在**每个用例前后**把绑定和账本都清干净，保证用例可以任意顺序执行。
    """
    token = set_current_trace(None)
    LEDGER.reset()
    yield
    LEDGER.reset()
    set_current_trace(None)
    ob._CURRENT_TRACE.reset(token)


# ---------------------------------------------------------------------------
# 1) 成本估算
# ---------------------------------------------------------------------------


class TestEstimateCost:
    def test_local_ollama_is_free(self):
        """本地推理成本必须是 0（README 的成本承诺：本地推理不花钱）。"""
        assert estimate_cost("ollama", 1_000_000, 1_000_000) == 0.0

    def test_local_model_with_tag_is_free(self):
        """带 ``:tag`` 的模型名一律视为本地模型，零价。"""
        assert estimate_cost("qwen3:8b", 500_000, 500_000) == 0.0
        assert estimate_cost("llama3.1:70b", 10**7, 10**7) == 0.0
        assert estimate_cost("my-custom-model:latest", 10**7, 10**7) == 0.0

    def test_cloud_model_is_not_free(self):
        cost = estimate_cost("deepseek-chat", 100_000, 20_000)
        assert cost > 0.0

    def test_cloud_cost_amount_is_sane(self):
        """一次综述 2~4 万 token 应该是"不到一毛钱"的量级，而不是几块钱。"""
        cost = estimate_cost("deepseek-chat", 30_000, 5_000)
        assert 0.0 < cost < 0.5

    def test_reasoner_more_expensive_than_chat(self):
        """reasoner 带思维链，单价比 chat 高，否则成本面板会系统性低估。"""
        chat = estimate_cost("deepseek-chat", 100_000, 100_000)
        reasoner = estimate_cost("deepseek-reasoner", 100_000, 100_000)
        assert reasoner > chat

    def test_prefix_match_longest_first(self):
        """``deepseek-chat-v3`` 必须命中 ``deepseek-chat``，而不是更粗的 ``deepseek``。"""
        assert estimate_cost("deepseek-chat-v3", 10**6, 0) == PRICES["deepseek-chat"][0]
        assert estimate_cost("deepseek-chat-0324", 0, 10**6) == PRICES["deepseek-chat"][1]

    def test_reasoner_prefix_not_swallowed_by_shorter_prefix(self):
        """``deepseek-reasoner`` 不能被 ``deepseek`` 抢走（那会少算一半）。"""
        assert estimate_cost("deepseek-reasoner", 10**6, 0) == PRICES["deepseek-reasoner"][0]

    def test_case_insensitive(self):
        assert estimate_cost("DeepSeek-Chat", 10**6, 0) == PRICES["deepseek-chat"][0]

    def test_unknown_model_is_zero_not_exception(self):
        """未知模型宁可少算，也绝不能让统计把主流程搞挂。"""
        assert estimate_cost("totally-unknown-model", 10**6, 10**6) == 0.0
        assert estimate_cost("", 100, 100) == 0.0

    def test_negative_tokens_do_not_produce_negative_cost(self):
        """脏数据（负数 token）不能算出负成本，否则总成本会被抵消。"""
        assert estimate_cost("deepseek-chat", -1000, -1000) == 0.0

    def test_output_priced_higher_than_input(self):
        assert estimate_cost("deepseek-chat", 10**6, 0) < estimate_cost("deepseek-chat", 0, 10**6)

    def test_empty_prices_table_models_only_local(self):
        """PRICES 里必须有本地零价条目和至少一个云端条目。"""
        assert PRICES["ollama"] == (0.0, 0.0)
        assert PRICES["deepseek-chat"][0] > 0.0
        assert PRICES["deepseek-reasoner"][0] > 0.0


class TestPercentile:
    def test_empty_returns_zero(self):
        assert ob._percentile([], 0.5) == 0.0

    def test_single_value(self):
        assert ob._percentile([7.5], 0.95) == 7.5

    @pytest.mark.parametrize(
        "q,expected",
        [(0.0, 0.0), (0.5, 50.0), (0.95, 95.0), (1.0, 100.0)],
    )
    def test_linear_interpolation_on_regular_series(self, q, expected):
        values = [float(i) for i in range(0, 101, 10)]
        assert ob._percentile(values, q) == expected

    def test_p95_is_interpolated_not_max(self):
        """小样本下 p95 应是插值结果而非最大值 —— 否则尾部延迟被夸大。"""
        assert ob._percentile([100.0, 200.0, 300.0, 400.0], 0.95) == 385.0

    def test_q_is_clamped(self):
        values = [1.0, 2.0, 3.0]
        assert ob._percentile(values, -1.0) == 1.0
        assert ob._percentile(values, 2.0) == 3.0


# ---------------------------------------------------------------------------
# 2) 失败分类
# ---------------------------------------------------------------------------


class TestClassifyFailure:
    def test_timeout_by_type(self):
        assert classify_failure(TimeoutError("read timed out")) == "timeout"
        assert classify_failure(asyncio.TimeoutError()) == "timeout"

    def test_cancelled_by_type(self):
        """CancelledError 继承 BaseException，必须靠类型判定。"""
        assert classify_failure(asyncio.CancelledError()) == "cancelled"

    def test_cancelled_by_text(self):
        assert classify_failure("request cancelled by client") == "cancelled"

    def test_rate_limited(self):
        assert classify_failure("429 Too Many Requests") == "rate_limited"
        assert classify_failure(RuntimeError("Rate limit exceeded, retry later")) == "rate_limited"

    def test_auth(self):
        assert classify_failure("401 Unauthorized") == "auth"
        assert classify_failure("Invalid API key provided") == "auth"

    def test_bad_request(self):
        assert classify_failure("400 Bad Request: invalid model parameter") == "bad_request"

    def test_not_found(self):
        assert classify_failure("404 Not Found") == "not_found"

    def test_server_error(self):
        assert classify_failure("503 Service Unavailable") == "server_error"

    def test_connection(self):
        assert classify_failure(ConnectionResetError("Connection reset by peer")) == "connection"
        assert classify_failure("connect timeout to host") == "connection"

    def test_parse(self):
        assert classify_failure("Unterminated string in JSON response") == "parse"

    def test_parse_from_json_decode_error(self):
        try:
            json.loads("{not json")
        except json.JSONDecodeError as exc:
            assert classify_failure(exc) == "parse"
        else:  # pragma: no cover - json.loads 必然抛
            pytest.fail("json.loads 应该抛 JSONDecodeError")

    def test_context_overflow_wins_over_bad_request(self):
        """上下文超长本身就是 400，必须先归类到 context_overflow 才能采取对应措施。"""
        assert (
            classify_failure("400 This model's maximum context length is 8192 tokens")
            == "context_overflow"
        )

    def test_context_overflow_chinese(self):
        assert classify_failure("请求失败：上下文超长，请缩短输入") == "context_overflow"

    def test_unknown(self):
        assert classify_failure(RuntimeError("something odd happened")) == "unknown"

    def test_none_returns_empty(self):
        """None 表示"没有失败信息"，返回空串而不是 unknown，避免污染错误统计。"""
        assert classify_failure(None) == ""

    def test_empty_string_is_unknown(self):
        assert classify_failure("   ") == "unknown"

    def test_timeout_checked_before_connection(self):
        """TimeoutError 是 OSError 子类，不能被误判成 connection。"""
        assert classify_failure(TimeoutError("connection attempt timed out")) == "timeout"

    def test_every_result_is_a_known_kind(self):
        """任何输入都必须落在有限值域内（空串只代表"没有失败信息"）。"""
        samples: list[object] = [
            "",
            "429",
            "401",
            "400",
            "404",
            "500",
            "boom",
            TimeoutError("x"),
            asyncio.CancelledError(),
            ValueError("x"),
            12345,  # 脏类型（本该是异常或字符串）也不能把函数搞崩
        ]
        for sample in samples:
            assert classify_failure(sample) in FAILURE_KINDS

    def test_none_is_outside_the_value_range(self):
        """None 返回空串，刻意不属于 FAILURE_KINDS，避免污染错误统计。"""
        assert classify_failure(None) == ""
        assert classify_failure(None) not in FAILURE_KINDS

    def test_failure_kinds_cover_required_set(self):
        required = {
            "timeout",
            "rate_limited",
            "auth",
            "bad_request",
            "not_found",
            "server_error",
            "connection",
            "parse",
            "context_overflow",
            "cancelled",
            "unknown",
        }
        assert required.issubset(set(FAILURE_KINDS))


# ---------------------------------------------------------------------------
# 3) LLMUsage 与账本
# ---------------------------------------------------------------------------


def _usage(
    model: str = "deepseek-chat",
    prompt: int = 1000,
    completion: int = 200,
    latency: float = 100.0,
    ok: bool = True,
    phase: str = "execute",
    error_kind: str = "",
    cache_hit: bool = False,
    provider: str = "deepseek",
) -> LLMUsage:
    """构造测试用记录，默认值固定，保证断言完全确定。"""
    return LLMUsage(
        provider=provider,
        model=model,
        prompt_tokens=prompt,
        completion_tokens=completion,
        latency_ms=latency,
        ok=ok,
        phase=phase,
        run_id="run-1",
        error_kind=error_kind,
        cache_hit=cache_hit,
    )


class TestLLMUsage:
    def test_total_tokens(self):
        assert _usage(prompt=1200, completion=300).total_tokens == 1500

    def test_cost_yuan_cloud(self):
        assert _usage().cost_yuan > 0.0

    def test_cost_yuan_local_is_zero(self):
        assert _usage(model="qwen3:8b", provider="ollama").cost_yuan == 0.0

    def test_frozen(self):
        """不可变：记账之后任何一层都不该改写历史记录。"""
        usage = _usage()
        with pytest.raises(Exception):
            usage.model = "other"  # type: ignore[misc]

    def test_to_dict_is_json_serializable(self):
        payload = _usage().to_dict()
        assert json.loads(json.dumps(payload, ensure_ascii=False))["model"] == "deepseek-chat"
        assert payload["total_tokens"] == 1200
        assert payload["cost_yuan"] > 0.0


class TestUsageLedger:
    def test_starts_empty(self):
        ledger = UsageLedger()
        summary = ledger.summary()
        assert summary["calls"] == 0
        assert summary["total_tokens"] == 0
        assert summary["cost_yuan"] == 0.0
        assert summary["latency_ms_p50"] == 0.0
        assert summary["latency_ms_p95"] == 0.0
        assert ledger.all() == []
        assert ledger.recent(10) == []

    def test_empty_summary_has_all_required_keys(self):
        summary = UsageLedger().summary()
        for key in (
            "calls",
            "ok_calls",
            "failed_calls",
            "cache_hits",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_yuan",
            "latency_ms_p50",
            "latency_ms_p95",
            "by_model",
            "by_phase",
            "by_error_kind",
        ):
            assert key in summary

    def test_record_and_all_preserves_order(self):
        ledger = UsageLedger()
        ledger.record(_usage(model="m1"))
        ledger.record(_usage(model="m2"))
        assert [u.model for u in ledger.all()] == ["m1", "m2"]

    def test_recent_returns_tail(self):
        ledger = UsageLedger()
        for index in range(10):
            ledger.record(_usage(model=f"m{index}"))
        assert [u.model for u in ledger.recent(3)] == ["m7", "m8", "m9"]
        assert ledger.recent(100) == ledger.all()
        assert ledger.recent(0) == []

    def test_summary_totals(self):
        ledger = UsageLedger()
        ledger.record(_usage(prompt=1000, completion=200, latency=100.0))
        ledger.record(_usage(prompt=2000, completion=400, latency=300.0))
        ledger.record(_usage(ok=False, error_kind="timeout", latency=500.0))
        ledger.record(_usage(cache_hit=True, latency=1.0))
        summary = ledger.summary()
        assert summary["calls"] == 4
        assert summary["ok_calls"] == 3
        assert summary["failed_calls"] == 1
        assert summary["cache_hits"] == 1
        # 1000+2000+1000+1000 = 5000，输出 200+400+200+200 = 1000
        assert summary["prompt_tokens"] == 5000
        assert summary["completion_tokens"] == 1000
        assert summary["total_tokens"] == 6000

    def test_summary_by_model(self):
        ledger = UsageLedger()
        ledger.record(_usage(model="deepseek-chat", prompt=1000, completion=100))
        ledger.record(_usage(model="deepseek-chat", prompt=1000, completion=100))
        ledger.record(_usage(model="qwen3:8b", provider="ollama", prompt=500, completion=500))
        by_model = ledger.summary()["by_model"]
        assert set(by_model) == {"deepseek-chat", "qwen3:8b"}
        assert by_model["deepseek-chat"]["calls"] == 2
        assert by_model["deepseek-chat"]["total_tokens"] == 2200
        assert by_model["deepseek-chat"]["cost_yuan"] > 0.0
        assert by_model["qwen3:8b"]["calls"] == 1
        assert by_model["qwen3:8b"]["cost_yuan"] == 0.0

    def test_summary_by_phase(self):
        ledger = UsageLedger()
        ledger.record(_usage(phase="plan", prompt=100, completion=10))
        ledger.record(_usage(phase="plan", prompt=100, completion=10, ok=False, error_kind="auth"))
        ledger.record(_usage(phase="synthesize", prompt=1000, completion=500))
        by_phase = ledger.summary()["by_phase"]
        assert by_phase["plan"]["calls"] == 2
        assert by_phase["plan"]["failed_calls"] == 1
        assert by_phase["plan"]["total_tokens"] == 220
        assert by_phase["synthesize"]["total_tokens"] == 1500

    def test_phase_missing_becomes_labelled_bucket(self):
        """空 phase 归到「(未标注)」，不能在统计里凭空消失。"""
        ledger = UsageLedger()
        ledger.record(_usage(phase=""))
        assert "(未标注)" in ledger.summary()["by_phase"]

    def test_summary_by_error_kind(self):
        ledger = UsageLedger()
        ledger.record(_usage(ok=False, error_kind="timeout"))
        ledger.record(_usage(ok=False, error_kind="timeout"))
        ledger.record(_usage(ok=False, error_kind="rate_limited"))
        ledger.record(_usage())  # 成功记录不带 error_kind，不能进统计
        by_error = ledger.summary()["by_error_kind"]
        assert by_error["timeout"]["calls"] == 2
        assert by_error["rate_limited"]["calls"] == 1
        assert "deepseek-chat" in by_error["timeout"]["models"]
        assert len(by_error) == 2

    def test_summary_percentiles(self):
        ledger = UsageLedger()
        for latency in (0.0, 10.0, 20.0, 30.0, 40.0, 50.0, 60.0, 70.0, 80.0, 90.0, 100.0):
            ledger.record(_usage(latency=latency))
        summary = ledger.summary()
        assert summary["latency_ms_p50"] == 50.0
        assert summary["latency_ms_p95"] == 95.0

    def test_summary_percentile_single_call(self):
        ledger = UsageLedger()
        ledger.record(_usage(latency=123.4567))
        summary = ledger.summary()
        assert summary["latency_ms_p50"] == summary["latency_ms_p95"] == 123.457

    def test_summary_cost_sums_models(self):
        ledger = UsageLedger()
        ledger.record(_usage(model="deepseek-chat", prompt=100_000, completion=0))
        ledger.record(_usage(model="qwen3:8b", provider="ollama", prompt=100_000, completion=0))
        expected = estimate_cost("deepseek-chat", 100_000, 0)
        assert ledger.summary()["cost_yuan"] == pytest.approx(expected, abs=1e-9)

    def test_summary_is_json_serializable(self):
        ledger = UsageLedger()
        ledger.record(_usage(ok=False, error_kind="parse"))
        payload = json.dumps(ledger.summary(), ensure_ascii=False)  # 不抛即通过
        assert json.loads(payload)["calls"] == 1

    def test_all_returns_copy(self):
        """``all()`` 返回快照，调用方改它不能影响账本。"""
        ledger = UsageLedger()
        ledger.record(_usage())
        snapshot = ledger.all()
        snapshot.clear()
        assert len(ledger.all()) == 1

    def test_reset_clears(self):
        ledger = UsageLedger()
        ledger.record(_usage())
        ledger.reset()
        assert ledger.all() == []
        assert ledger.summary()["calls"] == 0

    def test_bounded_window_drops_old_items_but_keeps_counter(self):
        """明细有上限（长期驻留的服务不能无界增长），但累计计数仍然准确。"""
        ledger = UsageLedger(max_recent=5)
        for index in range(20):
            ledger.record(_usage(model=f"m{index}"))
        summary = ledger.summary()
        assert summary["calls"] == 5
        assert summary["recorded_total"] == 20
        assert summary["window_size"] == 5
        assert [u.model for u in ledger.all()] == ["m15", "m16", "m17", "m18", "m19"]

    def test_thread_safety(self):
        """多线程同时记账不能丢记录（工作线程池里真的会这样调用）。"""
        ledger = UsageLedger(max_recent=10_000)
        per_thread = 50
        threads = [
            threading.Thread(target=lambda: [ledger.record(_usage(latency=1.0)) for _ in range(per_thread)])
            for _ in range(4)
        ]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()
        assert ledger.summary()["recorded_total"] == 4 * per_thread
        assert ledger.summary()["calls"] == 4 * per_thread

    def test_module_level_ledger_is_a_usage_ledger(self):
        assert isinstance(LEDGER, UsageLedger)
        LEDGER.record(_usage(model="deepseek-chat"))
        assert LEDGER.summary()["calls"] == 1


# ---------------------------------------------------------------------------
# 4) span 与 trace 树
# ---------------------------------------------------------------------------


class TestSpan:
    def test_duration_without_end_is_non_negative(self):
        span = Span(name="x", start=time.monotonic())
        assert span.duration_ms >= 0.0

    def test_finished_span_duration(self):
        span = Span(name="x", start=100.0, end=100.25)
        assert span.duration_ms == 250.0

    def test_never_negative(self):
        """时钟异常/手工构造出 end < start 时不能返回负耗时。"""
        assert Span(name="x", start=10.0, end=5.0).duration_ms == 0.0

    def test_to_dict_serializable(self):
        parent = Span(name="parent", start=1.0, end=1.5, attrs={"phase": "plan"})
        child = Span(name="child", start=1.1, end=1.2, status="error", error="boom")
        parent.children.append(child)
        payload = json.loads(json.dumps(parent.to_dict(), ensure_ascii=False))
        assert payload["name"] == "parent"
        assert payload["duration_ms"] == 500.0
        assert payload["attrs"] == {"phase": "plan"}
        assert payload["children"][0]["status"] == "error"
        assert payload["children"][0]["error"] == "boom"


class TestTraceRecorder:
    def test_root_is_ok_and_finished(self):
        recorder = TraceRecorder(name="run")
        assert recorder.root.status == "ok"
        assert recorder.root.end is not None
        assert recorder.trace_id

    def test_trace_ids_are_unique(self):
        assert TraceRecorder().trace_id != TraceRecorder().trace_id

    def test_span_nesting_builds_tree(self):
        recorder = TraceRecorder(name="run")
        with recorder.span("outer"):
            with recorder.span("inner"):
                pass
        assert [child.name for child in recorder.root.children] == ["outer"]
        outer = recorder.root.children[0]
        assert [child.name for child in outer.children] == ["inner"]
        assert outer.status == "ok"
        assert outer.end is not None
        assert outer.duration_ms >= 0.0

    def test_sibling_spans_are_not_nested(self):
        recorder = TraceRecorder()
        with recorder.span("a"):
            pass
        with recorder.span("b"):
            pass
        assert [child.name for child in recorder.root.children] == ["a", "b"]
        assert recorder.root.children[0].children == []

    def test_span_attrs_are_recorded(self):
        recorder = TraceRecorder()
        with recorder.span("fetch", source="pubmed", count=10) as span:
            assert span.attrs == {"source": "pubmed", "count": 10}
        assert recorder.to_dict()["root"]["children"][0]["attrs"]["source"] == "pubmed"

    def test_span_attrs_are_copied_not_aliased(self):
        """传进来的 dict 后续被改，不能让历史 span 跟着变。"""
        recorder = TraceRecorder()
        attrs = {"source": "pubmed"}
        with recorder.span("fetch", **attrs):
            attrs["source"] = "mutated"
        assert recorder.root.children[0].attrs["source"] == "pubmed"

    def test_error_span_marks_error_and_reraises(self):
        recorder = TraceRecorder()
        with pytest.raises(RuntimeError, match="boom"):
            with recorder.span("explode"):
                raise RuntimeError("boom")
        span = recorder.root.children[0]
        assert span.status == "error"
        assert span.error == "RuntimeError: boom"
        assert span.duration_ms >= 0.0

    def test_error_span_classifies_failure(self):
        """异常 span 顺手写入分类，排障时不用再去猜。"""
        recorder = TraceRecorder()
        with pytest.raises(RuntimeError):
            with recorder.span("call"):
                raise RuntimeError("429 Too Many Requests")
        assert recorder.root.children[0].attrs["error_kind"] == "rate_limited"

    def test_error_in_nested_span_reraises_to_outer(self):
        """内层抛出必须一路穿透到调用方（trace 不吞异常）。"""
        recorder = TraceRecorder()
        with pytest.raises(ValueError):
            with recorder.span("outer"):
                with recorder.span("inner"):
                    raise ValueError("deep failure")
        outer = recorder.root.children[0]
        assert outer.children[0].status == "error"
        assert outer.status == "error"
        assert outer.children[0].error == "ValueError: deep failure"

    def test_span_stack_is_cleaned_after_error(self):
        """异常之后 span 栈必须清空，否则后续 span 会挂到已结束的节点上。"""
        recorder = TraceRecorder()
        with pytest.raises(RuntimeError):
            with recorder.span("boom"):
                raise RuntimeError("x")
        assert current_span() is None
        with recorder.span("after"):
            pass
        assert [child.name for child in recorder.root.children] == ["boom", "after"]

    def test_error_text_is_truncated(self):
        """超长错误文本截断到 500 字符，避免一个 span 撑爆 JSONL。"""
        recorder = TraceRecorder()
        with pytest.raises(RuntimeError):
            with recorder.span("boom"):
                raise RuntimeError("x" * 5000)
        assert len(recorder.root.children[0].error) == 500

    def test_to_dict_round_trips_through_json(self):
        recorder = TraceRecorder(name="review")
        with recorder.span("search", source="pubmed"):
            with recorder.span("embed"):
                pass
        payload = json.loads(json.dumps(recorder.to_dict(), ensure_ascii=False))
        assert payload["trace_id"] == recorder.trace_id
        assert payload["root"]["name"] == "review"
        assert payload["root"]["children"][0]["name"] == "search"

    def test_summary_aggregates_same_name(self):
        recorder = TraceRecorder()
        with recorder.span("fetch"):
            pass
        with recorder.span("fetch"):
            pass
        with recorder.span("other"):
            pass
        summary = recorder.summary()
        assert summary["trace_id"] == recorder.trace_id
        spans = summary["spans"]
        assert spans["fetch"]["count"] == 2
        assert spans["fetch"]["total_ms"] >= 0.0
        assert spans["fetch"]["max_ms"] >= 0.0
        assert spans["fetch"]["total_ms"] >= spans["fetch"]["max_ms"]
        assert spans["other"]["count"] == 1

    def test_summary_counts_errors(self):
        recorder = TraceRecorder()
        with pytest.raises(RuntimeError):
            with recorder.span("call"):
                raise RuntimeError("x")
        assert recorder.summary()["spans"]["call"]["errors"] == 1

    def test_to_jsonl_appends_readable_lines(self, tmp_path: Path):
        """落盘后必须能被 json.loads 逐行读回，且中文不转义、不串行。"""
        path = tmp_path / "trace.jsonl"
        first = TraceRecorder(name="第一次运行")
        with first.span("search", source="pubmed"):
            pass
        first.to_jsonl(path)

        second = TraceRecorder(name="第二次运行")
        with second.span("embed"):
            pass
        second.to_jsonl(path)

        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 2
        records = [json.loads(line) for line in lines]
        assert [record["root"]["name"] for record in records] == ["第一次运行", "第二次运行"]
        assert records[0]["root"]["children"][0]["name"] == "search"
        assert "第一次运行" in lines[0]  # ensure_ascii=False

    def test_to_jsonl_creates_parent_directory(self, tmp_path: Path):
        path = tmp_path / "nested" / "dir" / "trace.jsonl"
        TraceRecorder().to_jsonl(path)
        assert path.exists()

    def test_to_jsonl_error_trace_keeps_error_info(self, tmp_path: Path):
        path = tmp_path / "err.jsonl"
        recorder = TraceRecorder()
        with pytest.raises(RuntimeError):
            with recorder.span("call"):
                raise RuntimeError("unterminated JSON")
        recorder.to_jsonl(path)
        record = json.loads(path.read_text(encoding="utf-8").strip())
        node = record["root"]["children"][0]
        assert node["status"] == "error"
        assert node["attrs"]["error_kind"] == "parse"


class TestTraceBinding:
    def test_create_trace_binds_current(self):
        recorder = create_trace("review")
        assert current_trace() is recorder
        assert current_trace().trace_id == recorder.trace_id

    def test_set_current_trace_none_unbinds(self):
        create_trace("review")
        set_current_trace(None)
        assert current_trace() is None

    def test_set_current_trace_returns_token(self):
        recorder = create_trace("outer")
        token = set_current_trace(None)
        assert current_trace() is None
        ob._CURRENT_TRACE.reset(token)
        assert current_trace() is recorder

    def test_context_manager_restores_previous(self):
        """``with`` 退出后要还原成外层 trace，而不是简单清空。"""
        outer = create_trace("outer")
        with TraceRecorder(name="inner") as inner:
            assert current_trace() is inner
        assert current_trace() is outer

    def test_finish_unbinds_and_returns_dict(self):
        recorder = create_trace("review")
        with recorder.span("search"):
            pass
        payload = recorder.finish()
        assert current_trace() is None
        assert payload["root"]["children"][0]["name"] == "search"

    def test_current_span_tracks_stack(self):
        recorder = create_trace("review")
        assert current_span() is None
        with recorder.span("outer") as outer:
            assert current_span() is outer
            with recorder.span("inner") as inner:
                assert current_span() is inner
            assert current_span() is outer
        assert current_span() is None

    def test_span_without_current_trace_still_works(self):
        """不绑定也能用：显式持有 recorder 的调用路径不该依赖全局状态。"""
        recorder = TraceRecorder()
        with recorder.span("solo"):
            pass
        assert current_trace() is None
        assert recorder.root.children[0].name == "solo"

    def test_run_in_trace_returns_result_and_recorder(self):
        def work() -> str:
            create_trace("inner")
            with current_trace().span("step"):
                return "done"

        result, recorder = run_in_trace(work, name="batch")
        assert result == "done"
        assert recorder.to_dict()["root"]["name"] == "batch"

    def test_worker_threads_have_isolated_contexts(self):
        """工作线程各自持有一份 contextvars 上下文，互不串台、无需加锁。"""
        results: dict[str, TraceRecorder] = {}

        def work(tag: str) -> str:
            # run_in_trace 已经绑定好一棵 trace，这里直接用它（再 create_trace 会换掉绑定）
            recorder = current_trace()
            assert recorder is not None
            with recorder.span(f"span-{tag}"):
                pass
            return tag

        def worker(tag: str) -> None:
            # 工作线程里 current_trace() 初始必须是 None（不继承主线程的绑定）
            assert current_trace() is None
            _, recorder = run_in_trace(work, tag, name=tag)
            results[tag] = recorder

        threads = [threading.Thread(target=worker, args=(tag,)) for tag in ("A", "B", "C")]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        assert set(results) == {"A", "B", "C"}
        assert len({recorder.trace_id for recorder in results.values()}) == 3
        for tag, recorder in results.items():
            assert [child.name for child in recorder.root.children] == [f"span-{tag}"]


# ---------------------------------------------------------------------------
# 5) contextvars 并发隔离（本项目踩过坑的地方）
# ---------------------------------------------------------------------------


class TestConcurrentIsolation:
    async def test_gather_traces_do_not_cross_contaminate(self):
        """并发任务各自的 trace 必须互不可见 —— 全局变量实现下这里必挂。"""
        async def task(tag: str) -> dict:
            recorder = create_trace(tag)
            with recorder.span(f"work-{tag}"):
                # 让出控制权，制造真正的交错执行
                await asyncio.sleep(0)
                await asyncio.sleep(0)
            return {
                "tag": tag,
                "trace_id": recorder.trace_id,
                "bound": current_trace(),
                "children": [child.name for child in recorder.root.children],
            }

        first, second = await asyncio.gather(task("A"), task("B"))

        assert first["trace_id"] != second["trace_id"]
        assert first["bound"] is not None and first["bound"].trace_id == first["trace_id"]
        assert second["bound"] is not None and second["bound"].trace_id == second["trace_id"]
        assert first["children"] == ["work-A"]
        assert second["children"] == ["work-B"]

    async def test_gather_span_stacks_are_isolated(self):
        """span 栈也必须隔离，否则并发任务的 span 会挂到对方的父节点下。"""
        async def task(tag: str) -> tuple[str, list[str]]:
            recorder = create_trace(tag)
            with recorder.span(f"outer-{tag}"):
                await asyncio.sleep(0)
                with recorder.span(f"inner-{tag}"):
                    await asyncio.sleep(0)
            return recorder.trace_id, [child.name for child in recorder.root.children]

        results = await asyncio.gather(task("A"), task("B"), task("C"))
        for tag, (_, children) in zip(("A", "B", "C"), results):
            assert children == [f"outer-{tag}"]

    async def test_gather_error_in_one_task_does_not_affect_other(self):
        """一个任务的失败不能把另一个任务的 span 标成 error。"""
        async def failing() -> None:
            recorder = create_trace("failing")
            with recorder.span("call"):
                await asyncio.sleep(0)
                raise RuntimeError("429 Too Many Requests")

        async def healthy() -> dict:
            recorder = create_trace("healthy")
            with recorder.span("call"):
                await asyncio.sleep(0)
            return {
                "status": recorder.root.children[0].status,
                "attrs": recorder.root.children[0].attrs,
            }

        healthy_task = asyncio.ensure_future(healthy())
        with pytest.raises(RuntimeError):
            await asyncio.gather(failing(), healthy_task)
        result = await healthy_task
        assert result["status"] == "ok"
        assert result["attrs"] == {}

    async def test_nested_task_inherits_parent_trace(self):
        """先建 trace 再派生任务：子任务属于同一条调用链，应挂在同一棵树上。"""
        recorder = create_trace("review")

        async def child() -> str | None:
            bound = current_trace()
            return bound.trace_id if bound else None

        with recorder.span("parent"):
            child_trace_id = await asyncio.create_task(child())
        assert child_trace_id == recorder.trace_id

    async def test_concurrent_tasks_produce_serializable_trees(self, tmp_path: Path):
        """并发跑完的两棵树都能落盘并被读回，行数与任务数一致。"""
        path = tmp_path / "concurrent.jsonl"

        async def task(tag: str) -> None:
            recorder = create_trace(tag)
            with recorder.span("work"):
                await asyncio.sleep(0)
            recorder.to_jsonl(path)

        await asyncio.gather(task("A"), task("B"), task("C"))
        lines = path.read_text(encoding="utf-8").splitlines()
        assert len(lines) == 3
        names = sorted(json.loads(line)["root"]["name"] for line in lines)
        assert names == ["A", "B", "C"]
