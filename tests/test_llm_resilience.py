"""LLM 客户端的**韧性与记账**集成测试。

前面几个模块是分开测的：``tests/test_resilience.py`` 测重试/熔断原语，
``tests/test_observability.py`` 测账本与失败分类，``tests/test_llm_json.py`` 测解析。
但"每个单元都对"不等于"接起来就对" —— 集成层最典型的坑是：

* 重试把**不该重试**的错误也重试了（401 重试 3 次 = 用户白等 3 次超时）；
* 熔断的失败计数记在"尝试"而不是"调用"上，于是**一次成功恢复的调用**也会留下失败记录，
  几次网络抖动就能把健康的熔断器打开，接下来 30 秒所有请求被误伤；
* 成功路径记账了、失败路径没记账 → 面板上"平均耗时"永远漂亮。

这里用 ``httpx.MockTransport`` 驱动**真实** :class:`LLMClient`，断言的是行为
（发了几次请求、熔断状态、账本里有什么），不是实现细节。
"""

from __future__ import annotations

import asyncio

import httpx
import pytest

from medscholar.llm import transport
from medscholar.llm.client import LLMClient, LLMError
from medscholar.llm.transport import breaker_for, reset_breakers
from medscholar.platform.config import LLMSettings
from medscholar.platform.observability import LEDGER


@pytest.fixture(autouse=True)
def _clean_state(monkeypatch):
    """每个用例都从干净的账本、熔断器与零退避开始。

    退避压到 0 是**必须**的：默认 0.8/1.6 秒的退避会让"连续 5 次失败"这个用例
    真等 12 秒，而慢测试的下场一定是被跳过或被无脑重跑。
    退避策略本身由 ``tests/test_resilience.py`` 用假时钟精确断言，
    这里只验证"接起来了没有"。
    """
    monkeypatch.setattr(transport, "RETRY_BASE_DELAY", 0.0)
    monkeypatch.setattr(transport, "RETRY_MAX_DELAY", 0.0)
    LEDGER.reset()
    reset_breakers()
    yield
    LEDGER.reset()
    reset_breakers()


class _Recorder:
    """记录发往 mock 后端的请求路径，用于区分"业务请求"与"诊断请求"。"""

    def __init__(self) -> None:
        self.posts = 0
        self.other: list[str] = []

    def handler(self, response_factory):
        def _handle(request: httpx.Request) -> httpx.Response:
            # 只数 POST：Ollama 的错误路径会顺带 GET /api/tags 去列已安装模型，
            # 把它算进"重试次数"会让断言莫名其妙地差 1（第一次写就踩了）。
            # 不按路径过滤：DeepSeek 走 /chat/completions，Ollama 走 /api/chat。
            if request.method == "POST":
                self.posts += 1
            else:
                self.other.append(f"{request.method} {request.url.path}")
            return response_factory(self.posts)

        return _handle


def _client_with(handler, *, provider: str = "ollama", model: str = "qwen3:8b") -> LLMClient:
    """构造一个把 httpx 换成 MockTransport 的客户端。

    必须连 ``_client_loop`` 一起设置：``_ensure_started`` 会在事件循环变化时
    丢弃旧客户端（那是"httpx 连接池绑定事件循环导致请求永久挂起"这个真实故障
    留下的逻辑），只设 ``_client`` 会被它静默重建掉。
    """
    settings = LLMSettings(
        provider=provider,
        model=model,
        api_key="" if provider == "ollama" else "sk-test",
    )
    client = LLMClient(settings)
    client._client = httpx.AsyncClient(
        base_url="http://127.0.0.1:11434",
        transport=httpx.MockTransport(handler),
    )
    client._client_loop = asyncio.get_running_loop()
    return client


def _ollama_ok() -> httpx.Response:
    return httpx.Response(
        200,
        json={
            "message": {"content": "回答"},
            "prompt_eval_count": 120,
            "eval_count": 40,
            "done": True,
        },
    )


class TestRetryIntegration:
    async def test_retries_on_503_then_succeeds(self):
        """5xx 属于可重试：一次失败后重试成功，用户看不到错误。"""
        rec = _Recorder()

        def factory(n: int) -> httpx.Response:
            return httpx.Response(503, text="service unavailable") if n == 1 else _ollama_ok()

        client = _client_with(rec.handler(factory), model="retry-503")
        text = await client.chat([{"role": "user", "content": "hi"}])
        assert text == "回答"
        assert rec.posts == 2, "应当重试一次后成功"
        await client.close()

    async def test_does_not_retry_on_401(self):
        """认证失败重试一百次还是失败，只会让用户白等 —— 必须立即抛出。"""
        rec = _Recorder()

        def factory(n: int) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        client = _client_with(rec.handler(factory), provider="deepseek", model="no-retry-401")
        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": "hi"}])
        assert rec.posts == 1, "401 绝不能重试"
        assert "401" in str(excinfo.value)
        await client.close()

    async def test_retry_all_attempts_then_reports(self):
        """全是 5xx 时用满尝试次数，并把最后一次响应交给调用方组织文案。"""
        rec = _Recorder()

        def factory(n: int) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = _client_with(rec.handler(factory), model="always-500")
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "hi"}])
        assert rec.posts == 3, "默认 3 次尝试"
        await client.close()


class TestCircuitBreakerIntegration:
    async def test_breaker_opens_after_repeated_call_failures(self):
        """连续 5 次**调用**失败后熔断；第 6 次不再发出任何业务请求。

        这一条同时守住"按调用而不是按尝试计数"：单次调用内部重试 3 次只算 1 次失败
        （否则两次失败就熔断了，阈值形同虚设）。
        """
        rec = _Recorder()

        def factory(n: int) -> httpx.Response:
            return httpx.Response(500, text="boom")

        client = _client_with(rec.handler(factory), model="breaker-open")
        for _ in range(5):
            with pytest.raises(LLMError):
                await client.chat([{"role": "user", "content": "hi"}])
        assert rec.posts == 15, "5 次调用 × 每次 3 个尝试"

        with pytest.raises(LLMError) as excinfo:
            await client.chat([{"role": "user", "content": "hi"}])
        assert "熔断" in str(excinfo.value)
        assert rec.posts == 15, "熔断打开后不应再发出请求"
        assert breaker_for(client._breaker_key).state.value == "open"
        await client.close()

    async def test_successful_retry_does_not_trip_breaker(self):
        """回归：一次"失败两次后成功"的调用不能留下失败记录。

        若按尝试计数，5 次这样的调用就会把健康的熔断器打开，接下来 30 秒所有请求
        被误伤 —— 这是最容易写错、也最难查的一类 bug（错误只在"网络有点抖"时出现）。
        """
        rec = _Recorder()

        def factory(n: int) -> httpx.Response:
            return _ollama_ok() if n % 3 == 0 else httpx.Response(503, text="flaky")

        client = _client_with(rec.handler(factory), model="flaky-but-recovers")
        for _ in range(6):
            await client.chat([{"role": "user", "content": "hi"}])
        assert breaker_for(client._breaker_key).state.value == "closed"
        await client.close()


class TestUsageLedgerIntegration:
    async def test_success_recorded_with_tokens_and_cost(self):
        rec = _Recorder()
        client = _client_with(rec.handler(lambda n: _ollama_ok()), model="ledger-ok")
        await client.chat([{"role": "user", "content": "hi"}])

        summary = LEDGER.summary()
        assert summary["calls"] == 1
        assert summary["prompt_tokens"] == 120
        assert summary["completion_tokens"] == 40
        # 本地模型成本必须为 0：否则"本地推理省钱"这个结论就是假的
        assert summary["cost_yuan"] == 0.0
        assert client.usage.prompt_tokens == 120
        assert client.calls == 1
        await client.close()

    async def test_failure_recorded_with_error_kind(self):
        """失败也要记账并带上分类 —— 否则面板上只有成功调用的漂亮数字。"""
        rec = _Recorder()

        def factory(n: int) -> httpx.Response:
            return httpx.Response(401, text="unauthorized")

        client = _client_with(rec.handler(factory), provider="deepseek", model="ledger-fail")
        with pytest.raises(LLMError):
            await client.chat([{"role": "user", "content": "hi"}])

        summary = LEDGER.summary()
        assert summary["failed_calls"] == 1
        assert summary["ok_calls"] == 0
        assert summary["by_error_kind"]["auth"]["calls"] == 1
        await client.close()

    async def test_streaming_records_usage(self):
        """流式路径同样要记账 —— 它是界面上的主路径，漏了就没法算真实成本。"""
        body = "\n".join(
            [
                '{"message": {"content": "加"}, "done": false}',
                '{"message": {"content": "速"}, "done": false}',
                '{"message": {"content": ""}, "done": true,'
                ' "prompt_eval_count": 50, "eval_count": 10}',
            ]
        )
        rec = _Recorder()
        client = _client_with(rec.handler(lambda n: httpx.Response(200, text=body)), model="ledger-stream")
        chunks = [chunk async for chunk in client.stream([{"role": "user", "content": "hi"}])]
        assert "".join(chunks) == "加速"
        summary = LEDGER.summary()
        assert summary["calls"] == 1
        assert summary["prompt_tokens"] == 50
        await client.close()
