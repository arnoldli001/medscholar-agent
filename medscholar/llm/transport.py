"""LLM 出网调用的韧性层：重试与熔断（只管"失败了怎么办"）。

两类重试必须分开：传输层重发同一请求（网络抖动/429/5xx）；语义层重试在
chat_json 里把 JSON 错误回灌模型重新生成——混用会让一次坏 JSON 白烧 token/时间。
熔断失败按"调用"而非"尝试"计数：仅整次调用最终失败记一次，否则重试后成功
也会误伤健康熔断器。流式不重试：吐字后重试会让同一段内容重复两遍，
故只在建立连接阶段（未 yield 任何字节）允许重试。
"""

from __future__ import annotations

import logging
from collections.abc import Awaitable, Callable

import httpx

from ..platform.resilience import CircuitBreaker, CircuitOpenError, retry_async
from .errors import LLMError

logger = logging.getLogger(__name__)

__all__ = [
    "RETRYABLE_STATUS",
    "RETRY_ATTEMPTS",
    "RETRY_BASE_DELAY",
    "RETRY_MAX_DELAY",
    "breaker_for",
    "breaker_stats",
    "failure_kind",
    "http_error_kind",
    "reset_breakers",
    "run_with_resilience",
]

#: 这些状态码值得重发同一个请求。其余 4xx 是"请求本身有问题"：
#: 401 重试一百次还是 401，只会让用户白等。
RETRYABLE_STATUS: frozenset[int] = frozenset({408, 409, 425, 429, 500, 502, 503, 504})

#: 默认尝试次数（含首次）。3 次足够覆盖"模型正在加载"这类瞬时故障。
RETRY_ATTEMPTS = 3

#: 退避参数；模块级常量便于测试压到 0，避免重试用例真等十几秒。
RETRY_BASE_DELAY = 0.8
RETRY_MAX_DELAY = 8.0

#: 每个后端一个熔断器：key = "provider|model|base_url"
_BREAKERS: dict[str, CircuitBreaker] = {}


def breaker_for(key: str) -> CircuitBreaker:
    """取得（或创建）某后端的熔断器。按后端隔离：本地与云端故障互不影响，
    避免局部故障升级成整个 LLM 不可用。"""
    breaker = _BREAKERS.get(key)
    if breaker is None:
        breaker = CircuitBreaker(f"llm:{key}")
        _BREAKERS[key] = breaker
    return breaker


def breaker_stats() -> dict[str, dict]:
    """所有后端的熔断状态（给 /api/metrics 用）。"""
    return {key: breaker.stats() for key, breaker in _BREAKERS.items()}


def reset_breakers() -> None:
    """清空所有熔断器（配置变更或测试用）。"""
    _BREAKERS.clear()


class _RetryableStatus(Exception):
    """内部信号：状态码属于可重试集合。用异常复用 retry_async 循环，并带出响应对象。"""

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def failure_kind(exc: BaseException) -> str:
    """异常归类为统一失败分类（账本/面板统计用），比 observability.classify_failure
    多认熔断打开与可重试状态码耗尽两种；分类准确才能区分限流、认证等处置动作。"""
    from ..platform.observability import classify_failure

    if isinstance(exc, CircuitOpenError):
        return "circuit_open"
    if isinstance(exc, _RetryableStatus):
        return http_error_kind(exc.response.status_code)
    return classify_failure(exc) or "unknown"


def http_error_kind(status: int) -> str:
    """HTTP 状态码映射为 failure_kind 同一套分类名；直接用状态码而非文本匹配，
    无歧义且不依赖错误文案恰好包含数字。"""
    if status in (401, 403):
        return "auth"
    if status == 404:
        return "not_found"
    if status == 429:
        return "rate_limited"
    if status == 413:
        return "context_overflow"
    if status >= 500:
        return "server_error"
    if status >= 400:
        return "bad_request"
    return ""


async def run_with_resilience(
    send: Callable[[], Awaitable[httpx.Response]],
    *,
    provider: str,
    breaker_key: str,
    attempts: int = RETRY_ATTEMPTS,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
) -> httpx.Response:
    """发送请求并按策略重试、维护熔断。

    流式调用只把"建立连接"包进来：连接建立后的读流失败不经过此处，天然不重试。
    send 用零参可调用对象而非 httpx 客户端：本模块不持有连接池（池与事件循环
    绑定，由 LLMClient 统一管理并处理跨循环重建）。返回成功或重试耗尽后拿到的
    响应（调用方自行判断 >=400 并组织文案）；熔断打开或连接层最终失败抛 LLMError。
    """
    breaker = breaker_for(breaker_key)

    async def _attempt() -> httpx.Response:
        if not breaker.allow():
            raise CircuitOpenError(breaker.name, breaker.retry_after)
        response = await send()
        if response.status_code in RETRYABLE_STATUS:
            raise _RetryableStatus(response)
        return response

    effective_attempts = max(1, attempts)
    try:
        response = await retry_async(
            _attempt,
            attempts=effective_attempts,
            base_delay=RETRY_BASE_DELAY,
            max_delay=RETRY_MAX_DELAY,
            retry_on=(Exception,),
            give_up_on=(CircuitOpenError,),
            on_retry=on_retry,
        )
    except CircuitOpenError as exc:
        raise LLMError(
            f"{provider} 连续失败已触发熔断，暂停调用 {exc.retry_after:.0f} 秒。\n"
            "    为什么这样做：继续重试只会让每次请求都等满超时，把整个流程一起拖慢。\n"
            "    请检查后端是否在运行（本地 `ollama serve`；云端查网络与余额），冷却后自动恢复。"
        ) from exc
    except _RetryableStatus as exc:
        # 重试耗尽仍是可重试状态码：调用方拿响应去组织文案，这里只记一次熔断失败
        breaker.on_failure()
        logger.warning(
            "%s 连续 %d 次返回 HTTP %s", provider, effective_attempts, exc.response.status_code
        )
        return exc.response
    except Exception as exc:
        breaker.on_failure()
        raise LLMError(f"连接 {provider} 失败：{exc}") from exc

    breaker.on_success()
    return response
