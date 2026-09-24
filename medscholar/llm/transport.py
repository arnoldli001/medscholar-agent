"""LLM 出网调用的韧性层：重试与熔断。

重试、熔断、限流不是"某个厂商的 API 细节"，而是所有出网调用共有的横切关注点。
写在 ``_ollama_chat`` 和 ``_openai_chat`` 里会重复两遍，以后每加一个后端就要再抄一遍。

分工：

* 本模块负责"失败了怎么办"；
* :mod:`medscholar.llm.client` 负责"报文长什么样、错误文案怎么说"；
* :mod:`medscholar.platform.observability` 负责"花了多少、慢在哪里"。

两类重试必须分开（踩过的坑）：

1. 传输层重试（本模块）：网络抖动、429、5xx，重发同一个请求是安全的；
2. 语义层重试（``LLMClient.chat_json``）：模型返回的 JSON 不合法，
   需要把错误回灌给模型重新生成。

混在一起的话，一次坏 JSON 会连带把网络层也重试一遍：云端按 token 计费就是
成本 ×N，本地模型则是白等几十秒。

熔断的计数粒度：按"调用"而不是按"尝试"

一个容易写错的地方：如果在每次尝试失败时都记一次熔断失败，那么
"重试第 2 次成功"的调用也会留下失败记录，5 次网络抖动就能把健康的熔断器打开，
接下来 30 秒所有请求被误伤。所以这里只在整次调用最终失败时记一次。
（单测构造过这个场景：attempts=3 且第 3 次成功，熔断器必须仍是 CLOSED。）

流式为什么不重试：

流式一开始吐字，用户屏幕上就有内容了。此时重试会让同一段文字出现两遍，
而且两遍内容还不一样。所以流式只在建立连接阶段（还没有 yield 任何字节）
允许重试；一旦开始读流就只报错。取舍是明确的：用户能接受一次失败后点重试，
不能接受生成的综述里同一段话出现两次。
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

#: 退避参数。抽成模块级常量，测试可以把它压到 0 —— 否则"连续 5 次失败"的用例
#: 要真等 12 秒（每次调用 0.8+1.6），而慢测试的下场一定是被跳过或被无脑重跑。
RETRY_BASE_DELAY = 0.8
RETRY_MAX_DELAY = 8.0

#: 每个后端一个熔断器：key = "provider|model|base_url"
_BREAKERS: dict[str, CircuitBreaker] = {}


def breaker_for(key: str) -> CircuitBreaker:
    """取得（或创建）某个后端的熔断器。

    按后端隔离而不是全局共用一个：本地 Ollama 挂了不该影响云端 DeepSeek 的调用，
    反之亦然。共用一个熔断器会把局部故障升级成"整个 LLM 不可用"。
    """
    breaker = _BREAKERS.get(key)
    if breaker is None:
        # 阈值 5：单次失败很常见（模型加载、网络抖动），连续 5 次才判定后端真的坏了。
        # 冷却 30s：足够让 `ollama serve` 重启或云端限流窗口过去，又不至于让用户等太久。
        breaker = CircuitBreaker(f"llm:{key}", failure_threshold=5, reset_timeout=30.0)
        _BREAKERS[key] = breaker
    return breaker


def breaker_stats() -> dict[str, dict]:
    """所有后端的熔断状态（给 /api/metrics 用）。"""
    return {key: breaker.stats() for key, breaker in _BREAKERS.items()}


def reset_breakers() -> None:
    """清空所有熔断器（配置变更或测试用）。"""
    _BREAKERS.clear()


class _RetryableStatus(Exception):
    """内部信号：HTTP 状态码属于可重试集合。

    用异常表达是为了复用 :func:`retry_async` 的重试循环（它只认异常），
    同时把响应对象带出来，交给调用方组织面向用户的错误文案。
    """

    def __init__(self, response: httpx.Response) -> None:
        super().__init__(f"HTTP {response.status_code}")
        self.response = response


def failure_kind(exc: BaseException) -> str:
    """把异常归类到统一的失败分类（供账本与面板统计）。

    比 :func:`observability.classify_failure` 多认两种 LLM 特有的情况：
    熔断打开、以及"HTTP 状态码属于可重试集合但重试耗尽了"。
    没有分类就无法做告警："失败 37 次" 对定位问题毫无帮助，
    "429 占 30 次" 才直接指向限流。
    """
    from ..platform.observability import classify_failure

    if isinstance(exc, CircuitOpenError):
        return "circuit_open"
    if isinstance(exc, _RetryableStatus):
        return http_error_kind(exc.response.status_code)
    return classify_failure(exc) or "unknown"


def http_error_kind(status: int) -> str:
    """把 HTTP 状态码映射到与 :func:`failure_kind` 同一套分类名。

    不复用字符串匹配的 :func:`classify_failure`：状态码本身信息完整且无歧义，
    走文本匹配反而要依赖"错误文案里恰好含 401"这种巧合。
    面板上"认证失败 12 次"和"限流 30 次"是两种完全不同的处置动作，
    分类必须准确。
    """
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
    """发送一次请求，按策略重试，并维护熔断状态。

    对流式调用，调用方只把"建立连接"这一步包进来：一旦连接建立，
    读流阶段的失败不会再经过这里，因此天然不会重试（避免重复文本）。

    Args:
        send: 真正发请求的零参可调用对象。用可调用对象而不是 httpx 客户端，
            是为了让本模块不持有连接池：连接池的生命周期与事件循环绑定，
            由 :class:`~medscholar.llm.client.LLMClient` 统一管理
            （那里已有"跨事件循环重建客户端"的处理，见 httpx 连接池绑循环的坑）。
        breaker_key: 后端标识（``provider|model|base_url``），用于隔离熔断状态。

    Returns:
        成功（HTTP < 400）或"重试耗尽但拿到了响应"时的 :class:`httpx.Response`。
        调用方需要自己判断 ``status_code >= 400`` 并组织错误文案：
        只有调用方知道该提示"跑 ollama pull"还是"检查 API Key"。

    Raises:
        LLMError: 熔断打开、或连接层最终失败（网络错误/超时）。
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
