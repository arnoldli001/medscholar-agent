"""学术 API 客户端公共基础：限流、重试、错误类型与统一接口。

所有数据源客户端都继承 :class:`BaseClient`，对上层只暴露 ``search()`` /
``references()`` / ``citations()`` / ``fulltext()`` 四个语义化方法，
因此 Scout Agent 不需要知道任何 HTTP 细节。
"""

from __future__ import annotations

import asyncio
import logging
import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Mapping

import httpx

from ..config import AppConfig, SourceSettings, get_config
from ..models import Paper

logger = logging.getLogger(__name__)

__all__ = [
    "SourceError",
    "RateLimited",
    "SearchFilters",
    "TokenBucket",
    "BaseClient",
    "USER_AGENT",
]

USER_AGENT = "MedScholarAgent/1.0 (academic research tool; +https://github.com/medscholar)"


class SourceError(RuntimeError):
    """某个数据源检索失败。携带数据源名，便于上层降级而不是整体失败。"""

    def __init__(self, source: str, message: str, *, status: int | None = None) -> None:
        super().__init__(f"[{source}] {message}")
        self.source = source
        self.message = message
        self.status = status


class RateLimited(SourceError):
    """被数据源限流（HTTP 429）。"""


@dataclass(slots=True)
class SearchFilters:
    """跨数据源统一的检索过滤条件。各客户端自行翻译为各自的参数语法。"""

    year_from: int | None = None
    year_to: int | None = None
    open_access_only: bool = False
    publication_types: list[str] = field(default_factory=list)
    language: str | None = None
    sort: str = "relevance"  # relevance | date | citations
    extra: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "year_from": self.year_from,
            "year_to": self.year_to,
            "open_access_only": self.open_access_only,
            "publication_types": list(self.publication_types),
            "language": self.language,
            "sort": self.sort,
        }


class TokenBucket:
    """异步令牌桶限流器。

    数据源的公开速率限制是**硬约束**（超限会被封或降级），因此这里按
    ``SourceSettings.rps`` 严格控制；桶容量固定为 1，避免突发流量触发 429。
    """

    def __init__(self, rps: float) -> None:
        self.rps = max(0.01, float(rps))
        self._interval = 1.0 / self.rps
        self._next_allowed = 0.0
        self._lock = asyncio.Lock()

    async def acquire(self) -> None:
        async with self._lock:
            now = time.monotonic()
            wait = self._next_allowed - now
            if wait > 0:
                await asyncio.sleep(wait)
                now = time.monotonic()
            self._next_allowed = now + self._interval

    def update_rps(self, rps: float) -> None:
        self.rps = max(0.01, float(rps))
        self._interval = 1.0 / self.rps


class BaseClient(ABC):
    """学术数据源客户端基类。

    子类需要实现 :meth:`search`；其余方法默认不可用（返回空），
    由具体客户端按能力覆盖。
    """

    #: 数据源短名，需与 ``SourceSettings`` 中的字段名一致（``s2`` 除外）
    name: str = "base"
    #: 展示名
    label: str = "Base"
    #: 检索时使用的来源标识（写入 ``papers.source``）
    source_id: str = "base"
    #: 基础 URL
    base_url: str = ""

    def __init__(
        self,
        settings: SourceSettings | None = None,
        *,
        config: AppConfig | None = None,
        client: httpx.AsyncClient | None = None,
    ) -> None:
        self.config = config or get_config()
        self.settings = settings or self.config.sources.get(self.name)
        self.bucket = TokenBucket(self.settings.rps)
        self._client = client
        self._owns_client = client is None
        self._client_loop: asyncio.AbstractEventLoop | None = None

    # ------------------------------------------------------------ 生命周期
    async def __aenter__(self) -> "BaseClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        if self._client is None:
            self._client = httpx.AsyncClient(
                timeout=httpx.Timeout(self.settings.timeout, connect=15.0),
                headers={"User-Agent": USER_AGENT, "Accept": "application/json"},
                follow_redirects=True,
                limits=httpx.Limits(max_connections=8, max_keepalive_connections=4),
            )
            self._owns_client = True
            self._client_loop = asyncio.get_running_loop()

    async def _ensure_client(self) -> None:
        """确保客户端存在且属于**当前**事件循环。

        httpx.AsyncClient 连接池绑定创建它的循环；跨循环复用会让请求永久挂起。
        客户端会被注册表长期缓存，因此每次请求前都校验一次循环归属。
        """
        loop = asyncio.get_running_loop()
        if self._client is not None and self._owns_client and self._client_loop is not loop:
            logger.debug("%s：事件循环变化，重建 HTTP 客户端", self.name)
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - 旧循环可能已关闭
                pass
            self._client = None
        if self._client is None:
            await self.start()

    async def close(self) -> None:
        if self._owns_client and self._client is not None:
            await self._client.aclose()
            self._client = None
            self._owns_client = False

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise SourceError(self.name, "HTTP 客户端尚未初始化，请先 await client.start()")
        return self._client

    # ------------------------------------------------------------ HTTP 封装
    async def request(
        self,
        method: str,
        url: str,
        *,
        params: Mapping[str, Any] | None = None,
        headers: Mapping[str, str] | None = None,
        expect: str = "json",
    ) -> Any:
        """带令牌桶限流 + 指数退避重试的请求。

        * 429 / 5xx / 网络错误 → 指数退避重试（含抖动），并尊重 ``Retry-After``
        * 4xx（除 429）→ 立即抛出 :class:`SourceError`，不浪费时间重试
        """
        clean_params = {k: v for k, v in (params or {}).items() if v is not None}
        last_error: Exception | None = None
        await self._ensure_client()

        # 文本类端点（JATS XML / Atom / HTML）的内容协商很严格：
        # 默认的 `Accept: application/json` 会被 Europe PMC 直接判为 406 Not Acceptable。
        # 实测同一个 URL：
        #     Accept: application/json → 406
        #     Accept: text/xml         → 406
        #     Accept: application/xml  → 200
        #     Accept: */*              → 200
        # 这个坑曾让**整条开放获取全文管道全部失效**（99 篇有 PMCID 的一篇都取不到），
        # 因此对非 JSON 请求统一改用 `*/*`。
        send_headers = dict(headers or {})
        if expect != "json":
            send_headers.setdefault("Accept", "*/*")

        for attempt in range(max(1, self.settings.retries)):
            await self.bucket.acquire()
            try:
                response = await self.client.request(
                    method, url, params=clean_params, headers=send_headers
                )
            except (httpx.TimeoutException, httpx.TransportError) as exc:
                last_error = SourceError(self.name, f"网络错误：{exc}")
                delay = self._backoff(attempt)
                logger.debug("%s 第 %d 次请求失败（%s），%.1fs 后重试", self.name, attempt + 1, exc, delay)
                await asyncio.sleep(delay)
                continue

            if response.status_code == 429:
                retry_after = _parse_retry_after(response.headers.get("Retry-After"))
                delay = max(retry_after, self._backoff(attempt))
                last_error = RateLimited(self.name, "触发速率限制（HTTP 429）", status=429)
                logger.debug("%s 被限流，%.1fs 后重试", self.name, delay)
                await asyncio.sleep(delay)
                continue

            if response.status_code >= 500:
                last_error = SourceError(
                    self.name, f"服务端错误 HTTP {response.status_code}", status=response.status_code
                )
                await asyncio.sleep(self._backoff(attempt))
                continue

            if response.status_code >= 400:
                raise SourceError(
                    self.name,
                    f"HTTP {response.status_code}：{response.text[:200]}",
                    status=response.status_code,
                )

            if expect == "json":
                try:
                    return response.json()
                except ValueError as exc:
                    raise SourceError(self.name, f"响应不是合法 JSON：{exc}") from exc
            return response.text

        raise last_error or SourceError(self.name, "请求失败且无可用错误信息")

    def _backoff(self, attempt: int) -> float:
        base = max(0.05, float(self.settings.backoff))
        return base * (2**attempt) + random.uniform(0.0, base * 0.4)

    # -------------------------------------------------------------- 语义接口
    @abstractmethod
    async def search(
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        """按关键词检索，返回规范化后的 :class:`Paper` 列表。"""

    async def references(self, paper: Paper) -> list[dict[str, Any]]:
        """该文献的参考文献（支持者少，默认空）。"""
        return []

    async def citations(self, paper: Paper) -> list[dict[str, Any]]:
        """引用该文献的文献（支持者少，默认空）。"""
        return []

    async def fulltext(self, paper: Paper) -> str:
        """开放获取全文纯文本；不支持时返回空串。"""
        return ""

    # ---------------------------------------------------------------- 工具
    def enabled(self) -> bool:
        return bool(self.settings.enabled)

    def describe(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "label": self.label,
            "source": self.source_id,
            "enabled": self.enabled(),
            "has_api_key": bool(self.settings.api_key),
            "rps": self.settings.rps,
            "base_url": self.base_url,
        }


def _parse_retry_after(value: str | None) -> float:
    """解析 ``Retry-After``（秒数或 HTTP 日期），失败返回 0。"""
    if not value:
        return 0.0
    value = value.strip()
    try:
        return max(0.0, float(value))
    except ValueError:
        pass
    try:
        from email.utils import parsedate_to_datetime

        target = parsedate_to_datetime(value)
        return max(0.0, target.timestamp() - time.time())
    except (TypeError, ValueError):
        return 0.0
