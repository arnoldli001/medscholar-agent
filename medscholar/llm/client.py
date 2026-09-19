"""统一 LLM 客户端。

支持三类后端，通过 ``llm.provider`` 切换：

====================== ====================================================
``ollama``             本地 Ollama（默认）。零成本、可离线，支持流式与 JSON 模式。
``deepseek``           DeepSeek 云端 API（OpenAI 兼容），需 API Key。
``openai-compatible``  任意 OpenAI 兼容端点（vLLM / One-API / 硅基流动 …）。
====================== ====================================================

设计要点：

* 对外只暴露 :meth:`LLMClient.chat` / :meth:`stream` / :meth:`chat_json`；
* Ollama 的 ``think`` 开关默认关闭 —— Qwen3 系列默认会输出大段思维链，
  在流式 UI 里既慢又吵，需要时可通过配置打开；
* :meth:`chat_json` 对模型输出做**容错解析**（剥离 ```json 围栏、截取首个
  平衡的花括号块、容忍尾随逗号），因为本地小模型几乎不会严格输出纯 JSON。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

from ..platform.config import AppConfig, LLMSettings, get_config
from ..platform.observability import (
    LEDGER,
    LLMUsage,
    current_span,
    current_trace,
)
from .errors import LLMError
from .ollama_backend import OllamaBackend
from .openai_backend import OpenAIBackend
from .transport import breaker_for

logger = logging.getLogger(__name__)

__all__ = ["Message", "LLMClient", "LLMError", "get_llm", "reset_llm", "extract_json"]

Message = Mapping[str, str]

# JSON 容错解析已拆到 medscholar/llm/json_parsing.py（纯算法、单独测试）。
# 这里继续重导出 extract_json：它是既有调用方与测试在用的名字。
from .json_parsing import extract_json  # noqa: E402  (重导出：对外 API 不变)
# LLMError 定义在 errors.py（为断开 json_parsing ↔ client 的环），此处重导出。


@dataclass(slots=True)
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion


class LLMClient(OllamaBackend, OpenAIBackend):
    """按配置路由到具体后端的统一客户端。

    出网调用的**重试与熔断**在 :mod:`medscholar.llm.transport`，
    **耗时与 token 记账**在 :mod:`medscholar.platform.observability`；
    本类只负责"报文长什么样、错误文案怎么说"。
    """

    def __init__(self, settings: LLMSettings | None = None, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.settings = settings or self.config.llm
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self.usage = _Usage()
        self.calls = 0
        # 熔断按后端隔离：key 里带上 model，这样"换一个模型"不会被上一个模型的
        # 连续失败拖累（实测场景：qwen3:8b 未拉取导致 400，换成已装模型仍应可用）。
        self._breaker_key = f"{self.settings.provider}|{self.settings.model}|{self.settings.base_url}"
        self.breaker = breaker_for(self._breaker_key)

    # ------------------------------------------------------------ 生命周期
    async def __aenter__(self) -> "LLMClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """建立 HTTP 客户端；顺带做 provider/model 一致性校验。

        校验放在这里而不是配置加载时，是为了让**任何**调用路径（Web / CLI / MCP）
        都能得到同一条可照做的中文提示，而不是各自去撞云端返回的英文 400。
        """
        if self._client is None:
            problem = self.settings.consistency_error()
            if problem:
                raise LLMError(problem)
            base = self._resolve_base_url()
            headers = {"Content-Type": "application/json"}
            if self.settings.provider in {"deepseek", "openai-compatible"}:
                if not self.settings.api_key:
                    raise LLMError(
                        f"{self.settings.provider} 需要 API Key。\n"
                        "    请在 config.yaml 的 llm.api_key 或环境变量 DEEPSEEK_API_KEY 中配置。\n"
                        f"    当前 base_url：{base}"
                    )
                headers["Authorization"] = f"Bearer {self.settings.api_key}"
            self._client = httpx.AsyncClient(
                base_url=base,
                headers=headers,
                timeout=httpx.Timeout(self.settings.timeout, connect=15.0),
            )

    async def _ensure_started(self) -> None:
        """对话方法自动确保已初始化，避免调用方忘记 ``await start()``。

        同时检测事件循环是否变化：``httpx.AsyncClient`` 的连接池绑定在创建它的
        循环上，若被跨循环复用，请求会**永久挂起**（这是实测踩到的真实故障，
        触发路径是在 worker 线程里 ``asyncio.run`` 跑一个同步封装的异步流程）。
        检测到循环变化就丢弃旧客户端并重建。
        """
        loop = asyncio.get_running_loop()
        if self._client is not None and self._client_loop is not loop:
            logger.debug("检测到事件循环变化，重建 LLM HTTP 客户端")
            try:
                await self._client.aclose()
            except Exception:  # pragma: no cover - 旧循环可能已关闭
                pass
            self._client = None
        if self._client is None:
            await self.start()
            self._client_loop = loop

    async def close(self) -> None:
        if self._client is not None:
            await self._client.aclose()
            self._client = None

    @property
    def client(self) -> httpx.AsyncClient:
        if self._client is None:
            raise LLMError("LLM 客户端尚未初始化，请先 await client.start()")
        return self._client

    def _resolve_base_url(self) -> str:
        base = (self.settings.base_url or "").rstrip("/")
        if self.settings.provider == "ollama" and not base:
            base = "http://127.0.0.1:11434"
        if self.settings.provider == "deepseek" and (
            not base or base == "http://127.0.0.1:11434"
        ):
            base = "https://api.deepseek.com/v1"
        if self.settings.provider == "deepseek" and not base.endswith("/v1") and "deepseek" in base:
            base = base + "/v1"
        return base

    def describe(self) -> dict[str, Any]:
        return {
            "provider": self.settings.provider,
            "model": self.settings.model,
            "base_url": self._resolve_base_url(),
            "has_api_key": bool(self.settings.api_key),
            "temperature": self.settings.temperature,
        }

    # ------------------------------------------------------------ 记账
    def _record(
        self,
        *,
        prompt_tokens: int,
        completion_tokens: int,
        started: float,
        ok: bool = True,
        error_kind: str = "",
    ) -> None:
        """记一次调用到全局账本 + 本地累计器。

        **成功与失败都记**：只统计成功调用会得到"平均耗时很漂亮、实际体验很差"的
        假象（慢的往往正是失败重试的那几次）。失败时 token 记 0，
        但耗时与错误分类照记，这样"哪个阶段在烧钱/在超时"才看得出来。

        阶段（plan/execute/reflect/synthesize）与运行号从当前 trace 上下文推断，
        调用方不必为了记账多传参数。账本是纯内存操作，不会拖慢主流程。
        """
        latency_ms = (time.monotonic() - started) * 1000
        if ok:
            self.calls += 1
            self.usage.add(prompt_tokens, completion_tokens)
        span = current_span()
        trace = current_trace()
        LEDGER.record(
            LLMUsage(
                provider=self.settings.provider,
                model=self.settings.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                ok=ok,
                phase=span.name if span is not None else "",
                run_id=trace.trace_id if trace is not None else "",
                error_kind=error_kind,
            )
        )

    # ---------------------------------------------------------------- 对话
    async def chat(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        json_mode: bool = False,
    ) -> str:
        """一次性返回完整回复。"""
        await self._ensure_started()
        payload_messages = self._prepare(messages, system)
        if self.settings.provider == "ollama":
            data = await self._ollama_chat(payload_messages, temperature, max_tokens, json_mode, stream=False)
            return data
        return await self._openai_chat(payload_messages, temperature, max_tokens, json_mode)

    async def stream(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ) -> AsyncIterator[str]:
        """流式产出增量文本。"""
        await self._ensure_started()
        payload_messages = self._prepare(messages, system)
        if self.settings.provider == "ollama":
            async for chunk in self._ollama_stream(payload_messages, temperature, max_tokens):
                yield chunk
        else:
            async for chunk in self._openai_stream(payload_messages, temperature, max_tokens):
                yield chunk

    async def chat_json(
        self,
        messages: Sequence[Message],
        *,
        system: str | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
        retries: int = 2,
        expect: str = "object",
    ) -> Any:
        """要求模型输出 JSON，并做容错解析。

        解析失败、或顶层形状不符合 ``expect`` 时，会把错误回灌给模型重试
        （本地小模型首次输出常带解释性文字，也常在嵌套的检索式里写坏 JSON）。
        四个调用方（规划 / 大纲 / 评审 / 反思）要的都是顶层对象。
        """
        history = list(messages)
        last_error: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            text = await self.chat(
                history,
                system=system,
                temperature=temperature if temperature is not None else 0.1,
                max_tokens=max_tokens,
                json_mode=True,
            )
            try:
                return extract_json(text, expect=expect)
            except LLMError as exc:
                last_error = exc
                logger.debug("JSON 解析失败（第 %d 次），要求模型重试", attempt + 1)
                history = [
                    *history,
                    {"role": "assistant", "content": text[:1500]},
                    {
                        "role": "user",
                        "content": (
                            "上面的输出不是合法 JSON（或顶层类型不对）。请**只**输出 JSON 本身，"
                            "不要任何解释文字、不要 Markdown 代码围栏；"
                            "顶层必须是一个 JSON 对象（以 { 开始、以 } 结束）；"
                            "字符串内部不要出现未转义的英文双引号——"
                            "检索式里需要引号时请改用单引号。"
                        ),
                    },
                ]
        raise LLMError(f"模型连续 {retries + 1} 次未返回合法 JSON：{last_error}")

    def _prepare(self, messages: Sequence[Message], system: str | None) -> list[dict[str, str]]:
        prepared: list[dict[str, str]] = []
        if system:
            prepared.append({"role": "system", "content": system})
        for message in messages:
            role = str(message.get("role", "user"))
            content = str(message.get("content", ""))
            if not content:
                continue
            prepared.append({"role": role, "content": content})
        if not prepared:
            raise LLMError("消息列表为空")
        return prepared


    # ---------------------------------------------------------------- 自检
    async def health(self) -> tuple[bool, str]:
        """探测后端与模型是否可用。"""
        problem = self.settings.consistency_error()
        if problem:
            return False, problem
        try:
            await self.start()
            text = await self.chat(
                [{"role": "user", "content": "回复两个字：可用"}],
                max_tokens=16,
                temperature=0.0,
            )
        except LLMError as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover
            return False, f"{type(exc).__name__}: {exc}"
        return True, f"{self.settings.provider}/{self.settings.model} 可用（回复：{text.strip()[:20]}）"


# --------------------------------------------------------------- 全局单例
_LLM: LLMClient | None = None
_CACHE_KEY: str = ""


def get_llm(config: AppConfig | None = None) -> LLMClient:
    """获取 LLM 客户端单例（配置变更时自动重建）。"""
    global _LLM, _CACHE_KEY
    cfg = config or get_config()
    key = f"{cfg.llm.provider}:{cfg.llm.model}:{cfg.llm.base_url}"
    if _LLM is None or _CACHE_KEY != key:
        _LLM = LLMClient(cfg.llm, config=cfg)
        _CACHE_KEY = key
    return _LLM


def reset_llm() -> None:
    global _LLM, _CACHE_KEY
    _LLM = None
    _CACHE_KEY = ""
