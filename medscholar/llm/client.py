"""统一 LLM 客户端，按 llm.provider 路由：ollama（本地默认，支持流式/JSON 模式）、
deepseek、openai-compatible（任意 OpenAI 兼容端点）。

Ollama think 默认关闭（Qwen3 默认思维链又慢又吵）；chat_json 做容错解析
（剥围栏、截取平衡花括号块、容忍尾随逗号），本地小模型几乎不严格输出纯 JSON。
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

from ..constants import (
    HEALTH_REPLY_PREVIEW,
    HTTP_CONNECT_TIMEOUT,
    JSON_CORRECTION_CONTEXT,
    LLM_JSON_RETRIES,
    LLM_MAX_TOKENS_HEALTH,
    LLM_TEMPERATURE_JSON_DEFAULT,
)
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

# 容错解析拆在 json_parsing.py（纯算法、单独测试）；此处重导出旧名字保持对外 API 不变。
# LLMError 定义在 errors.py 以断开 json_parsing ↔ client 的环。
from .json_parsing import extract_json  # noqa: E402  (重导出：对外 API 不变)


@dataclass(slots=True)
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion


class LLMClient(OllamaBackend, OpenAIBackend):
    """按配置路由后端的统一客户端。重试/熔断在 llm.transport，
    耗时与 token 记账在 platform.observability，本类只管报文与错误文案。"""

    def __init__(self, settings: LLMSettings | None = None, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.settings = settings or self.config.llm
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self.usage = _Usage()
        self.calls = 0
        # 熔断按 provider|model|base_url 隔离：换模型不应被前一个模型的连续失败拖累。
        self._breaker_key = f"{self.settings.provider}|{self.settings.model}|{self.settings.base_url}"
        self.breaker = breaker_for(self._breaker_key)

    # ------------------------------------------------------------ 生命周期
    async def __aenter__(self) -> "LLMClient":
        await self.start()
        return self

    async def __aexit__(self, *exc: object) -> None:
        await self.close()

    async def start(self) -> None:
        """建立 HTTP 客户端并做 provider/model 一致性校验。放在此处而非配置加载时，
        保证 Web/CLI/MCP 各路径拿到同一条可照做的中文提示，而非云端英文 400。"""
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
                timeout=httpx.Timeout(self.settings.timeout, connect=HTTP_CONNECT_TIMEOUT),
            )

    async def _ensure_started(self) -> None:
        """自动确保客户端已初始化；并检测事件循环变化——httpx.AsyncClient 连接池
        绑定创建时的循环，跨循环复用会永久挂起（如 worker 线程内 asyncio.run），
        检测到换循环就丢弃旧客户端重建。"""
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
        """记账到全局 LEDGER 与本地累计器，成功失败都记（失败 token=0 但耗时/错误照记，
        否则重试耗时会被统计掩盖）。阶段与 run_id 从当前 trace 上下文推断。

        当调用方没建 trace（P0-4 缺陷的现场）时，从调用栈推断一个 phase 标签：
        取最近一个 agent/ 下的方法名作为 phase、``_record`` 自身的调用者作为 trace_id 的 fallback。
        这避免 observability 里两列长期为空。
        """
        latency_ms = (time.monotonic() - started) * 1000
        if ok:
            self.calls += 1
            self.usage.add(prompt_tokens, completion_tokens)
        span = current_span()
        trace = current_trace()
        phase = span.name if span is not None else ""
        run_id = trace.trace_id if trace is not None else ""
        # 调用栈推断（P0-4 兜底）
        if not phase or not run_id:
            try:
                import inspect as _inspect
                frame = _inspect.currentframe()
                # 跳过 _record 自身 + 客户端方法 + 后端方法
                for _ in range(8):
                    if frame is None:
                        break
                    frame = frame.f_back
                inferred_phase = ""
                if frame is not None:
                    fn = frame.f_code.co_qualname or frame.f_code.co_name
                    # WriterAgent.write_review -> "write_review"；agent.run -> "agent.run"
                    inferred_phase = fn.split(".")[-1]
                if inferred_phase and not phase:
                    phase = inferred_phase
                if frame is not None and not run_id:
                    run_id = f"untraced:{frame.f_code.co_filename.split('medscholar')[-1]}:{frame.f_lineno}"
            except Exception:  # pragma: no cover - 栈推断失败绝不影响记账
                pass
        LEDGER.record(
            LLMUsage(
                provider=self.settings.provider,
                model=self.settings.model,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                latency_ms=latency_ms,
                ok=ok,
                phase=phase,
                run_id=run_id,
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
        retries: int = LLM_JSON_RETRIES,
        expect: str = "object",
    ) -> Any:
        """要求 JSON 并容错解析；解析失败或顶层形状不符合 expect 时把错误回灌模型重试
        （小模型常带解释文字或在嵌套检索式里写坏 JSON）。"""
        history = list(messages)
        last_error: Exception | None = None
        for attempt in range(max(1, retries + 1)):
            text = await self.chat(
                history,
                system=system,
                temperature=temperature if temperature is not None else LLM_TEMPERATURE_JSON_DEFAULT,
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
                    {"role": "assistant", "content": text[:JSON_CORRECTION_CONTEXT]},
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
                max_tokens=LLM_MAX_TOKENS_HEALTH,
                temperature=0.0,
            )
        except LLMError as exc:
            return False, str(exc)
        except Exception as exc:  # pragma: no cover
            return False, f"{type(exc).__name__}: {exc}"
        return True, f"{self.settings.provider}/{self.settings.model} 可用（回复：{text.strip()[:HEALTH_REPLY_PREVIEW]}）"


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
