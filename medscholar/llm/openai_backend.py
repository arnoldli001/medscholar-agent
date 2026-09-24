"""OpenAI 兼容后端（DeepSeek / vLLM / One-API 等）：报文构造与响应解析。

作为 Mixin 混入 :class:`~medscholar.llm.client.LLMClient`，宿主契约同
:mod:`medscholar.llm.ollama_backend`。

这一套里有一处必须保留的防御：云端推理模型（DeepSeek 的推理系列）把思维链
放在 ``reasoning_content``，与正文共用 ``max_tokens`` 预算。预算不够时
``content`` 会是空字符串——早期版本会静默返回空文本，表现为"综述一个字都没写"
却没有任何报错。所以这里对"空 content + finish_reason=length"单独报错并给出调参建议。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator

from .errors import LLMError
from .transport import failure_kind, http_error_kind, run_with_resilience

logger = logging.getLogger(__name__)


class OpenAIBackend:
    """OpenAI 兼容的 ``/chat/completions`` 实现（含流式）。"""

# --------------------------------------------------- OpenAI 兼容（DeepSeek）
    def _openai_payload(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        *,
        stream: bool,
        ) -> dict[str, Any]:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "temperature": self.settings.temperature if temperature is None else temperature,
            "top_p": self.settings.top_p,
            "max_tokens": max_tokens or self.settings.max_tokens,
            "stream": stream,
        }
        if json_mode:
            payload["response_format"] = {"type": "json_object"}
        return payload
    async def _openai_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        ) -> str:
        payload = self._openai_payload(messages, temperature, max_tokens, json_mode, stream=False)
        started = time.monotonic()
        try:
            response = await run_with_resilience(
                lambda: self.client.post("/chat/completions", json=payload),
                provider=self.settings.provider,
                breaker_key=self._breaker_key,
            )
        except LLMError as exc:
            self._record(
                prompt_tokens=0,
                completion_tokens=0,
                started=started,
                ok=False,
                error_kind=failure_kind(exc),
            )
            raise
        if response.status_code >= 400:
            self._record(
                prompt_tokens=0,
                completion_tokens=0,
                started=started,
                ok=False,
                error_kind=http_error_kind(response.status_code),
            )
            raise LLMError(
                f"{self.settings.provider} 返回 HTTP {response.status_code}：{response.text[:300]}"
            )
        data = response.json()
        usage = data.get("usage") or {}
        self._record(
            prompt_tokens=int(usage.get("prompt_tokens") or 0),
            completion_tokens=int(usage.get("completion_tokens") or 0),
            started=started,
        )
        choices = data.get("choices") or []
        if not choices:
            raise LLMError(f"{self.settings.provider} 未返回 choices：{str(data)[:200]}")

        choice = choices[0]
        message = choice.get("message") or {}
        content = str(message.get("content") or "")
        reasoning = str(message.get("reasoning_content") or message.get("reasoning") or "")
        finish = str(choice.get("finish_reason") or "")

        # 云端推理模型（DeepSeek 的 deepseek-flash / deepseek-v4-pro 都是）会把
        # 思维链放在独立的 reasoning_content 字段里，content 只放最终答案。
        # 如果 max_tokens 不够，模型会把预算全花在思考上，content 直接是空字符串：
        # 早期版本会静默返回空文本，表现为"综述一个字都没写"却没有任何报错。
        if not content.strip():
            if finish == "length":
                raise LLMError(
                    f"{self.settings.provider}/{self.settings.model} 输出为空："
                    f"max_tokens={payload.get('max_tokens')} 全部被思维链消耗掉了。\n"
                    f"    该模型是推理模型，思维链与正文共用 max_tokens 配额。\n"
                    f"    请在 config.yaml 中把 llm.max_tokens 调大（建议 ≥4000），"
                    f"或改用非推理模型。"
                )
            if reasoning.strip():
                raise LLMError(
                    f"{self.settings.provider}/{self.settings.model} 只返回了思维链、没有正文"
                    f"（finish_reason={finish or '未知'}）。请调大 llm.max_tokens 后重试。"
                )
        if reasoning:
            logger.debug("模型返回了 %d 字思维链（已忽略，只用 content）", len(reasoning))
        return content
    async def _openai_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        ) -> AsyncIterator[str]:
        payload = self._openai_payload(messages, temperature, max_tokens, False, stream=True)
        started = time.monotonic()
        # 与 Ollama 流式同理：只在建立连接阶段允许重试，开始吐字后不再重试。
        request = self.client.build_request("POST", "/chat/completions", json=payload)
        try:
            response = await run_with_resilience(
                lambda: self.client.send(request, stream=True),
                provider=self.settings.provider,
                breaker_key=self._breaker_key,
                attempts=2,
            )
        except LLMError as exc:
            self._record(
                prompt_tokens=0,
                completion_tokens=0,
                started=started,
                ok=False,
                error_kind=failure_kind(exc),
            )
            raise
        try:
            if response.status_code >= 400:
                body = await response.aread()
                self._record(
                    prompt_tokens=0,
                    completion_tokens=0,
                    started=started,
                    ok=False,
                    error_kind=http_error_kind(response.status_code),
                )
                raise LLMError(
                    f"{self.settings.provider} 返回 HTTP {response.status_code}："
                    f"{body.decode('utf-8', 'replace')[:300]}"
                )
            async for line in response.aiter_lines():
                line = line.strip()
                if not line.startswith("data:"):
                    continue
                data = line[5:].strip()
                if data == "[DONE]":
                    break
                try:
                    chunk = json.loads(data)
                except json.JSONDecodeError:
                    continue
                for choice in chunk.get("choices") or []:
                    piece = (choice.get("delta") or {}).get("content")
                    if piece:
                        yield piece
                usage = chunk.get("usage")
                # OpenAI 兼容流式默认不返回 usage，只有显式 stream_options 才有；
                # 所以拿不到时记 0 而不是编一个数 —— 宁可少算也不要假数据。
                if usage:
                    self._record(
                        prompt_tokens=int(usage.get("prompt_tokens") or 0),
                        completion_tokens=int(usage.get("completion_tokens") or 0),
                        started=started,
                    )
                    break
        finally:
            await response.aclose()
