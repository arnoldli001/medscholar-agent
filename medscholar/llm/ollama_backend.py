"""Ollama 后端（本地推理）：报文构造、响应解析、分段耗时日志。

作为 Mixin 混入 :class:`~medscholar.llm.client.LLMClient`。宿主必须提供：

* ``self.settings`` —— LLMSettings（model / keep_alive / think / num_ctx …）
* ``self.client`` —— 已建立的 ``httpx.AsyncClient``（属性，未初始化时会抛 LLMError）
* ``self._breaker_key`` —— 熔断器标识
* ``self._resolve_base_url()`` / ``self._record(...)``

拆出来的原因：Ollama 与 OpenAI 兼容这两套后端互不相关，
挤在同一个类里会让"改 DeepSeek 的分支要在 Ollama 的代码里找位置"。
Mixin 让每套后端的实现各自成文件，宿主类只保留生命周期与路由。
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, AsyncIterator, Mapping

import httpx

from .errors import LLMError
from .transport import failure_kind, http_error_kind, run_with_resilience

logger = logging.getLogger(__name__)


class OllamaBackend:
    """Ollama 的 ``/api/chat`` 实现（含流式）。"""

# ------------------------------------------------------------- Ollama
    async def _ollama_chat(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        json_mode: bool,
        *,
        stream: bool,
        ) -> str:
        payload: dict[str, Any] = {
            "model": self.settings.model,
            "messages": messages,
            "stream": stream,
            "think": bool(self.settings.think),
            "keep_alive": self.settings.keep_alive,
            "options": self._ollama_options(temperature, max_tokens),
        }
        if json_mode:
            payload["format"] = "json"

        started = time.monotonic()
        try:
            # 重试/熔断交给 transport；这里只管报文与文案。
            response = await run_with_resilience(
                lambda: self.client.post("/api/chat", json=payload),
                provider=f"Ollama（{self._resolve_base_url()}）",
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
            raise LLMError(
                f"{exc}\n请确认已安装并运行 Ollama：`ollama serve`，"
                f"以及已拉取模型：`ollama pull {self.settings.model}`"
            ) from exc

        if response.status_code >= 400:
            self._record(
                prompt_tokens=0,
                completion_tokens=0,
                started=started,
                ok=False,
                error_kind=http_error_kind(response.status_code),
            )
            raise LLMError(await self._ollama_error(response))

        data = response.json()
        self._record(
            prompt_tokens=int(data.get("prompt_eval_count") or 0),
            completion_tokens=int(data.get("eval_count") or 0),
            started=started,
        )
        self._log_timing(data)
        return str((data.get("message") or {}).get("content") or "")
    def _log_timing(self, data: Mapping[str, Any]) -> None:
        """把 Ollama 的分段耗时写进日志。

        没有这些数字时，「规划很慢」只能靠猜：真正的元凶可能是模型重新加载
        （keep_alive 到期）、显存不足、或系统内存吃紧导致权重页被换出。
        """
        try:
            load = float(data.get("load_duration") or 0) / 1e9
            prompt = float(data.get("prompt_eval_duration") or 0) / 1e9
            gen = float(data.get("eval_duration") or 0) / 1e9
            gen_tokens = int(data.get("eval_count") or 0)
            prompt_tokens = int(data.get("prompt_eval_count") or 0)
            speed = gen_tokens / gen if gen else 0.0
            logger.info(
                "LLM 耗时：加载 %.1fs | 提示 %.1fs(%d tok) | 生成 %.1fs(%d tok, %.1f tok/s)",
                load,
                prompt,
                prompt_tokens,
                gen,
                gen_tokens,
                speed,
            )
            if load > 20:
                # 重新加载模型是纯浪费，且往往比生成本身还慢
                logger.warning(
                    "模型重新加载耗时 %.0fs（keep_alive=%s）。"
                    "把 llm.keep_alive 调大（如 30m）可避免反复加载。",
                    load,
                    self.settings.keep_alive,
                )
            if gen_tokens >= 50 and speed < 10:
                logger.warning(
                    "生成速度仅 %.1f tok/s，远低于本机应有水平（8B 模型在 RTX 4060 上约 45 tok/s）。"
                    "常见原因：系统内存不足导致模型权重被换出、或 GPU 被桌面/浏览器抢占。",
                    speed,
                )
        except Exception:  # pragma: no cover - 遥测失败绝不影响主流程
            logger.debug("耗时统计失败", exc_info=True)
    async def _ollama_stream(
        self,
        messages: list[dict[str, str]],
        temperature: float | None,
        max_tokens: int | None,
        ) -> AsyncIterator[str]:
        payload = {
            "model": self.settings.model,
            "messages": messages,
            "stream": True,
            "think": bool(self.settings.think),
            "keep_alive": self.settings.keep_alive,
            "options": self._ollama_options(temperature, max_tokens),
        }
        started = time.monotonic()
        # 只把"建立连接"这一步交给韧性层：一旦开始吐字就不允许重试
        # （重试会让同一段文字出现两遍），这是流式的取舍。
        request = self.client.build_request("POST", "/api/chat", json=payload)
        try:
            response = await run_with_resilience(
                lambda: self.client.send(request, stream=True),
                provider="Ollama",
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
            raise LLMError(
                f"{exc}。请确认 `ollama serve` 正在运行。"
            ) from exc

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
                    f"Ollama 返回 HTTP {response.status_code}：{body.decode('utf-8', 'replace')[:200]}"
                )
            async for line in response.aiter_lines():
                if not line.strip():
                    continue
                try:
                    chunk = json.loads(line)
                except json.JSONDecodeError:
                    continue
                piece = (chunk.get("message") or {}).get("content") or ""
                if piece:
                    yield piece
                if chunk.get("done"):
                    self._record(
                        prompt_tokens=int(chunk.get("prompt_eval_count") or 0),
                        completion_tokens=int(chunk.get("eval_count") or 0),
                        started=started,
                    )
                    self._log_timing(chunk)
                    break
        finally:
            await response.aclose()
    def _ollama_options(self, temperature: float | None, max_tokens: int | None) -> dict[str, Any]:
        return {
            "temperature": self.settings.temperature if temperature is None else temperature,
            "top_p": self.settings.top_p,
            "num_predict": max_tokens or self.settings.max_tokens,
            "num_ctx": self.settings.num_ctx,
        }
    async def _ollama_error(self, response: httpx.Response) -> str:
        detail = response.text[:300]
        try:
            available = (await self.client.get("/api/tags")).json().get("models", [])
            names = [m.get("name", "") for m in available]
        except Exception:  # pragma: no cover
            names = []
        hint = ""
        if names and self.settings.model not in names:
            hint = (
                f"\n当前已安装模型：{', '.join(names[:8])}\n"
                f"请执行 `ollama pull {self.settings.model}`，"
                f"或把 config.yaml 的 llm.model 改为已安装的模型。"
            )
        return f"Ollama 返回 HTTP {response.status_code}：{detail}{hint}"
