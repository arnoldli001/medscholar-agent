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
import json
import logging
import re
from dataclasses import dataclass
from typing import Any, AsyncIterator, Mapping, Sequence

import httpx

from ..config import AppConfig, LLMSettings, get_config

logger = logging.getLogger(__name__)

__all__ = ["Message", "LLMClient", "LLMError", "get_llm", "reset_llm", "extract_json"]

Message = Mapping[str, str]


class LLMError(RuntimeError):
    """LLM 调用失败。"""


_FENCE_RE = re.compile(r"```(?:json|JSON)?\s*(.*?)```", re.DOTALL)


def extract_json(text: str, *, expect: str = "any") -> Any:
    """从模型输出中尽力提取 JSON。

    依次尝试：直接解析 → 剥离 Markdown 围栏 → 截取首个平衡的 ``{...}`` 或 ``[...]``
    → 补全被截断的对象 → 修复尾随逗号 / 中文引号后重试。

    ``expect`` 为 ``"object"`` 或 ``"array"`` 时会**只**接受该形状的顶层结果。
    这一点很关键：实测 qwen3:8b 生成 ``queries`` 时写坏过 JSON，而残缺对象里
    第一个配平的 ``[...]`` 恰好是 ``pico.outcomes``；若不限定形状，就会把
    结局指标数组当成整份计划返回，``topic_zh`` / ``queries`` / ``outline`` 全部丢失。
    """
    if not text:
        raise LLMError("模型返回为空，无法解析 JSON")
    text = text.strip()

    last_error: json.JSONDecodeError | None = None
    for candidate in _json_candidates(text):
        parsed: Any = None
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError as exc:
            last_error = exc
            fixed = _repair(candidate)
            if fixed != candidate:
                try:
                    parsed = json.loads(fixed)
                except json.JSONDecodeError:
                    continue
            else:
                continue
        if _matches(parsed, expect):
            return parsed

    want = {"object": "JSON 对象", "array": "JSON 数组"}.get(expect, "JSON")
    detail = f"（{last_error}）" if last_error else ""
    raise LLMError(f"无法从模型输出中解析出{want}{detail}：{text[:400]}")


def _matches(value: Any, expect: str) -> bool:
    """顶层形状是否符合调用方的期望。"""
    if expect == "object":
        return isinstance(value, dict)
    if expect == "array":
        return isinstance(value, list)
    return True


def _leading_opener(text: str) -> str | None:
    """文本自己声明的顶层形状：第一个非空白字符是 ``{`` 还是 ``[``。"""
    for ch in text:
        if ch in "{[":
            return ch
        if not ch.isspace():
            return None
    return None


def _close_truncated(fragment: str) -> str | None:
    """补全被截断的 JSON：退到最后一个完整的值，再补上未闭合的括号。

    本地小模型偶尔会在生成到一半时陷入空白循环，把 token 预算烧完（实测
    qwen3:8b 在 ``"queries"`` 里输出 ``"query": "("`` 之后就只剩换行）。
    这时整个对象虽然不合法，但前面已经生成好的 ``topic_zh`` / ``pico``
    都是完好的，值得捞回来。
    """
    stack: list[str] = []
    in_string = False
    escaped = False
    cut = -1  # 最后一个完整值的结束位置
    for index, ch in enumerate(fragment):
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
                cut = index + 1
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append("}" if ch == "{" else "]")
        elif ch in "}]":
            if not stack:
                return None
            stack.pop()
            if not stack:
                return None  # 顶层已经闭合，说明不是截断
            cut = index + 1
        elif ch == "," and stack:
            cut = index
    if in_string or not stack or cut <= 0:
        return None
    # 回退到最后一个完整值，去掉悬空的逗号与半截成员
    repaired = fragment[:cut].rstrip().rstrip(",")
    return repaired + "".join(reversed(stack))


def _json_candidates(text: str) -> list[str]:
    bases = [text]
    fenced = _FENCE_RE.search(text)
    if fenced:
        bases.append(fenced.group(1).strip())

    # 只按文本自己声明的形状找块：以 `{` 开头就只认对象。否则残缺对象里第一个
    # 配平的 `[...]` 会被当成答案（见 extract_json 的说明）。
    lead = _leading_opener(text)
    if lead == "{":
        pairs = [("{", "}")]
    elif lead == "[":
        pairs = [("[", "]")]
    else:
        pairs = [("{", "}"), ("[", "]")]

    candidates: list[str] = list(bases)
    for base in bases:
        for opener, closer in pairs:
            block = _balanced_block(base, opener, closer)
            if block:
                candidates.append(block)
                continue  # 已配平，无需再尝试补全
            start = base.find(opener)
            if start >= 0:
                closed = _close_truncated(base[start:])
                if closed:
                    candidates.append(closed)

    seen: set[str] = set()
    unique: list[str] = []
    for item in candidates:
        if item and item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def _balanced_block(text: str, opener: str, closer: str) -> str | None:
    """截取第一个括号配平的块（跳过字符串字面量内的括号）。"""
    start = text.find(opener)
    if start < 0:
        return None
    depth = 0
    in_string = False
    escaped = False
    for index in range(start, len(text)):
        ch = text[index]
        if in_string:
            if escaped:
                escaped = False
            elif ch == "\\":
                escaped = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch == opener:
            depth += 1
        elif ch == closer:
            depth -= 1
            if depth == 0:
                return text[start : index + 1]
    return None


def _repair(text: str) -> str:
    """修复常见 JSON 瑕疵：尾随逗号、中文引号。"""
    repaired = re.sub(r",\s*([}\]])", r"\1", text)
    repaired = repaired.replace("“", '"').replace("”", '"')
    return repaired


@dataclass(slots=True)
class _Usage:
    prompt_tokens: int = 0
    completion_tokens: int = 0

    def add(self, prompt: int, completion: int) -> None:
        self.prompt_tokens += prompt
        self.completion_tokens += completion


class LLMClient:
    """按配置路由到具体后端的统一客户端。"""

    def __init__(self, settings: LLMSettings | None = None, *, config: AppConfig | None = None) -> None:
        self.config = config or get_config()
        self.settings = settings or self.config.llm
        self._client: httpx.AsyncClient | None = None
        self._client_loop: asyncio.AbstractEventLoop | None = None
        self.usage = _Usage()
        self.calls = 0

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

        try:
            response = await self.client.post("/api/chat", json=payload)
        except httpx.TransportError as exc:
            raise LLMError(
                f"无法连接 Ollama（{self._resolve_base_url()}）：{exc}\n"
                "请确认已安装并运行 Ollama：`ollama serve`，"
                f"以及已拉取模型：`ollama pull {self.settings.model}`"
            ) from exc

        if response.status_code >= 400:
            raise LLMError(await self._ollama_error(response))

        data = response.json()
        self.calls += 1
        self.usage.add(
            int(data.get("prompt_eval_count") or 0), int(data.get("eval_count") or 0)
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
        try:
            async with self.client.stream("POST", "/api/chat", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
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
                        self.calls += 1
                        self.usage.add(
                            int(chunk.get("prompt_eval_count") or 0),
                            int(chunk.get("eval_count") or 0),
                        )
                        break
        except httpx.TransportError as exc:
            raise LLMError(
                f"连接 Ollama 失败：{exc}。请确认 `ollama serve` 正在运行。"
            ) from exc

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
        try:
            response = await self.client.post("/chat/completions", json=payload)
        except httpx.TransportError as exc:
            raise LLMError(f"连接 {self.settings.provider} 失败：{exc}") from exc
        if response.status_code >= 400:
            raise LLMError(
                f"{self.settings.provider} 返回 HTTP {response.status_code}：{response.text[:300]}"
            )
        data = response.json()
        self.calls += 1
        usage = data.get("usage") or {}
        self.usage.add(
            int(usage.get("prompt_tokens") or 0), int(usage.get("completion_tokens") or 0)
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
        # 如果 max_tokens 不够，模型会把预算全花在思考上，content 直接是空字符串 ——
        # 早期版本会**静默返回空文本**，表现为"综述一个字都没写"却没有任何报错。
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
        try:
            async with self.client.stream("POST", "/chat/completions", json=payload) as response:
                if response.status_code >= 400:
                    body = await response.aread()
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
        except httpx.TransportError as exc:
            raise LLMError(f"连接 {self.settings.provider} 失败：{exc}") from exc

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
