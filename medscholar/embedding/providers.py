"""向量嵌入提供方。

三种后端，按 ``embedding.provider`` 选择：

============== ==========================================================
``ollama``     本地 Ollama（默认）。零成本、离线可用，768 维与需求文档一致。
``sentence-transformers``
               本地 PubMedBERT（``neuml/pubmedbert-base-embeddings``），
               医学语义更强，但需额外安装 torch（约 2GB）。
``hashing``    纯 Python 哈希嵌入，无任何外部依赖，仅用于离线自检与
               无模型环境下的链路验证，**不具备真实语义检索能力**。
============== ==========================================================

所有提供方都实现同一接口：``dim`` 属性 + ``async embed(texts) -> list[list[float]]``。
"""

from __future__ import annotations

import asyncio
import hashlib
import logging
import math
import re
from abc import ABC, abstractmethod
from typing import Sequence

import httpx

from ..config import AppConfig, EmbeddingSettings, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "EmbeddingProvider",
    "OllamaEmbedding",
    "SentenceTransformerEmbedding",
    "HashingEmbedding",
    "build_provider",
    "get_provider",
    "reset_provider",
]

_TOKEN_RE = re.compile(r"[a-z0-9]+|[\u4e00-\u9fff]")


class EmbeddingProvider(ABC):
    """嵌入提供方接口。"""

    name: str = "base"

    def __init__(self, settings: EmbeddingSettings, *, config: AppConfig | None = None) -> None:
        self.settings = settings
        self.config = config or get_config()

    @property
    def dim(self) -> int:
        return int(self.settings.dim)

    @property
    def model(self) -> str:
        return self.settings.model

    @abstractmethod
    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        """把一批文本转成向量（顺序与输入一致）。"""

    async def embed_one(self, text: str) -> list[float]:
        vectors = await self.embed([text])
        return vectors[0] if vectors else [0.0] * self.dim

    async def probe(self) -> tuple[bool, str]:
        """探测提供方是否可用，返回 (是否可用, 说明)。"""
        try:
            vectors = await self.embed(["健康检查"])
        except Exception as exc:
            return False, f"{type(exc).__name__}: {exc}"
        if not vectors or len(vectors[0]) != self.dim:
            got = len(vectors[0]) if vectors else 0
            return False, f"维度不符：期望 {self.dim}，实际 {got}"
        return True, f"{self.name}/{self.model} 可用（{self.dim} 维）"

    async def close(self) -> None:  # pragma: no cover - 由子类按需覆盖
        return None


class OllamaEmbedding(EmbeddingProvider):
    """通过 Ollama 的 ``/api/embed`` 生成向量。

    .. important::
       **不要跨事件循环复用 httpx.AsyncClient。**
       AsyncClient 的连接池、锁与流都绑定在创建它的那个事件循环上。
       本项目存在"同步上下文里起一个新循环"的路径（``run_embedding_pipeline``
       在 worker 线程中 ``asyncio.run``），如果复用主循环创建的 client，
       请求会**永久挂起**（既不报错也不返回）—— 这是实测踩到的真实故障。
       因此这里每次调用都新建一个短连接客户端：批量嵌入的次数很少，
       连接复用带来的收益远小于正确性风险。
    """

    name = "ollama"

    def _client(self) -> httpx.AsyncClient:
        return httpx.AsyncClient(
            base_url=self.settings.base_url.rstrip("/"),
            timeout=httpx.Timeout(self.settings.timeout, connect=10.0),
        )

    async def list_models(self) -> list[str]:
        """列出 Ollama 已安装的模型（用于给出可操作的错误提示）。"""
        try:
            async with self._client() as client:
                response = await client.get("/api/tags")
                response.raise_for_status()
                return [m.get("name", "") for m in response.json().get("models", [])]
        except Exception as exc:  # pragma: no cover
            logger.debug("列出 Ollama 模型失败：%s", exc)
            return []

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        payload = {
            "model": self.settings.model,
            "input": [t[: self.settings.max_chars] for t in texts],
        }
        try:
            async with self._client() as client:
                response = await asyncio.wait_for(
                    client.post("/api/embed", json=payload),
                    timeout=self.settings.timeout + 30.0,
                )
                if response.status_code == 404:
                    # 旧版 Ollama 没有 /api/embed，退回 /api/embeddings（单条）
                    return await self._embed_legacy(client, texts)
                if response.status_code >= 400:
                    detail = response.text[:200]
                    installed = await self._list_models_with(client)
                    hint = ""
                    if installed and self.settings.model not in installed:
                        hint = (
                            f" 当前已安装：{', '.join(installed[:6])}。"
                            f"请执行 `ollama pull {self.settings.model}`。"
                        )
                    raise RuntimeError(
                        f"Ollama 嵌入失败 HTTP {response.status_code}：{detail}{hint}"
                    )
                data = response.json()
        except asyncio.TimeoutError as exc:
            raise RuntimeError(
                f"Ollama 嵌入超时（{self.settings.timeout + 30:.0f}s）。"
                "可能是模型过大或并发过高，可调小 embedding.batch_size。"
            ) from exc
        except httpx.TransportError as exc:
            raise RuntimeError(
                f"无法连接 Ollama（{self.settings.base_url}）：{exc}。"
                "请确认 `ollama serve` 正在运行。"
            ) from exc

        vectors = data.get("embeddings")
        if not vectors:
            raise RuntimeError(f"Ollama 未返回 embeddings 字段：{str(data)[:200]}")
        return [list(map(float, v)) for v in vectors]

    async def _list_models_with(self, client: httpx.AsyncClient) -> list[str]:
        try:
            response = await client.get("/api/tags")
            return [m.get("name", "") for m in response.json().get("models", [])]
        except Exception:  # pragma: no cover
            return []

    async def close(self) -> None:  # pragma: no cover - 无可释放的常驻资源
        return None

    async def _embed_legacy(
        self, client: httpx.AsyncClient, texts: Sequence[str]
    ) -> list[list[float]]:
        out: list[list[float]] = []
        for text in texts:
            response = await client.post(
                "/api/embeddings",
                json={"model": self.settings.model, "prompt": text[: self.settings.max_chars]},
            )
            response.raise_for_status()
            out.append(list(map(float, response.json()["embedding"])))
        return out


class SentenceTransformerEmbedding(EmbeddingProvider):
    """本地 PubMedBERT（``sentence-transformers``），首次调用时懒加载。"""

    name = "sentence-transformers"

    def __init__(self, settings: EmbeddingSettings, *, config: AppConfig | None = None) -> None:
        super().__init__(settings, config=config)
        self._model = None
        self._lock = asyncio.Lock()

    def _load(self):
        if self._model is None:
            try:
                from sentence_transformers import SentenceTransformer  # type: ignore
            except ImportError as exc:
                raise RuntimeError(
                    "未安装 sentence-transformers。请执行：\n"
                    "    .python\\python.exe -m pip install sentence-transformers\n"
                    "或把 embedding.provider 改为 ollama。"
                ) from exc
            logger.info("加载本地嵌入模型 %s（首次较慢）…", self.settings.model)
            self._model = SentenceTransformer(self.settings.model, trust_remote_code=True)
            actual = int(self._model.get_sentence_embedding_dimension())
            if actual != self.dim:
                logger.warning(
                    "模型实际维度 %d 与配置 %d 不一致，已自动采用 %d", actual, self.dim, actual
                )
                self.settings.dim = actual
        return self._model

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        if not texts:
            return []
        async with self._lock:
            model = await asyncio.to_thread(self._load)
            vectors = await asyncio.to_thread(
                model.encode,
                [t[: self.settings.max_chars] for t in texts],
                batch_size=self.settings.batch_size,
                normalize_embeddings=False,
                show_progress_bar=False,
            )
        return [list(map(float, v)) for v in vectors]


class HashingEmbedding(EmbeddingProvider):
    """确定性哈希嵌入（无外部依赖）。

    用「词元 → 哈希桶」构造稀疏词袋向量，再叠加字符二元组以缓解未登录词。
    它**能反映词面重叠**，适合在无模型环境下跑通链路与做回归测试，
    但**不具备真正的语义泛化能力**，不要用于生产检索质量评估。
    """

    name = "hashing"

    async def embed(self, texts: Sequence[str]) -> list[list[float]]:
        return [self._embed_one(t) for t in texts]

    def _embed_one(self, text: str) -> list[float]:
        dim = self.dim
        vector = [0.0] * dim
        tokens = _TOKEN_RE.findall((text or "").lower())
        if not tokens:
            return vector

        for token in tokens:
            bucket = int(hashlib.blake2b(token.encode("utf-8"), digest_size=8).hexdigest(), 16)
            vector[bucket % dim] += 1.0
            # 字符二元组：让 "rehabilitation"/"rehabilitative" 这类词形变化更接近
            for i in range(len(token) - 1):
                gram = token[i : i + 2]
                gbucket = int(
                    hashlib.blake2b(gram.encode("utf-8"), digest_size=8).hexdigest(), 16
                )
                vector[gbucket % dim] += 0.5

        norm = math.sqrt(sum(v * v for v in vector))
        if norm > 0:
            vector = [v / norm for v in vector]
        return vector


_PROVIDERS: dict[str, type[EmbeddingProvider]] = {
    "ollama": OllamaEmbedding,
    "sentence-transformers": SentenceTransformerEmbedding,
    "hashing": HashingEmbedding,
}

_CACHE: dict[str, EmbeddingProvider] = {}


def build_provider(
    settings: EmbeddingSettings | None = None, *, config: AppConfig | None = None
) -> EmbeddingProvider:
    """按配置创建一个新的嵌入提供方实例。"""
    cfg = config or get_config()
    settings = settings or cfg.embedding
    provider_cls = _PROVIDERS.get(settings.provider)
    if provider_cls is None:
        raise ValueError(
            f"未知的嵌入提供方：{settings.provider}（可选：{', '.join(_PROVIDERS)}）"
        )
    return provider_cls(settings, config=cfg)


def get_provider(config: AppConfig | None = None) -> EmbeddingProvider:
    """获取缓存的嵌入提供方单例。"""
    cfg = config or get_config()
    key = f"{cfg.embedding.provider}:{cfg.embedding.model}:{cfg.embedding.dim}"
    provider = _CACHE.get(key)
    if provider is None:
        provider = build_provider(cfg.embedding, config=cfg)
        _CACHE[key] = provider
    return provider


def reset_provider() -> None:
    """清空提供方缓存（配置变更后调用）。"""
    _CACHE.clear()
