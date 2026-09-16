"""嵌入层：本地向量生成与增量嵌入管道。"""

from __future__ import annotations

from .pipeline import (
    EmbeddingReport,
    embed_paper,
    embed_papers,
    embed_query,
    embedding_status,
    run_embedding_pipeline,
    run_embedding_pipeline_async,
)
from .providers import (
    EmbeddingProvider,
    HashingEmbedding,
    OllamaEmbedding,
    SentenceTransformerEmbedding,
    build_provider,
    get_provider,
    reset_provider,
)

__all__ = [
    "EmbeddingReport",
    "embed_paper",
    "embed_papers",
    "embed_query",
    "embedding_status",
    "run_embedding_pipeline",
    "run_embedding_pipeline_async",
    "EmbeddingProvider",
    "OllamaEmbedding",
    "SentenceTransformerEmbedding",
    "HashingEmbedding",
    "build_provider",
    "get_provider",
    "reset_provider",
]
