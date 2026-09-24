"""嵌入层：本地向量生成与增量嵌入管道。

导入时由嵌入层主动向数据层 register_embed_hooks 注册实现，避免数据层反向依赖造成环。
"""

from __future__ import annotations

from ..db.repositories.papers import register_embed_hooks
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


def _embed_one(paper_id: int, database) -> None:
    embed_paper(paper_id, db=database)


def _embed_many(ids, database):
    return run_embedding_pipeline(ids=list(ids), db=database)


#: 注册给数据层的钩子（幂等：模块只会被导入一次）
register_embed_hooks(_embed_one, _embed_many)

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
