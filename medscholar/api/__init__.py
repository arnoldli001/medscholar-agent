"""学术数据源客户端。

统一入口::

    from medscholar.api import search_all, SearchFilters

    outcome = await search_all("accelerated rTMS post-stroke depression",
                               sources=["pubmed", "europepmc"], limit=30)
    for paper in outcome.papers:
        print(paper.title)

单独使用某个客户端::

    from medscholar.api import EuropePMCClient

    async with EuropePMCClient() as client:
        papers = await client.search("stroke rehabilitation", limit=10)
        text = await client.fulltext(papers[0])
"""

from __future__ import annotations

from .arxiv_client import ArxivClient
from .base import BaseClient, RateLimited, SearchFilters, SourceError, TokenBucket
from .cnki_client import CNKI_STATUS, CnkiClient
from .crossref_client import CrossrefClient
from .europepmc_client import EuropePMCClient
from .openalex_client import OpenAlexClient, reconstruct_abstract
from .pubmed_client import PubMedClient
from .registry import (
    ALL_SOURCES,
    CHINESE_SOURCES,
    DEFAULT_SOURCES,
    SOURCE_ALIASES,
    SearchOutcome,
    SourceRegistry,
    SourceStatus,
    close_registry,
    get_registry,
    normalize_source_name,
    search_all,
)
from .semantic_scholar_client import SemanticScholarClient

__all__ = [
    "BaseClient",
    "SourceError",
    "RateLimited",
    "SearchFilters",
    "TokenBucket",
    "PubMedClient",
    "EuropePMCClient",
    "SemanticScholarClient",
    "OpenAlexClient",
    "CrossrefClient",
    "ArxivClient",
    "CnkiClient",
    "CNKI_STATUS",
    "reconstruct_abstract",
    "SourceRegistry",
    "SourceStatus",
    "SearchOutcome",
    "search_all",
    "get_registry",
    "close_registry",
    "normalize_source_name",
    "SOURCE_ALIASES",
    "DEFAULT_SOURCES",
    "ALL_SOURCES",
    "CHINESE_SOURCES",
]
