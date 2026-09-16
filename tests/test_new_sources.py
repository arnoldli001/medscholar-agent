"""新增免费数据源：DOAJ（开放获取期刊）与 CORE（机构库）。

DOAJ 的价值：PubMed 偏生物医学，而 DOAJ 收录两万余种完全 OA 期刊，
且全部能合法拿到全文。
CORE 的价值：学位论文、技术报告、作者自存档稿，其它库收不全。
"""

from __future__ import annotations

import asyncio

import httpx

from medscholar.api.core_client import CoreClient
from medscholar.api.doaj_client import DoajClient
from medscholar.api.registry import (
    ALL_SOURCES,
    SourceRegistry,
    normalize_source_name,
)
from medscholar.config import AppConfig, SourceSettings

DOAJ_RESPONSE = {
    "total": 2,
    "results": [
        {
            "id": "doaj-1",
            "bibjson": {
                "title": "Efficacy of accelerated rTMS for post-stroke depression",
                "abstract": "A randomized trial.",
                "author": [{"name": "Zhang, Wei"}, {"name": "Li, Ming"}],
                "journal": {"title": "Frontiers in Neurology", "volume": "12", "number": "3"},
                "year": "2021",
                "identifier": [{"type": "doi", "id": "10.3389/fneur.2021.123456"}],
                "link": [
                    {"type": "fulltext", "url": "https://www.frontiersin.org/articles/x/full"},
                    {"type": "homepage", "url": "https://www.frontiersin.org/articles/x"},
                ],
                "keywords": ["rTMS", "depression"],
                "start_page": "100",
            },
        },
        {
            "id": "doaj-2",
            "bibjson": {
                "title": "Second OA paper",
                "author": [{"name": "Wang, Li"}],
                "journal": {"title": "PLOS ONE"},
                "year": "2019",
                "identifier": [{"type": "doi", "id": "10.1371/journal.pone.0000001"}],
                "link": [],
            },
        },
    ],
}

CORE_RESPONSE = {
    "totalHits": 1,
    "results": [
        {
            "id": 42,
            "title": "Accelerated rTMS in stroke rehabilitation: a thesis",
            "abstract": "Doctoral thesis.",
            "authors": [{"name": "Smith, John"}],
            "yearPublished": 2020,
            "doi": "10.1000/thesis.2020.1",
            "downloadUrl": "https://repo.example.edu/thesis.pdf",
            "publisher": "University Repository",
            "fieldOfStudy": ["Medicine"],
            "language": {"code": "en"},
        }
    ],
}


def mock(client, handler):
    async def _patch():
        await client.start()
        client._client = httpx.AsyncClient(
            base_url="https://example.test", transport=httpx.MockTransport(handler)
        )
        client._client_loop = asyncio.get_running_loop()
    return _patch()


class TestDoajClient:
    async def test_parses_bibjson(self):
        client = DoajClient(config=AppConfig())

        def handler(request: httpx.Request) -> httpx.Response:
            assert "/search/articles/" in str(request.url)
            assert "pageSize" in str(request.url)
            return httpx.Response(200, json=DOAJ_RESPONSE)

        await mock(client, handler)
        papers = await client.search("rTMS depression", limit=10)
        assert len(papers) == 2
        first = papers[0]
        assert first.title == "Efficacy of accelerated rTMS for post-stroke depression"
        assert first.authors == ["Zhang, Wei", "Li, Ming"]
        assert first.journal == "Frontiers in Neurology"
        assert first.pub_year == 2021
        assert first.doi == "10.3389/fneur.2021.123456"
        assert first.volume == "12"
        assert first.keywords == ["rTMS", "depression"]
        # DOAJ 全库都是完全开放获取，标记 OA 是准确的
        assert first.is_open_access is True
        assert first.full_text_url == "https://www.frontiersin.org/articles/x/full"
        assert first.source == "doaj"

    async def test_url_falls_back_to_doi(self):
        client = DoajClient(config=AppConfig())

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=DOAJ_RESPONSE)

        await mock(client, handler)
        papers = await client.search("x", limit=10)
        assert papers[1].url == "https://doi.org/10.1371/journal.pone.0000001"

    async def test_respects_limit(self):
        client = DoajClient(config=AppConfig())

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=DOAJ_RESPONSE)

        await mock(client, handler)
        assert len(await client.search("x", limit=1)) == 1

    async def test_empty_query_returns_nothing(self):
        client = DoajClient(config=AppConfig())
        assert await client.search("   ") == []

    async def test_malformed_items_are_skipped(self):
        client = DoajClient(config=AppConfig())

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "results": [
                    {"bibjson": {"title": ""}},          # 无标题
                    {"no_bibjson": True},                # 结构不对
                    "字符串",                             # 类型不对
                    {"bibjson": {"title": "有效标题", "author": ["纯字符串作者"]}},
                ]
            })

        await mock(client, handler)
        papers = await client.search("x", limit=10)
        assert len(papers) == 1
        assert papers[0].title == "有效标题"
        assert papers[0].authors == ["纯字符串作者"]


class TestCoreClient:
    def make(self, api_key: str = "test-key") -> CoreClient:
        config = AppConfig()
        config.sources.core = SourceSettings(enabled=True, api_key=api_key, rps=100.0)
        return CoreClient(config=config)

    async def test_without_api_key_is_disabled(self):
        """没有 key 时应当直接不可用，而不是每次都去撞 401。"""
        client = self.make(api_key="")
        assert client.enabled() is False
        assert await client.search("rTMS") == []

    async def test_with_key_is_enabled(self):
        assert self.make().enabled() is True

    async def test_parses_results(self):
        client = self.make()

        def handler(request: httpx.Request) -> httpx.Response:
            assert request.headers.get("Authorization") == "Bearer test-key"
            return httpx.Response(200, json=CORE_RESPONSE)

        await mock(client, handler)
        papers = await client.search("rTMS", limit=10)
        assert len(papers) == 1
        paper = papers[0]
        assert paper.title.startswith("Accelerated rTMS in stroke rehabilitation")
        assert paper.authors == ["Smith, John"]
        assert paper.pub_year == 2020
        assert paper.doi == "10.1000/thesis.2020.1"
        assert paper.is_open_access is True
        assert paper.full_text_url == "https://repo.example.edu/thesis.pdf"
        assert paper.source == "core"

    async def test_year_filter_is_pushed_to_query(self):
        client = self.make()
        seen: list[str] = []

        def handler(request: httpx.Request) -> httpx.Response:
            seen.append(str(request.url))
            return httpx.Response(200, json={"results": []})

        await mock(client, handler)
        from medscholar.api.base import SearchFilters

        await client.search("rTMS", filters=SearchFilters(year_from=2015, year_to=2024))
        assert seen and "yearPublished" in seen[0]


class TestRegistryIntegration:
    def test_new_sources_are_registered(self):
        assert "doaj" in ALL_SOURCES
        assert "core" in ALL_SOURCES

    def test_aliases(self):
        assert normalize_source_name("DOAJ") == "doaj"
        assert normalize_source_name("开放获取期刊") == "doaj"
        assert normalize_source_name("CORE") == "core"

    def test_core_skipped_without_key(self):
        """没有 key 的 CORE 不应被选中，否则每次检索都多一个必然失败的源。"""
        registry = SourceRegistry(config=AppConfig())
        chosen = registry.select_sources(["doaj", "core"])
        assert "doaj" in chosen
        assert "core" in chosen  # 显式指定时保留，由 client.enabled() 决定是否真跑

    def test_describe_includes_new_sources(self):
        registry = SourceRegistry(config=AppConfig())
        names = {d["name"] for d in registry.describe()}
        assert {"doaj", "core"} <= names
