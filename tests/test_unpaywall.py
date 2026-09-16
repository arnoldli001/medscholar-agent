"""Unpaywall：按 DOI 找合法 OA 全文。

回归背景：库里大量文献只有 DOI（既无 PMCID 也未被标记 OA），
以前这四种路径都覆盖不到，永远取不到全文。
"""

from __future__ import annotations

import httpx
import pytest

from medscholar.api.base import SourceError
from medscholar.api.unpaywall_client import OALocation, UnpaywallClient, is_valid_email
from medscholar.config import AppConfig, SourceSettings


def make_client(**source_kwargs) -> UnpaywallClient:
    config = AppConfig()
    config.sources.unpaywall = SourceSettings(
        enabled=True, email="researcher@example.edu", rps=100.0, retries=1,
        **source_kwargs,
    )
    return UnpaywallClient(config=config)


def patch(handler):
    """把客户端的 httpx 客户端换成 MockTransport。"""
    async def _patch(client: UnpaywallClient) -> None:
        await client.start()
        client._client = httpx.AsyncClient(
            base_url="https://api.unpaywall.org",
            transport=httpx.MockTransport(handler),
        )
        import asyncio
        client._client_loop = asyncio.get_running_loop()
    return _patch


class TestEmailValidation:
    def test_accepts_normal_email(self):
        assert is_valid_email("a@b.com")
        assert is_valid_email("first.last@fudan.edu.cn")

    def test_rejects_bad_shapes(self):
        for bad in ("", "   ", "noatsign", "@nodomain", "nodot@domain", None):
            assert not is_valid_email(bad), bad

    async def test_lookup_without_email_raises_actionable_error(self):
        config = AppConfig()
        config.sources.unpaywall = SourceSettings(enabled=True, email="")
        client = UnpaywallClient(config=config)
        await client.start()
        with pytest.raises(SourceError, match="未配置有效邮箱"):
            await client.lookup("10.1/x")

    async def test_lookup_without_doi_returns_none(self):
        client = make_client()
        await client.start()
        assert await client.lookup("") is None


class TestBestOALocation:
    async def test_picks_best_pdf(self, monkeypatch):
        client = make_client()

        def handler(request: httpx.Request) -> httpx.Response:
            assert "email=" in str(request.url)
            return httpx.Response(200, json={
                "doi": "10.1/x",
                "is_oa": True,
                "best_oa_location": {
                    "url_for_pdf": "https://repo.example.edu/x.pdf",
                    "url": "https://repo.example.edu/x",
                    "host_type": "repository",
                    "version": "acceptedVersion",
                    "license": "cc-by",
                },
            })

        await patch(handler)(client)
        loc = await client.best_oa_location("10.1/x")
        assert isinstance(loc, OALocation)
        assert loc.preferred_url == "https://repo.example.edu/x.pdf"
        assert loc.host_type == "repository"
        assert loc.is_best is True

    async def test_not_oa_returns_none(self):
        client = make_client()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"doi": "10.1/x", "is_oa": False})

        await patch(handler)(client)
        assert await client.best_oa_location("10.1/x") is None

    async def test_404_is_normal_not_an_exception(self):
        """DOI 不在 Unpaywall 索引里是常见情况，不该当成错误炸出去。"""
        client = make_client()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(404, json={"message": "not found"})

        await patch(handler)(client)
        assert await client.best_oa_location("10.1/nope") is None

    async def test_422_placeholder_email_gives_actionable_message(self):
        """实测：Unpaywall 会拒绝 example.com 这类占位邮箱（HTTP 422）。

        必须给出能照做的中文提示，而不是把原始 JSON 抛给用户。
        """
        client = make_client()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(422, json={
                "HTTP_status_code": 422,
                "error": True,
                "message": "Please use your own email address in API calls.",
            })

        await patch(handler)(client)
        with pytest.raises(SourceError) as info:
            await client.best_oa_location("10.1/x")
        message = str(info.value)
        assert "真实邮箱" in message
        assert "sources.unpaywall.email" in message
        # 要告诉用户可以关掉，不要让它变成阻断性故障
        assert "enabled" in message

    async def test_falls_back_to_oa_locations_when_best_missing(self):
        client = make_client()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "is_oa": True,
                "oa_locations": [
                    {"url": "https://a.example/1", "host_type": "repository"},
                    {"url_for_pdf": "https://b.example/2.pdf", "host_type": "repository"},
                ],
            })

        await patch(handler)(client)
        loc = await client.best_oa_location("10.1/x")
        assert loc is not None
        assert loc.preferred_url == "https://a.example/1"
        assert loc.is_best is False

    async def test_landing_page_only_still_returned(self):
        """只有落地页没有 PDF 直链也要给出来——落地页里还能再找 PDF 链接。"""
        client = make_client()

        def handler(request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={
                "is_oa": True,
                "best_oa_location": {"url": "https://repo.example/x"},
            })

        await patch(handler)(client)
        loc = await client.best_oa_location("10.1/x")
        assert loc is not None
        assert loc.preferred_url == "https://repo.example/x"


class TestReaderIntegration:
    async def test_reader_uses_unpaywall_for_doi_only_paper(self, tmp_path, monkeypatch):
        """只有 DOI 的文献：应当经 Unpaywall 找到 OA 副本并成功入库全text。"""
        from medscholar.agent.reader import ReaderAgent
        from medscholar.db.connect import Database
        from medscholar.models import Paper

        config = AppConfig(data_dir=str(tmp_path), offline=False)
        config.sources.unpaywall = SourceSettings(enabled=True, email="a@b.com", rps=100.0)
        db = Database(tmp_path / "r.db", config=config)

        calls: list[str] = []

        class FakeUnpaywall:
            async def start(self):
                return None

            async def best_oa_location(self, doi):
                calls.append(doi)
                return OALocation(url_for_pdf="https://repo.example/x.pdf", host_type="repository")

        agent = ReaderAgent(config=config, db=db)
        monkeypatch.setattr(agent, "_client", lambda name: _return(FakeUnpaywall()))

        # 让 PDF 下载返回一段可解析文本
        async def fake_fetch_pdf(paper, *, persist):
            from medscholar.agent.reader import FullTextResult
            return FullTextResult(paper.paper_id or 0, "这是正文内容", origin="unpaywall-pdf")

        monkeypatch.setattr(agent, "_fetch_pdf", fake_fetch_pdf)

        paper = Paper(title="Only DOI paper", source="crossref", doi="10.1/only")
        result = await agent.fetch_fulltext(paper, persist=False)
        assert calls == ["10.1/only"], "应当按 DOI 查询 Unpaywall"
        assert result.ok
        assert "正文内容" in result.content


async def _return(value):
    return value
