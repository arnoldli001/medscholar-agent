"""Unpaywall 客户端：按 DOI 查合法的开放获取副本。

本地库里大量文献只有 DOI、没有 PMCID，Europe PMC 覆盖不到，也没有
``full_text_url``。Unpaywall 索引了机构库/预印本/期刊的 OA 副本，
按 DOI 一查即可定位免费版本。

只返回开放获取链接，不做认证绕过；API 免费，要求带联系邮箱
（``sources.unpaywall.email``）。

接口：``GET https://api.unpaywall.org/v2/{doi}?email=you@example.com``
文档：https://unpaywall.org/products/api
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

from ..models import Paper
from .base import BaseClient, SearchFilters, SourceError

logger = logging.getLogger(__name__)

__all__ = ["UnpaywallClient", "OALocation", "is_valid_email"]


def is_valid_email(value: str | None) -> bool:
    """Unpaywall 要求真实邮箱；这里只做基本形状校验。

    邮箱为空时不猜不编，直接跳过并提示用户配置 ——
    用假邮箱调用不礼貌，也可能被限流。
    """
    text = (value or "").strip()
    if "@" not in text:
        return False
    local, _, domain = text.partition("@")
    return bool(local) and "." in domain and not domain.endswith(".")


@dataclass(slots=True)
class OALocation:
    """一个开放获取位置。"""

    url_for_pdf: str = ""
    url: str = ""
    host_type: str = ""      # repository / journal
    version: str = ""        # publishedVersion / acceptedVersion / submittedVersion
    license: str = ""
    is_best: bool = False

    @property
    def preferred_url(self) -> str:
        """优先用 PDF 直链；没有就用落地页（落地页里还能再找 PDF 链接）。"""
        return self.url_for_pdf or self.url

    def to_dict(self) -> dict[str, Any]:
        return {
            "url_for_pdf": self.url_for_pdf,
            "url": self.url,
            "host_type": self.host_type,
            "version": self.version,
            "license": self.license,
            "is_best": self.is_best,
        }


class UnpaywallClient(BaseClient):
    """Unpaywall 客户端。

    注意：它不是"检索数据源"，只按 DOI 查 OA 位置，
    因此不注册到 ``SourceRegistry._CLIENT_TYPES``，也不会出现在
    数据源选择里；只有 Reader 取全文时会用到它。
    """

    name = "unpaywall"
    label = "Unpaywall"
    source_id = "unpaywall"
    base_url = "https://api.unpaywall.org/v2"

    async def search(  # pragma: no cover - 不参与关键词检索
        self,
        query: str,
        *,
        limit: int = 20,
        filters: SearchFilters | None = None,
    ) -> list[Paper]:
        """Unpaywall 不支持关键词检索，永远返回空列表。"""
        return []

    def email(self) -> str:
        return (self.settings.email or "").strip()

    async def lookup(self, doi: str) -> dict[str, Any] | None:
        """查询一个 DOI，返回 Unpaywall 原始记录；未收录返回 ``None``。

        404 表示"这个 DOI 不在 Unpaywall 里"，是正常结果，不抛异常。
        """
        clean = (doi or "").strip()
        if not clean:
            return None
        if not is_valid_email(self.email()):
            raise SourceError(
                self.name,
                "未配置有效邮箱，无法调用 Unpaywall。"
                "请在 config.yaml 的 sources.unpaywall.email 填入你的邮箱"
                "（Unpaywall 要求用它来识别调用方，免费且不会发广告）。",
            )
        try:
            data = await self.request(
                "GET",
                f"{self.base_url}/{clean}",
                params={"email": self.email()},
            )
        except SourceError as exc:
            if exc.status == 404:
                logger.debug("Unpaywall 未收录 DOI %s", clean)
                return None
            if exc.status == 422:
                # 实测：填 example.com 这类占位邮箱会被明确拒绝（HTTP 422，
                # 提示 "Please use your own email address"）。Unpaywall 会校验
                # 邮箱真实性，所以这里必须给用户一条能照做的中文提示。
                raise SourceError(
                    self.name,
                    "Unpaywall 拒绝了当前邮箱（HTTP 422）。它要求填**你自己的真实邮箱**"
                    "（不接受 example.com 之类的占位地址）。"
                    "请在 config.yaml 的 sources.unpaywall.email 填入你的邮箱后重试；"
                    "若暂时不想配，可把 sources.unpaywall.enabled 设为 false，"
                    "其余数据源与全文获取不受影响。",
                    status=422,
                ) from exc
            raise
        return data if isinstance(data, dict) else None

    async def best_oa_location(self, doi: str) -> OALocation | None:
        """取该 DOI 的最佳合法 OA 位置；没有开放获取版本时返回 ``None``。"""
        data = await self.lookup(doi)
        if not data or not data.get("is_oa"):
            return None

        best = data.get("best_oa_location")
        if isinstance(best, dict):
            location = self._to_location(best, is_best=True)
            if location.preferred_url:
                return location

        # best_oa_location 缺失时退而求其次：挑第一个有链接的
        for item in data.get("oa_locations") or []:
            if not isinstance(item, dict):
                continue
            location = self._to_location(item, is_best=False)
            if location.preferred_url:
                return location
        return None

    @staticmethod
    def _to_location(raw: dict[str, Any], *, is_best: bool) -> OALocation:
        return OALocation(
            url_for_pdf=str(raw.get("url_for_pdf") or "").strip(),
            url=str(raw.get("url") or "").strip(),
            host_type=str(raw.get("host_type") or "").strip(),
            version=str(raw.get("version") or "").strip(),
            license=str(raw.get("license") or "").strip(),
            is_best=is_best,
        )
