"""/api/prisma/flow 的接口测试：用真实检索日志装配 PRISMA 数字。

这个接口的价值不在"能返回 JSON"，而在两件事：
1. 数字来自**真实检索日志**（研究者不必拿 Excel 手工数）；
2. 数字**不自洽时提前拦住** —— 一张对不上的 PRISMA 图交到审稿人手里，
   质疑的是整篇的可信度。
"""

from __future__ import annotations

import httpx
import pytest

from medscholar.db import repo
from medscholar.models import Paper, SearchLogEntry
from medscholar.server.app import app


@pytest.fixture()
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


@pytest.fixture(autouse=True)
def clean_search_logs():
    """清空检索日志，让每个用例的 PRISMA 数字可预测。

    为什么必须清：测试库是**会话级共享的**（`conftest` 把 MEDSCHOLAR_HOME 指向一个
    带 PID 的临时目录，同一进程内所有用例共用）。用例自己插日志又不清理的话，
    第二个用例会看到"前一个用例 + 自己"的数字 —— 这正是我第一次跑这个文件时的失败：
    期望 120 实际 720（6 倍）。测试依赖执行顺序是最经典的 flaky 来源。
    """
    from medscholar.db.connect import get_db

    get_db().execute("DELETE FROM search_logs")
    yield
    get_db().execute("DELETE FROM search_logs")


def _seed_logs() -> None:
    """写入真实的检索日志：两个源，各返回若干条、其中一部分是新入库的。"""
    repo.insert_paper(
        Paper(title="加速rTMS治疗卒中后抑郁", abstract="HAMD 下降", authors=["王伟"], source="pubmed")
    )
    repo.log_search(
        SearchLogEntry(
            query="rTMS post-stroke depression",
            source="pubmed",
            result_count=120,
            new_count=40,
            duration_ms=180,
        )
    )
    repo.log_search(
        SearchLogEntry(
            query="rTMS post-stroke depression",
            source="openalex",
            result_count=88,
            new_count=30,
            duration_ms=220,
        )
    )


class TestPrismaFlowEndpoint:
    async def test_returns_flow_text_and_checklist(self, client):
        _seed_logs()
        response = await client.get("/api/prisma/flow")
        assert response.status_code == 200
        payload = response.json()
        for key in ("flow", "text", "warnings", "checklist", "search_summary", "note"):
            assert key in payload, f"响应缺少 {key}"

    async def test_identified_counts_come_from_real_search_logs(self, client):
        _seed_logs()
        flow = (await client.get("/api/prisma/flow")).json()["flow"]
        assert flow["identified"] == {"pubmed": 120, "openalex": 88}
        assert flow["identified_total"] == 208

    async def test_duplicates_derived_from_logged_new_counts(self, client):
        """重复移除数 = 返回条数 − 新入库条数（日志里能拿到的最接近口径）。"""
        _seed_logs()
        flow = (await client.get("/api/prisma/flow")).json()["flow"]
        assert flow["duplicates_removed"] == 208 - 70
        assert flow["records_after_dedup"] == 70
        assert flow["screened"] == 70

    async def test_manual_counts_are_passed_through(self, client):
        _seed_logs()
        flow = (
            await client.get(
                "/api/prisma/flow",
                params={"included": 12, "excluded_screening": 40, "not_retrieved": 3},
            )
        ).json()["flow"]
        assert flow["excluded_at_screening"] == 40
        assert flow["not_retrieved"] == 3
        assert flow["sought_for_retrieval"] == 30
        assert flow["assessed_for_eligibility"] == 27
        assert flow["included"] == 12

    async def test_inconsistent_numbers_are_flagged(self, client):
        """把数字填成矛盾的是常见操作，接口必须给出可读的中文警告而不是默默接受。"""
        _seed_logs()
        payload = (
            await client.get("/api/prisma/flow", params={"included": 999})
        ).json()
        assert payload["warnings"], "矛盾数字必须被指出"
        assert any("纳入" in w for w in payload["warnings"])
        assert "inconsisten" in payload["text"].lower()

    async def test_text_is_submission_ready_shape(self, client):
        _seed_logs()
        payload = (await client.get("/api/prisma/flow")).json()
        text = payload["text"]
        assert "Identification" in text and "Screening" in text and "Included" in text
        assert "Records identified from pubmed (n = 120)" in text

    async def test_checklist_marks_automatable_items(self, client):
        _seed_logs()
        rows = {row["code"]: row for row in (await client.get("/api/prisma/flow")).json()["checklist"]}
        assert rows["16a"]["status"] == "auto", "PRISMA 流程数字应当被标为可自动填"
        assert rows["11"]["status"] == "manual", "偏倚风险评估必须由研究者决定"

    async def test_empty_library_gives_actionable_warning(self, client):
        """全新用户打开这个接口时，应当被告知"先去检索"，而不是拿到一张全 0 的图。

        检索日志已由 autouse fixture 清空，所以这里确定性地面向"零记录"场景。
        """
        payload = (await client.get("/api/prisma/flow")).json()
        assert payload["flow"]["identified_total"] == 0
        assert any("检索" in w for w in payload["warnings"])
