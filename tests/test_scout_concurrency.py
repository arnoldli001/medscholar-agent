"""Scout 多条检索式必须并发执行，且结果顺序保持确定。

回归背景：规划会给出 3~4 条检索式，而旧实现是一条一条串行 await。
每条检索式内部已经并发查 5~6 个数据源，但检索式之间是纯网络等待，
串行意味着总耗时 = 各条之和，并发后 = 最慢的那条。
"""

from __future__ import annotations

import asyncio

from medscholar.agent.scout import ScoutAgent
from medscholar.agent.state import PlanQuery, ResearchPlan
from medscholar.api import SourceStatus
from medscholar.config import AppConfig
from medscholar.models import Paper


class FakeOutcome:
    def __init__(self, query: str, papers: list[Paper], delay: float = 0.0) -> None:
        self.query = query
        self.papers = papers
        self.raw_count = len(papers)
        self.duration_ms = int(delay * 1000)
        self.statuses = [
            SourceStatus(name="pubmed", label="PubMed", ok=True, count=len(papers))
        ]
        self.errors: dict[str, str] = {}


class FakeRegistry:
    """记录调用顺序与并发峰值。"""

    def __init__(self, *, delay: float = 0.3, fail_on: str = "") -> None:
        self.delay = delay
        self.fail_on = fail_on
        self.calls: list[str] = []
        self._live = 0
        self.peak = 0

    async def search(self, query: str, **kwargs) -> FakeOutcome:
        self.calls.append(query)
        self._live += 1
        self.peak = max(self.peak, self._live)
        try:
            await asyncio.sleep(self.delay)
            if self.fail_on and query == self.fail_on:
                raise RuntimeError("上游 500")
            return FakeOutcome(query, [Paper(title=f"论文-{query}", source="pubmed")])
        finally:
            self._live -= 1


def make_plan(queries: list[str]) -> ResearchPlan:
    return ResearchPlan(
        topic_zh="加速rTMS治疗卒中后抑郁",
        topic_en="accelerated rTMS for post-stroke depression",
        queries=[PlanQuery(query=q, sources=["pubmed"]) for q in queries],
    )


def make_agent(registry: FakeRegistry, tmp_path) -> ScoutAgent:
    from medscholar.db.connect import Database

    config = AppConfig(data_dir=str(tmp_path), offline=False)
    db = Database(tmp_path / "scout.db", config=config)
    return ScoutAgent(config=config, registry=registry, db=db)


class TestConcurrentQueries:
    async def test_queries_run_concurrently(self, tmp_path):
        registry = FakeRegistry(delay=0.3)
        agent = make_agent(registry, tmp_path)
        plan = make_plan(["q1", "q2", "q3"])

        result = await agent.search_plan(plan, persist=False, embed=False, limit=10)

        assert len(result.papers) == 3
        # 用「并发峰值」判定，而不是墙钟：峰值与机器负载无关，墙钟断言在
        # 负载高的机器上会偶发失败（实测同一台机器从 0.3s 抖到 5s）。
        assert registry.peak >= 2, f"并发峰值只有 {registry.peak}，说明仍是串行"
        assert len(registry.calls) == 3

    async def test_result_order_is_deterministic(self, tmp_path):
        """并发不能打乱统计与日志的顺序（否则界面/日志会跳来跳去）。"""
        registry = FakeRegistry(delay=0.05)
        agent = make_agent(registry, tmp_path)
        plan = make_plan(["aaa", "bbb", "ccc"])

        result = await agent.search_plan(plan, persist=False, embed=False, limit=10)
        queries = [s.get("query") for s in result.stats]
        assert queries == ["aaa", "bbb", "ccc"]

    async def test_one_failure_is_isolated(self, tmp_path):
        """单条检索式失败不能中断其余检索式。"""
        registry = FakeRegistry(delay=0.05, fail_on="bbb")
        agent = make_agent(registry, tmp_path)
        plan = make_plan(["aaa", "bbb", "ccc"])

        result = await agent.search_plan(plan, persist=False, embed=False, limit=10)
        assert len(result.papers) == 2, "另外两条检索式的结果必须保留"
        failed = [s for s in result.stats if s.get("query") == "bbb"]
        assert failed and "上游 500" in failed[0].get("error", "")

    async def test_single_query_still_works(self, tmp_path):
        registry = FakeRegistry(delay=0.01)
        agent = make_agent(registry, tmp_path)
        result = await agent.search_plan(
            make_plan(["only"]), persist=False, embed=False, limit=10
        )
        assert len(result.papers) == 1
        assert registry.peak == 1


class TestPlanFallbackQueries:
    async def test_empty_queries_falls_back_to_topic(self, tmp_path):
        registry = FakeRegistry(delay=0.01)
        agent = make_agent(registry, tmp_path)
        plan = ResearchPlan(topic_zh="加速rTMS", topic_en="accelerated rTMS")

        result = await agent.search_plan(plan, persist=False, embed=False, limit=10)
        assert registry.calls == ["accelerated rTMS"]
        assert len(result.papers) == 1
