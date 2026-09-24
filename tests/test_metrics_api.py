"""/api/metrics 与 /api/metrics/schema 的接口契约测试。

为什么值得单独测：这两个接口是**可观测性的出口**，一旦字段名漂移，
前端的指标面板会静默显示空白（不报错），而"指标看起来是 0"和"真的没有调用"
在界面上长得一模一样 —— 这种失败最难发现。

用 ``ASGITransport`` 而不是真实 uvicorn：这两个接口是普通 JSON，
不像 SSE 那样会被 ASGI 传输缓冲（``scripts/smoke_http.py`` 里已说明为什么 SSE 必须打真实服务）。
"""

from __future__ import annotations

import httpx
import pytest

from medscholar.server.app import app


@pytest.fixture()
async def client():
    transport = httpx.ASGITransport(app=app)
    async with httpx.AsyncClient(transport=transport, base_url="http://test") as http:
        yield http


class TestMetricsEndpoint:
    async def test_metrics_returns_all_sections(self, client):
        response = await client.get("/api/metrics")
        assert response.status_code == 200
        payload = response.json()
        for section in ("llm", "llm_recent", "breakers", "caches", "injection", "uptime_s"):
            assert section in payload, f"/api/metrics 缺少 {section} 字段"

    async def test_llm_summary_exposes_cost_and_latency(self, client):
        """这三个数字是"花了多少钱 / 慢在哪 / 为什么失败"的答案，字段名不能漂移。"""
        summary = (await client.get("/api/metrics")).json()["llm"]
        for key in (
            "calls",
            "ok_calls",
            "failed_calls",
            "prompt_tokens",
            "completion_tokens",
            "total_tokens",
            "cost_yuan",
            "latency_ms_p50",
            "latency_ms_p95",
            "by_model",
            "by_phase",
            "by_error_kind",
        ):
            assert key in summary, f"账本摘要缺少 {key}"

    async def test_metrics_does_not_leak_content(self, client):
        """指标接口会被贴到 issue/群里求助，绝不能顺手泄漏文献内容或提示词正文。

        断言方式是**结构性的**：遍历返回的 JSON，任何字符串值都不该超过 120 字符。
        指标只可能是短标签（模型名、阶段名、失败类型、缓存名），
        出现长文本就说明把摘要、材料块或提示词正文带出来了 ——
        这比逐个 grep 敏感词更可靠（新字段加进来时也守得住）。
        """
        payload = (await client.get("/api/metrics")).json()
        too_long: list[tuple[str, int]] = []

        def walk(node, path: str = "$") -> None:
            if isinstance(node, dict):
                for key, value in node.items():
                    walk(value, f"{path}.{key}")
            elif isinstance(node, list):
                for index, value in enumerate(node):
                    walk(value, f"{path}[{index}]")
            elif isinstance(node, str) and len(node) > 120:
                too_long.append((path, len(node)))

        walk(payload)
        assert not too_long, f"指标里出现了长文本，疑似泄漏内容：{too_long}"

        text = (await client.get("/api/metrics")).text.lower()
        assert "abstract" not in text and "untrusted" not in text
        assert "sk-" not in text, "绝不能出现疑似密钥"

    async def test_usage_is_recorded_into_the_ledger(self, client):
        """记账链路必须真的接上：用一个新的用量记录驱动，然后在接口里看到它。"""
        from medscholar.platform.observability import LEDGER, LLMUsage

        before = (await client.get("/api/metrics")).json()["llm"]["calls"]
        LEDGER.record(
            LLMUsage(
                provider="ollama",
                model="qwen3:8b",
                prompt_tokens=100,
                completion_tokens=50,
                latency_ms=123.0,
                phase="plan",
            )
        )
        after = (await client.get("/api/metrics")).json()
        assert after["llm"]["calls"] == before + 1
        assert after["llm"]["total_tokens"] >= 150
        assert after["llm_recent"], "最近调用列表不能为空"
        assert after["llm_recent"][-1]["phase"] == "plan"


class TestSchemaStatusEndpoint:
    async def test_schema_status_is_readable(self, client):
        response = await client.get("/api/metrics/schema")
        assert response.status_code == 200
        payload = response.json()
        # 迁移框架未启用时也必须返回结构化结果，而不是 500：
        # 监控系统打到一个 500 会一直告警，而真实原因只是"这个库还没迁移过"。
        assert isinstance(payload, dict)
        assert "available" in payload or "current_version" in payload


class TestMetricsPanelWiring:
    """界面上的「运行指标」面板必须真的接到这个接口上。

    为什么值得测：面板与接口之间是**字符串约定**（标签名、容器 id、请求路径）。
    任何一端改名都不会报错 —— 界面上只会静静显示"加载中…"或空白，
    而"指标确实是 0"和"接口没接上"在屏幕上长得一模一样。
    这类断裂靠人眼 review 基本抓不住，所以用测试钉住三件事：
    ① 首页有那个标签页；② 有对应容器；③ 前端真的去请求 /api/metrics。
    """

    async def test_index_has_metrics_tab_and_container(self, client):
        response = await client.get("/")
        assert response.status_code == 200
        html = response.text
        assert 'data-mtab="metrics"' in html, "设置弹窗里缺少「运行指标」标签"
        assert 'data-mpane="metrics"' in html, "缺少对应的面板容器"
        assert 'id="metricsDetail"' in html, "缺少渲染容器（前端会找不到挂载点）"

    async def test_frontend_actually_calls_the_endpoint(self, client):
        response = await client.get("/static/app.js")
        assert response.status_code == 200
        script = response.text
        assert "/api/metrics" in script, "前端没有请求 /api/metrics —— 面板会是空的"
        assert "function refreshMetrics" in script
        assert "function renderMetricsDetail" in script

    async def test_panel_renders_every_section_the_endpoint_returns(self, client):
        """接口返回的每个区块，前端都要有对应的渲染分支。

        漏渲染一个区块的后果不是报错，而是**那个数字永远不出现在界面上** ——
        例如熔断状态没渲染，用户就只能等到失败才发现后端在熔断。
        """
        payload = (await client.get("/api/metrics")).json()
        script = (await client.get("/static/app.js")).text
        section_to_marker = {
            "llm": "llm.by_phase",
            "breakers": "m.breakers",
            "caches": "m.caches",
            "injection": "m.injection",
            "llm_recent": "m.llm_recent",
        }
        for section in payload:
            marker = section_to_marker.get(section)
            if marker:
                assert marker in script, f"接口返回 {section}，但前端没有渲染它（找不到 {marker}）"
