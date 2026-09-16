"""HTTP 契约冒烟测试（不联网、不依赖 LLM）。

全程打**真实 uvicorn 服务**，逐条核对 ``docs/API.md``。

为什么不用 ``httpx.ASGITransport``：它会把整个响应体缓冲完再返回，
而 SSE 是无限流，用它在进程内测 SSE 必然死锁；真实服务器才能顺带验证
lifespan、静态文件挂载与流式响应。

Agent 工作流在这里走**离线模式**，目的是验证「图 → 运行时 → SSE → 审批往返」
这条链路本身是通的；真实联网 + LLM 的端到端测试见 ``scripts/e2e.py``。

    .python\\python.exe scripts\\smoke_http.py
"""

from __future__ import annotations

import asyncio
import faulthandler
import json
import os
import shutil
import socket
import sys
import tempfile
import threading
from pathlib import Path

faulthandler.dump_traceback_later(240, exit=True)

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

# 必须在导入 medscholar 之前指定数据目录
SANDBOX = Path(tempfile.gettempdir()) / "medscholar_smoke"
if SANDBOX.exists():
    shutil.rmtree(SANDBOX, ignore_errors=True)
SANDBOX.mkdir(parents=True, exist_ok=True)
os.environ["MEDSCHOLAR_HOME"] = str(SANDBOX)

import httpx  # noqa: E402
import uvicorn  # noqa: E402

from medscholar.models import Paper  # noqa: E402
from medscholar.server.app import app  # noqa: E402

PASSED: list[str] = []
FAILED: list[str] = []


def check(label: str, condition: bool, detail: str = "") -> None:
    if condition:
        PASSED.append(label)
        print(f"  ok   {label}", flush=True)
    else:
        FAILED.append(f"{label} — {detail}")
        print(f"  FAIL {label}  {detail}", flush=True)


def section(title: str) -> None:
    print("\n" + "=" * 72, flush=True)
    print(title, flush=True)


def free_port() -> int:
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        return int(probe.getsockname()[1])


async def main() -> int:
    from medscholar.db import repo
    from medscholar.db.connect import get_db

    db = get_db()
    seeds = [
        Paper(
            title="Accelerated rTMS for post-stroke depression: a randomized trial",
            abstract="Accelerated repetitive transcranial magnetic stimulation improved HAMD scores.",
            authors=["Zhang Wei", "Li Ming"], journal="Brain Stimulation", pub_year=2023,
            source="pubmed", pmid="37123456", doi="10.1016/j.brs.2023.001",
            mesh_terms=["Depression", "Stroke"], cited_by_count=42, is_open_access=True,
        ),
        Paper(
            title="加速rTMS治疗卒中后抑郁的临床疗效观察",
            abstract="目的：探讨加速重复经颅磁刺激治疗卒中后抑郁的临床疗效。结果：HAMD评分显著降低。",
            authors=["王伟", "李静"], journal="中国康复医学杂志", pub_year=2022,
            source="openalex", keywords=["卒中后抑郁", "重复经颅磁刺激"],
        ),
        Paper(
            title="Deep brain stimulation for treatment-resistant depression",
            abstract="DBS of the subcallosal cingulate showed sustained response.",
            authors=["Malone DA"], journal="Biological Psychiatry", pub_year=2019,
            source="europepmc", doi="10.1016/j.biopsych.2019.05.011", cited_by_count=310,
        ),
    ]
    inserted = repo.insert_papers(seeds, db=db, embed=False)
    paper_ids = inserted["ids"]
    print(f"种子数据：{inserted['new']} 篇，paper_id={paper_ids}", flush=True)

    port = free_port()
    config = uvicorn.Config(app, host="127.0.0.1", port=port, log_level="warning")
    server = uvicorn.Server(config)
    thread = threading.Thread(target=server.run, daemon=True, name="uvicorn-smoke")
    thread.start()

    base = f"http://127.0.0.1:{port}"
    for _ in range(150):
        if server.started:
            break
        await asyncio.sleep(0.1)
    check("uvicorn 启动成功", server.started, base)
    if not server.started:
        return 1

    try:
        async with httpx.AsyncClient(base_url=base, timeout=120.0) as client:
            # ---------------------------------------------------------- 1
            section("1) 健康与配置")
            response = await client.get("/api/health")
            check("GET /api/health 200", response.status_code == 200, str(response.status_code))
            health = response.json()
            for key in ("ok", "version", "offline", "llm", "embedding", "database", "sources"):
                check(f"health.{key} 存在", key in health)
            check("health.database.papers == 3", health["database"]["papers"] == 3,
                  str(health["database"]["papers"]))

            response = await client.get("/api/config")
            check("GET /api/config 200", response.status_code == 200)
            raw = response.text.replace('"has_api_key"', "")
            check("config 不含 API Key 明文", '"api_key"' not in raw, "疑似泄露")

            response = await client.get("/api/sources")
            check("GET /api/sources 200", response.status_code == 200)
            # 断言"必需的数据源都在"，而不是"恰好 N 个"：
            # 写死数量会在新增数据源时误报（加 DOAJ/CORE 时就踩到了），
            # 而移除数据源这种真回归仍然会被抓住。
            source_names = {item["name"] for item in response.json().get("items", [])}
            required_sources = {
                "pubmed", "europepmc", "openalex", "crossref",
                "semantic_scholar", "arxiv", "cnki", "doaj", "core",
            }
            missing = required_sources - source_names
            check(
                f"sources 含全部 {len(required_sources)} 个必需数据源",
                not missing,
                f"缺少 {sorted(missing)}" if missing else f"实际 {len(source_names)} 个",
            )

            # ---------------------------------------------------------- 2
            section("2) 文献库")
            response = await client.get("/api/papers", params={"limit": 10})
            body = response.json()
            check("GET /api/papers 200", response.status_code == 200)
            check("total=3", body["total"] == 3, str(body["total"]))
            check("Paper 含 source_label", "source_label" in body["items"][0])

            response = await client.get(f"/api/papers/{paper_ids[0]}")
            check("GET /api/papers/{id} 200", response.status_code == 200)
            check("Paper.title 正确", "rTMS" in response.json()["title"])
            check("不存在的 ID 返回 404",
                  (await client.get("/api/papers/999999")).status_code == 404)

            response = await client.post("/api/papers/search", json={"query": "rTMS 抑郁", "top_k": 5})
            check("POST /api/papers/search 200", response.status_code == 200, response.text[:200])
            payload = response.json()
            check("混合检索有命中", payload["count"] >= 1, f"count={payload.get('count')}")
            if payload["items"]:
                for key in ("score", "matched_by", "fts_rank", "vector_rank"):
                    check(f"检索结果含 {key}", key in payload["items"][0])

            response = await client.get(f"/api/papers/{paper_ids[0]}/fulltext")
            check("GET fulltext 结构正确",
                  response.status_code == 200 and "char_count" in response.json())

            response = await client.get(f"/api/papers/{paper_ids[0]}/references")
            check("GET references 含 local_references",
                  response.status_code == 200 and "local_references" in response.json())

            # ---------------------------------------------------------- 3
            section("3) 引用与导出")
            response = await client.get("/api/cite/styles")
            check("GET /api/cite/styles 含 6 种样式",
                  response.status_code == 200 and len(response.json()["styles"]) == 6)

            for style in ("apa7", "vancouver", "gb7714", "bibtex", "ris"):
                response = await client.post(
                    "/api/cite", json={"paper_ids": paper_ids[:2], "style": style}
                )
                check(f"POST /api/cite style={style}",
                      response.status_code == 200 and len(response.json()["content"]) > 20,
                      response.text[:120])

            for fmt in ("bibtex", "ris", "csv", "json", "gb7714"):
                response = await client.post(
                    "/api/export", json={"paper_ids": paper_ids, "format": fmt, "name": "smoke"}
                )
                ok = response.status_code == 200 and Path(response.json()["path"]).exists()
                check(f"POST /api/export format={fmt}", ok, response.text[:120])

            check("非法导出格式返回 400",
                  (await client.post("/api/export",
                                     json={"paper_ids": paper_ids, "format": "docx"})).status_code == 400)

            # ---------------------------------------------------------- 4
            section("4) 课题管理")
            response = await client.post(
                "/api/projects",
                json={"name": "卒中后抑郁", "description": "测试课题", "keywords": ["rTMS"]},
            )
            check("POST /api/projects 200", response.status_code == 200, response.text[:120])
            project_id = response.json()["id"]
            response = await client.get("/api/projects")
            check("GET /api/projects 含新课题",
                  any(p["id"] == project_id for p in response.json()["items"]))
            response = await client.post(
                f"/api/projects/{project_id}/papers", json={"paper_ids": paper_ids[:2]}
            )
            check("添加文献到课题", response.json()["added"] == 2, response.text[:120])
            response = await client.get("/api/papers", params={"project_id": project_id})
            check("按课题筛选（total 与 items 一致）",
                  response.json()["total"] == 2 and len(response.json()["items"]) == 2,
                  f"total={response.json().get('total')} items={len(response.json().get('items', []))}")

            # ---------------------------------------------------------- 5
            section("5) 会话与产物")
            check("GET /api/sessions 200", (await client.get("/api/sessions")).status_code == 200)
            response = await client.post("/api/sessions", json={"title": "冒烟会话"})
            session_id = response.json()["id"]
            check("POST /api/sessions 200", response.status_code == 200)
            check("GET messages 200",
                  (await client.get(f"/api/sessions/{session_id}/messages")).status_code == 200)
            check("GET /api/artifacts 200", (await client.get("/api/artifacts")).status_code == 200)

            # ---------------------------------------------------------- 6
            section("6) 维护与静态资源")
            check("POST /api/maintenance/reindex 200",
                  (await client.post("/api/maintenance/reindex")).status_code == 200)
            check("GET /api/stats 200", (await client.get("/api/stats")).status_code == 200)

            response = await client.get("/")
            check("GET / 返回前端页面",
                  response.status_code == 200 and "<html" in response.text.lower(),
                  f"HTTP {response.status_code}")
            check("GET /static/app.js 200",
                  (await client.get("/static/app.js")).status_code == 200)
            check("GET /static/style.css 200",
                  (await client.get("/static/style.css")).status_code == 200)

            # ---------------------------------------------------------- 7
            section("7) Agent 工作流 + SSE + 人工审批")
            response = await client.post(
                "/api/agent/run",
                json={"topic": "加速rTMS治疗卒中后抑郁", "offline": True, "require_approval": True},
            )
            check("POST /api/agent/run 200", response.status_code == 200, response.text[:200])
            run_id = response.json()["run_id"]
            run_session = response.json()["session_id"]
            check("返回 run_id 与 session_id", bool(run_id) and bool(run_session))

            events: list[str] = []
            plan_seen: dict = {}
            approved = False

            async with client.stream("GET", f"/api/agent/stream/{run_id}") as stream:
                check("SSE Content-Type 正确",
                      "text/event-stream" in stream.headers.get("content-type", ""),
                      stream.headers.get("content-type", ""))
                current = None
                async for line in stream.aiter_lines():
                    if line.startswith("event: "):
                        current = line[7:].strip()
                        events.append(current)
                    elif line.startswith("data: ") and current == "plan" and not plan_seen:
                        # 契约：plan 事件的数据形如 {"plan": {...}}
                        payload = json.loads(line[6:])
                        plan_seen = payload.get("plan", payload)
                    elif current == "awaiting_approval" and not approved:
                        approved = True
                        answer = await client.post(
                            f"/api/agent/approve/{run_id}",
                            json={"decision": "approve", "feedback": ""},
                        )
                        check("审批请求返回 200", answer.status_code == 200, answer.text[:120])
                    if current == "done":
                        break

            check("SSE 收到 phase 事件", "phase" in events, str(events[:10]))
            check("SSE 收到 awaiting_approval（人工审批节点生效）",
                  "awaiting_approval" in events, str(events))
            check("SSE 收到 plan 事件", "plan" in events, str(events))
            check("plan 含中英文课题",
                  bool(plan_seen.get("topic_zh") or plan_seen.get("topic_en")), str(plan_seen)[:120])
            check("plan 含检索式", bool(plan_seen.get("queries")), str(plan_seen)[:120])
            check("plan 含大纲", bool(plan_seen.get("outline")), str(plan_seen)[:120])
            check("SSE 收到 done 事件（审批后流程继续）", "done" in events, str(events[-8:]))

            check("重复审批返回 409",
                  (await client.post(f"/api/agent/approve/{run_id}",
                                     json={"decision": "approve"})).status_code == 409)

            response = await client.get("/api/agent/runs")
            check("运行列表含本次运行",
                  any(r["run_id"] == run_id for r in response.json()["runs"]))
            check("GET /api/agent/runs/{id} 200",
                  (await client.get(f"/api/agent/runs/{run_id}")).status_code == 200)

            response = await client.get(f"/api/sessions/{run_session}/messages")
            check("会话已记录 user 消息",
                  any(m["role"] == "user" for m in response.json()["items"]),
                  response.text[:160])

            # ---------------------------------------------------------- 8
            section("8) 审批「取消」路径")
            response = await client.post(
                "/api/agent/run",
                json={"topic": "取消路径测试", "offline": True, "require_approval": True},
            )
            cancel_run = response.json()["run_id"]
            cancel_sent = False
            async with client.stream("GET", f"/api/agent/stream/{cancel_run}") as stream:
                current = None
                async for line in stream.aiter_lines():
                    if line.startswith("event: "):
                        current = line[7:].strip()
                    elif current == "awaiting_approval" and not cancel_sent:
                        cancel_sent = True
                        await client.post(
                            f"/api/agent/approve/{cancel_run}", json={"decision": "cancel"}
                        )
                    if current == "done":
                        break
            check("取消后流程正常结束", cancel_sent)

            # ---------------------------------------------------------- 9
            section("9) 删除")
            response = await client.request(
                "DELETE", "/api/papers", json={"ids": [paper_ids[2]]}
            )
            check("DELETE /api/papers 200", response.status_code == 200)
            check("删除计数为 1", response.json()["deleted"] == 1, response.text[:120])
    finally:
        server.should_exit = True
        thread.join(timeout=15)

    section(f"结果：通过 {len(PASSED)} 项，失败 {len(FAILED)} 项")
    for item in FAILED:
        print(f"  - {item}", flush=True)
    print("SMOKE_HTTP:", "PASS" if not FAILED else "FAIL", flush=True)
    return 0 if not FAILED else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
