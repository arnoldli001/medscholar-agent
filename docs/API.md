# MedScholar Agent — HTTP API 契约（v1.0）

前端（`medscholar/web/`）与自动化脚本均依赖本契约。**改动此文件必须同步改前端。**

- 基地址：`http://127.0.0.1:8760`
- 所有请求/响应均为 UTF-8 JSON，`Content-Type: application/json`
- 错误统一返回 `{"detail": "错误说明"}`，HTTP 状态码语义化
- 所有时间戳为 `YYYY-MM-DD HH:MM:SS`（本地时间）

---

## 1. 健康与配置

### `GET /api/ping`
极轻量的连通性探测：**不读数据库、不碰模型**，用于前端判断后端是否在线。
页面首屏应先用它握手，再去取 `/api/health` 详情。
```json
{ "ok": true, "version": "1.0.0", "uptime_s": 12.3 }
```

### `GET /api/health`
查询参数：`probe=true` 强制立即刷新模型探测（默认使用最多 120 秒的缓存）。

> ⚠️ **`llm.ok` / `embedding.ok` 可能是 `null`**，表示"后台正在探测"。
> 早期版本会在本接口里同步跑一次 LLM 生成，实测耗时 4~5.6 秒（冷启动 25~35 秒），
> 导致页面一打开就满屏"连不上后端"。现在改为立即返回缓存 + 后台异步刷新，
> 前端遇到 `null` 应显示"检测中"而不是"不可用"，并在数秒后重新拉取。

```json
{
  "ok": true,
  "version": "1.0.0",
  "offline": false,
  "llm":     { "ok": true,  "provider": "ollama", "model": "qwen3:8b",  "message": "..." },
  "embedding": { "ok": true, "provider": "ollama", "model": "nomic-embed-text:latest", "dim": 768, "message": "..." },
  "database": { "papers": 128, "embedded": 128, "vector_backend": "sqlite-vec", "db_path": "...", "db_size_mb": 1.2 },
  "sources": [ { "name": "pubmed", "label": "PubMed", "enabled": true, "has_api_key": false, "rps": 3.0 } ]
}
```

### `GET /api/stats` → 知识库统计（文献数、全文数、向量覆盖率、数据源分布）

### `GET /api/metrics` — 运行指标（可观测性面板的数据源）
只返回**聚合数字**，不含任何文献内容或提示词正文（这个接口常被贴到 issue 或群里求助）：
```json
{
  "llm": {
    "calls": 42, "ok_calls": 40, "failed_calls": 2, "cache_hits": 0,
    "prompt_tokens": 38210, "completion_tokens": 9140, "total_tokens": 47350,
    "cost_yuan": 0.0,
    "latency_ms_p50": 1830.5, "latency_ms_p95": 6210.0,
    "by_model": { "qwen3:8b": { "calls": 42, "...": "..." } },
    "by_phase": { "plan": { "calls": 2, "...": "..." }, "synthesize": { "calls": 9, "...": "..." } },
    "by_error_kind": { "timeout": { "calls": 2, "models": ["qwen3:8b"] } }
  },
  "llm_recent": [ { "provider": "ollama", "model": "qwen3:8b", "phase": "synthesize", "latency_ms": 1830.5, "ok": true } ],
  "breakers": { "llm:ollama|qwen3:8b|": { "name": "...", "state": "closed", "failures": 0 } },
  "caches": { "embed_query": { "hits": 12, "misses": 3, "hit_rate": 0.8, "size": 3, "ttl": 3600.0 } },
  "injection": { "blocks_with_findings": 0, "total_findings": 0, "last_finding": null },
  "uptime_s": 812.4
}
```
用途：回答"这次综述花了多少 token / 多少钱"、"慢在哪个阶段"、"哪个后端在连续失败"、
"检索内容里有没有被塞进指令"。**账本在进程内存**，重启即清零（见架构文档 ADR-8）。

### `GET /api/metrics/schema` — 数据库 schema 版本与待执行迁移（只读）
```json
{ "current_version": 2, "applied": [ { "version": 1, "name": "baseline", "..." : "..." } ],
  "pending": [], "dirty": false, "exists": true }
```
单独一个路径而不是塞进 `/api/metrics`：它要打开数据库读元表，比纯内存指标重得多，
监控可以只轮询轻量的那个。

### `GET /api/config`
返回脱敏配置（**绝不含 API Key 明文**）：
```json
{
  "app_name": "MedScholar Agent",
  "language": "zh-CN",
  "offline": false,
  "llm": { "provider": "ollama", "model": "qwen3:8b", "base_url": "...", "has_api_key": false },
  "embedding": { "provider": "ollama", "model": "nomic-embed-text:latest", "dim": 768 },
  "retrieval": { "rrf_k": 60, "top_k": 20 },
  "sources": { "pubmed": { "enabled": true, "has_api_key": false, "rps": 3.0 }, "...": {} }
}
```

---

## 2. 文献库

### `GET /api/papers`
查询参数：`q` `limit`(默认 20) `offset` `year_from` `year_to` `min_cited` `open_access`(bool)
`source` `sources`(逗号分隔) `journal` `project_id` `order_by`（`created_desc|year_desc|year_asc|cited_desc|title_asc`）

```json
{ "total": 128, "limit": 20, "offset": 0, "items": [ Paper, ... ] }
```

### `POST /api/papers/search` — 本地知识库**混合检索**（BM25 + 向量 + RRF）
请求体：`{ "query": "加速rTMS 卒中后抑郁", "top_k": 20, "filters": { "year_from": 2015, "open_access": true } }`
```json
{ "query": "...", "count": 12, "items": [ { ...Paper, "score": 0.032, "matched_by": "bm25+vector",
  "fts_rank": 1, "vector_rank": 3 } ] }
```

### `GET /api/papers/{paper_id}` → `Paper`
### `DELETE /api/papers` 请求体 `{ "ids": [1,2,3] }` → `{ "deleted": 3 }`
### `GET /api/papers/{paper_id}/references` → `{ "items": [...] }`
### `GET /api/papers/{paper_id}/fulltext` → `{ "paper_id": 1, "content": "...", "char_count": 1234 }`

### `Paper` 对象
```json
{
  "paper_id": 1, "pmid": "37123456", "pmcid": "PMC10239808", "doi": "10.1016/j.brs.2023.001",
  "title": "...", "abstract": "...", "authors": ["Zhang Wei", "Li Ming"],
  "journal": "Brain Stimulation", "pub_year": 2023, "source": "pubmed", "source_label": "PubMed",
  "mesh_terms": ["..."], "keywords": ["..."], "cited_by_count": 42, "is_open_access": true,
  "full_text_url": "", "url": "https://pubmed.ncbi.nlm.nih.gov/37123456/",
  "volume": "", "issue": "", "pages": "", "publication_type": "Journal Article", "language": "eng",
  "note": "", "paper_id": 1, "created_at": "...", "updated_at": "...",
  "dedup_key": "doi:10.1016/...", "citation_label": "Zhang 2023", "short_authors": "Zhang Wei, Li Ming"
}
```

---

## 3. 联网检索（Scout）

### `POST /api/search/live`
请求体：
```json
{ "query": "accelerated rTMS post-stroke depression",
  "sources": ["pubmed","europepmc","openalex","crossref"],
  "limit": 40, "per_source_limit": 20,
  "filters": { "year_from": 2015, "year_to": null, "open_access": false, "sort": "relevance" },
  "save": true, "embed": true }
```
响应：
```json
{ "query": "...", "count": 39, "raw_count": 46, "duration_ms": 4210,
  "saved": { "new": 30, "updated": 9 }, "embedded": { "embedded": 30 },
  "sources": [ { "name":"pubmed","label":"PubMed","ok":true,"count":6,"error":"","duration_ms":2265,"skipped":false } ],
  "items": [ Paper, ... ] }
```

---

## 3.5 检索式解析（组合检索）

### `POST /api/query/preview`
把用户的检索输入解析成结构化查询，并给出**各数据源实际会发出的检索式**。
前端用它做「将检索：…」的实时回显，解析逻辑与真正检索同源，所见即所发。

```json
{ "query": "加速rTMS 卒中后抑郁 -大鼠 \"randomized trial\"",
  "sources": ["pubmed", "openalex"] }
```
```json
{ "query": "...",
  "parsed": { "must": ["加速rTMS", "卒中后抑郁"], "any_groups": [],
              "exclude": ["大鼠"], "phrases": ["randomized trial"],
              "is_simple": false, "description": "\"randomized trial\" 且 加速rTMS 且 卒中后抑郁 且 排除 大鼠" },
  "help": "空格或逗号 = AND（同时包含）｜ | 或 OR = 任选其一｜ - 或 NOT = 排除｜ \"引号\" = 精确短语",
  "boolean_sources": ["arxiv", "europepmc", "pubmed"],
  "per_source": {
    "pubmed": "(\"randomized trial\" AND 加速rTMS AND 卒中后抑郁) NOT 大鼠",
    "openalex": "randomized trial 加速rTMS 卒中后抑郁"
  } }
```

**输入语法**（与 PubMed / Web of Science 习惯一致）：

| 写法 | 含义 |
|---|---|
| 空格、`,`、`，` | AND（同时包含） |
| `\|`、`OR`、`；`、`;`、`或者` | OR（任选其一，用于同义词） |
| `-词`、`NOT 词` | 排除 |
| `"词组"` | 精确短语 |

**各库能力**：PubMed / Europe PMC / arXiv 支持完整布尔（arXiv 用 `ANDNOT`），
OpenAlex / Semantic Scholar / Crossref / CNKI 只做相关度检索，
会退化为"核心词"（OR 组的全部同义词都会带上，便于排序）。

---

## 4. Agent 工作流（异步 + SSE）

四节点：`Plan → 用户审批 → Execute → Reflect → Synthesize`

### `POST /api/agent/run`
```json
{ "topic": "加速rTMS治疗卒中后抑郁",
  "sources": ["pubmed","europepmc","openalex","crossref"],
  "project_id": null, "session_id": null,
  "require_approval": true, "offline": false }
```
→ `{ "run_id": "0f3c…", "session_id": 12 }`

### `GET /api/agent/stream/{run_id}` — **SSE**（`text/event-stream`）
每条事件：`event: <type>` + `data: <JSON>`。事件类型：

| type | data 字段 | 说明 |
|---|---|---|
| `phase` | `phase`, `label` | 阶段切换（plan/execute/reflect/synthesize/done） |
| `status` | `message` | 进度文字 |
| `token` | `text` | 流式增量文本（写作阶段） |
| `plan` | `plan`（见下） | 规划完成，若需审批随后会发 `awaiting_approval` |
| `awaiting_approval` | `run_id`, `plan` | **等待用户审批**，前端应显示「批准 / 修改后继续 / 取消」 |
| `search_result` | `query`, `sources`, `count`, `saved`, `items` | 单条检索式的结果 |
| `papers` | `items`, `count` | 汇总后的文献卡片列表 |
| `critique` | `assessments`, `overall` | Critic 评估结果 |
| `artifact` | `artifact_id`, `title`, `content`, `fmt` | 生成产物（综述草稿） |
| `review` | `verdict`, `score`, `issues` | 自我审查结果 |
| `error` | `message` | 错误（非致命错误也会发，随后继续） |
| `done` | `run_id`, `elapsed_ms`, `usage` | 结束 |

`plan` 结构：
```json
{ "topic_zh": "...", "topic_en": "...",
  "pico": { "population": "", "intervention": "", "comparator": "", "outcomes": [] },
  "queries": [ { "query": "...", "sources": ["pubmed"], "rationale": "..." } ],
  "mesh_terms": ["..."], "year_from": 2015,
  "key_questions": ["..."],
  "outline": [ { "title": "1 引言", "points": ["..."] } ] }
```

### `POST /api/agent/approve/{run_id}`
```json
{ "decision": "approve" | "revise" | "cancel", "feedback": "可选，revise 时的修改意见" }
```
→ `{ "ok": true, "decision": "approve" }`

### `POST /api/agent/cancel/{run_id}` → `{ "ok": true }`
### `GET /api/agent/runs` → `{ "runs": [ { "run_id", "topic", "phase", "status", "created_at" } ] }`

---

## 5. 会话与产物

- `GET /api/sessions` → `{ "items": [ { id, title, project_id, topic, created_at, updated_at, message_count } ] }`
- `POST /api/sessions` `{ "title": "...", "topic": "..." }` → `{ "id": 12 }`
- `GET /api/sessions/{id}/messages` → `{ "items": [ { id, role, content, meta, created_at } ] }`
- `DELETE /api/sessions/{id}` → `{ "ok": true }`
- `GET /api/artifacts?session_id=12` → `{ "items": [ { id, title, kind, fmt, char_count, created_at } ] }`
- `GET /api/artifacts/{id}` → `{ id, title, content, kind, fmt, meta, ... }`

---

## 6. 引用与导出

### `POST /api/cite`
```json
{ "paper_ids": [1,2,3], "style": "gb7714", "format": "list" }
```
`format`: `list`（参考文献表）| `bibtex` | `ris` | `inline`
→ `{ "style": "gb7714", "label": "GB/T 7714-2015", "content": "..." }`

### `GET /api/cite/styles` → `{ "styles": [ { "key": "apa7", "label": "APA 7th" } ] }`

### `POST /api/export`
```json
{ "paper_ids": [1,2,3], "format": "bibtex", "name": "rtms综述" }
```
→ `{ "path": "D:\\...\\exports\\rtms综述_20260210_113000.bib", "format": "bibtex", "bytes": 1234 }`

支持的 `format`：`bibtex` `ris` `apa7` `vancouver` `gb7714` `csv` `json` `markdown` `html`

### `GET /api/prisma/flow` — PRISMA 2020 流程数字（系统评价投稿必需）
查询参数：`since`（只统计该时间之后的检索）`included` `excluded_screening` `not_retrieved`（研究者判断，需手工填）

```json
{
  "flow": {
    "identified_total": 208, "identified": { "pubmed": 120, "openalex": 88 },
    "duplicates_removed": 138, "records_after_dedup": 70, "screened": 70,
    "excluded_at_screening": 40, "sought_for_retrieval": 30, "not_retrieved": 3,
    "assessed_for_eligibility": 27, "excluded_at_fulltext": { "研究设计不符": 12 },
    "included": 12, "excluded_total": 52
  },
  "text": "Identification\n  Records identified from pubmed (n = 120) ...",
  "warnings": [],
  "checklist": [ { "code": "16a", "name": "Study selection results", "status": "auto" } ],
  "search_summary": { "total_results": 208, "total_new": 70, "failures": 0 },
  "note": "「题目/摘要排除」「未获取到全文」「最终纳入」需要研究者自己判断后填入…"
}
```
**能自动算的**：各数据库识别到的记录数（真实检索日志）、重复移除数。
**必须人工填的**：题目/摘要排除、未获取到全文、最终纳入（是学术判断，工具不代替）。
`warnings` 在数字**不自洽时**给出中文提示 —— 一张对不上的 PRISMA 图交到审稿人手里，质疑的是整篇的可信度。
`checklist` 是 PRISMA 2020 的 27 个条目，`status` 取 `auto`（工具能填）/ `manual`（需人来写）/ `done`（已覆盖）。

---

## 7. 课题管理

- `GET /api/projects` → `{ "items": [ { id, name, description, keywords, paper_count, created_at } ] }`
- `POST /api/projects` `{ "name": "...", "description": "...", "keywords": [] }` → `{ "id": 3 }`
- `POST /api/projects/{id}/papers` `{ "paper_ids": [1,2] }` → `{ "added": 2 }`
- `DELETE /api/projects/{id}` → `{ "ok": true }`

---

## 8. 维护

- `GET /api/stats` → 数据库统计（同 `/api/health` 的 `database` 字段）
- `GET /api/metrics` → 运行指标（LLM 用量与成本、熔断状态、缓存命中率、注入扫描）
- `GET /api/metrics/schema` → schema 版本与待执行迁移
- `POST /api/maintenance/embed` `{ "limit": 200 }` → `{ "embedded": 200, "failed": 0, "skipped": 0 }`
- `POST /api/maintenance/reindex` → `{ "ok": true }`
