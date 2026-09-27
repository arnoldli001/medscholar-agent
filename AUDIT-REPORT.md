# MedScholar Agent 深度审计报告

> 审计时间：本机实测。审计范围：`medscholar/` 全量 229 个文件 + `medscholar/web/` 前端三件套 + 19 个 scripts + 154 个测试文件 + docs 12 份。
> 基线核对：`pytest tests -q` → **1384 passed in 25.65s**（与 README 一致）；
> 所有缺陷结论均可由仓库根目录的 `verify_findings.py` 当场复现（12/12 复现）。

---

## 0. 结论摘要

这个项目的**工程质量基线明显高于同类简历项目**：分层由 AST 强制、1384 测试全绿、评测跑生产原语并做 parity 校验、拒绝抓订阅资源。它的问题不在"做得少"，而在**三处结构性缺口**：

| 缺口 | 一句话 | 后果 |
|---|---|---|
| **编号一致性没有单一真相源** | 正文用 `citation_map` 的编号，参考文献表被 `enumerate` 重编号 | 成稿的 `[16]` 指向第 1 条参考文献——**每一篇综述的引用都是错的** |
| **最贵的资源（全文）没有消费方** | 预取 82 篇全文 / 197 万字符，写作只吃 800 字摘要 | 全文抓取+解析+建 FTS 索引的开销 100% 未产生价值 |
| **声明与实现之间没有可执行对账** | trace/span、Bulkhead、claim-level 校验、生产 FTS 权重、min_score、"严格一一对应" | 文档承诺的能力在真实运行中不存在；面试追问即崩 |

以及一条贯穿性的成本问题：**预算是按"检索到多少篇"分配的，不是按"用上多少篇"分配的**——150 篇进评估、138 篇做启发式、12 篇过 LLM、最后 8~15 篇出现在成稿里。

---

## 1. 可复现的验证

```bat
.python\python.exe -X utf8 verify_findings.py
```

实测输出（摘要）：

```
✔ 复现  引用编号错位
✔ 复现  检索式解析器不识别 AND
✔ 复现  trace/span 是死代码
✔ 复现  预取全文未进入提示词
✔ 复现  账本缺阶段/运行归属
✔ 复现  前端 api.metrics 未定义
✔ 复现  SSE 补播不幂等
✔ 复现  resume 可并发覆盖同一 run
✔ 复现  协程内同步 SQLite
✔ 复现  Bulkhead 无调用者
✔ 复现  文档与实现不符
✔ 复现  换模型即丢向量
共 12 项，复现 12 项。
```

---

## 2. 严重度分级问题清单

严重度定义：
- **P0** = 产出错误结果 / 核心能力静默失效 / 数据丢失，且界面上看不出来
- **P1** = 成本翻倍、显著体验缺陷、单点故障
- **P2** = 性能与可维护性劣化
- **P3** = 文档与一致性

### 2.1 P0（7 条）

| # | 问题 | 证据 | 影响 | 建议方案 | 工时 |
|---|---|---|---|---|---|
| **P0-1** | **成稿引用编号与参考文献表全部错位** | `agent/graph.py:483` 用 `only_cited` 过滤 entries → `domain/citation/styles.py:330` `enumerate(items, start=1)` 从 1 重编号。实测 artifact 2：正文引 `[12,13,16,19,20]`，参考文献表只有 `[1..8]`；artifact 1：15 条正文引用中 5 条越界 | **每一篇综述的引用都对不上**——学术场景下这是产品定义级别的失败。selfcheck 只校验"草稿内部"（`formatter.py:210` 拿 `citation_map` 全部编号），Formatter 又只校验"内部变量"，两道检查都不看最终产物 | 让编号有唯一真相源：`build_references(entries, style, only_cited=...)` 保留原始 index，改为 `for index, paper in entries: format_citation(paper, style, index=index)`；`format_reference_list` 增加 `indices` 参数；补一条端到端断言「成稿正文编号集合 ⊆ 参考文献表编号集合」 | 0.5d |
| **P0-2** | **检索式解析器不识别显式 `AND`，真实规划输出全被破坏** | `domain/query.py` 的 `_OR_WORDS` 只含 `or/或者/｜`，`AND` 落进 `must` 当普通词，`for_source` 又用 `" AND ".join(...)` 拼接。实测 3/3 条真实规划式产出 `(rTMS AND AND AND "post-stroke" ...)`；`[openalex]` 类源收到 `...(疗效 安全性...)` 含裸 `AND` | 四个真实检索式在 **Europe PMC 全部 hitCount=0**（HTTP 200，静默返回空）→ OA 全文主力源被静默废掉，检索结果只来自 PubMed/OpenAlex/Crossref 三个源。`(meta-analysis OR ...)` 被切成 `("(meta-analysis" OR systematic))` 括号错配 | `query.py` 增加 `_AND_WORDS = {"and","且","&"}` 归入分隔符（跳过而非入词）；对非布尔源（openalex/crossref/s2）剥离 `AND/OR/NOT/括号` 只留核心词；补一条**表达式合法性**断言：`assert " AND AND" not in expr` | 0.5d |
| **P0-3** | **`warm_fulltext` 预取的全文从未进入任何提示词** | `retrieval.build_context_digest` → `llm/prompts.digest_papers` 只消费 `Paper.to_dict()`（title/abstract/journal/年份/被引），没有全文参数。实测：库内 82 篇全文 / **1,967,575 字符**，与成稿逐字重合（60 字窗口）**0 篇** | 全文下载 + PDF/JATS 解析 + `paper_fulltext` 落库 + `fulltext_fts` 建索引的全部开销**未产生任何价值**；`config` 里 `warm_fulltext: true` / `fulltext_top_n: 4` 是纯浪费旋钮。写作只能看到 800 字摘要，"深度引用"的承诺落空 | 二选一：① **接上**——`build_context_digest` 支持 `fulltext_by_id`，写作阶段对 `select_papers` 后的 top-N 注入全文前 N 千字（并同步把全文纳入注入扫描，现在只扫 title+abstract）；② **摘掉**——删除预取链路，把省下的时间预算换更多文献的摘要分辨率 | 接上 1.5d / 摘掉 0.5d |
| **P0-4** | **trace/span 在生产路径零调用点** | 全仓 `grep create_trace\|run_in_trace\|\.span(`——`medscholar/` 下**无调用者**，只有 `tests/test_observability.py` 与 `observability.py` 自身定义。`llm/client.py:169` 的 `current_span()` 因此恒为 `None` | README「trace/span 树」、`ARCHITECTURE.md` 的调用链还原能力**在真实运行中不存在**；`LLMUsage.phase` 与 `run_id` 恒为空串 → `/api/metrics` 的 `by_phase` 永远只有 `(未标注)` 一个桶，"按阶段聚合成本"不可用 | 在 `graph.run()` 入口 `with create_trace(state.run_id) as tr:`，各阶段 `with tr.span("plan"):` …，`finally: tr.finish(); tr.to_jsonl(data_home/"traces.jsonl")`。约 40 行 | 0.5d |
| **P0-5** | **`resume()` 不检查内存活句柄 → 同一 run 并发两个 task + 句柄泄漏** | `agent/runtime.py:243-346` 的 `resume()` 无 `self._runs.get(run_id)` 检查，`:337` 直接覆盖；`server/deps.py:30-40` 的 `is_resumable` 对运行中的 run 恒返回 True；`routes/agent.py:195` 的 `/api/agent/latest` 只查 DB 不看内存 | 界面点「继续」→ 两个 task 跑同一 run_id；旧 task 停在无人 resolve 的审批 Future 上，`handle.closed` 永远 False，`_prune()`（`runtime.py:548`）永远清不掉 → **内存泄漏 + 旧任务无法 cancel**；云端模型下重复计费。配合前端 P0-7，"运行中刷新页面"没有安全恢复路径 | `resume()` 首行加活句柄检查：已存在且 `not closed` → 直接返回现有 handle（幂等）；`/api/agent/latest` 改为合并内存 live 状态 | 0.5d |
| **P0-6** | **协程里同步跑 SQLite，阻塞整个事件循环** | `retrieval.py:49` `hybrid_search`、`scout.py:230` `insert_paper` 循环、`graph.py:501/522` `save_artifact`+`add_message`、`pipeline.py:99/129` `get_paper`+`store_embeddings`——均未包 `asyncio.to_thread`；`scout.py:161` 的 `log_search` 在 async 路径直接调用，5 条检索式 × 6 源 = **30 次 `BEGIN IMMEDIATE` 写事务**串行 | 一次本地检索（含 vec 回退、`get_papers_by_ids`）或一次嵌入批次会**冻结整个 HTTP/SSE 服务**；SSE 心跳、审批响应、健康检查全部卡住 | 统一包 `await asyncio.to_thread(...)`；`log_search` 改为批量写一次；并给 `scripts/check_arch.py` 加一条 AST 规则：**`async def` 内出现 `repo.<fn>(`/`db.<fn>(` 且未被 `to_thread` 包裹即违规** | 1d |
| **P0-7** | **SSE 全量补播 × 前端不幂等 → 正文被拼接两遍** | 后端 `runtime.py:511` `delivered = 0` 每次订阅从第 0 条重放；`routes/agent.py:74` 不发 `id:`、不处理 `Last-Event-ID`；前端 `app.js` 无条件 `buffer += text`，重连时不重置视图 | 运行中一次网络抖动/笔记本休眠 → 浏览器自动重连 → 服务端从 0 补播 → **已收到的正文再拼一遍**，plan/critique/review 消息与工具面板条目全部重复。用户拿到坏草稿且无提示 | 前端抽唯一入口 `attachStream(runId)` = `closeStream` + `resetRunViewForReplay` + 新建 EventSource（开始/续跑/刷新恢复/手动重连全走它）；后端加 `id: <seq>` + 支持 `Last-Event-ID` 续播 | 1d |

### 2.2 P1（14 条）

| # | 问题 | 证据 | 影响 | 建议方案 | 工时 |
|---|---|---|---|---|---|
| **P1-1** | **FTS「级联放宽」第一级非空即 return，宽召回路径被丢弃** | `db/repositories/search.py:64-83` `for index, expression in enumerate(strategies): ... if rows: return`。实测 28 条查询：phrase 候选 15 条、bigram-AND **也是 15 条**（从未多召回一条）、bigram-OR 有 **760 条**但仅在前两级皆空时启用 | 多相关文献查询召回腰斩；而评测集每条查询只标 1 篇 relevant，**结构上无法发现** | 三路并集后再进 RRF，删掉短路；`ORDER BY score, paper_id` 消除同分非确定性；`min_score` 语义从"绝对分"改为"相对最佳分比例" | 0.5d |
| **P1-2** | **评估覆盖率与写作预算不匹配：150→12→25 的漏斗全部白算** | 真实 run：检索 150 篇 → `critic.assess` 对 **150 篇**算启发式（`critic.py:271`）→ LLM 只点评 12 篇 → `writer_max_papers=25` 重编号 → 成稿实际引用 8 篇。另 `_llm_assess` 的排序键（`critic.py:352`）**又重算一遍**启发式 | 138 篇的评估与 125 篇的编号工作全部废弃；更严重的是 **selected 里 LLM 只覆盖 12/25**，其余按纯启发式（`quality` 由出版类型/被引/样本量决定）竞争，selection 由"是否被 LLM 看过"决定而非相关性 | 评估预算按"将进入成稿的 N 篇"动态分配（`critique_max_papers = writer_max_papers` 或按 context 预算反推）；`_llm_assess` 复用已算的启发式分数而非重算 | 0.5d |
| **P1-3** | **单模型路由：配置注释推荐双模型，代码无路由能力** | `config.example.yaml:200-202` 建议 `deepseek-flash` 做大批量生成、`deepseek-v4-pro` 做规划与批判；但 `get_llm()`（`llm/client.py:304`）只有一个全局单例，`LLMSettings` 只有单一 `provider/model` | 成本与质量双向损失：规划/批判这类"想深"环节得不到强模型，正文这类"量大"环节又用了贵模型 | `LLMSettings` 加 `routing: {plan: "model-a", critique: "model-a", section: "model-b", ...}`，`get_llm(stage=...)` 按阶段取；`breaker_for` 的 key 已含 model，天然隔离 | 1d |
| **P1-4** | **成本账本按运行/阶段聚合不可用** | 同 P0-4：`phase` 与 `run_id` 恒空。`/api/metrics` 的 `by_phase` 只有 `(未标注)`；`LEDGER.summary()` 还是**全局**的，`window_size=1000` 条明细被多次运行挤满后，早期运行的用量被丢弃 | 无法回答"这次综述花了多少钱/哪个阶段最贵"——成本优化的前提数据缺失 | 修 P0-4 即可（账本字段天生就绪）；再加 `?run_id=` 过滤与按 run 聚合的 `/api/metrics/runs/{id}` | 0.25d（依赖 P0-4） |
| **P1-5** | **流式路径的 token 记账在失败/中断时丢失** | `llm/ollama_backend.py:_ollama_stream` 只在 `chunk.get("done")` 分支 `_record`；`_openai_stream` 只在收到含 `usage` 的 chunk 时 `_record`（且立即 `break`）。中途异常、客户端断开、`finally: await response.aclose()` 都不补记 | **恰好在最需要看清成本时的失败场景下记账为 0**；"成功与失败都记账"的设计在流式这条最贵的路径上破功 | 用 `finally` 补一条 `ok=False, error_kind=...` 的记录（token 拿不到就记 0 但**耗时与失败分类必须落账**）；或按已 yield 的字符数估算并标注 `estimated=True` | 0.25d |
| **P1-6** | **参考编号错位导致"未引用条目"告警也是假的** | 同 P0-1。`artifact 1` 的 review 报「参考文献表中有 17 条未被正文引用」——这个 17 是编号错位的产物，不是真实问题 | LLM 自审 + 规则自检同时被同一条编号缺陷污染，**自审结论不可信** | 修 P0-1 后回归；并把 selfcheck 的输入从"草稿"改成"最终产物" | 随 P0-1 |
| **P1-7** | **前端「运行指标」面板引用未定义标识符** | `app.js` 唯一一处小写 `api.metrics()`，实际对象是 `const API = {...}`（33 处大写调用） | 点击该标签同步抛 `ReferenceError`，面板永久停在"加载中…"，**成本可观测性的唯一 UI 出口是坏的**；现有测试用 `"/api/metrics" in script` 子串断言，恰好放过 | 改大小写；把测试从子串断言升级为"标识符必须已声明"的静态检查 | 10min |
| **P1-8** | **刷新页面无法重新接上正在运行的 SSE，"继续"按钮是陷阱** | `restoreLatestRun` 全程不调用 `openStream`；`/api/agent/latest` 只查 DB | 5~10 分钟的任务里**不能刷新、不能切走**（切走再回来丢失直播）；点「继续」触发 P0-5 的并发 | 前端：`status ∈ {running, awaiting_approval}` 直接 `attachStream(run_id)`（服务端补播已就绪，恢复是免费的） | 0.5d（依赖 P0-7） |
| **P1-9** | **前端每次刷新全量重下 + 重新解析 260KB** | `server/deps.py` 注入 `?v=<mtime>`，但 `server/app.py:143-151` 又对 `/static/*` 强制 `Cache-Control: no-store, must-revalidate`，两者互斥；全项目无 `GZipMiddleware` | 每次刷新重下 198KB JS + 62KB CSS 并重新解析；本地回环下瓶颈是**解析**不是传输 | 去掉 `no-store`，对带 `?v=` 的静态资源用 `public, max-age=31536000, immutable`（mtime 变了 URL 就变，天然失效）；加 `GZipMiddleware` | 0.25d |
| **P1-10** | **换嵌入模型/维度即 DROP 向量表，静默降级为纯 BM25** | `db/connect.py:198-251` `_ensure_vector_table` 在 provider/model/dim 变化时 `DROP TABLE IF EXISTS paper_embeddings`；`sync_embedding_dim` 同样 | 1 万篇的库换模型后**向量全部丢失**，在后台重嵌完成前检索静默退化为纯 BM25，界面无任何告警 | 影子索引：vec0 表名带 `hash(model\|dim)` 后缀，只建新表不 DROP 旧表；检索读 `active` 表，覆盖率 <80% 时继续用旧表并在 `/api/health` 提示 | 1.5d |
| **P1-11** | **Bulkhead 零调用者；数据源扇出无整体超时；`Retry-After` 无上限** | 全仓引用 `Bulkhead` 的文件：**无**。`api/registry.py:271` 的 `gather` 无 `wait_for`；`api/base.py:223-229` `delay = max(retry_after, backoff)` 直接 sleep | 一个慢源决定整轮墙钟（单源最坏 3×30s + 退避）；恶意/异常的超大 `Retry-After` 可让进程睡任意久 | `gather` 包 `wait_for(config.sources.<name>.timeout * 2 + 30)`，超时源按失败上报；`delay = min(max(retry_after, backoff), 120)`；9 个客户端挂 `Bulkhead` | 1d |
| **P1-12** | **限流配置被静默上调（把用户的保守设置覆盖掉）** | `pubmed_client.py:40-41`、`semantic_scholar_client.py:38-39` 用 `max(settings.rps, N)`；`api/registry.py:257` 取 pubmed 的 `max_results=100` 套给所有源 | 用户为保护 IP 段设的更保守 rps 被程序覆盖；S2 的速率下调是**单向**的（`_consecutive_429` 只降不升），被限流一次后本次会话永久变慢 | 改 `min(settings.rps, 上限)`；S2 加"连续 N 次成功后 `rps = min(configured, rps * 1.5)`"的恢复路径；`per_source` 改 `sources.get(name).max_results` | 0.5d |
| **P1-13** | **HTTP 层零缓存；引用关系表零调用者** | 缓存注册表唯一使用者是 `embedding/pipeline.py:255-261`（查询向量）；`db/repositories/citations.py:38 add_citations` 生产**零调用者** → `get_references` 恒返回空 | 跨会话重复请求同一检索式/全文/Unpaywall；`citations` 表建了永不写入；合规面还有一个未限流的出网口（PDF 下载每篇新建连接池+无桶） | `api/base.py request()` 加 GET 缓存（检索 `ttl=600`、全文/引用 `ttl=86400`，`maxsize=512`）；`reader` 复用注册表客户端而不是自建第二套 TokenBucket（现在同 host 实际速率是配置的 2 倍） | 1.5d |
| **P1-14** | **评测语料是英文单相关小集合，却承担了跨语言/融合/权重结论** | `eval/datasets/regression.json` corpus **40/40 全英文**，但 cases 带 `zh`/`es` 的 `cross-lingual` 标签（实测这些查询 FTS 命中 0）。`.cache/eval-report.json`：**5 个融合配置数值逐位相同**（production/rrf-k10/rrf-k100/weighted-fts2/weighted-vec2 全为 0.8889/0.8678/0.8611/0.8611） | 「7 配置消融证明融合有效」「跨语言能力已评测」**都超出语料能支撑的范围**；`hybrid_search` 的 `fts_weight/vector_weight` 全仓只有评测传值、生产恒为 1.0 | 语料加中英对照条目并断言"查询语言在语料中存在"；多相关标注（每查询 ≥3 篇）才能让 rrf_k/权重产生区分度；`min_score` 改相对阈值 | 3d |

### 2.3 P2（12 条，摘要）

| # | 问题 | 证据 | 影响 | 建议 |
|---|---|---|---|---|
| P2-1 | 候选池固定 100，不随库规模自适应；limit 公式在生产与评测两处独立演算 | `search.py:221-222` vs `eval/harness.py:246-252` | 库到 1 万篇时候选不足；任一侧改动使 parity 静默失效 | 抽 `candidate_limit()` 一处 + `~sqrt(N)` 自适应 |
| P2-2 | `async` 路径用"每次新建 `ThreadPoolExecutor(max_workers=1)` + 新事件循环" | `retrieval.py:77`、`pipeline.py:219`、`tools.py:489` | 这是 `docs/VALIDATION.md` 记录的跨循环死锁故障的温床；每调用创建线程 | 改模块级单例 executor |
| P2-3 | 每线程 SQLite 连接默认 32MB 页缓存且工作线程连接永不 close | `db/connect.py:72-73`、`:91-112` | 工作线程数 × 32MB，理论上可达 GB 级 | 工作线程连接用后显式 close 或降 `cache_size` |
| P2-4 | 无界增长：`search_logs`/`agent_runs`/`run_steps` 只插不删 | `runs.py:96 delete_run` 零调用者；`search.py:270` 只插 | 长期自用必然膨胀；`run_steps` 单行 40KB（含草稿快照） | 加保留策略（如保留最近 50 次运行 + 90 天检索日志） |
| P2-5 | `run_steps` 快照含 `draft` 全文，单行 40KB+，无压缩 | 实测 5 条 step = 132KB | 几百次运行后 DB 臃肿 | 草稿只存 `artifact_id` 引用；旧 step 折叠 |
| P2-6 | SSE 用 200ms 轮询 `history`；多标签页各自独立轮询同一 list | `runtime.py:542` | CPU 与延迟浪费；`Event`/`Future` 跨循环的恐惧导致了这个设计 | 单事件循环内用 `asyncio.Event` 唤醒（记录创建 loop，跨 loop 才回落轮询） |
| P2-7 | 前端流式渲染每 120ms 整棵子树重建 + `scroll-behavior:smooth` 与 `scrollTop` 互殴 | `app.js` `clear(body)`+`renderMarkdown(整个 buffer)`；`style.css:737` | **不会卡死**，真实症状是持续掉帧 + 自动跟随莫名停住 + 用户选区每 120ms 被清 | rAF 节流 + 只重渲染尾部（保留稳定前缀）+ `scroll-behavior:auto` |
| P2-8 | 「思考中」气泡只在首个 token 时被替换，CSS 无限动画 | `app.js` 创建点唯一、消费点唯一 | 取消/失败/无 token 的运行永久转圈；正常运行时跨过审批等待一直显示"正在规划…" | 在 `onDone`/`cancelRun`/`onError`/`onAwaitingApproval` 都调 `settleThinking()` |
| P2-9 | 失败/超时的运行被报成"已完成" | `runtime.py:507/536/402` 失败路径也发 `done`（带 `phase:"error"`），`onDone` 不读 `data.phase` | 用户以为成功，绿色 chip "已完成" | `onDone` 按 `data.phase` 分支 |
| P2-10 | `api/registry.py` 泛 `except` 吞掉源错误类型；`_apply_vector_filters` 在 KNN 后置过滤会返回不足 k 条 | 代码注释已自认 | 统计口径与真实召回不符 | 保留 `SourceError` 子类型；KNN 时把过滤条件下推或超采样 |
| P2-11 | 迁移完整性在"有数据的库"被自动放行；失败不阻断启动；迁移用 DEFERRED BEGIN | `db/connect.py:164-176` `allow_checksum_change=has_data`；`migrate.py:500` 裸 `BEGIN` | 改过的已发布迁移在有库上静默放行 | 改由显式环境变量控制；迁移用 `BEGIN IMMEDIATE` |
| P2-12 | numpy 回退路径对 blob 长度零校验 | `search.py:135-137` `np.frombuffer(b"".join(...)).reshape(...)` | 一条坏 blob → 整个检索抛异常；2 万篇 × 768 维每次解 61MB | 分块 2048 条 + 长度校验 + 跳过计数 |

### 2.4 P3（文档与实现不符，8 处）

| # | 文档声明 | 实现/实测 | 严重度 |
|---|---|---|---|
| P3-1 | README「正文 `[n]` 与参考文献表严格一一对应」 | 实测 5/15 错位（P0-1） | **硬矛盾** |
| P3-2 | `ARCHITECTURE.md` 阴性对照"应返回 0 命中" | 实测返回 10.0（= `top_k`），`EVALUATION.md` 写对了 | **硬矛盾** |
| P3-3 | `ARCHITECTURE.md`「生产配置给标题精确匹配更高的 FTS 权重」 | `fts_weight` 全仓只有评测 harness 传值，生产恒 1.0 | **硬矛盾** |
| P3-4 | `ARCHITECTURE.md` 把 claim-level 校验画进请求链路 | `analyse_draft` 调用者只有评测脚本，**不在 agent 图里** | **硬矛盾** |
| P3-5 | `ARCHITECTURE.md` 把库规模"474 篇"写成消融语料 | 消融实际跑在 40/60/25 篇上 | 口径错 |
| P3-6 | `EVALUATION.md` 声称"两个发现已固化为测试" | `known-item` 在 `tests/` 里 grep 0 命中 | 无支撑 |
| P3-7 | README 测试数 1384 vs `ci.yml` 608；API 路径 57 vs 49 | 本机实测 1384 passed | 口径不一致 |
| P3-8 | README「缓存策略」含"不缓存 LLM 生成结果" | 属实，但 HTTP 层也零缓存、引用关系零调用者 | 表述不完整 |

> **这三处（P3-1/2/3/4）必须优先修**：它们是"声明与实现之间没有可执行对账"的直接体现，会连带削弱项目里所有真实材料的可信度。修完成本约半小时。

---

## 3. Agent 成本深度分析

### 3.1 单次综述的 token 成本模型（默认 config）

默认 `require_approval: true`、`critique_max_papers: 12`、`writer_max_papers: 25`、`review_min/max_chars: 4000/8000`、`num_ctx: 8192`，正文目标 8000 字 ÷ 5 节 = 每节约 1600 字 ≈ 1066 token。

| 阶段 | 输入 token（估） | 输出 token（估） | 说明 |
|---|---|---|---|
| Plan | ~2.5k | ~1.5k | `PLAN_SYSTEM` + schema + 课题；`chat_json(retries=2)` |
| Execute（检索） | 0 | 0 | 纯网络 + 嵌入 |
| Reflect | ~11k | ~3k | `CRITIQUE_SYSTEM` + 12 篇 × 600 字摘要（≈6.4k）+ 输出 schema；**每次 JSON 纠错重试整份重发** |
| Outline 细化 | ~4k | ~1.2k | 20 篇 × 280 字摘要 |
| Synthesize（5 节） | ~29k | ~5.3k | 单节 ≈5.8k 输入 + 1066 输出；`_fit_digest` 阶梯收缩材料 |
| Review | ~9k | ~1.2k | 草稿截 6000 字 + 400 字摘要 ×25 |
| Auto-revise | ~11k | ~2.7k | `DRAFT_TRUNCATE_REVISE=8000` + 材料 + 问题清单 |
| **合计** | **≈66k** | **≈15k** | **≈81k token / 次** |

**成本结论**：
- **本地 Ollama**：边际成本 0（`_is_local_model` 对带 `:` 的模型名返回 0），唯一成本是电费。墙钟 5~10 分钟。
- **云端 DeepSeek（¥2 / ¥8 per Mtok）**：`66k × 2 + 15k × 8` ÷ 1M ≈ **¥0.25**。README 说的"一次综述不到一毛钱"**对不上**——除非模型名落到 `qwen-plus`(0.8/2.0) 之类的更低价档，或走缓存命中价。**这是一处需要修正的成本口径**。
- 无效 token 占比：全文预取 100% 无效（P0-3）；评估漏斗 150→12 篇的启发式 + 125 篇的编号工作废弃（P1-2）；`_llm_assess` 排序重算启发式（P1-2）。

### 3.2 三个成本优化（按 ROI 排序）

**优化 1：让预算正比于"进入成稿的篇数"，而非"检索到的篇数"（P1-2 + P0-3）**

现状漏斗：

```
检索 150 篇 → 启发式评估 150 篇 → LLM 点评 12 篇 → 重编号 25 篇 → 成稿引用 8 篇
                ↑ 138 篇白算              ↑ 12/25 覆盖不均     ↑ 125 篇编号废弃
```

改法：
1. `critique_max_papers` 从固定 12 改为**由 context 预算反推**（`num_ctx` − 输出预算 − 系统提示词 ÷ 单篇摘要 token）；
2. LLM 评估**只对将进入 `select_papers` 的候选**做（即先按启发式粗排一次，再对 top-N 做 LLM 精评），使 `selected ⊆ LLM 覆盖`；
3. `_llm_assess` 复用已算的启发式分数（现在 `critic.py:352` 在排序 key 里重新调用 `heuristic_assessment`）。

**收益**：LLM 评估从"覆盖 12/25 且丢弃 88% 输入"变成"覆盖 25/25"，评估阶段 token 几乎不变（12→min(25, 预算允许)）但**质量提升**（消除"是否被 LLM 看过"这一隐性 selection bias）；同时消掉 138 篇启发式冗余计算。

**优化 2：提示词前缀重排 → 命中云端前缀缓存（免费，收益最大）**

现状：`section_user()` 的拼接顺序是「课题 → 章节标题 → 要点 → **材料块** → 字数要求」，而系统提示词在每个请求里都一样。5 个章节的请求，前缀在第 3 段（章节标题）就分叉，**缓存命中率 0**。

DeepSeek / OpenAI 的自动前缀缓存要求**前缀逐字节相同**。把顺序改成：

```
[SECTION_SYSTEM（固定）] + [材料块 digest（5 节完全相同）] + [字数要求（固定）] + [章节标题 + 要点（唯一变化部分）]
```

后 4 个章节的输入全部命中缓存（DeepSeek 缓存命中价约为标准价 10%）。

**收益**：Synthesize 阶段输入从 ~29k 降到 ~6k（首个章节全价 + 4 节缓存价），**总输入 token 降约 35%，总成本降约 30%**。改动量：`llm/prompts.py` 的 `section_user()` 调换参数顺序，约 10 行。

**优化 3：流式失败也记账 + 按运行聚合，才能定位成本（P1-4 + P1-5 + P0-4）**

现在无法回答"这次综述花了多少钱"。修 P0-4（trace 挂上）后，`/api/metrics` 立刻能给出 `by_phase` 的真实分布，这才是做成本优化的前提。**先有度量，再谈优化**——这本身就是面试可讲的判断。

### 3.3 成本与质量的取舍矩阵

| 档位 | 配置 | 单次 token | 云端成本 | 墙钟（本地 8B / 云端） | 质量 |
|---|---|---|---|---|---|
| 极简 | 小模型 + `critique_max_papers: 6` + 4000 字 | ~50k | ≤¥0.05 | 2~4 min / 40s | 中（引用纪律可能破坏，`llama3.2:3b` 会把 `[n]` 原样写进正文） |
| **默认（推荐）** | 当前 config | ~81k | ~¥0.25 | 5~10 min / 2~3 min | 良 |
| 高质量 | 双模型路由（规划/批判用 pro，正文用 flash）+ 25 篇评估 + 8000 字 | ~150k | ~¥0.5 | 8~15 min / 3~5 min | 优 |
| 生产级（应做） | 上述 + prompt cache + 按需全文 + 评估覆盖对齐 | ~65k | **~¥0.15** | 5~10 min / 2 min | 优 |

> 关键洞察：**"更高质量"和"更低成本"在这里不冲突**——把评估覆盖面从 12 提到 25、把全文用起来，同时靠前缀缓存与漏斗对齐把 token 压下来。真正冲突的是**延迟**（本地串行解码是硬件天花板：272 GB/s ÷ 6.19 GB ≈ 44 tok/s，实测 45~47，已证无优化空间）。

---

## 4. 性能分析

| 环节 | 现状 | 瓶颈 | 优化 |
|---|---|---|---|
| LLM 生成 | 本地 45~47 tok/s（硬件极限，README 已算清） | 显存带宽，**无优化空间** | 不优化；把预算投到"不产生价值的时间" |
| Synthesize 墙钟 | 5 节串行 ≈ 2~4 min（本地） | 串行流式 | **章节并发（信号量 2~3）**：本地因带宽受限不会线性加速，但云端可 3~5 倍降低墙钟。**质量-成本-延迟三角**：并发会同时抬高瞬时成本 |
| 检索 | 4 源并发 ~10.5s（README 实测） | 网络等待 | 修 P0-2 后 Europe PMC 复活，数据源命中率 3/5 → 5/5，时间升至 ~13-15s——**用时间换召回，值得** |
| 本地知识库检索 | `hybrid_search` 同步阻塞事件循环（P0-6） | 同步 SQLite | 统一 `to_thread`；`log_search` 30 次写事务 → 1 次 |
| FTS 召回 | 级联短路丢弃 760 条候选（P1-1） | 设计缺陷 | 并集 + 相对分阈值 |
| 候选池 | 固定 100（P2-1） | 不自适应 | `~sqrt(N)` 自适应 |
| 前端加载 | 260KB 每次刷新全量重下+解析（P1-9） | `no-store` 与 `?v=` 互斥 | `immutable` + GZip |
| 前端流式渲染 | 每 120ms 整棵子树重建（P2-7） | 全量 re-parse | rAF + 只渲染尾部 |
| SSE 唤醒 | 200ms 轮询（P2-6） | 跨循环恐惧 | 同 loop 用 `asyncio.Event` |

---

## 5. 健壮性与稳定性

**做得好的（应保留并讲）**：
- 全抖动指数退避（防重试风暴）、熔断按"调用"而非"尝试"计数、401 绝不重试、流式不重试（避免文字重复两遍）——这四条是**真实踩坑换来的设计**，含金量高。
- 事件循环归属检测（`_client_loop`）——跨循环死锁这个坑修得干净。
- 失败分类 11 类 + 成功失败都记账（虽在流式路径破功，见 P1-5）。
- RAG 注入防御放在四条链路**共用的唯一出口**（`build_context_digest`），一处生效不漏——这是正确的架构直觉。
- 阶段快照 + 断点续跑；启动时把"运行中"标成 `interrupted`。
- schema 迁移校验和、迁移前备份、dry-run。

**关键缺口**：
1. **恢复路径不完整**（P0-5 + P0-7 + P1-8）：断线可补播（后端对了），但前端不幂等、刷新丢直播、"继续"会并发。合起来是"5~10 分钟的任务不能刷新"这条硬限制。
2. **单写者争用**：`BEGIN IMMEDIATE` × 30（P0-6）+ 进程内 RLock；WAL 下读者不受影响，但写路径串行。
3. **向量重建是破坏性的**（P1-10）。
4. **配置可被静默覆盖**（P1-12）。
5. **无鉴权 + CORS `allow_origins` 可空则 `["*"]`**：绑 `127.0.0.1` 是可接受的边界，但 `cors_origins` 默认写死 8760 端口——改端口后 CORS 失效，而代码会回落到 `["*"]`。

---

## 6. 优化路线图（按 ROI 排序）

### 第 0 阶段：止血（1 天，8 处小改）
| # | 改动 | 工时 | 收益 |
|---|---|---|---|
| 1 | 前端 `api.metrics()` → `API.metrics()` | 10min | 恢复成本面板（唯一 UI 出口） |
| 2 | `query.py` 增加 `AND` 识别 + 表达式合法性断言 | 0.5d | Europe PMC 复活，召回 +40% |
| 3 | `_llm_assess` 复用启发式分数、`critique_max_papers` 按预算反推 | 0.5d | 评估覆盖 12→预算值，消除 selection bias |
| 4 | 前端 `settleThinking()` 四处补调 + `scroll-behavior:auto` | 30min | 消除"永久转圈"与"跟随失效" |
| 5 | 静态资源 `immutable` + GZip | 0.25d | 刷新 260KB 全量重下 → 0 |
| 6 | `/api/agent/latest` 合并内存 live 状态 | 0.25d | 恢复入口正确性 |
| 7 | `resume()` 幂等（活句柄检查） | 0.25d | 消除并发双跑 + 内存泄漏 |
| 8 | 修 4 处文档硬矛盾（P3-1~4） | 0.5h | 消除面试最易被抓的矛盾 |

### 第 1 阶段：正确性（3~4 天）
9. **P0-1 引用编号单一真相源** + 端到端断言（0.5d）——**最高优先级，它让每一篇产出都是错的**
10. **P0-3 全文接上或摘掉**（1.5d / 0.5d）
11. **P0-4 trace 挂上** → 账本 phase/run_id 生效（0.5d）
12. **P0-6 统一 `to_thread`** + check_arch AST 规则焊住（1d）
13. **P0-7 + P1-8 前端 `attachStream` 唯一入口** + 后端 `id:`/`Last-Event-ID`（1d）

### 第 2 阶段：成本与质量（3~5 天）
14. **提示词前缀重排** 命中云端缓存（10min，成本 −30%）
15. **双模型路由** `LLMSettings.routing`（1d，成本 −40% 或质量 +1 档）
16. **流式失败也记账**（0.25d）
17. **P1-1 FTS 级联改并集**（0.5d，召回 +40%）
18. **P1-14 评测语料多语言 + 多相关标注**（3d，让所有检索结论重新成立）

### 第 3 阶段：健壮性（2~3 天）
19. Bulkhead 挂上 + `gather` 超时 + `Retry-After` 上限（1d）
20. 限流改 `min()` + S2 恢复路径（0.5d）
21. HTTP 缓存 + 引用关系落库（1.5d）
22. 影子向量索引（1.5d）
23. 无界表保留策略（0.5d）

**总投入约 12~16 人天**，其中第 0+1 阶段（4~5 天）解决全部 P0 并恢复成本可观测性。

---

## 7. 面试可讲性：哪些结论站得住

### 站得住（可以讲，附口径）
1. **"评测跑的是生产原语"** —— `harness.verify_production_parity` 有可执行证据（把 `hybrid_search` 换成倒序仍能发现）。**但要说"由单测强制"，不要说"CI 门禁"**：CLI 侧只打印不改退出码。
2. **"CI 回归门禁与真实质量评测刻意分离"** —— CI 用 `--embed-provider hashing` 跑确定性回归 + 阈值门禁；真实质量评测在本机。这是成熟团队的判断。
3. **指标实现数学正确** —— `dcg_at_k` 折损、`ndcg_at_k` 的 IDCG 取理想排序、支持分级相关性、无标注返回 `None` 而非 0、`retrieved_ids` 去重保序。已逐函数核对。
4. **"我测了我自己的校验器"** —— 27 条人工标注 + 混淆矩阵 + 逐类 P/R + 门禁。方向正确且真抓到过 bug（CJK 贪婪匹配导致中文规则静默失效）。
5. **数字可复现、无造假** —— `eval_faithfulness.py --labels` 实跑与文档逐位一致（24/27=88.9%、P=0.89、R=1.00、acc=0.93）。
6. **硬件天花板算清后停止瞎调** —— 272 GB/s ÷ 6.19 GB ≈ 44 tok/s，实测 45~47；`num_ctx` 8192/4096/2048 吞吐 46.7/46.5/46.9 无差异。**这是全项目最值钱的一条工程判断。**

### 站不住（必须主动降级表述）
1. **不说**"7 配置消融证明 RRF k=60 最优" → 说"BM25 在标题派生查询上更强（nDCG +7.9pt）、向量在自然语言查询上更强（+6.1pt）；融合配置在单相关小语料上无法区分"。
2. **不说**"recall@10 饱和说明小语料没区分度" → 项目自己的 40 篇语料上 recall 有区分度（0.85/0.91/0.89）。限定为"标题派生查询 + 该本地库语料下饱和"。
3. **不说**"跨语言/中文能力已评测" → 语料 40/40 全英文，这些查询实测命中 0。
4. **不说**"校验器零漏报/类别级 P=R=1.00" → 说"27 条无歧义案例上可判定子集零漏报、精确率 0.89；已知盲区是语义改写"。`unverifiable` 被踢出问题集，`fn=0` 不等于检测力完美；逐类 support 为 6/2/3/**1**，n=1 的类仍印满比率。
5. **不说**"claim-level 校验已接进产品" → 说"实现完成并自评估，尚未接入生成链路"。
6. **不说**"换模型自动校正维度无代价" → 主动说"DROOP 旧表是代价，修法是影子索引"。
7. **不说**"正文 [n] 与参考文献表严格一一对应"（这是 P0-1）。

> **最值钱的姿态**：主动说出 P0-1（编号错位）和 P1-1（级联短路）这两个自己找出来的缺陷。**能自己指出评测与产出的失效边界，比任何指标数字都更能证明工程判断力。**
