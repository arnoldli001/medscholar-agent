# 验证报告

> **这是一份历史快照，不是当前状态。** 本文记录的是**早期一次完整验证**的过程与结果
> （当时的用例数是 307、数据源 7 个、引用样式 5 种）。保留它的价值在于
> **方法论与真实日志**：每一节都写了"怎么验的、发现了什么真实缺陷、怎么修的",
> 这部分内容至今没有过时。
>
> 但它里面的**数字已经变了**，引用前请以当前状态为准（README 顶部的规模表、
> `pytest tests -q`、`scripts/check_arch.py`）：
>
> | 项 | 本文快照 | 当前 |
> |---|---|---|
> | 单元测试 | 307 项 | **1374 项** |
> | 数据源 | 7 个 | **9 个**（+Crossref / DOAJ / CORE） |
> | 引用样式 | 5 种 | **6 种**（+Chicago） |
> | HTTP 契约 | 71 项 | 71 项（未变） |
> | 架构约束 | 当时还没有校验器 | `ARCH: PASS`（层次 / 循环 / 规模三类检查） |
>
> 我不会去"更新"下面的日志行 —— 那些命令行输出是**当时真实跑出来的**，
> 改掉它们等于伪造证据。当前状态请用命令现场复测。

本文档记录 MedScholar Agent v1.0.0 的**实际验证过程与结果**，包括验证方法、
发现的真实缺陷及其修复。所有结论都来自实际运行，不是设计意图的复述。

---

## 一、验证手段

| 手段 | 脚本 | 覆盖范围 | 依赖 |
|---|---|---|---|
| 静态自检 | `scripts/check.py` | 语法编译、模块导入、纯函数断言 | 无 |
| 数据层冒烟 | `scripts/smoke_db.py` | 建表 → 入库 → 三路检索 → 统计 | 无 |
| **单元测试** | `pytest`（307 项） | 文本处理、数据层、引用、去重、解析器、Agent、配置 | 无 |
| 数据源联调 | `scripts/smoke_api.py` | 7 个数据源真实请求 | 联网 |
| **HTTP 契约测试** | `scripts/smoke_http.py`（71 项） | 全部 REST 端点 + SSE + 人工审批往返 | 无（离线模式） |
| **端到端测试** | `scripts/e2e.py` | 真实课题全流程：检索 → 嵌入 → 混合检索 → 六智能体工作流 → 成稿 | 联网 + Ollama |
| 环境自检 | `medscholar doctor` | 逐项检查运行时、扩展、模型、数据源连通性 | 可选联网 |

**为什么 HTTP 契约测试要打真实 uvicorn 而不是 `httpx.ASGITransport`**：
ASGITransport 会把整个响应体缓冲完再返回，而 SSE 是无限流，
在进程内用它测 SSE **必然死锁**（这是实测踩到的，不是理论推断）。

---

## 二、验证结果

### 2.1 单元测试：307 项全部通过

```
$ .python\python.exe -m pytest -q
307 passed in 3.25s
```

### 2.2 HTTP 契约测试：71 项全部通过

```
$ .python\python.exe scripts\smoke_http.py
结果：通过 71 项，失败 0 项
SMOKE_HTTP: PASS
```

覆盖：健康检查、配置脱敏、7 个数据源枚举、文献 CRUD、混合检索、全文、
引用（5 种样式）、导出（5 种格式）、课题管理、会话与产物、维护接口、
静态资源、SSE 全部事件类型、人工审批（批准/取消/重复审批 409）。

### 2.3 端到端（真实课题「加速rTMS治疗卒中后抑郁」）

| 环节 | 结果 |
|---|---|
| 嵌入后端 | `ollama/nomic-embed-text`，768 维，3 条 0.3s |
| LLM 后端 | `ollama/qwen3:8b`，CPU 推理 |
| 规划（Plan） | 4 条检索式 + 5 个章节大纲，45.5 s |
| 多源检索 | 原始 **383 条 → 去重 320 篇**，4 个数据源全部命中 |
| 中文文献通路 | OpenAlex `language:zh` 返回 30 篇真实中文文献 |
| 入库 + 真实向量 | 新增 273 篇 / 更新 47 篇 |
| **嵌入吞吐** | **273 篇 / 5.6 秒**（约 49 篇/秒），失败 0 |
| 向量覆盖率 | 100%（后端 `sqlite-vec v0.1.9`） |
| 开放获取全文 | 3 篇，40736 / 49314 / **139960** 字 |
| 批判性评估（Reflect） | 150 篇中 93 篇纳入写作，整体证据质量「低」，95 s |
| **撰写（Synthesize）** | 大纲细化为 5 章 → 逐章流式生成，共 **7872 字** |
| **自我审查（Review）** | `pass`，**9.0/10**，发现 1 个问题 |
| **引用校验** | **通过**：正文引用 **18 篇，全部可对应参考文献表**，零越界引用 |
| 产物落库 | `artifact_id=1`，可在「产物」面板调阅 |
| **全流程耗时** | **721 秒（约 12 分钟）** |

**引用零越界的意义**：这是 LLM 写综述最容易翻车的地方（编造 `[27]` 这种不存在的编号）。
本项目用三重机制保证：① 写作前按筛选结果**重新连续编号**；② 写作后 `_sanitize()`
剔除所有越界编号；③ Formatter 再独立校验一次。端到端实测 18 条引用全部合法。

生成草稿的实际片段（节选）：

> ## rTMS在卒中后抑郁中的应用现状
>
> rTMS在卒中后抑郁（PSD）中的应用现状显示，其具有一定的临床应用价值。现有文献表明，
> rTMS能够改善PSD患者的抑郁症状，尤其是在针对左侧背外侧前额叶皮层（DLPFC）的
> 高频（HF）rTMS治疗中表现出较好的疗效 [2][3]。……
>
> 在具体疗效方面，一项系统综述与网络meta分析（NMA）纳入了12项研究，涉及四种不同的
> rTMS模式，结果显示，不同rTMS模式对PSD的疗效存在显著差异 [2]。……

中文查询「rTMS 治疗卒中后抑郁的疗效」的 TOP5 —— 注意**全部为双路融合命中**，
且都是真正相关的中文文献（不是靠查询里那个英文词 `rTMS` 蹭到的）：

```
score=0.03175 [bm25+vector] bm25#4  vec#2  针刺治疗脑卒中后抑郁机制研究进展
score=0.03154 [bm25+vector] bm25#6  vec#1  分析用超低频经颅磁刺激(ILF-rTMS)疗法、帕罗西汀联合治疗抑郁症的近期疗效
score=0.03150 [bm25+vector] bm25#3  vec#4  多靶点无创神经调控治疗脑卒中后抑郁状态的临床观察
score=0.03128 [bm25+vector] bm25#2  vec#6  高频重复经颅磁刺激联合高压氧治疗脑卒中后抑郁效果观察
score=0.03028 [bm25+vector] bm25#1 vec#12  针刺"五心穴"联合四逆散治疗缺血性脑卒中后抑郁临床疗效观察
```

**修复前**同样的查询返回的是「Accelerated and Intensive rTMS Treatment Protocols」这类英文文献，
原因是中文二元组检索式永远匹配不到（见 P0-2），实际只靠查询里的 `rTMS` 一词命中。

---

## 三、验证中发现并修复的真实缺陷

这些不是"写的时候就知道"的问题，而是**跑起来才暴露**的。按严重程度排列。

### 🔴 P0-0 三个 `.bat` 启动器在中文 Windows 上完全无法运行

**现象**（用户实测）：双击 `run-doctor.bat` 满屏报错，且中文全是乱码：

```
"dp0" 不是内部或外部命令，也不是可运行的程序
e" set "PY" 不是内部或外部命令，也不是可运行的程序
系统找不到指定的路径。
cho 不是内部或外部命令，也不是可运行的程序
```

`run.bat` 同理。**这是交付层的致命缺陷 —— 朋友拿到包根本打不开。**

**根因（两个叠加的问题）**：

1. **批处理文件里不能放中文。** cmd.exe 用**控制台 OEM 代码页**（中文 Windows 是 936/GBK）
   解析 `.bat`，而文件是 UTF-8。更糟的是：汉字在 GBK 下常以「前导字节」结尾，这个字节会把
   紧随其后的 CR 当作自己的第二个字节**吞掉** —— 行边界因此错乱，下一行命令被截断执行，
   于是 `cd /d "%~dp0"` 变成 `"dp0"`、`echo` 变成 `cho`。`chcp 65001` 救不了：
   cmd 的批处理解析器在 65001 下有已知缺陷。
2. **文件是 LF 换行，不是 CRLF。** cmd.exe 要求批处理用 CRLF；LF-only 会让 `goto` /
   标签跳转解析异常。生成文件时没有指定换行符。

**修复**：三个 `.bat` 改成**纯 ASCII + CRLF**，只保留「找解释器 → 转发参数 → 出错时 pause」
这几件纯逻辑的事；**所有中文提示移到 Python**（新增 `scripts/bootstrap.py`）—— Python 在
Windows 上通过 `WriteConsoleW` 直接写 Unicode，与控制台代码页无关，任何 locale 都不会乱码。

**修复后实测**：解压分享包 → 双击 `run-doctor.bat` → 中文自检全部正常；双击 `run.bat` →
服务启动、`GET /` 返回 200。

---

### 🔴 P0-1 嵌入提供方跨事件循环复用 HTTP 客户端 → 请求永久挂起

**现象**：78 篇文献入库后卡住不动。15 分钟后数据库里 `paper_embeddings` 仍为 **0 条**，
而单独测 Ollama 批量嵌入（16 条）只需 1.0 秒。Python 进程 CPU 时间仅 1.3 秒 —— 是阻塞，不是计算。

**根因**：`OllamaEmbedding` 缓存了 `httpx.AsyncClient`。该客户端的连接池、锁与流都绑定在
**创建它的那个事件循环**上。而 FastAPI 的 `/api/search/live`、`/api/maintenance/embed`
都通过 `asyncio.to_thread` 调用同步封装 `insert_papers(embed=True)`，
后者在线程里 `asyncio.run()` **新起了一个事件循环** —— 复用旧循环的客户端，请求就永远不返回
（既不报错也不超时）。

**影响面**：Web 端的「联网检索并入库」「补全向量」两个功能会直接卡死。

**修复**：`OllamaEmbedding.embed()` 每次调用新建短连接客户端（批量嵌入次数很少，
连接复用的收益远小于正确性风险）；同时给 `LLMClient` 与 `BaseClient` 加了事件循环归属校验，
检测到循环变化就丢弃旧客户端重建；并给嵌入请求加了硬超时，保证这类问题**最多超时，不会无限挂起**。

**回归防护**：`e2e.py` 的入库步骤显式走 `asyncio.to_thread(repo.insert_papers, ..., embed=True)`
这条路径并设置 600s 超时，`test_db`/`test_agent` 覆盖了同步封装。

---

### 🔴 P0-2 中文二元组检索式永远匹配不到任何东西

**现象**：中文查询的 BM25 一路几乎全空（`bm25#None`），只有查询里带英文词（如 `rTMS`）时才靠英文词命中。

**根因**：索引侧 `segment_cjk()` 把汉字切成**单字 token**（`加 速 治 疗`），
而查询侧 `bigram` 策略生成的是**双字 token**（`"治疗"`）—— 索引里根本不存在这样的 token，
所以二元组这一级永远是死代码。

**修复**：二元组改为「两个单字 token 组成的短语」：`"治 疗"` 而不是 `"治疗"`。

**回归防护**：`test_textutil.py::test_cjk_bigram` 断言精确字符串；
`test_produces_valid_fts5_syntax` 参数化验证构造出的表达式真的能被 FTS5 解析并命中自身。

---

### 🔴 P0-3 环境变量劫持 LLM 后端 → 前端报「大语言模型不可用」

**现象**（用户实测）：前端横幅提示 LLM 不可用，展开后是：

```
deepseek 返回 HTTP 400：{"error":{"message":"The supported API model names are
deepseek-flash, deepseek-v4-pro, but you passed qwen3:8b."}}
```

用户本地明明装着 `qwen3:8b`，配置里写的也是 `provider: ollama`，为什么请求发去了 deepseek？

**根因**：`_env_overrides()` 里有一行自以为贴心的逻辑 ——
「看到 `DEEPSEEK_API_KEY` 就自动把 provider 切成 deepseek」。
而用户机器上**另一个项目**（`Moss-finagent-research`）把 `DEEPSEEK_API_KEY`
设成了**用户级环境变量**，于是：

1. provider 被悄悄改成 `deepseek`；
2. `model` 仍是 config.yaml 里的 `qwen3:8b`（Ollama 的模型名）；
3. `_resolve_base_url()` 又把 base_url 从 Ollama 地址改写成云端地址；
4. 三者组合必然 400，且云端返回的是英文错误、不提示怎么改。

**修复**（三步）：

1. **删掉自动切换逻辑。** 环境里有某个 Key，不等于用户想让本程序用它。
   后端从此**只由 config.yaml 决定**；Key 仅在被显式选用时生效。
2. **新增 provider/model 一致性校验** `LLMSettings.consistency_error()`：
   云端 provider 配着 Ollama 风格模型名（含 `:` 或 `-7b` 结尾）时，
   在任何调用路径（Web / CLI / MCP）都给出**可照做的中文提示**，
   而不是让用户去猜云端的英文 400。
3. `.env.example` 与 README 补充说明「为什么不看环境变量自动切后端」。

**回归防护**：新增 24 项测试（`TestLlmConsistency` + `TestEnvOverrides`），
覆盖 11 组 provider/model 组合、消息可操作性、以及「注入 Key 后 provider 不变」。

**实测验证**：把用户级 `DEEPSEEK_API_KEY` 注入进程后启动服务，
`/api/health` 返回 `llm.ok=true`、`provider=ollama`、`model=qwen3:8b`。

---

### 🟠 P1-1 中文精确短语检索过严 → 常见改写查不到

**现象**：库里存的是「针刺治疗**脑**卒中后抑郁机制研究进展」，用户查「治疗卒中后抑郁」时 BM25 返回 0 条。

**根因**：只用「连续子串」一种策略。中间的「脑」字一插，短语就断了。

**修复**：改为**逐级放宽**：`phrase`（精确子串）→ `bigram`（二元组 AND）→ `bigram OR`（召回兜底），
任何一级有结果就停。常见查询仍只跑一次 FTS（毫秒级）。

**效果**：端到端验证中，中文查询「rTMS 治疗卒中后抑郁的疗效」的 TOP5 全部变为 `bm25+vector` 双路融合命中。

---

### 🟠 P1-2 离线模式形同虚设 —— 离线运行仍会真的联网

**现象**：以 `offline: true` 启动的 Agent 运行，仍然发起了真实的学术库请求。

**根因**：`SourceRegistry` 是**跨运行共享的单例**，它的 `self.config.offline` 是**第一次创建时**的配置。
而每次 Agent 运行可能有自己的离线开关（运行时通过 `model_copy` 复制配置），
这个逐次设置从未传达到注册表。

**修复**：`SourceRegistry.search()` 增加显式 `offline` 参数，由 Scout 逐次传入。

---

### 🟠 P1-3 对同步函数 `await` → 向量生成静默失败

**现象**：日志里出现 `向量生成失败（不影响关键词检索）：object EmbeddingReport can't be used in 'await' expression`。

**根因**：`run_embedding_pipeline` 是**同步**封装（内部自己处理事件循环），
但 `scout._persist()` 与 `/api/maintenance/embed` 都写了 `await run_embedding_pipeline(...)`。

**修复**：改用异步版本 `run_embedding_pipeline_async`。

**备注**：这个错误的**降级处理是正确的** —— 向量生成失败没有阻断入库，工作流继续跑到成稿。
但功能确实没生效，属于必须修的问题。

---

### 🟠 P1-4 CNKI 解析器把「作者行」当成「摘要」

**现象**：「若页面结构恢复则应能解析」的测试失败：`abstract == "作者:王伟;李静;"`，
且 `authors` 里混进了摘要全文。

**根因**：摘要正则把 `class="info"`（作者行）也算了进去，而它在文档里排在 `class="summary"` 之前；
作者正则用 `[^<\n]{1,200}` 贪婪匹配，剥掉标签后会把后面的摘要整段吞下。

**修复**：摘要只认 `summary|abstract`；作者/关键词改为独立匹配 `info` 段落，
且限定在分号分隔的、每段 ≤60 字的片段内。

---

### 🟡 P2-1 Europe PMC 全文 URL 形态错误

**排查过程**：

| URL | 结果 |
|---|---|
| `/PMC/PMC7431489/fullTextXML` | 404 |
| `/MED/32849235/fullTextXML` | 404 |
| `/PMC7431489/fullTextXML` | **200，89 KB JATS 正文** ✅ |

即全文端点**不带 source 段**。修复后已有 3 篇全文成功抓取（4~5 万字/篇）。

（同期发现：Europe PMC 的 `references` 端点在验证期间返回
`503 This API is temporarily unavailable due to maintenance.` —— 服务端维护，非本项目缺陷，代码已做优雅降级。）

---

### 🟡 P2-2 arXiv 相关性差

**现象**：长自然语言查询返回「High-Fidelity Transcranial Ultrasound」「Head phantoms for EEG」等完全无关的论文。

**根因**：`all:"整句" OR all:整句` 的宽松 OR 会让 arXiv 返回anything；
而单用精确短语又常常 0 命中。

**修复**：渐进放宽查询（精确短语 → 全部实词 AND → 前 3 个实词 AND），
并对结果做**词面相关度校验**（标题+摘要中至少命中 2 个实词）。
修复后返回「Personalized rTMS for Depression: A Review」等真正相关的论文。

---

### 🟡 P2-5 右栏引用列表只剩一个编号

**现象**：端到端跑完后，`artifact` 事件里的 `reference_entries` 每条的 `text` 字段
只输出 `[1]` `[2]` `[3]`，前端「引用列表」因此只显示编号、看不到文献信息。

**根因**：`FormatterAgent.reference_entries()` 在数字制样式下用了 `format_inline()`
（那是**文内引用短标**，数字制下就是 `[1]`），而应该用 `format_citation()`
输出完整参考文献。同一分支在作者-年份制样式下恰好是对的（`format_inline` 返回
`(Zhang, 2023)`，看起来像正常内容），所以只看一种样式不容易发现。

**修复**：`text` 统一用 `format_citation()`；另外单独提供 `inline` 字段给需要
文内短标的场景。新增回归测试断言 `text` 必须包含标题与期刊名、长度 > 40。

---

### 🟡 P2-3 引用键去重后缀失效

**现象**：同一篇文献连续两次生成 BibTeX 引用键，得到完全相同的结果。

**根因**：`citation_key()` 里写的是 `if not taken: return base` ——
**空集合是 falsy**，所以即使调用方传入了 `taken=set()`（意图去重），
第一次也直接返回且不登记，第二次自然还是同一个键。

**修复**：改为 `if taken is None`。

---

### 🟡 P2-4 GB/T 7714 的序号前缀被丢弃

**现象**：GB/T 7714 输出的参考文献没有 `[1]` `[2]` 序号。

**根因**：`prefix` 变量算出来了却从未拼进返回值（Vancouver 分支用到了，GB/T 分支漏了）。

---

### ⚪ P3 其他修正

| 问题 | 修复 |
|---|---|
| `sqlite_vec.load(conn)` 报 `not authorized` | 它**不会**自行开启扩展加载权限，必须先 `conn.enable_load_extension(True)` |
| `Paper` 缺 `full_text_path` 字段导致入库崩溃 | 补齐字段并把 `from_row` 改为容错读取 |
| `count_papers()` 忽略 `project_id` | 按课题筛选时 `total` 恒等于全库总数（`items` 却是对的），补齐过滤条件 |
| `LLMClient` 必须先 `await start()` 才能 `chat()` | 改为自动初始化，消除调用方踩坑 |
| `RunHandle` 用 `asyncio.Event` 做唤醒 | Event/Future 绑定创建它的事件循环，跨循环访问会静默死锁；改为按 200ms 轮询 `history` 列表 |
| 中文向量维度变更需手工删表 | 新增维度自探测与自动重建，换 `bge-m3`（1024 维）只需改一行配置 |

---

## 四、环境层面的实测结论（非代码缺陷）

| 结论 | 证据 | 处置 |
|---|---|---|
| **CNKI 公开检索不可用** | `/Search/Result` 返回 200 但只有 JS 外壳（零条文献记录，无任何 `kns.cnki.net`/`dbcode=` 链接）；`/Search/ListResult`、`kns.cnki.net/kns8` 均返回 **403** | 客户端保留实现但默认关闭，被调用时抛出**带处置建议**的错误；改用 OpenAlex `language:zh`（实测 367 条真实中文文献）+ PubMed `chinese[la]` + Crossref 三条通路 |
| **Semantic Scholar 无 Key 限流严重** | 实测频繁 `HTTP 429`，重试后仍可能失败 | 保留自适应降速（连续 429 自动把速率减半，下限 0.05 次/秒）与单源失败隔离；文档中强烈建议申请免费 Key |
| **`nomic-embed-text` 跨语言能力弱** | 同主题跨语言余弦 0.3788 **低于** 无关主题同语言 0.5832 —— 模型按**语言**聚类而非按主题 | 默认配置保持 768 维（与需求文档一致、开箱可用）；README 显著标注该限制并给出 `bge-m3`（1024 维，多语言）的切换方法；维度自动适配已实现 |
| **`sqlite_vec` 0.1.9 的 `load()` 有坑** | 见上表 P3 | 连接层自行 `enable_load_extension`，失败时自动回退纯 Python 向量检索 |
| **Python 环境隔离** | 系统 Python 位于工作区外，沙箱限制下无法直接执行 | 构建了自包含便携运行时 `.python/`，同时成为分享包的一部分（朋友无需安装 Python） |

---

## 五、尚未验证 / 未实现

诚实列出，避免误以为已覆盖：

1. **Web 前端未在真实浏览器里点过**：前端由子代理交付并做了 headless DOM 测试（评审路径 42/42、附加路径 20/20、Markdown/XSS 17/17），但**没有人用鼠标实际点击过**。`run.bat` 启动、`GET /` 返回 200 已实测，页面内的交互流程待人工确认。
2. **P2 定时追踪未实现**：`subscriptions` 表结构已就位，但没有调度器与简报生成。
3. **PDF 解析未实测**：PyMuPDF 是可选依赖，本次验证走的是 Europe PMC JATS 通道。
   代码路径存在但未跑过真实 PDF。
4. **MCP Server 未与真实客户端联调**：`mcp` 包为可选依赖，未安装；
   工具 schema 与注册逻辑已就绪，但没有用 Claude Desktop 实际连过。
5. **Web 前端为子代理交付并独立验证**（5135 行，三文件）：它做了 headless DOM 测试
   （评审路径 42/42、附加路径 20/20、Markdown/XSS 17/17），但**没有人在真实浏览器里点过**。
6. **前端未做浏览器兼容矩阵测试**：仅按标准 API 编写。
7. **未做大库压力测试**：最大规模为端到端验证中的 252 篇；万篇级的知识库性能未测。
8. **macOS / Linux 未验证**：便携运行时是 Windows x64。
