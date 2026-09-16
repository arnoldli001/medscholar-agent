# MedScholar Agent

**面向医学研究者的本地化学术智能体** — 自动检索文献、构建本地可语义检索的知识库、辅助综述与论文写作、并把用户的反馈变成系统的长期记忆。

[![CI](https://github.com/arnoldli001/medscholar-agent/actions/workflows/ci.yml/badge.svg)](https://github.com/arnoldli001/medscholar-agent/actions/workflows/ci.yml)

> 一句话概括工程目标：**让「AI 辅助写作」在医学场景下可信、可审计、可复现。**
> 医学写作的失败代价是学术不端，所以这个项目的重点不在"生成得多漂亮"，
> 而在**幻觉治理**：引用必须对得上、数字必须有出处、错误必须被记住。

| | |
|---|---|
| 语言/规模 | Python 16.4k 行 · 原生 JS 4.2k 行 · 测试 5.2k 行 |
| 测试 | **687 个测试**，CI 全绿（**6 个作业**：lint / test / **eval** / smoke / package / test-linux） |
| 检索评测 | recall@k · nDCG@k · MRR · MAP + 7 种配置消融 + 阴性对照；**每次 CI 跑门禁** |
| 后端 | FastAPI + uvicorn（异步）· Pydantic v2 · httpx |
| 存储 | SQLite（16 张表 + 2 张 FTS5 虚拟表）· sqlite-vec · 单文件、可整目录拷走 |
| 检索 | FTS5 BM25 ⊕ sqlite-vec KNN → RRF(k=60) · 中文字符级切分 |
| 模型 | Ollama（本地，默认）/ DeepSeek / 任意 OpenAI 兼容端点 · 多模态嵌入回退 |
| 数据源 | **9 个官方检索源** + Unpaywall（DOI→OA 全文）· 限流/退避/自适应降速 |
| 交付 | 自包含便携运行时，双击 `run.bat` 即用；朋友无需装 Python |

**先看这几份文档**：架构深挖与选型理由见本文
[关键设计决策](#关键设计决策选型与权衡) 与 [深水区问题](#深水区问题与解决)；
**检索质量怎么量化**见 [`docs/EVALUATION.md`](docs/EVALUATION.md)；
面试问答准备见 [`docs/INTERVIEW-FAQ.md`](docs/INTERVIEW-FAQ.md)；
HTTP 契约见 [`docs/API.md`](docs/API.md)。

---

## 项目要解决的问题

医学研究生/临床医生写综述与论文时的真实痛点，按痛苦程度排序：

1. **检索分散** — PubMed、Europe PMC、Scopus、CNKI 各查一遍，还要手动去重；
2. **证据不可追溯** — 读了几十篇，写的时候已经记不清哪句话出自哪篇；
3. **"AI 帮你写"不可信** — 通用大模型会编造参考文献、编造 P 值，医学场景**直接构成学术不端**；
4. **越用越笨** — 通用工具不会记住"你上次指出过它把这个结论说反了"；
5. **数据主权与成本** — 云端工具把未发表数据传出去，且按量计费。

本项目对这五条的对应设计：**多源并发检索 + 本地知识库**、**引用编号与参考文献表强一致校验**、
**数字溯源校验**、**反馈→记忆闭环**、**全本地 SQLite + 本地模型**。

---

## 系统架构

```
┌────────────── 前端：免构建三列工作台（HTML + CSS + 原生 JS，零 CDN、零打包） ───────────────┐
│  对话视口(SSE 流式) · 文献卡片 · 综述草稿 · 引用列表 · 论文写作 · 质疑/纠错工具条 · 学习闭环徽标  │
└────────────────────────────────────┬───────────────────────────────────────────────────┘
                                     │ REST + SSE（事件列表是唯一事实来源，断线可完整补播）
┌────────────────────────────────────▼───────────────────────────────────────────────────┐
│ FastAPI 服务端 │ 55+ 端点 · SSE · 后台运行管理 · 人工审批 Future · 静态资源 mtime 版本号      │
└────────────────────────────────────┬───────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────────────────────┐
│ 工作流层：Plan →〔人工审批〕→ Execute → Reflect → Synthesize → Review → 成稿落库           │
│             │         │            │          │             │            │              │
│           LLM 规划   用户决定     Scout       Critic       Writer      Formatter         │
│         (结构化 JSON) 批准/改/取消 + Reader   (双轨评估)  (逐节流式)  + 引用越界剔除        │
│                                                                                          │
│ 每个阶段结束写一次**快照**（run_steps）→ 支持断点续跑：已完成的阶段不重跑                     │
└────────────────────────────────────┬───────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────────────────────┐
│ 检索与知识层                                                                              │
│  · 9 个检索源客户端 + Unpaywall：令牌桶限流 · 指数退避+抖动 · 429 自适应降速 · 单源失败隔离    │
│  · 检索式之间并发(信号量=3) · 单检索式内多源并发(asyncio.gather)                            │
│  · 混合检索：FTS5 BM25 ⊕ sqlite-vec KNN → RRF(k=60)；中文逐字切分 + 三级降级查询              │
│  · 嵌入管道：增量、批量、跨事件循环安全                                                     │
│  · 合规导入：题录文件(RIS/BibTeX/EndNote/WoS/CSV) · 官方批量包(PubMed baseline/PMC OA) · Zotero │
└────────────────────────────────────┬───────────────────────────────────────────────────┘
                                     │
┌────────────────────────────────────▼───────────────────────────────────────────────────┐
│ 存储：单个 SQLite 文件                                                                    │
│  papers · papers_fts(FTS5) · paper_fulltext · fulltext_fts · run_steps(阶段快照)           │
│  feedback(反馈/纠错) · manuscripts(论文) · citations · search_logs · artifacts · sessions    │
└──────────────────────────────────────────────────────────────────────────────────────────┘
```

**分层原则**：`api/` 只负责 HTTP 与字段映射，`agent/` 只依赖 `Paper` 模型，
`db/` 是唯一的事实来源，`server/` 不含业务逻辑。因此**新增一个数据源不需要改动下游任何代码**。

---

## 关键设计决策（选型与权衡）

每一条都包含"当时的备选方案"和"这个选择的代价"——**代价比理由更能说明工程判断**。

| 决策 | 备选方案 | 为什么这么选 | 代价 / 边界 |
|---|---|---|---|
| **不用 LangChain / LangGraph，自研工作流** | LangChain、LlamaIndex、LangGraph | 本项目工作流是**固定 5 阶段线性 + 一个人工审批点**，不是任意图。框架带来的是几百 MB 依赖、版本漂移、（曾是）隐式提示词拼装，以及"出问题不知道错在哪一层"。自研 `graph.py` 只有数百行，且每个阶段可单独测试 | 没有现成的 checkpointer / 可视化；`run_steps` 快照与断点续跑要自己写（已实现）。若未来要做**动态分支/多智能体自由协商**，再引入图引擎更合适 |
| **不用 Postgres/pgvector，用单文件 SQLite** | Postgres + pgvector、Milvus、Qdrant、Chroma | 目标用户是**单机个人研究者**：没有运维能力，备份必须简单（拷一个文件）。数万篇规模下 sqlite-vec 的 KNN 足够；`papers` 表天然是"唯一事实来源"，不需要再多一套服务的一致性 | **写并发是单写者**，不能水平扩展；>百万级向量会吃力。这是"个人工具"的合理边界，不是通用平台的选择；扩展路线见[已知局限](#已知局限与路线图) |
| **不用 BERT/微调模型做嵌入，用可切换的嵌入提供方** | 训练领域嵌入、Sentence-Transformers | 医学与中文场景对嵌入模型敏感（实测 `nomic-embed-text` 中英跨语言弱），所以把**嵌入提供方做成可替换接口**（Ollama / 本地 ST / 哈希回退），并自动同步向量维度 | 默认通用模型的跨语言检索质量一般；换模型需要重算全库向量（已提供 `db --optimize` / 增量管道） |
| **不用 FAISS，用 sqlite-vec + 纯 Python 回退双通路** | FAISS、hnswlib | 朋友的机器可能缺 VC 运行库或架构不符导致原生扩展加载失败。**功能可用性优先于峰值性能**：加载失败自动回退纯 Python 余弦检索（有 numpy 则加速），结果保持单调一致（向量入库前归一化） | 大数据量下回退路径慢；因此给出"降级但可用"而不是"报错退出" |
| **不用 React/Vite/Next.js，免构建原生 JS** | React + TS + Vite | 这个应用的价值在**本地化与可分发**：朋友解压即用，不该再要求 Node 工具链。原生 JS 4.2k 行足够撑三栏交互，且调试链路最短（改完刷新即可） | 组件复用靠手写约定；不是复杂前端的合适选择。**取舍点是"分发成本 < 开发体验"就选前者** |
| **不用 Celery/Redis 做任务队列，用进程内 asyncio + 阶段快照** | Celery、RQ、Arq、Temporal | 单机单用户场景下引入 broker 只是增加运维负担。真正的需求是**"服务重启后不丢进度"**，这靠 `run_steps` 阶段快照 + `mark_interrupted_runs` 解决，而不是靠队列 | 进程崩溃会丢失**当前阶段**的中间结果（已完成阶段可复用）；不能跨机负载均衡。多租户/高并发场景必须换成真队列 |
| **不用 `asyncio.Event` 唤醒 SSE，用 200ms 轮询事件列表** | Event / Future / 消息总线 | `Event`/`Future` **绑定创建它的事件循环**，一旦跨循环访问（测试里同时跑 ASGITransport 与 uvicorn）就静默死锁。轮询的代价可以忽略，换来的是彻底消除这类隐患 | 有最多 200ms 的推送延迟；事件表需要内存上限控制（已有保留期与裁剪） |
| **引用编号在 Critic 筛选之后重新连续分配** | 沿用检索阶段的编号 | 这是引用准确性最容易崩的地方：筛选掉几篇后编号出现空洞，正文 `[n]` 就会与参考文献表错位。**先筛选、再重编号**，写完还要再过一遍越界校验 | 需要额外一轮校验与 `_sanitize`；换来的是"正文引用与文末列表严格一一对应"这一硬保证 |
| **Critic 用「启发式 + LLM」双轨而非纯 LLM** | 纯 LLM 评估 | 启发式从出版类型、研究设计关键词、样本量、被引、时效性这些**可验证信号**打分，不幻觉、离线可用；LLM 只补充"核心发现/局限"这类需要理解的内容，并按 `critique_max_papers` 限量以控制时间 | 启发式对领域细微差异不敏感；两者等权融合是经验值，未做系统调参（这正是[评估体系](#已知局限与路线图)要补的） |
| **合规优先：不做订阅资源抓取，改做"导出→导入"与官方批量包** | 用机构账号抓取订阅全文 | 学校电子资源管理办法明文禁止批量下载，且出版商会封禁**整个学校**的 IP 段。技术上能做、但**后果由全校承担**，因此主动放弃这条路径，改提供四条合规替代（题录导入 / 官方批量包 / Zotero 本地库 / 链接解析器跳转） | 覆盖不了订阅内容的**全文**自动化获取；这是有意识的取舍，并写进了[合规声明](#合规声明)与产品文案 |
| **数字溯源校验（论文模块）** | 靠提示词约束模型别编数据 | 医学论文里编一个 P 值就是学术不端。提示词是概率约束，**程序校验才是确定性约束**：正文每个数字都要能在用户提供的数据/文献里找到出处，找不到就单独列出来给人工核对 | 会误报（同义表述、换算单位）；但对"防编造"来说，**误报比漏报可接受** |

---

## 深水区问题与解决

按"现象 → 根因 → 方案 → 验证"记录。这些都是**实测踩出来的**，不是设计文档里的假想。

### 1. 中文检索完全失效（FTS5 把整段汉字当成一个 token）

* **现象**：库里有几百篇中文文献，搜"抑郁"一条都搜不到；英文正常。
* **根因**：SQLite FTS5 的 `unicode61` 分词器把连续 CJK 字符视为**单个 token**，`卒中后抑郁` 是一个 token，查询 `抑郁` 无法匹配任何前缀/子串。
* **方案**：索引与查询**两侧**都做字符级切分（`segment_cjk` 在汉字间插空格），查询侧用短语（`"抑 郁"`）还原子串语义。再叠加三级降级级联（短语 → bigram → bigram-OR），避免长句精确匹配失败就返回空。
* **验证**：`tests/test_db.py` 覆盖"中文片段出现在更长短语中"等场景；`test_textutil.py` 锁定 `build_match_query` 的三种模式。

### 2. 78 篇文献嵌入跑 15 分钟没有一条向量（跨事件循环死锁）

* **现象**：嵌入管道挂住，CPU 占用 1.3s 几乎为零，日志无报错。
* **根因**：缓存的 `httpx.AsyncClient` 连接池**绑定创建它的那个事件循环**。测试与服务先后在不同循环里复用它，请求永久挂起而不抛异常。
* **方案**：客户端记录 `_client_loop`，每次请求前校验循环归属，不一致就重建；嵌入提供方改为**每次调用创建短生命周期客户端**。
* **验证**：`78 篇入库 + 全部生成向量 = 1.6 秒`（见[性能与质量](#性能与质量实测数据)）；同类问题在 `runtime.stream()` 里也被主动规避（见选型表最后第 3 行）。

### 3. 「AI 写论文」差点把编造的 P 值写进正文

* **现象**：生成的论文初稿出现了用户从未提供过的"下降 4.2 分""n=128"。
* **根因**：LLM 的补全倾向。医学写作的填空式表达（"治疗后 HAMD 下降 __ 分"）会诱使模型生成一个**看起来合理**的数字；仅靠系统提示词无法可靠约束。
* **方案**：`check_number_provenance()` 用正则抽出正文全部数字（含 `%`、小数、P 值），逐个在「用户提供的数据/统计结果」与「本地文献材料」里找出处；`12%` 与 `12` 视为等价。找不到的数字单独列成清单并附上**它所在的句子**，返回 `pass/warn/fail` 三级结论，同时明确告知"不要直接投稿"。
* **验证**：`tests/test_learning_and_manuscript.py` 用"编造的 4.2 分"与"9999 例"作回归用例；实测生成的草稿会列出无法溯源的数字。

### 4. 规划阶段偶发崩溃：`'list' object has no attribute 'get'`

* **现象**：约 1/3 的运行在规划阶段报这个错，随后静默降级成模板大纲，用户以为"规划成功了"。
* **根因**：分两层。**表层**是 `extract_json` 在对象解析失败后会退到"第一个配平的 `[...]`"，而残缺对象里第一个配平数组恰好是 `pico.outcomes`——于是一个**嵌套数组被当成了整份答案**返回，`topic_zh/queries/outline` 全被丢掉。**深层**是模型在 `queries` 里写坏了 JSON（检索式内用了未转义双引号），以及在 `"query": "("` 之后陷入空白循环把 token 预算烧完。
* **方案**：① `extract_json` 改为**形状感知**——文本以 `{` 开头就只认对象，绝不把嵌套数组当答案；② 新增截断补全（退到最后一个完整值再补括号），救回已生成好的字段；③ `chat_json(expect="object")` 校验顶层类型，不符就**触发纠错重试**，重试完仍不行才干净降级；④ 规划异常的日志补上 `exc_info=True`（原来只有 `%s`，定位不到抛点）。
* **验证**：用**真实抓取的两种坏输出**（截断版与畸形版）做回归；修复后生产 Plan 节点实测 `state.errors=[]`、稳定产出 3 条布尔检索式 + 5 章大纲。

### 5. 模型反复重新加载，比生成本身还慢

* **现象**：规划"有时 20 秒，有时 6 分钟"，差异巨大。
* **根因**：分两个。① Ollama 默认 `keep_alive` 只有 5 分钟，一旦超时卸载，**重新加载 8B 模型要 60~120 秒**；② 更隐蔽的是：`num_ctx` 是**提示词与输出的共享预算**，曾经 15,676 字的材料块占掉 4,129 token，把输出挤到只剩几百字。
* **方案**：① 显式设 `keep_alive: 30m` 并记录 `load_duration`，加载超 20 秒直接告警；② 新增 `_fit_digest()` 在预算内**按阶梯收缩材料**（摘要 800→500→400→300→200 字，篇数不限→18→12→8），优先保正文篇幅；③ 补充 `estimate_tokens()` 做预算估算（用真实测量校准，误差 8%）。
* **验证**：生产环境实测 **45.5 tok/s**，与硬件理论上限吻合（见下）；规划墙钟从 5~6 分钟降到 **22.6 秒**；字数范围调到 12000~20000 时材料自动从 5661 降到 4204 token。

### 6. 硬件天花板被算清楚了，于是不再瞎调

* **现象**：本地 8B 模型慢，怀疑是程序效率问题。
* **根因**：`nvidia-smi` 显示 `utilization.memory 100%`、SM 频率满血 —— 说明**显存带宽打满**。RTX 4060 带宽 272 GB/s ÷ 模型 6.19 GB ≈ **44 tok/s 理论上限**，实测 45~47。
* **方案（结论）**：**不再尝试用并发/调参提升单次生成速度**——生成是带宽受限的串行过程，并发只会排队。转而优化"不产生价值的时间"（模型加载、坏输出的重试、串行等待的网络）与"生成多少 token"（字数范围、材料裁剪）。
* **验证**：`num_ctx` 取 8192/4096/2048 时吞吐为 46.7/46.5/46.9 tok/s（**无差异**，直接否掉"减小上下文能提速"的猜想）；据此把优化预算投到别处，并把这套测量写进 README 供换机器时复测。

### 7. 多条检索式串行等待网络（纯浪费）

* **现象**：4 条检索式各等 7~10 秒，一轮检索要 30 秒以上。
* **根因**：单检索式内部的多数据源已经 `asyncio.gather` 并发了，但**检索式之间**是串行 `await`。这部分是纯网络 I/O，不占 GPU。
* **方案**：检索式之间也并发（信号量限 3 —— 每条检索式内部还会并发 5~6 个源，限太高容易触发上游限流），结果**按原顺序整理**，保证日志、统计与界面事件顺序确定。
* **验证**：用用户库里真实记录的耗时反推：4 条检索式串行 32.7s → 并发约 **10.5s**，省约 22 秒。`tests/test_scout_concurrency.py` 用**并发峰值**断言（不用墙钟 —— 墙钟在负载高的机器上会偶发失败，这是踩过的坑）。

### 8. 修好了前端，用户却还在跑旧 JS

* **现象**：明明已经修复的 bug，用户复现的还是老错误。
* **根因**：两个叠加。① 浏览器缓存了旧 `app.js`；`Cache-Control: no-store` 只能防**再次**缓存，对"已经缓存了旧副本"无能为力。② 一次 Windows 批处理脚本的编码陷阱：`.bat` 里写 UTF-8 中文会被 cmd 的 936 代码页吃字符，更严重的是 LF 换行导致 `goto`/标签失效，报出 `"dp0" 不是内部或外部命令`。
* **方案**：① 服务端渲染 `index.html` 时按文件 mtime 给静态资源加 `?v=` 版本号（换新 URL 才能强制更新）+ `no-store`；② `.bat` 一律改为**纯 ASCII + CRLF**，所有中文提示移到 Python 侧（用 `WriteConsoleW` 正确输出）。
* **验证**：线上校验页面版本令牌与文件 mtime 一致；`run.bat` / `run-doctor.bat` / `run-cli.bat` 中文输出正常。

### 9. XML/HTML 里"看起来对"的字段其实全错（真实数据才暴露）

* **现象**：导入 Web of Science 导出的题录后，DOI 全部为空，去重失效、全文获取全失败。
* **根因**：**WoS 用 `DI` 表示 DOI，而 RIS 规范是 `DO`**；同理期刊名用 `SO`（不是 `JO`）、页码用 `BP/EP`（不是 `SP`）。另外 WoS 的**纯文本导出没有短横线**（`AU Zhang, Wei`），且多位作者是**缩进续行**——按标准 RIS 解析会一条都读不出，或把三位作者拼成一个名字。
* **方案**：补全标签映射；把"续行"语义按字段区分——**可重复字段（作者/关键词）的续行是新的一项**，单值字段（摘要/标题）的续行是拼接；格式识别改为**内容特征优先于扩展名**（纯文本 WoS 存成 `.txt` 时会命中"首行含 title+逗号"而被误判成 CSV，这个坑也修了）。
* **验证**：`tests/test_importers.py` 用五个数据库的真实导出样式做用例；同类问题在 Zotero 上重演了一次——`storage/<目录>` 用的是**附件条目自己的 key**，不是父文献的 key（传错会一篇 PDF 都索引不到），已修正并加注释。

### 10. Unpaywall 拒绝占位邮箱（外部依赖的真实约束）

* **现象**：按 DOI 查开放获取全文时报 HTTP 422，错误体是一段英文 JSON。
* **根因**：Unpaywall **会校验邮箱真实性**，`example.com` 这类占位地址被明确拒绝（"Please use your own email address"）。这类约束文档里不显眼，只有真调用才会暴露。
* **方案**：捕获 422 并转成**可照做的中文提示**（去填 `sources.unpaywall.email`），同时告诉用户"也可以直接 `enabled: false` 关掉，其余功能不受影响"——外部依赖不可用时不该变成阻断性故障。
* **验证**：`tests/test_unpaywall.py` 用真实 422 响应体做回归，断言提示里包含配置项名与关闭方式。

---

## 性能与质量（实测数据）

> 所有数字都是**本机实测**，不是文档抄来的。换机器后用 `scripts/bench_models.py`、
> `scripts/check.py` 可以复测；`medscholar doctor` 随时复查各数据源状态。

### 非 LLM 环节：几乎不是瓶颈

| 环节 | 实测结果 |
|---|---|
| 嵌入（`nomic-embed-text`，768 维） | **约 60 条/秒** |
| 78 篇文献入库 + 全部生成向量 | **1.6 秒** |
| 跨库检索（4 源并发） | **2~3 秒** |
| 多条检索式并发后的一轮检索 | 约 **10.5 秒**（串行为 32.7 秒，实测数据反推） |

**结论：瓶颈从来不是检索或嵌入，而是 LLM 生成。**

### LLM 环节：硬件天花板是可以算出来的

| 场景 | 生成速度 | 规划阶段墙钟 |
|---|---|---|
| 纯 CPU（`qwen3:8b`） | 约 7.4 tok/s | 48 秒 |
| **RTX 4060（`qwen3:8b` Q4，实测）** | **45~47 tok/s** | **22.6 秒** |

RTX 4060 显存带宽 272 GB/s ÷ 模型 6.19 GB ≈ **44 tok/s 理论上限**，实测 45~47 ——
**说明这已经是硬件极限**。据此可以断定：任何并发、调 `num_ctx`（8192/4096/2048 实测
46.7/46.5/46.9 tok/s，**无差异**）、调线程数都不会更快。

于是优化预算被投到真正有浪费的地方，并各自拿到收益：

| 浪费点 | 优化 | 收益 |
|---|---|---|
| 模型被卸载后重新加载 | `keep_alive: 30m` | 省掉 **60~120 秒/次** |
| 提示词吃掉输出预算 | `_fit_digest()` 按阶梯收缩材料 | 正文不再被挤到几百字 |
| 坏输出的重试 | 形状感知解析 + 截断补全 + 纠错重试 | 规划从"偶发崩溃降级"变为稳定成功 |
| 检索式串行等待 | 检索式之间并发（信号量 3） | 每轮省约 **22 秒** |

### 端到端耗时与质量

| 指标 | 实测 |
|---|---|
| 完整「课题 → 综述」（本地 GPU） | 约 **5~10 分钟**（取决于字数范围与数据源数量） |
| 综述正文篇幅 | 默认目标 **4000~8000 字**（可配置 800~40000）；实测单节 1192 字（默认档）/2195 字（大字数档） |
| 引用一致性 | 正文 `[n]` 与参考文献表严格一一对应，越界引用被自动剔除 |
| 数字溯源 | 论文模块逐数字校验出处，无法溯源的单独列出 |

### 检索质量：从"感觉准了"到可复现的数字

```bat
:: CI 同款：确定性离线回归 + 阈值门禁（秒级，不需要 Ollama）
.python\python.exe scripts\eval_retrieval.py --dataset regression --k 10 --embed-provider hashing --check

:: 本机真实评测：在你自己的库上跑（需要 Ollama）
.python\python.exe scripts\eval_retrieval.py --from-library --limit 60 --k 10 --mode title-terms
```

指标是 `recall@k` / `precision@k` / `MRR` / `MAP` / `nDCG@k`，配 7 种配置的消融矩阵。
方法论、偏差说明与"这个评测不能证明什么"见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。

**实测结果（本地库 60 篇真实文献，`nomic-embed-text`，title-terms 查询，@10）**：

| 配置 | recall@10 | nDCG@10 | MRR | 相对 BM25 |
|---|---|---|---|---|
| bm25-only | 1.0000 | 0.9754 | 0.9667 | — |
| vector-only | 1.0000 | 0.8967 | 0.8611 | **−7.9pt nDCG** |
| production（RRF k=60） | 1.0000 | 0.9815 | 0.9750 | **+0.6pt nDCG** |
| weighted-fts2（BM25 权重 ×2） | **1.0000** | **0.9877** | **0.9833** | **+1.2pt nDCG** |

**这张表说了三件事，其中两件是对项目自身的批评**：

1. **融合确实优于单路** —— production 好过 bm25-only 与 vector-only。
   这是"BM25 ⊕ 向量 → RRF"有效果的**第一次量化证据**（此前只有原理判断）。
2. **向量单路明显弱于 BM25**（nDCG −7.9pt）：语料是英文医学文献、查询由标题派生，
   词面重叠仍然偏高，所以语义路拿不到便宜。**说明"上向量检索总是更好"是错的**。
3. **recall@10 全部饱和在 1.0，没有区分度** —— 60 篇语料下 top-10 基本覆盖全部相关文献。
   结论：**小语料上应该看 nDCG/MRR，不要看 recall**。

同一套指标还抓出了两个"数据集本身不合格"的事实（都已固化为测试）：
`known-item` 模式下 BM25 直接满分 1.0（查询就是标题，评测无法区分配置）；
以及**小语料上改 `rrf_k` 完全不改变排序**（k 只是对 `1/(k+rank)` 做单调缩放）——
所以"调 k"必须在真实规模的库上做。

**阴性对照**：一条故意无关的查询（量子色动力学）在语料中确无相关文献，
生产配置平均仍返回 **10.0 条**（= top_k）。这暴露了一个必须显式盯住的系统特性：
**RRF 融合天然总会给出答案，没有"我不知道"的出口** ——
所以单靠检索层无法拒绝无关问题，必须由下游的引用越界剔除与数字溯源校验兜住。

---

## 快速开始（给使用者）

Windows 下双击 **`run.bat`**。它会自动找到 Python、检查依赖、生成配置、启动服务并打开浏览器。

启动后是三列工作台：左栏会话/文献库/知识库统计，中栏对话视口与流式输出，
右栏产物预览（**文献卡片 · 综述草稿 · 引用列表 · 论文写作**）。

**第一次使用建议先做两件事**：

1. 双击 **`run-doctor.bat`** 做环境自检 —— 逐项告知 LLM、嵌入模型、各数据源是否就绪，
   以及没就绪时**具体该执行什么命令**。
2. 确认 Ollama 已启动并拉好模型：

   ```bat
   ollama serve
   ollama pull qwen3:8b
   ollama pull nomic-embed-text
   ```

### 想更快的话

| 方案 | 效果 | 代价 |
|---|---|---|
| 换小模型：`llm.model: qwen3.5:4b` | 约 2 倍 | 质量略降（实测相当） |
| 调小 `agent.critique_max_papers`（12 → 6） | 评估省一半时间 | 更多文献只能用启发式评分 |
| 调小 `agent.writer_max_papers`（25 → 12） | 写作上下文更短 | 可引用文献变少 |
| 调小 `agent.review_max_chars` | 生成 token 更少 | 正文更短 |
| 关掉 `agent.auto_revise` | 省一轮全稿修订 | 失去自动查错订正 |
| **改用 DeepSeek 云端 API** | **快一个数量级，质量更好** | 需 Key，按量付费（一次综述不到一毛钱） |

> ⚠️ **不要只看 tok/s 就换小模型**：`llama3.2:3b` 快 5 倍，但它会把占位符 `[n]`
> 原样写进正文——看起来有引用，实际读者根本对不上参考文献表。
> 综述写作对**引用纪律**的要求比速度更硬。

---

## 常见故障排查

### 前端提示「大语言模型不可用」

先双击 **`run-doctor.bat`**，它会直接告诉你原因。常见有三种：

**① 报错里出现 `deepseek`，但模型名是本地的（例如 `qwen3:8b`）**

```
deepseek 返回 HTTP 400：The supported API model names are ... but you passed qwen3:8b
```

这是 **provider 与 model 不配套**：`config.yaml` 里写着 `provider: deepseek`
（云端），却把本地 Ollama 的模型名填给了它。

> **为什么会变成这样？** 你机器上可能有的**别的项目**把 `DEEPSEEK_API_KEY`
> 设成了用户级环境变量。早期版本会因为"看到这个变量"就把后端悄悄切成 deepseek，
> 从而踩这个坑。**现在已修复：后端永远只由 `config.yaml` 决定，不受环境变量影响。**

改法二选一（编辑 `config.yaml`）：

```yaml
# 方案 A：用本地 Ollama（推荐，零成本）
llm:
  provider: ollama
  model: qwen3:8b
  base_url: http://127.0.0.1:11434
```

```yaml
# 方案 B：真的要用云端 —— provider / model / base_url 三者必须配套
llm:
  provider: deepseek
  model: deepseek-chat            # 换成你所用网关支持的模型名
  base_url: https://api.deepseek.com/v1
  api_key: sk-xxxxxxxx
```

程序现在会在启动时检查这种不一致，并给出中文提示，不会再让你去猜英文报错。

**② 提示连不上 Ollama**

```bat
ollama serve
ollama pull qwen3:8b
```

**③ 模型名对但 Ollama 里没装**

```bat
ollama list
ollama pull <config.yaml 里写的模型名>
```

### 想换一个本地模型

改 `config.yaml` 的 `llm.model` 一行即可，重启生效。用同一段中文医学写作任务实测
（`scripts/bench_models.py`，纯 CPU、关闭 thinking）：

| 模型 | 稳态生成速度 | 220 字耗时 | 引用纪律 | 冷加载 | 结论 |
|---|---|---|---|---|---|
| `qwen3:8b`（默认） | 7.6 tok/s | 18.7 s | ✅ 全部合法 | 0 s | 质量基准 |
| **`qwen2.5:7b`** | **9.6 tok/s** | **14.1 s** | ✅ 全部合法 | — | **比默认快 26%，质量相当** |
| **`qwen3.5:4b`** | **15.9 tok/s** | **9.4 s** | ✅ 全部合法 | 26~32 s | **快 2 倍，质量相当，性价比最高** |
| `llama3.2:3b` | 41.0 tok/s | 3.5 s | ❌ **输出字面量 `[n]` 未替换** | 12 s | 快但**引用不可用**，不建议 |

> ⚠️ **不要只看 tok/s 就换小模型**：`llama3.2:3b` 快 5 倍，但它会把占位符
> `[n]` 原样写进正文 —— 看起来有引用，实际读者根本对不上参考文献表。
> 综述写作对引用纪律的要求比速度更硬。
>
> 「冷加载」是模型不在内存时的一次性成本。8 GB 内存的机器上同时只挂得住一个模型，
> **换模型后第一次请求要多等 20~30 秒**，之后才是上表的稳态速度。

> **关于「模型被别的项目占用」**：Ollama 会把并发请求排队处理，不会冲突或报错，
> 只是你的请求可能要等。换成另一个模型**不一定更快** —— Ollama 需要把新模型也
> 载入内存，8 GB 内存上同时挂两个 5 GB 模型反而会拖慢（还要付冷加载成本）。
> 真嫌慢的话，换更小的模型（`qwen3.5:4b`）或改用云端 API 效果更直接。

自己随时可以复测：

```bat
.python\python.exe -X utf8 scripts\bench_models.py
```

---

## 环境要求

| 组件 | 要求 | 说明 |
|---|---|---|
| 操作系统 | Windows 10/11（macOS / Linux 理论可用，未验证） | 便携运行时是 Windows 版 |
| Python | 3.10+（**已内置 3.13.8 便携版**） | 朋友无需自己装 Python |
| 内存 | 8 GB 起（跑 qwen3:8b） | 4B 模型 6 GB 即可 |
| 磁盘 | 约 1 GB（含运行时与依赖） | 模型另计（qwen3:8b 约 5 GB） |
| Ollama | 可选但推荐 | 不装则无法生成综述，检索与知识库仍可用 |
| 网络 | 检索时需要 | 可切离线模式，只用本地库 |

---

## 能做什么

| 能力 | 说明 |
|---|---|
| **多源检索** | 一次课题并发检索 PubMed / Europe PMC / OpenAlex / Crossref / Semantic Scholar / arXiv，自动跨库去重合并 |
| **中英文双通路** | 中文课题自动走 OpenAlex `language:zh` 通路，英文课题走 PubMed 主通路 |
| **本地知识库** | 全部文献存入单个 SQLite 文件，FTS5 全文索引 + sqlite-vec 向量索引 |
| **混合检索** | BM25 关键词 + 向量语义 + RRF 融合（k=60），再叠加年份/期刊/被引/开放获取过滤 |
| **六智能体协作** | Orchestrator 编排 / Scout 检索 / Reader 全文解析 / Critic 证据评估 / Writer 写作 / Formatter 格式化 |
| **人工审批节点** | Plan 阶段产出检索策略与大纲后**暂停**，你确认或提修改意见后才继续执行 |
| **批判性评估** | 逐篇给出相关性、方法学质量、证据等级、核心发现与局限，决定哪些文献进综述 |
| **带引用综述** | 流式生成分章节综述草稿，正文引用编号与参考文献表严格一一对应，自动剔除越界引用 |
| **开放获取全文** | Europe PMC JATS → PMC → OA PDF 三级获取；非开放获取文献只保留元数据与出版商链接 |
| **多格式引用** | APA 7th / Vancouver / GB-T 7714-2015 / Chicago / BibTeX / RIS |
| **多格式导出** | BibTeX、RIS、CSV、JSON、Markdown、可直接粘进 Word 的 HTML |
| **MCP Server** | 把 10 个原子工具通过 Model Context Protocol 暴露给 Claude Desktop / Cursor 等客户端 |
| **命令行** | 服务器之外还提供 `search` / `run` / `kb` / `cite` / `export` / `doctor` 等子命令 |

---

## 数据源实测状态

> 下表是**实测结果**（2026-02），不是照抄文档。各 API 的策略会变，可用 `medscholar doctor` 随时复查。

| 数据源 | 状态 | 覆盖 | 速率限制 | 需要 Key |
|---|---|---|---|---|
| **PubMed** | ✅ 正常 | 3600 万+ 生物医学文献 | 3 次/秒；填免费 Key 后 10 次/秒 | 否（建议填） |
| **Europe PMC** | ✅ 正常 | 4000 万+，含 800 万+ 开放获取全文 | 约 10 次/秒 | 否 |
| **OpenAlex** | ✅ 正常 | 2.5 亿+，**中文文献通路主力** | 约 10 次/秒 | 否（建议填邮箱） |
| **Crossref** | ✅ 正常 | DOI 官方注册库，元数据权威 | 约 5 次/秒 | 否（建议填邮箱） |
| **arXiv** | ✅ 正常 | 预印本 | 每 3 秒 1 次 | 否 |
| **Semantic Scholar** | ⚠️ 限流 | 2 亿+，强在引用图谱 | 无 Key 约 100 次/5 分钟 | **强烈建议申请免费 Key** |
| **DOAJ** | ✅ 正常 | 两万余种完全开放获取期刊（补 PubMed 不收的 OA 期刊） | 约 2 次/秒 | 否 |
| **CORE** | ⚙️ 需 Key | 全球机构库：学位论文、技术报告、自存档稿 | 约 1 次/秒 | 是（免费） |
| **Unpaywall** | ✅ 正常（不参与关键词检索） | 按 DOI 找**合法**开放获取副本 | 约 2 次/秒 | 需填邮箱 |
| **CNKI** | ❌ 不可用（默认关闭） | — | — | — |

## 扩充数据来源（合规路径）

学校订阅的数据库（Web of Science / Scopus / Embase / Cochrane / CNKI / 万方）
**没有开放 API，也不能用账号脚本化抓取**：多数学校的电子资源管理办法明文禁止
批量下载，处罚包括停用校外访问权限；出版商会封禁整个学校的 IP 段，
影响的是全校师生。因此本项目提供下面四条合规途径来扩充数据。

### 1. 从图书馆数据库导出题录 → 导入

WoS、Scopus、Embase、CNKI、万方 都支持导出 RIS / BibTeX / CSV / EndNote。
导入后会自动去重、入库、生成向量，**立刻可被本地检索和综述引用**。

```bash
medscholar import savedrecs.ris              # Web of Science 的 RIS 导出
medscholar import scopus.bib                 # Scopus 的 BibTeX
medscholar import cnki.enw wanfang.txt       # CNKI/万方的标记文本，可一次多个
medscholar import scopus.csv --dry-run       # 先看能识别多少条，不写库
```

自动识别 **RIS / BibTeX / EndNote 标记文本 / WoS 纯文本 / CSV** 五种格式
（含 GBK 编码的中文导出）。网页端「文献库」面板也有「导入」按钮。

### 2. 官方批量数据包（真正的"合法爬取"）

NLM 与 Europe PMC **主动发布**这些数据包给批量使用，README 明确允许文本挖掘：

```bash
# PubMed baseline（每年一次，3700 万+ 题录，解压后约 40 GB）
medscholar bulk pubmed D:\pubmed\ --match "rTMS,depression" --year-from 2015 --limit 5000

# PMC Open Access Subset（几百万篇**全文**，含 JATS 正文）
medscholar bulk pmc-oa D:\pmc_oa\ --match "transcranial magnetic"
```

* 下载：<https://pubmed.ncbi.nlm.nih.gov/download/>、
  <https://www.ncbi.nlm.nih.gov/pmc/tools/ftp/>
* 必须给 `--match`（或 `--limit`）：全量 3700 万条不可能都入库。
* 采用**流式解析**，内存占用与数据包大小无关；先 `--dry-run` 看命中量再跑。

### 3. Zotero 本地库

你用学校代理把 PDF 合法收进 Zotero 之后，MedScholar 只读本机数据库，
索引**你已经持有的文件**：

```bash
medscholar zotero                  # 导入题录 + 解析本地 PDF 全文
medscholar zotero --no-pdf         # 只导题录
```

只读，不修改 Zotero；Zotero 运行时会先复制副本再读，避免锁库。

### 4. 网页端「通过图书馆获取全文」

在网页「设置」里填学校的**链接解析器（OpenURL）**或**校外访问代理前缀**，
文献卡片上就会出现跳转按钮 —— 点击后用**你自己的会话**在浏览器里阅读订阅全文。
这是代理被设计出来的用途：只生成链接，不下载、不抓取。

复旦大学相关信息：校外访问方式见
[图书馆说明](https://library.fudan.edu.cn/e8/b5/c42805a518325/page.htm)，
图书馆代理为 `libproxy.fudan.edu.cn:8080`，链接解析器为
`sfx-86fdu.hosted.exlibrisgroup.com.cn`（具体地址请在图书馆点一次
「Find it」后复制问号前的部分）。


### ⚠️ 关于 CNKI 的重要说明

需求文档要求通过 `search.cnki.com.cn` 公开接口获取中文文献元数据。**实测结论：该接口已不可用。**

具体验证过程与结果：

| 尝试 | 结果 |
|---|---|
| `GET /Search/Result?content=卒中后抑郁` | HTTP 200，返回 39648 字节 HTML，但**只有检索表单和页脚，零条文献记录** |
| 校验页面内是否含文献条目链接（`kns.cnki.net` / `/KCMS/detail` / `dbcode=`） | **否** —— 结果由前端 JS 动态渲染 |
| `GET /Search/ListResult`（数据接口） | **HTTP 403** |
| `POST /Search/ListResult`（带 X-Requested-With / Referer / Origin） | **HTTP 403** |
| `GET kns.cnki.net/kns8/defaultresult/index` | **HTTP 403** |

因此，在不引入浏览器渲染（Playwright/Puppeteer）的前提下**无法稳定获取 CNKI 元数据**。

**本项目的处理方式**：`CnkiClient` 保留完整实现（含 HTML 解析器），但默认 `enabled: false`；
被调用时会抛出**携带具体处置建议**的错误，而不是静默返回空结果。
同时提供了**三条实测可用**的中文文献通路：

1. **OpenAlex `language:zh`** —— 实测对「卒中后抑郁」返回 367 条真实中文文献，
   能拿到「高频重复经颅磁刺激联合高压氧治疗脑卒中后抑郁效果观察」这类正确的中文标题。
   程序会在检测到中文课题时**自动追加**这条路。
2. **PubMed `chinese[la]`** —— 被 PubMed 收录的中文期刊（如《生理学报》《针刺研究》《南方医科大学学报》）。
3. **Crossref** —— 注册了 DOI 的中文期刊。

如果你自建了渲染服务或拥有机构授权接口，可把 `sources.cnki.base_url` 指向它并置 `enabled: true`，客户端即可正常工作。

---

## 配置

首次运行会由 `config.example.yaml` 自动生成 `config.yaml`。**所有字段都可省略。**

优先级：内置默认值 < `config.yaml` < 环境变量 / `.env`

### 提升检索速度（推荐）

三个免费 Key 能让检索明显更快、更稳（尤其是 Semantic Scholar）：

```bat
:: NCBI（PubMed 3 次/秒 → 10 次/秒）
::   https://www.ncbi.nlm.nih.gov/account/settings/
set NCBI_API_KEY=你的key

:: Semantic Scholar（100 次/5 分钟 → 1000 次/5 分钟）
::   https://www.semanticscholar.org/product/api
set S2_API_KEY=你的key

:: OpenAlex（配额 ×10）
::   https://openalex.org/
set OPENALEX_API_KEY=你的key
```

也可以写进项目根目录的 `.env` 文件（参考 `.env.example`）。

### 切换推理后端

```yaml
# 本地（默认，零成本）
llm:
  provider: ollama
  model: qwen3:8b

# 云端 DeepSeek（成本极低，需要 Key）
llm:
  provider: deepseek
  model: deepseek-chat
  base_url: https://api.deepseek.com/v1
  api_key: sk-xxxxxxxx
```

### 切换嵌入模型

默认 `nomic-embed-text`（768 维，与需求文档的 768 维设定一致，且无需 torch）。

> **注意**：`nomic-embed-text` 以英文为主，**中文语义检索能力一般**。
> 实测跨语言相似度区分度较弱（中文 rTMS 句与英文 rTMS 句的余弦相似度 0.38，
> 与无关的 DBS 句为 0.40，基本落在噪声区间）。中文文献的语义召回因此主要依赖
> BM25 与同语种向量。
>
> 如果你的课题以中文文献为主，建议换用多语言模型（需要 `ollama pull`）：
>
> ```yaml
> embedding:
>   provider: ollama
>   model: bge-m3          # 多语言，1024 维
>   dim: 1024
> ```
>
> 或改用需求文档推荐的医学专用模型：
>
> ```bat
> .python\python.exe -m pip install sentence-transformers
> ```
> ```yaml
> embedding:
>   provider: sentence-transformers
>   model: neuml/pubmedbert-base-embeddings   # 768 维
>   dim: 768
> ```
>
> 更换模型或维度会**自动重建向量表**，下次检索时重新嵌入（原文与元数据不受影响）。

---

## 命令行用法

```bat
:: 环境自检（排查问题的第一步）
.python\python.exe -X utf8 -m medscholar doctor

:: 联网检索并入库
.python\python.exe -X utf8 -m medscholar search "accelerated rTMS post-stroke depression" --limit 30

:: 本地知识库混合检索
.python\python.exe -X utf8 -m medscholar kb "rTMS 抑郁" -n 15

:: 跑完整工作流（终端流式输出，会暂停等你确认检索计划）
.python\python.exe -X utf8 -m medscholar run "加速rTMS治疗卒中后抑郁"
.python\python.exe -X utf8 -m medscholar run "加速rTMS治疗卒中后抑郁" --yes   :: 跳过审批

:: 单篇速读
.python\python.exe -X utf8 -m medscholar summarize 12

:: 生成参考文献
.python\python.exe -X utf8 -m medscholar cite 1 2 3 --style apa7

:: 导出
.python\python.exe -X utf8 -m medscholar export 1 2 3 -f bibtex

:: 知识库统计
.python\python.exe -X utf8 -m medscholar stats
```

也可以双击 `run-cli.bat` 用菜单操作。

---

## 接入其他 MCP 客户端

MedScholar 提供 MCP Server，可把检索与格式化能力交给 Claude Desktop、Cursor、Cherry Studio 等客户端：

```bat
.python\python.exe -m pip install mcp
.python\python.exe -m medscholar mcp          :: stdio 传输
.python\python.exe -m medscholar mcp --http   :: 流式 HTTP
```

Claude Desktop 配置示例（`claude_desktop_config.json`）：

```json
{
  "mcpServers": {
    "medscholar": {
      "command": "D:\\code\\medscholar-agent\\.python\\python.exe",
      "args": ["-m", "medscholar", "mcp"]
    }
  }
}
```

暴露的 10 个工具：`search_literature`、`search_knowledge_base`、`get_paper`、`list_library`、
`library_stats`、`fetch_fulltext`、`fetch_references`、`format_citation`、`save_paper`、`save_fulltext`。

---

## 打包分享给朋友

```bat
.python\python.exe scripts\pack_share.py
```

会生成 `dist/MedScholar-Agent-v1.0.0-win64.zip`，其中包含：

* 自包含的 Python 3.13.8 便携运行时（**朋友无需安装 Python**）
* 全部依赖（fastapi / uvicorn / httpx / pydantic / sqlite-vec …）
* 程序源码、前端页面、配置文件模板、三个 .bat 启动器
* 使用说明

朋友拿到后：**解压 → 双击 `run.bat`**。唯一的外部依赖是 Ollama（不装也能用检索与知识库）。

---

## 目录结构

```
medscholar-agent/
├── run.bat / run-doctor.bat / run-cli.bat   一键启动 / 自检 / 菜单式 CLI
├── config.example.yaml                      完整注释的配置模板
├── .env.example                             API Key 环境变量模板
├── .python/                                 自包含便携运行时（随包分发）
├── data/                                    数据目录（数据库、全文、导出）
│   ├── medscholar.db                        你的整个知识库就是这一个文件
│   ├── fulltext/                            开放获取全文缓存
│   └── exports/                             导出文件
├── docs/API.md                              HTTP 契约（前端与后端共同依据）
├── medscholar/
│   ├── config.py        配置加载             models.py        数据模型
│   ├── textutil.py      中文字段处理          dedupe.py        跨库去重合并
│   ├── retrieval.py     混合检索门面          tools.py         10 个原子工具
│   ├── cli.py           命令行
│   ├── db/              SQLite + FTS5 + sqlite-vec + 迁移
│   ├── api/             9 个检索源 + Unpaywall（限流/退避/降级）
│   ├── embedding/       嵌入提供方 + 增量管道
│   ├── llm/             统一 LLM 客户端 + 容错 JSON 解析 + 提示词
│   ├── agent/           工作流（graph）+ 六智能体 + 运行时（含断点续跑）
│   ├── importers/       题录文件解析（RIS/BibTeX/EndNote/WoS/CSV）
│   ├── bulk.py          官方批量包导入（PubMed baseline / PMC OA，流式）
│   ├── zotero.py        Zotero 本地库桥接（只读）
│   ├── feedback.py      反馈/质疑 → 纠错记忆 + 偏好对 + 重加权
│   ├── manuscript.py    基于实验数据的论文生成 + 数字溯源校验
│   ├── cite/            引用格式化引擎
│   ├── export/          导出
│   ├── server/          FastAPI 服务端
│   ├── mcp/             MCP Server
│   └── web/             前端（纯静态，免构建）
└── scripts/             自检 / 冒烟 / 端到端 / 基准 / 打包
```

---

## 测试与工程质量

```bat
.python\python.exe -m pytest tests -q               :: 608 个测试
.python\python.exe scripts\check.py                 :: 语法 + 模块导入 + 纯函数断言
.python\python.exe scripts\check_bat.py             :: .bat 必须纯 ASCII + CRLF
.python\python.exe -X utf8 scripts\smoke_http.py    :: 71 项真实 uvicorn 端到端
```

### CI（GitHub Actions，6 个作业）

| 作业 | 平台 | 内容 |
|---|---|---|
| `lint` | ubuntu | `ruff`(F,E9) + `compileall` + `check.py` + `check_bat.py` |
| `test` | windows / 3.13 | 687 个测试（与随包便携运行时同版本，保证"CI 绿 = 用户能用"） |
| **`eval`** | ubuntu | **检索质量回归**：消融评测 + 阈值门禁 + 与 `hybrid_search` 的一致性自检 |
| `smoke` | windows | 真实 uvicorn，逐条核对 `docs/API.md`（离线、不联网） |
| `package` | windows | 打包回归 + 断言分享包**不含 `data/`**（合规） |
| `test-linux` | ubuntu | 实验性（`continue-on-error`）——README 已声明 Linux 未验证 |

配置见 [`.github/workflows/ci.yml`](.github/workflows/ci.yml)。
提交前门禁见 [`.pre-commit-config.yaml`](.pre-commit-config.yaml)：

```bat
.python\python.exe -m pip install pre-commit
.python\python.exe -m pre_commit install
```

**`eval` 作业为什么用哈希嵌入而不是真实模型**：CI 里装几 GB 的嵌入模型既慢又不稳定，
一旦因为环境问题频繁红掉，大家就会开始习惯性忽略它。所以 CI 只做**确定性回归**
（指标与融合逻辑有没有被改坏），真实语义质量在本机用 `--from-library` 测。
这两件事刻意分开，混为一谈就会得出虚假结论。

**CI 第一次跑就抓到了真回归**：`smoke_http.py` 里写死了"7 个数据源"，
而我刚加了 DOAJ/CORE（共 9 个）——本地测试全绿、只有这条契约检查失败。
已改为断言"必需数据源集合齐全"，新增源不再误报、移除源仍会被抓住。

### 换行符策略（`core.autocrlf` 的一个真实陷阱）

本机 `core.autocrlf=true`，会把 `.bat` 归一化成 **LF 存进仓库**；
而 `lint` 作业跑在 **Linux** 上执行 `check_bat.py`，检出后就是 LF → **CI 必然失败，且本地怎么都复现不出来**。

[`.gitattributes`](.gitattributes) 用 `*.bat text eol=crlf` 固定语义：
**仓库存 LF、任何平台检出都还原 CRLF**。可用 `git ls-files --eol run.bat` 验证：
`i/lf  w/crlf  attr/text eol=crlf`。

**测试策略**（不是"为了覆盖率写测试"）：

* **用真实抓取到的坏数据做回归** —— 规划崩溃的两种坏输出、WoS 的四种导出格式、
  Unpaywall 的 422 响应体，都是从生产/log 里抓下来的原文，不是手编的示意数据。
* **断言"不该发生的调用"** —— 断点续跑测试直接断言"续跑时**不得**再调用 plan/execute"，
  这样"跳过已完成阶段"才是被证明的，而不是被假设的。
* **优先用不依赖时序的信号** —— 并发测试断言**并发峰值**而不是墙钟耗时
  （墙钟在负载高的机器上会偶发失败，踩过）；`test_config.py` 用
  `MockTransport` 断言请求体里确实带上了 `keep_alive`。
* **故意喂脏数据** —— 截断的 JSON、损坏的 tar、非法 JSON 的数据库字段、
  GBK 编码的导出文件、`example.com` 邮箱，都要有明确的降级行为而不是崩溃。

**已建立的工程习惯**：`compileall` + `ruff`（F/E9）+ `check.py` 三件套、
每个新模块配套测试、`.bat` 纯 ASCII+CRLF 约定（已有守卫脚本强制）、
静态资源 mtime 版本号、配置与代码同步（`config.example.yaml` 是唯一文档源）、
**CI 五作业 + pre-commit 提交门禁**。

**尚未建立的**：容器化、schema 迁移工具（见下节路线图）。

---

## 已知局限与路线图

**主动写出来**，因为"知道自己的边界在哪"比"声称什么都行"更可信。

### 当前的真实局限

| 局限 | 具体表现 | 影响 |
|---|---|---|
| **检索评测只到文献级** | 有 recall@k / nDCG / MRR，但没有 **claim-level 忠实度**（`[n]` 是否真的支持那句话） | 引用"存在性"可查，"支持性"不可查；需要 NLI/LLM 逐句核验 |
| **没有端到端可观测性** | 有 LLM 分段耗时日志，但没有 trace、token/成本核算、失败分类 | 线上问题定位靠翻日志 |
| **单进程、单写者** | agent 跑在 `asyncio.create_task`；SQLite 单写 | 进程崩溃丢当前阶段的中间结果；不能水平扩展 |
| **无鉴权与多租户** | REST 全开放，无用户隔离、配额、审计 | 仅适合单机个人使用 |
| **检索较基础** | 无 cross-encoder 重排、无 query 改写/HyDE、无多跳 | 复杂查询的召回还有提升空间 |
| **无 prompt 版本管理 / A-B 实验** | 提示词是代码里的常量 | 改提示词无法灰度、无法归因 |
| **无容器化 / 无 schema 迁移** | 靠 `CREATE TABLE IF NOT EXISTS`，改列会痛 | 部署与演进规范性不足 |
| **CNKI 不可用** | 公开检索页改为 JS 渲染、接口 403 | 中文文献主要靠 OpenAlex + 题录导入覆盖 |

### 路线图（按投入产出排序）

1. ~~**CI + pre-commit**~~ ✅ **已完成**——6 个作业，含检索质量门禁与 `.bat` 换行符守卫。
2. ~~**RAG 评估体系**~~ ✅ **已完成**——指标 + 消融 + 阴性对照 + CI 门禁，
   方法论与偏差说明见 [`docs/EVALUATION.md`](docs/EVALUATION.md)。
   下一步是补 **claim-level 忠实度**（下面第 3 项）。
3. **引用忠实度**（2~3 天）—— 用 NLI 模型或 LLM 逐句核验"这句话是否被所引文献支持"，
   把现在的"引用存在性校验"升级为"引用支持性校验"。
4. **可观测性**（2 天）—— OpenTelemetry 打点 + 每次运行的可视化 trace + token/成本核算 + 失败分类。
5. **容器化 + schema 迁移**（1~2 天）—— Dockerfile 固化环境；Alembic 或轻量迁移表。
6. **横向扩展**（1 周）—— PostgreSQL + pgvector、任务队列（Arq/Temporal）、无状态服务、语义缓存。
7. **检索质量提升**（2~3 天）—— cross-encoder 重排、query 改写、chunk 策略消融。
   评测框架已就位，因此每一项都能立刻给出"涨了多少"的数字。
8. **领域专业度**（持续）—— claim-evidence 逐句对齐、GRADE 证据分级、
   PRISMA/CONSORT/STROBE 报告规范检查、统计报告一致性校验。
9. **记忆升级**—— 从"单条纠错记忆"扩展到长期研究记忆（跨会话课题上下文、期刊/审稿人偏好）。

---

## 组合检索（多关键词 / 布尔检索）

检索框支持与 PubMed、Web of Science 一致的写法，**输入时会实时回显解析结果**，
不用记语法：

| 写法 | 含义 | 例子 |
|---|---|---|
| 空格、`,`、`，` | **AND**（同时包含） | `rTMS 卒中后抑郁` |
| `\|`、`OR`、`；`、`;` | **OR**（任选其一，用于同义词） | `rTMS \| 经颅磁刺激 卒中后抑郁` |
| `-词`、`NOT 词` | **排除** | `rTMS 卒中后抑郁 -大鼠` |
| `"词组"` | **精确短语** | `"post-stroke depression" rTMS` |

实测效果（真实 PubMed 请求）：

```
输入   rTMS | 经颅磁刺激 卒中后抑郁
解析   卒中后抑郁 且 (rTMS 或 经颅磁刺激)
PubMed (卒中后抑郁 AND (rTMS OR 经颅磁刺激))

输入   transcranial magnetic stimulation, depression -animal
PubMed (transcranial AND magnetic AND stimulation AND depression) NOT animal
```

**各库能力不同，前端会分别显示实际发出的检索式**：PubMed / Europe PMC / arXiv
支持完整布尔；OpenAlex / Semantic Scholar / Crossref 只做相关度检索，
会退化为核心词（OR 组的同义词都会带上以便排序）。

数据源选择：点击顶部「数据源」下拉，或直接点检索面板上的「数据源：…」按钮。

---

## 使用限制与注意事项

架构层面的局限见[已知局限与路线图](#已知局限与路线图)，下面是**使用者会实际碰到**的几条：

1. **`nomic-embed-text` 的中文语义能力一般**，中文语义召回偏弱；中文文献多时建议换 `bge-m3`（维度会自动同步），详见「切换嵌入模型」。
2. **本地 8B 模型的写作深度有限**：结构完整、引用规范，但论述深度不及云端大模型。对成稿质量要求高时建议切到 DeepSeek（一次综述不到一毛钱）。
3. **Semantic Scholar 无 Key 时限流严重**：程序会自动降速并让其它数据源继续工作（单源失败不影响整体），但建议申请免费 Key。
4. **PDF 全文解析需要可选依赖 PyMuPDF**（`pip install pymupdf`）；未安装时只走 JATS / PMC 通道。
5. **CNKI 不可用**（见上文），中文文献覆盖率不等同于知网；补足方式是题录导入。
6. **P2 的定时追踪尚未实现**：`subscriptions` 表结构已就位，但还没有调度器与简报生成。
7. **便携运行时为 Windows x64**；macOS / Linux 需自备 Python 3.10+。

---

## 合规声明

* 所有文献数据通过各平台的**官方开放 API** 获取，不含任何绕过付费墙的功能。
* 系统对非开放获取文献**仅存储元数据与指向出版商页面的链接**，不下载、不缓存、不再分发全文。
* 开放获取全文来自 Europe PMC / PMC / arXiv 等明确授权再分发的来源。
* 请遵守各 API 的使用条款，将获取的文献**仅用于个人学习和研究目的**，不得用于商业用途或批量再分发。
* CNKI 相关实现只访问面向公众开放的检索页，不登录、不破解验证码、不绕过任何访问控制。
* **不使用图书馆账号批量抓取订阅资源**：本项目不提供任何以机构凭据认证并
  批量下载订阅全文的功能。订阅资源只支持「导出题录后导入」与「跳转链接阅读」
  两种用法；官方批量数据包（PubMed baseline、PMC OA Subset）与 DOI 开放获取
  发现（Unpaywall）是获取大量全文的合规替代方案。
* 导入的题录与本地 PDF 索引**只存在你自己的 `data/` 目录**，打包分享时不会
  随程序分发（`scripts/pack_share.py` 排除 `data`），每位使用者各自建库。

---

## 许可

MIT
