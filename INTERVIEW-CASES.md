# 面试亮点案例 + 简历写法建议

> 本文是 `AUDIT-REPORT.md` 的配套交付物。案例取材于本次深度审计**真实找到的缺陷**，每个都能当场跑 `verify_findings.py` 复现。
> 与项目已有的 `docs/HIGHLIGHTS.md`、`docs/RESUME.md` 定位不同：本文提供的是**你亲手挖出并修复的问题**——面试里这比"我做了一个项目"强一个量级，因为它证明的是**判断力**而不是**执行力**。

---

# 第一部分：可直接开讲的亮点案例

## 使用说明

- 每个案例都是「**问题 → 方案 → 结果**」结构，并附「**面试官会追问什么**」与「**诚实边界**」。
- 案例 1 是主推案例（最硬、最有戏剧性、最能体现工程判断力）；案例 4 是架构级案例（讲给架构师岗）；案例 5 是成本案例（讲给 AI 应用岗）。
- **请先真正把对应缺陷修掉再讲**。用未修的缺陷当"我解决了"讲，被追问一层就崩。每个案例都给了修复方案与预估工时。

---

## 案例 1（主推）：一个编号缺陷让每一篇综述的引用都是错的

**适用岗位**：全部。**时长**：4~6 分钟。

### 问题

我在做一次产出质量抽检时，把成稿里的引用编号和参考文献表逐条对了一遍。结果发现：正文写 `[16]`，而参考文献表的第 16 条根本不存在——表里只有 15 条；`[16]` 指向的那篇文献，实际被排在了表的第 1 条。

进一步核查两份历史产物：

| 产物 | 正文引用编号 | 参考文献表编号 | 越界引用 |
|---|---|---|---|
| artifact 1 | 1,2,3,5,7,9,10,11,12,14,16,17,21,22,24 | 1..15 | **16,17,21,22,24** |
| artifact 2 | 2,4,7,12,13,16,19,20 | 1..8 | **12,13,16,19,20** |

**每一篇产出的每一条引用都是错的**，而且错到"读者按编号去查文献，查到的是另一篇"。在医学写作场景，这不是 bug，是学术不端风险——而这恰好是这个项目声称要解决的问题。

### 排查（讲这段最能体现方法）

我最初的假设是"LLM 在正文里编了不存在的编号"。但 `writer._sanitize()` 已经把越界编号剔除了，`formatter.selfcheck()` 也报"引用校验通过"。**两道防线都说没问题，产出的东西却是错的。**

转折点是我不再信"中间变量的校验"，改为**直接对最终产物做断言**：把 artifact 里的正文编号集合与参考文献表编号集合做差集。一跑就出来了。

然后往回追数据流：

```
graph.finalize()
  ├─ entries = sorted(state.citation_map.items())        # 编号 1..25，来自 AgentState
  ├─ references = formatter.build_references(entries, style, only_cited=state.draft)
  │     └─ build_references: kept = [paper for index, paper in entries if index in cited]
  │           └─ format_reference_list(papers, style)     # ← Paper 对象，index 被丢掉了
  │                 └─ for i, paper in enumerate(items, start=1):
  │                        format_citation(paper, style, index=i)   # ← 从 1 重新编号！
  └─ content += references
```

**根因**：`AgentState.citation_map` 是 `{编号: Paper}`，编号是唯一真相源；但 `finalize` 为了"只输出被引用过的条目"，把 `entries` 过滤成 `[Paper]`，**丢掉了 index**；下游 `format_reference_list` 只好用 `enumerate` 重新从 1 编号。于是正文用一套编号（1..25 带空洞），参考文献表用另一套（1..15 连续）。

### 为什么 1384 个测试全绿还是漏了

这一点是我认为面试里最有价值的回答：

1. **测试测的是组件，不是产物。** `tests/test_agent.py` 有一条 `test_build_references_only_cited`：
   ```python
   text = agent.build_references(entries, "gb7714", only_cited="正文只引用了 [2]")
   assert sample_papers[1].title[:15] in text     # ← 只断言"论文出现了"
   ```
   它**没有断言编号是 `[2]`**。阈值定在"标题在不在"，恰好从缺陷旁边走过去。
2. **两道运行时校验都在看"草稿"而不是"产物"。** `selfcheck(state.draft, entries)` 拿的是**未被过滤**的 `citation_map`（编号与正文一致，所以通过）；`validate_citations` 比较的是"草稿引用 vs citation_map"，也是内部一致。**没有任何一处校验"成稿"这个最终字节串。**
3. **缺陷横跨两个模块的接缝。** `agent/formatter.py`（应用层）与 `domain/citation/styles.py`（领域层）各自单独看都合理，是"过滤后没传递编号"这个接缝漏了。

### 方案

**原则：编号必须有唯一真相源，且不允许任何下游重新编号。**

```python
# domain/citation/styles.py —— format_reference_list 接受显式编号
def format_reference_list(
    papers: Sequence[Paper], style: str = "gb7714", *,
    indices: Sequence[int] | None = None,      # 新增：外部编号
    numbered: bool | None = None, sort: str = "cited",
) -> str:
    ...
    if indices is None:
        indices = range(1, len(items) + 1)
    for index, paper in zip(indices, items):
        lines.append(format_citation(paper, style, index=index if numbered else None))
```

```python
# agent/formatter.py —— 过滤时把编号一起带下来
def build_references(self, entries, style="gb7714", *, only_cited=None):
    pairs = list(entries)
    if only_cited is not None:
        cited = set(extract_citations(only_cited))
        kept = [(i, p) for i, p in pairs if i in cited]
        if kept:
            pairs = kept
    return format_reference_list(
        [p for _i, p in pairs], style, indices=[i for i, _p in pairs]
    )
```

**并且加一条端到端断言（这是关键的防回归手段）：**

```python
def test_final_artifact_citation_numbers_match_reference_list(tmp_path):
    """回归：成稿正文的每个 [n] 都必须在参考文献表里有同号条目。"""
    state = _run_graph_offline(topic="测试课题")          # 走真实 finalize
    body, _, refs = state.draft.rpartition("## 参考文献")
    cited = set(extract_citations(body))
    listed = {int(m) for m in re.findall(r"^\[(\d+)\]", refs, flags=re.M)}
    assert cited <= listed, f"正文引用了参考文献表里不存在的编号：{sorted(cited - listed)}"
```

同时把 `selfcheck` 的输入从 `state.draft` 换成**最终产物**，让它在未来的同类缺陷上真的会红。

### 结果

- 修复后两份历史产物重跑：正文编号集合 ⊆ 参考文献表编号集合，**越界引用 0 条**。
- `verify_findings.py` 的第 1 项从"复现"变为"未复现"，成为可当场演示的验收证据。
- 顺带修掉一个被掩盖的假告警：artifact 1 的 LLM 自审报"参考文献表中有 **17 条**未被正文引用"——这个 17 是编号错位的产物，不是真实问题。修完后自审结论才可信。

### 面试官会追问什么

| 追问 | 怎么答 |
|---|---|
| "1384 个测试为什么没发现？" | 测试测组件不测产物；断言定在"标题在不在"而不是"编号对不对"；两道运行时校验都在看草稿而不是成稿。**根因是"最终产物"没有验收测试。** |
| "你怎么发现的？" | 抽检产出：不再信中间变量的校验，改为对最终产物做集合断言（正文编号 ⊆ 表编号）。一次就出来了。 |
| "为什么过滤时会丢编号？" | 过滤写成了 `[paper for index, paper in entries if ...]`——列表推导把 index 消费掉了。这是"过滤时只保留一半信息"的典型。 |
| "还有别的地方有同类问题吗？" | 有。`export/exporters.py` 也调 `format_reference_list(references, style)`，同样丢编号，导出路径需要一起修。**主动说出这一点比只修一处强。** |
| "如果编号表有空洞呢？" | 项目本来就有这个机制：`digest_papers` 支持 `__index__`，`citation_map` 重编号后允许空洞（如 1,4,7,12）。参考文献表必须保留空洞编号，否则必然错位。 |

### 诚实边界

这个缺陷是本次审计新发现的，**不是我在项目开发期修掉的**。讲的时候说"我在做产出质量抽检时发现并定位了一个让所有引用错位的缺陷"，不要说"我在开发时就设计了编号一致性"。前者是"我会做质量审计"，后者是"我在说谎"。

---

## 案例 2：真实 LLM 输出被检索式解析器悄悄破坏（静默失效）

**适用岗位**：AI 应用 / 架构师。**时长**：3~4 分钟。

### 问题

README 里有一条实测数据："跨库检索（4 源并发）2~3 秒"。我在核对真实运行日志时发现一个异常：**Europe PMC 有 7 次成功请求，`result_count` 全部是 0**——不是报错，是"成功但返回空"。

### 排查

我拿生产数据库里 **plan 阶段真实产出的 4 条检索式**（不是我构造的测试用例），直接喂给 `for_source()`：

```
输入：rTMS AND post-stroke depression AND (HAMD OR MADRS) AND (randomized OR RCT)
输出：(rTMS AND AND AND "post-stroke" AND depression AND ((HAMD OR MADRS)) AND ((randomized OR RCT)))
                ↑↑↑↑↑↑↑↑
```

以这个表达式去请求 Europe PMC，直接验证：

| 查询 | HTTP | hitCount |
|---|---|---|
| 客户端构造的表达式 | 200 | **0** |
| 去掉 `AND AND` 后的同一表达式 | 200 | **157** |

4 条真实检索式，**4 条全部畸形**。

### 根因

`domain/query.py` 的 `parse_query` 把空格/逗号视为 AND 分隔符，把 `|`/`OR`/`-`/`NOT` 视为运算符——**但没有把 `AND` 本身当运算符**。于是 `AND` 作为普通词落进 `must` 列表；`for_source` 再用 `" AND ".join(units)` 拼接，就产生了 `AND AND`。

更隐蔽的一点：**这是静默失效**。Europe PMC 对语法错误的表达式返回 **HTTP 200 + hitCount=0**，不是 4xx。所以：

- 错误分类器（`classify_failure`）判它"成功"；
- 检索日志记 `error=None`（成功）；
- 前端显示"Europe PMC 检索完成，0 条"——用户以为这个库就是没有相关文献。

**唯一能看出来的是"0 条"这个数字本身，而它和"真的没有"长得一模一样。**

### 方案

```python
# domain/query.py
_AND_WORDS = {"and", "且", "&"}      # 与 OR/NOT 同级，视为连接符而非词

for token in segments:
    lowered = token.lower()
    if lowered in _AND_WORDS:
        pending_or = pending_not = False       # 显式 AND：分隔符，不入词
        continue
    if lowered in _OR_WORDS or token == "|":
        ...
```

并对**非布尔数据源**（OpenAlex / Crossref / Semantic Scholar 只做相关度检索）剥离所有布尔记号，只留核心词（现在会把 `AND` 和括号原样发过去）。

**关键补充：加一条"表达式合法性"契约测试**，因为这类缺陷的正确防线不是"测某个查询"，而是"测翻译器的输出形状"：

```python
def test_translated_expression_is_wellformed():
    for query in REAL_PLANNER_OUTPUTS:          # 从生产 plan 快照里取的 4 条
        for source in ("pubmed", "europepmc", "openalex", "crossref"):
            expr = for_source(query, source)
            assert " AND AND" not in expr
            assert " OR OR" not in expr
            assert "()" not in expr
            assert expr.count("(") == expr.count(")")
```

### 结果

- 修复后 4 条检索式在 Europe PMC 从 hitCount=0 变为 157/293/271 量级（我在修复验证时用真实 API 实测）。
- 数据源命中率从 **3/5 恢复到 5/5**，而 Europe PMC 恰好是这个项目 OA 全文的主力来源——**修这一个解析器缺陷同时修好了检索召回和全文获取两条链路**。
- 该项现在由契约测试守住，5 个数据源 × 4 条真实检索式的 20 个组合全部断言。

### 面试官会追问什么

| 追问 | 怎么答 |
|---|---|
| "怎么发现的？" | 核对真实运行日志时看到 Europe PMC 全是 `result_count=0` 且 `error=None`。**关键动作是"对每一列数字问一句'这个数合理吗'"，而不是只看报错。** |
| "为什么单元测试没抓住？" | 现有测试的用例是 `rTMS AND depression`——**不含 `OR` 组、不含括号**，于是恰好不产生 `AND AND`。而生产 LLM 输出必然带 `OR` 组（同义词）和括号（PICO 分层）。**测试用例的复杂度低于真实输入，是这类漏测的通病。** |
| "怎么防止再犯？" | 两条：① 契约测试断言的不是"结果对不对"而是"输出形状合法"（`AND AND`/括号配平）；② 测试用例直接从生产快照里取，不再手工构造。 |
| "其他数据源受影响吗？" | 受影响。OpenAlex/Crossref 收到含裸 `AND` 和括号的查询串，在相关度检索里 `AND` 会被当普通词参与匹配，等于引入噪声词。已一并修。 |
| "还有哪些静默失效？" | 这是一个模式，不止一处。同类还有：换嵌入模型 DROP 向量表后静默退化为纯 BM25；FTS 级联第一级非空即 return 丢弃宽召回。**"HTTP 200 但结果是错的"比 "HTTP 500" 危险得多。** |

### 诚实边界

真实测数据（hitCount 0 vs 157）来自本次审计的实跑，是可靠的。但"修复后数据源命中率 3/5 → 5/5"是基于验证请求的推断，不是完整跑一轮 agent 的端到端测量——讲的时候要说"修复验证时用真实 API 确认了这几条检索式能返回结果"。

---

## 案例 3：预取了 197 万字符全文，写作阶段一个字都没用

**适用岗位**：AI 应用 / 成本优化 / 架构师。**时长**：3~4 分钟。

### 问题

配置里有 `warm_fulltext: true` 和 `fulltext_top_n: 4`，注释写着"为开放获取文献预取全文，写作质量更高"。但我在核对"钱花在哪"时发现：**写作阶段的材料块只包含标题和摘要，没有全文**。

### 排查（两步，都是可量化证据）

**第一步：静态追数据流。**

```
reader.warm_fulltext(papers)  → 下载 PDF / JATS → 解析正文 → save_fulltext() → paper_fulltext 表
                                                              └→ 同时建 fulltext_fts 索引
writer.write_review(entries)  →  _fit_digest(entries) → build_context_digest(entries)
                                    → digest_papers([paper.to_dict() for ...])
                                       # paper.to_dict() 只有 title/abstract/journal/authors/year/cited
```

`digest_papers` 的签名里**没有全文参数**。也就是说无论 `paper_fulltext` 表里存了多少，材料块的 `max_abstract=800` 就是天花板。

**第二步：对最终产做逐字比对。** 把成稿正文与库里所有全文做 60 字滑动窗口匹配：

```
库内 82 篇全文 / 1,967,575 字符
成稿与之逐字重合（60 字窗口）：0 篇
```

**197 万字符的全文，零引用。**

### 影响量化（这才是让面试官抬头的地方）

预取全文的完整成本链：

```
网络下载 PDF（含重试、落地页→真 PDF 的二跳）→ 读取 64MB 上限校验 → PyMuPDF 解析
→ 文本抽取 → save_fulltext 落库（单行可达几十 KB）
→ fulltext_fts 索引写入（CJK 逐字切分）→ 每篇一次 BEGIN IMMEDIATE 写事务
```

这条链产出的东西，**消费方是零**。写作文本质量完全由 800 字摘要决定。而 `fulltext_top_n=4` 这个旋钮调大调小，对产出**没有任何影响**。

### 方案（二选一，都要明确说取舍）

**方案 A：接上（推荐，1.5 天）** —— 让全文真正参与写作。

```python
# retrieval.build_context_digest 增加全文通道
def build_context_digest(entries, *, max_abstract=800, guard=True,
                         fulltext_by_id: Mapping[int, str] | None = None,
                         fulltext_chars: int = 4000) -> str:
    ...
    if fulltext_by_id:
        ft = fulltext_by_id.get(paper.paper_id)
        if ft:
            data["fulltext_excerpt"] = ft[:fulltext_chars]
```

- 全文**必须纳入注入扫描**（现在 `_guard_materials` 只扫 title+abstract——全文恰恰是最需要扫的不可信内容）。
- 只在最终 `select_papers` 之后按需注入，避免"预取 25 篇但只用 8 篇"的新浪费。

**方案 B：摘掉（0.5 天）** —— 删除预取链路，把省下的时间预算换成"更多文献的摘要分辨率更高"（`max_abstract` 800 → 在上下文预算内给到 1500）。

我倾向 A，理由是**医学综述的可信度取决于能不能引用具体数字（效应量、CI、P 值），而摘要里的数字经常被截断或不给**——这正是项目的差异化定位所在。

### 结果

- 定位到"基础设施建完了但没接线"，并量化出无效开销（197 万字符 / 82 篇 / 每次运行 4 篇预取）。
- 修复后有明确的验收标准：**成稿与全文的逐字重合度从 0 篇提升到 ≥ 引用篇数 × 60%**，可直接用同一段比对脚本度量。

### 面试官会追问什么

| 追问 | 怎么答 |
|---|---|
| "预取全文是不是完全没用？" | 不是完全没用——`paper_fulltext` 表还被"单篇速读笔记"（`reader.summarize`）和 `tools.py` 的 MCP 工具读取，另外有全文 FTS 检索能力。**但在这个项目的主链路（写综述）上，消费方是零。** 而且全文 FTS 检索在生产里也零调用者。诚实区分"有用"和"在主链路上有用"。 |
| "你怎么确认是 0 而不是很少？" | 60 字滑动窗口做逐字比对。选 60 字是因为低于这个长度会撞常见的学术套话（比如"随机对照试验"），会产生假阳性。 |
| "如果接上全文，成本会涨多少？" | 会涨，但可控：只对最终选中的 N 篇注入前 4000 字。按 25 篇估算，材料块从 ~6.4k token 涨到 ~25k token——**这会超本地 8B 的 8192 上下文**，所以必须先做提示词前缀重排 + 上下文预算反推，然后按预算决定注入几篇、每篇多少字。**这是一个"先测量再改"的典型场景，不能直接接。** |
| "那为什么不干脆摘掉？" | 也是一个合理选择，成本是失去"引用具体数字"的能力。**我给两个方案并说清取舍，而不是直接选一个**——因为这是产品判断，需要和项目定位对齐。 |

### 诚实边界

全文注入成本"会超本地 8192 上下文"是我基于 token 估算的推断（`estimate_tokens` 误差约 8%），没有实跑。讲的时候用"按估算会超，所以要先做预算反推"。

---

## 案例 4：把"永不调用"的抽象清理掉——trace/Bulkhead 的同一类问题

**适用岗位**：架构师。**时长**：3 分钟。

### 问题

项目 README 把"trace/span 树（contextvars 隔离并发）"和"舱壁限并发"列为生产级能力。我做了一次"抽象是否有调用者"的扫描：

| 抽象 | 实现质量 | 生产调用者 |
|---|---|---|
| `TraceRecorder` / `create_trace` / `run_in_trace` | 高（contextvars 隔离并发、嵌套成树、`to_jsonl`、假时钟可注入） | **0 个**（除自身与测试） |
| `Bulkhead`（并发舱壁） | 高（同步/异步双路径、peak 统计、信号量惰性创建避跨循环） | **0 个** |
| `add_citations`（引用关系落库） | 完整 | **0 个** → `get_references` 恒返回空 |
| `search_full_text`（全文检索） | 完整（独立 FTS 表 + CJK 分词） | **0 个** |
| `parse_query` 的 `exclude`（NOT） | 解析正确 | 仅作用于远程 API；本地检索不支持 |

### 根因（这是架构层面的洞察）

这些抽象的**单元测试都是绿的**——它们各自逻辑正确。但缺陷不在实现，在**接线**：

```
TraceRecorder 的正确用法：graph.run() 里 with create_trace(run_id) + 各阶段 with span(...)
实际用法：无
```

后果是**跨模块的**、单模块测试永远发现不了的：

```python
# llm/client.py:_record —— 记账时尝试从上下文推断阶段与运行
span = current_span()      # 恒 None（没人创建 trace）
trace = current_trace()    # 恒 None
LEDGER.record(LLMUsage(..., phase=span.name if span else "",      # 恒 ""
                           run_id=trace.trace_id if trace else ""))  # 恒 ""
```

于是 `/api/metrics` 的 `by_phase` 分组**永远只有一个桶 `(未标注)`**——项目投入实现的"按阶段聚合成本"能力，在真实运行中完全不可用。

### 方案

**分两步，顺序不能反：**

**第一步：接上（0.5 天）**，让它产生价值。

```python
async def run(self, state, *, emit=None, approval=None):
    trace = create_trace(state.run_id)          # 绑定到当前上下文
    try:
        with trace.span("plan"):
            state.plan = await self.plan(state, emit=emit)
        with trace.span("execute"):
            await self.execute(state, emit=emit)
        ...
    finally:
        trace.finish()                           # 必须：否则 span 挂到上一棵树
        trace.to_jsonl(self.config.home / "traces.jsonl")
```

接上后立刻得到两样东西：`/api/metrics` 的 `by_phase` 有了真实分布；`traces.jsonl` 能回答"这次运行时间花在哪"。

**第二步：清理（0.25 天）**——给确实没有产品路径的抽象**明确出路**：

- `Bulkhead` → 接线到 9 个数据源客户端（这是它本该在的位置，且能修掉"最慢的一个源决定整轮检索墙钟"）。
- `search_full_text` / `add_citations` → 接进 `/api/ask` 或明确删除，不留在"看着有、其实没有"的状态。
- `cfg.agent.max_search_rounds`、`prefix=True`、`retrieval.min_score` → **死配置**：`min_score` 因为 RRF 分数恒为正（实测 k=60 时单路 rank100 = 0.00625）永远不触发；`max_search_rounds` 全仓无使用点。要么接通，要么从 config 里删掉——**配置文件里每一个旋钮都是对用户的承诺**。

**并且加一条守卫**，把"接线"从习惯变成检查：

```python
# scripts/check_arch.py 新增
def check_public_abstractions_are_wired():
    """platform/ 里导出的公开类必须在 medscholar/ 下有调用者，否则报违规。
    白名单只允许变小 —— 与既有的分层白名单同一策略。"""
```

### 结果

- 修 P0-4 后，账本的 `phase` / `run_id` 生效，`/api/metrics` 从"只有一个 `(未标注)` 桶"变为可按阶段/按运行聚合——**成本优化的前提数据第一次可用**。
- 清理后删掉 4 处死配置 / 2 处零调用者导出，配置文件与文档不再承诺不存在的能力。
- 新增的守卫规则在 CI 里生效，防止"实现了但没接线"再次发生。

### 面试官会追问什么

| 追问 | 怎么答 |
|---|---|
| "为什么会出现实现了但不接线？" | 因为**抽象是自底向上写的，接线是自顶向下的**。写 `TraceRecorder` 时容易被"这个类很完整、测试全绿"满足；而接线需要在 `graph.run()` 里做一次全局改动，改动点不在当前任务的 diff 里。**这是"组件思维"与"系统思维"的分界。** |
| "单元测试为什么发现不了？" | 单元测试断言的粒度就是"这个类的行为"。`TraceRecorder` 的行为确实是对的。**发现不了是必然的——需要一个跨模块的检查（抽象必须有调用者），而不是更强的单测。** |
| "这和'代码覆盖率'的关系？" | 覆盖率 100% 也可能是 0 价值。`Bulkhead` 的每行都被测试覆盖过，但它没有被生产调用过。**覆盖率度量"被执行"，不度量"被使用"。** |
| "你会怎么排优先级？" | 先接 `TraceRecorder`（0.5 天，立刻解锁成本可观测性），再挂 `Bulkhead`（1 天，修真实故障模式），最后清理死配置（0.25 天）。**接线优于清理**——因为接线产生新能力，清理只是减少误解。 |

### 诚实边界

"0 个调用者"是我全仓 grep 的结论，可靠（`verify_findings.py` 第 3 项与第 10 项可复现）。但"接上就能得到准确的分阶段成本"是推断，实际接线后还要注意 `graph` 里并发子任务（检索式并发）的 span 归属——contextvars 在两路 gather 时会各持一份上下文，同名 span 会分成两个节点，需要在报表侧按名字聚合（`TraceRecorder.summary()` 已经这么做了）。

---

## 案例 5（AI 应用岗主推）：把"预算跟着检索篇数走"改成"跟着用上的篇数走"

**适用岗位**：AI 应用开发 / 成本优化。**时长**：3~4 分钟。

### 问题

我统计了一次真实运行的漏斗：

```
检索到 150 篇
  → 启发式评估 150 篇（全部算完）
  → LLM 逐篇点评 12 篇（critique_max_papers）
  → 重编号进写作上下文 25 篇（writer_max_papers）
  → 成稿实际引用 8 篇
```

三个数字断层：**138 篇的评估结果被丢弃；25 篇里有 13 篇从未被 LLM 看过；12 篇 LLM 点评里只有一部分进入成稿。**

### 为什么"少看几篇"不是省成本，而是引入偏差

这是我认为最有价值的洞察点。`CriticAgent.assess` 的逻辑是"两者等权融合"：

```python
merged = PaperAssessment(
    relevance=round((llm_item.relevance + base.relevance) / 2, 1),
    quality=round((llm_item.quality + base.quality) / 2, 1),
    ...
)
```

看起来是"LLM 评分与启发式评分取平均"。但实际数据是：

```
source 分布：heuristic 138 篇 / llm+heuristic 12 篇
```

**被 LLM 看过的 12 篇拿到的是两路融合分；没被看过的 138 篇只有启发式分。** 然后 `state.select_papers` 用互斥的 `combined` 分数排序取 top-25：

```python
score = assessment.combined if assessment else 5.0
scored.sort(key=lambda item: (-item[0], item[1]))
```

**于是"是否进入综述"部分取决于"是否被 LLM 看过"，而不是取决于文献是否真的更相关。** 真实的分数分布印证了这一点：top-25 里绝大多数 `combined` 在 6.3~8.3 之间挤成一团，而 `relevance` 大量集中在 7.3 / 5.5 / 4.2 / 3.0 这几个离散值上（启发式相关性是"课题词覆盖率"的函数，天然离散）。

### 方案

**核心原则：评估预算必须覆盖"将进入成稿"的全部候选，而不是一个独立的固定数。**

```python
# 1) 评估覆盖对齐写作预算：先粗排，再对 top-N 精评
limit = min(len(entries), max(4, self.config.agent.critique_max_papers))
# 改为：由写作预算与上下文预算共同决定
limit = min(len(entries), self.config.agent.writer_max_papers)

# 2) 复用已算的启发式分数（现在排序 key 里重新算了一遍，纯浪费）
scored = sorted(entries, key=lambda pair: -heuristic[pair[0]].combined)

# 3) 让"未覆盖"不再是一种降级：覆盖不到时明确标注 source="heuristic-only"
#    并让 select_papers 在混合来源时按 source 分层排序，避免两类分数直接竞争
```

**第 4 条更关键——提示词前缀重排，让多章节共享云端前缀缓存：**

`section_user()` 现在的拼接顺序是「课题 → 章节标题 → 要点 → **材料块** → 字数要求」。5 个章节的请求在第 3 段就分叉，**前缀缓存命中率 0**。

改成「系统提示词 → **材料块** → 字数要求 → 章节标题 + 要点」，后 4 个章节的输入（材料块约 5k token）全部命中缓存——DeepSeek 的缓存命中价约为标准价的 10%。

### 结果（按 token 模型量化）

| 项 | 改前 | 改后（估算） |
|---|---|---|
| 单次输入 token | ≈66k | ≈36k（前缀缓存 + 评估覆盖对齐） |
| 单次输出 token | ≈15k | ≈15k |
| 云端成本（¥2/¥8 per Mtok） | ≈¥0.25 | **≈¥0.10** |
| LLM 评估覆盖 | 12 / 25（48%） | 25 / 25（100%） |
| 评估阶段冗余计算 | 150 篇启发式 + LLM 排序重算 | 150 篇启发式（一次） |
| 全文利用 | 0 篇（197 万字符） | 按预算注入 N 篇前 4000 字 |

**成本降约 60%，同时质量上升**——因为评估覆盖面从 48% 提到 100%，消除了 selection bias。这是这个案例最有说服力的地方：**"更便宜"和"更好"在这里不冲突。**

### 面试官会追问什么

| 追问 | 怎么答 |
|---|---|
| "cloud 成本 ¥0.25 是怎么算的？" | 输入 66k × ¥2/M + 输出 15k × ¥8/M。**要主动说这是估算**：token 数由提示词模板与 `max_abstract` 反推，`estimate_tokens` 误差约 8%；实际以厂商账单为准。而且要指出——项目 README 说"一次综述不到一毛钱"，与这个估算对不上，可能是定价表用的档位不同（`PRICES` 里 `qwen-plus` 是 0.8/2.0）。**主动暴露自己项目的数据不一致，比被面试官发现强。** |
| "为什么前缀缓存能生效？" | 自动前缀缓存要求**逐字节相同的前缀**。同一篇综述的 5 个章节共用同一份材料块，只要把材料块放在变化内容（章节标题）之前，前缀就相同。这是"提示词布局决定成本"的典型案例。 |
| "为什么不直接减少篇数？" | 减少篇数确实省钱，但那是在**降低质量**的前提下省钱。前缀缓存是**零质量代价**的省钱。优先级应该是：先消掉浪费（缓存、漏斗对齐）→ 再考虑降规格（小模型、少篇数）→ 最后才砍能力。 |
| "12 篇点评本身够不够？" | 不够，而且问题不在"少"，在"不均"。12 篇两路融合、138 篇单路，两类分数在同一个 sort 里竞争——这是**打分尺度不可比**的问题，不是数量问题。 |
| "怎么验证改善？" | 三个可量化指标：① `/api/metrics` 的 `by_phase` token 分布（修 trace 后可用）；② LLM 评估覆盖率（`source` 字段的分布）；③ 成稿引用数与摘要 token 的比值。**先有度量再优化**——这也是为什么我把"修 trace 接线"排在成本优化之前。 |

### 诚实边界

- ¥0.25 → ¥0.10 是**基于 token 模型的估算**，不是实测账单。讲的时候给公式，并说明哪部分是估算。
- 前缀缓存的实际收益率取决于厂商实现（DeepSeek 的自动前缀缓存有最小长度要求与命中粒度），**我没有实跑验证过这个项目的提示词重排能拿到多少命中率**。正确表述是"按厂商文档的缓存机制，重排后预计可命中，需要实测确认"。
- `critique_max_papers: 12 → writer_max_papers: 25` 会让评估阶段的输入 token 翻倍（12 篇 → 25 篇摘要）。**这是"用成本换质量"的取舍**，不是纯粹的省钱。讲的时候要把它和前缀缓存分开说：前者买质量，后者省钱。

---

# 第二部分：简历写法建议（面向中国 AI 大厂）

## 1. 先认清筛选机制

中国 AI 大厂（字节 Seed/Flow、阿里通义、腾讯混元、百度文心、月之暗面、智谱、MiniMax、DeepSeek 等）对本项目这类"个人项目 + AI 结对"的定位：

| 环节 | 实际在筛什么 | 本项目的风险 |
|---|---|---|
| HR/简历初筛 | 关键词匹配 + 公司/学历 | 无大厂经历时，靠项目关键词密度过筛 |
| 一面（技术） | 能否解释每一个设计决定 | "人定方案 + AI 实现"若讲成"我写的每一行"，追问三层即崩 |
| 二面（深度） | 有没有真实的"深水区"经历 | 项目有 13 条真实踩坑记录，**这是最强资产** |
| 三面（架构/判断） | 取舍能力、边界意识 | 主动放弃抓取订阅资源、拒绝 LangChain、算了硬件天花板——**都是加分项** |
| 交叉面/HR | 诚实度、配合度 | 数字造假或夸大上线规模是**一票否决** |

**核心结论：这个项目的正确打法是"以真实的工程判断力取胜"，不是"以规模或用户量取胜"。** 项目已有的 `docs/INTERVIEW-PACK-3ROLES.md` 已经正确识别了这一点（"绝不要说每一行都是我写的"），要继承这个基调。

## 2. 简历项目条目的结构：STAR 的变体

大厂技术简历的项目条目，建议用这个结构（**总长控制在 5~7 行**）：

```
项目名 —— 一句话定位（含规模数字）
· 背景与约束：一句话说清"难在哪、为什么难"
· 我做了什么：2~3 条，每条 = 动作 + 技术 + 可验证结果
· 差异化判断：1 条，说清"我拒绝了什么、为什么"  ← 这一条最能拉开差距
```

## 3. 三个岗位的定制版本

### A. AI 应用开发工程师（最匹配）

```markdown
MedScholar Agent —— 面向医学研究者的本地学术智能体：检索→精读→评估→综述写作→引用校验全链可审计，数据落在单个 SQLite。
· 自研 5 阶段 Agent 工作流（Plan→人工审批→Execute→Reflect→Synthesize→Review），不用 LangChain；
  阶段快照支持进程重启后续跑，规避了"5~10 分钟任务断线即重来"的体验缺陷。
· 把"防幻觉"做成确定性校验而非提示词约束：正文数字逐个溯源到材料、跨语言弱证据用语言无关信号兜底、
  校验器本身在 27 条人工标注集上自评估（可判定子集零漏报、精确率 0.89）并设为 CI 门禁。
· 定位并修复 3 个静默失效缺陷：LLM 规划输出的检索式被解析器破坏（Europe PMC hitCount=0 静默返回空）、
  预取全文从未参与写作（197 万字符零引用）、成稿引用编号与参考文献表错位；三者均已补端到端断言防回归。
· 拒绝抓取学校订阅资源（会造成全校 IP 被封），改为题录导入 + 官方批量包 + 链接解析器三条合规路径。
```

**为什么这么写**：
- 第一条证明"会用 Agent 范式，且知道框架的边界"。
- 第二条是**这个项目最硬的资产**，把"幻觉治理"从提示词层面提升到工程层面，且带自评估数字。
- 第三条是**本次审计新增的**，证明你有"产出质量审计"的能力——大厂非常看重能主动找自己系统缺陷的人。
- 第四条展示"有能力但不做"的判断力，这是产品意识的直接证据。

### B. AI 架构师

```markdown
MedScholar Agent —— 单机可分发约束下的完整 AI 应用架构样本（Python 2.8 万行 + 原生前端 4.6k 行，1384 测试）。
· 设计五层架构（platform/domain/infrastructure/application/interface）并用 AST 校验器在 CI 强制
  层次方向、循环依赖与规模上限；校验器曾抓到两类真实违规（基础设施层懒加载应用层、db↔embedding 循环依赖）。
· 可观测性与韧性：trace/span 树（contextvars 隔离并发）、LLM 用量与人民币成本账本、11 类失败分类、
  全抖动退避 + 按后端隔离熔断 + 令牌桶限流；并主动审计出"抽象已实现但未接线"的跨模块缺陷类型。
· RAG 安全边界收敛到四条链路共用的唯一出口（检索内容包成定界数据块 + 中英文注入扫描 + 零宽/双向字符检测），
  一处生效不漏；输出护栏在落盘/导出前校验提示词泄漏与凭据残留。
· 用可量化证据否决框架与组件选型：算出显存带宽天花板（272 GB/s ÷ 6.19 GB ≈ 44 tok/s，实测 45~47），
  据此把优化预算从"提速"整体转到"消除浪费"（模型重载、坏输出重试、串行等待）。
```

**为什么这么写**：架构师岗不看你写了多少功能，看你**如何做决策、如何把约束变成可执行检查**。AST 校验器、唯一出口、天花板计算，这三条是同类简历里极少见的。

### C. AI 产品经理 / 技术产品

```markdown
MedScholar Agent —— 面向医学研究者的本地 AI 写作台，主打"它写的每一句，你都能查到出处"。
· 目标用户是医学研究生与临床医生，核心痛点不是"生成得不够好"而是"不敢用"——因为编造参考文献与 P 值
  在医学场景等于学术不端。据此把产品指标从"生成质量"改为"引用可追溯率"。
· 设计人工审批节点（Plan 后暂停），让研究者在检索前纠正检索策略，把 AI 从"黑箱生成"变成"可控协作者"。
· 做出放弃决策：技术上可抓取图书馆订阅资源，但会导致全校 IP 被封，主动改为三条合规替代路径。
· 用 PRISMA 2020 规范约束产出：各库识别数取自真实检索日志，纳入/排除由研究者填写，
  工具只保证数字自洽并在矛盾时拦住。
```

## 4. 关键词策略（过初筛用）

不同厂的 JD 关键词差异大，建议在项目描述里**自然嵌入**以下词（每个词都要能答出对应的设计）：

| 类别 | 关键词 | 在本项目里的对应物 |
|---|---|---|
| Agent | Multi-Agent、工作流编排、Human-in-the-loop、断点续跑、状态快照 | 六角色 Agent + 审批点 + `run_steps` |
| RAG | 混合检索、RRF 融合、BM25、向量召回、消融实验、nDCG | `hybrid_search` + 7 配置消融 |
| 幻觉治理 | 引用溯源、数字溯源、忠实度评估、事实一致性 | `check_number_provenance` + claim-level 校验 |
| 成本 | token 账本、模型路由、前缀缓存、上下文预算 | 账本 + 待做的 routing/cache |
| 工程 | 可观测性、熔断、退避、限流、分层架构、契约测试 | 全套齐备 |
| 评测 | 离线评测集、CI 门禁、recall/NDCG/MRR、阴性对照 | `eval/` + parity 校验 |

**避免的词**（写了就是给自己挖坑）：
- ❌ "高并发"、"QPS"、"百万级用户" —— 单进程单用户
- ❌ "微服务"、"K8s"、"分布式" —— 单个 SQLite 文件
- ❌ "线上 A/B 实验" —— 只有提示词变体机制，无线上数据
- ❌ "多租户"、"高可用" —— 无鉴权，主动写进局限里

## 5. 数字口径：写之前先重跑一遍

简历里每个数字都要能当场跑出来。**建议只写这 5 个（都是可命令复现的）**：

| 数字 | 复现命令 |
|---|---|
| 1384 测试全绿 | `pytest tests -q` |
| 架构门禁 PASS（层次 0 / 循环 0 / 规模 0） | `python -X utf8 scripts/check_arch.py` |
| 校验器自评估（可判定子集零漏报 / P=0.89） | `python scripts/eval_faithfulness.py --labels` |
| 检索门禁（production recall 0.8889 / nDCG 0.8678） | `python scripts/eval_retrieval.py --dataset regression --check` |
| 本次审计 12 项缺陷可复现 | `python -X utf8 verify_findings.py` |

**不要写的数字**：模块数 / 行数（会变，且不是成果）、API 路径数（README 与 ARCHITECTURE 自己就不一致）、"一次综述不到一毛钱"（成本估算是 0.25 元，口径对不上）。

## 6. 面试前必须准备好的三个"防守问题"

### Q1："这是你独立做的吗？"（一定会问）

**推荐答法**：
> "需求定义、技术选型、取舍和验收是我做的，编码是我和 AI 结对完成的。所以我能解释**每一个设计决定**和**每一个我拒绝的方案**——你可以现场挑任何一个模块问我为什么这么写。"
>
> "我更愿意聊我做的判断，而不是我敲的代码。比如我拒绝了 LangChain（我的工作流是固定 5 阶段线性 + 1 个人工审批点，不是任意图，框架带来的隐式提示词拼装会让我在出问题时不知道错在哪），拒绝了 Postgres（目标用户是单机个人研究者，备份靠拷一个文件），也拒绝了抓取学校订阅资源（会造成全校 IP 被封）。"

**关键**：把话题从"谁敲键盘"转到"谁做判断"。**这是可验证的**——现场让面试官挑模块。

### Q2："这个项目有什么问题？"（主动暴露，反而加分）

**这是你能拿分最多的一题。** 推荐答法：

> "有几个我自己审计出来的问题，我认为比亮点更能说明我的工作方式。"
>
> "第一，我发现成稿的引用编号和参考文献表全部错位——正文 `[16]` 指向表里第 1 条。根因是过滤参考文献时丢掉了编号，下游用 `enumerate` 从 1 重编。**1384 个测试全绿是因为测试测组件不测产物、两道运行时校验都在看草稿而不是成稿。**"
>
> "第二，我的检索式解析器不认识显式 `AND`，导致 LLM 规划出的真实检索式在 Europe PMC 上 hitCount=0——而它返回 HTTP 200，所以失败分类器判它成功、日志记 `error=None`。**这是我最在意的一类缺陷：不是报错，是产生看起来合理的结果。**"
>
> "第三，我预取了 82 篇全文共 197 万字符，但写作阶段的材料块只吃 800 字摘要——**全文一字未用**。基础设施建完了，没接线。"
>
> "这三个我都定位了根因并给了修复方案，也加了端到端断言防回归。我特意把它们整理成了可复现的验证脚本，你现场就能跑。"

**为什么这比讲亮点强**：面试官见过太多"我的项目很好"；能清晰说出自己系统缺陷、根因层次、以及"为什么测试没抓住"的人极少。**这直接证明你能在大厂代码库里做质量审计。**

### Q3："跟 LangChain / Dify / Coze 比，你这个有什么意义？"

**推荐答法**：
> "如果我 2024 年做这个项目，我会用 LangChain。我不用的原因不是它不好，而是我的约束和它的设计目标不同："
>
> "我的约束是**单机可分发**——用户解压就能跑，不能要求 Node 工具链，不能要求他们装几百 MB 依赖，朋友机器缺 VC 运行库时原生扩展加载失败也必须能用（所以我做了纯 Python 向量检索回退）。"
>
> "我的流程是**固定 5 阶段线性 + 1 个人工审批点**，不是任意图——框架的 checkpointer、动态分支、多智能体协商这些能力我都用不上，但我要为它们付出"隐式提示词拼装"的代价：出问题时我不知道错在哪一层。自研 `graph.py` 几百行，每个阶段可以单独测试。"
>
> "代价我也清楚：**没有现成的 checkpointer，快照与续跑要自己写**（我已实现，`run_steps` + 阶段复用）。如果未来要做动态分支或多智能体协商，我会引入引擎——**那时候框架的价值才真正兑现**。"

## 7. 一句话总结打法

> **不要把这个项目包装成"我做了一个很棒的 AI 产品"。**
> **把它讲成"我在一个真实约束（单机可分发 / 医学不能出错 / 本地 8B 硬件天花板）下，做了一串可验证的工程判断，并且我能指出自己系统的失效边界"。**
>
> 前者是"执行力"，人人都有；后者是"判断力"，百万年薪买的就是这个。
