---
name: audit-repair-discipline
description: 修复深度代码审计中发现的"静默失效"类缺陷时必须遵守的工作纪律。覆盖：(1) 修代码不要引发 SyntaxError 拖垮全部测试，(2) 审计脚本的"复现"语义与"修复后"的 PASS 含义要分清，(3) 端到端测试的硬/软断言取决于可执行环境，(4) 栈推断等"实现细节"代码必须 fallback 到标准做法且有界，(5) 文档数字与代码基线要同源更新。
when_to_use: 当用户让你修复深度审计 / 安全审计 / 性能审计报告里的 P0 / 静默失效类缺陷，或让你把"复现脚本"扩展为"回归契约"时。任何涉及"修代码 + 改测试 + 改文档 + 改 verify 脚本"四件套的复合任务，先调此 skill。
---

# audit-repair-discipline

> 我自己（MiniMax-M3）在修 `medscholar-agent` 4 个 P0 时**真犯过**的错。下面的每条都标了"为什么这是错、当时怎么栽、收尾时的真相"，便于未来 AI 一次看穿。

## 1. 大块缩进改动 = 全测试收集阶段 SyntaxError

**为什么这是错**：当一段代码已经是 `try / except / finally` 已闭合的块时，在外层加一行（例如 `with trace.span(...)` 或 `async with ...`），看起来"只多一行"，但 Python 用缩进表达所有块的开始/结束。**只多外层一行 = 内层全部要重缩进 = 一旦你漏一行就会出现 `else:` 浮在 `if` 之外**。

**我当时怎么栽**：在 `medscholar/agent/graph.py` 的 `ResearchGraph.run` 里加了 `with trace.span("plan"):` 又只对一行代码缩进，剩下几十行没动。pytest 收集阶段就 `SyntaxError: invalid syntax` 爆掉，1384 测试全部无法运行——CI 红屏一片但你看不到哪个测试挂了。

**怎么避免**：
- 永远不要"在已闭合块外层只加一行"。要么**整块重缩进**（用 IDE 的 block indent），要么**用不改变缩进的策略**——比如：
  1. 把 `create_trace()` 放在 try 块**之前**，trace 上下文靠 `with` 进入时绑定即可；
  2. 用 `with current_trace().span("xxx"):` 包单个方法调用（不包多行），保持外部缩进不变；
  3. 用装饰器或 contextmanager 在被调函数**内部**打开 span（侵入最小）。
- 改完 30 秒内跑一次 `python -c "import medscholar.agent.graph"` 或 `pytest --collect-only`——**先确保能收集，再跑测试**。这一步在 `medscholar-agent` 那次栽了 90 秒才发现。
- **强规则**：任何修改同时影响 5 行以上缩进的 PR，必须先在本地跑 `pytest --collect-only` 再 `pytest -q`。collect-only 是语法错误的"快速触电测试"。

## 2. 审计脚本的"PASS 含义"必须显式区分复现 / 修复后

**为什么这是错**：`verify_findings.py` 写的时候是**反向断言**——"缺陷应仍存在"。改完代码后它自然变 FAIL，但脚本不告诉你这是好事还是坏事，导致：
- 用户看到 FAIL 误以为修复回退了；
- 下次 AI 接手会把它当成"测试挂了"去 debug；
- 长期会变成"脚本老报错，谁都不敢碰"。

**我当时怎么栽**：第一次跑修复后脚本变 FAIL，我没改脚本就跑去别的事。下次再跑出来还是 FAIL——已经在脑子里形成"verify_findings 是出了名的烂脚本"的错觉，差点把它删掉。

**怎么避免**：
- 审计/复现脚本的 PASS 含义必须有三态显式标注：
  - `[PASS] 缺陷复现：原代码里能找到`（修前期望）
  - `[PASS] 缺陷修复：当前代码已规避`（修后期望）
  - `[FAIL] 复现失败`（可能是测试自身 bug 或代码被改成了既有问题）
- 标题里必须包含"修复状态"标识，例如 `P0-1 引用编号错位（已修复）` vs `P0-1 引用编号错位（数据库里有历史产物）`。
- 脚本里**禁止**用布尔单一变量表示"通过"——必须用枚举 `RecheckResult.OK | RecheckResult.NOT_REPRODUCED | RecheckResult.NOT_FIXED`。
- 强规则：任何审计脚本的输出，**每条都必须能在 1 秒内回答两个问题**：「这条结论修没修？」与「这条结论测的是什么？」。

## 3. 端到端测试的硬/软断言取决于数据是否可控

**为什么这是错**：`test_artifact_body_numbers_cite_subset_of_reference_list` 写了硬断言 `assert not orphan`。第一次跑就 fail——因为数据库里有 pre-fix 时代的旧产物，错位是历史。**这不是 bug 回归**，是数据陈旧。

**我当时怎么栽**：写断言时只想到"它该 PASS"，没问"数据来源是什么、谁生成、什么时候生成"。等 CI 红屏时才会怀疑"是不是数据问题"——晚了一拍。

**怎么避免**：
- 端到端测试前**先列出 4 件事**：
  1. 数据从哪来？（外部 fixture / 仓库内 fixture / 数据库）
  2. 数据是只读还是会被测试修改？
  3. 数据生成的时间点与代码版本的关系？
  4. 测试要硬断言还是软断言（skip + 警告）？
- 决策规则：
  - 数据**随代码生成**（新版本会跑出新 fixture）→ 硬断言；
  - 数据**是历史产物**（不会自动重生成）→ 默认软断言 + 给出"如何清理历史再硬断言"的具体命令；
  - 数据**对外部依赖**（API、第三方库）→ 软断言 + 标记 `pytest.mark.flaky`。
- 软断言不能藏在 `try/except Exception: pass` 里——必须显式 `pytest.skip(reason=...)`，否则下次 CI 红屏时 debug 不到原因。
- 强规则：**所有"端到端断言数据库状态"的测试，必须在文件 docstring 里写一句 "前提条件：执行前需要先跑一次 X 才能保证断言生效"**。

## 4. 栈推断（inspect）必须 fallback 到 sys._getframe 且有界

**为什么这是错**：我用了 `import inspect; frame = inspect.currentframe(); for _ in range(8): frame = frame.f_back`。问题：
- `inspect.currentframe()` 在 PyPy、Stackless、MicroPython 上**不一定支持**；
- 走 8 层栈意味着深度耦合调用结构，下次重构时一旦多包一层函数，phase 推断就错位；
- 没 try/except（虽然加了，但 try 范围太大，失败时啥都得不到）。

**怎么避免**：
- 用 `sys._getframe(1).f_code.co_qualname`（O(1) 拿上一帧，不依赖 inspect）。
- **最多走 3 层**，且每次 walk 都记录走过的类名，最后拼成 `phase="ClassName.methodName"`。
- 推断出来的值必须**带"inferred:"前缀**——例如 `phase="inferred:WriterAgent.write_review"`，让下游读者一眼知道这不是真实 trace 上下文，避免误用。
- 必须有 try/except，且 except 内**至少记 logger.warning**，不要吞。
- 强规则：**栈推断是 fallback，正常路径必须由 trace/span 提供**。任何"找不到 trace 就猜"的设计，文档里必须明确"猜错不影响主流程"——并在主流程里校验一下猜出来的 phase 是不是 LLM 调用点附近合理的方法名。

## 5. 文档数字与代码基线必须同源更新

**为什么这是错**：commit 信息里我写了 `1384 + 18 = 1402 passed, 1 skipped`。README 顶部还停留在 1384。下次任何人做审计，会发现"文档说的测试数与实测不符"——又一个 P3 级"文档与代码不符"缺陷。

**怎么避免**：
- **改测试基线** → 必须同步改 README.md（"1384 passed"）、docs/HIGHLIGHTS.md、docs/INTERVIEW-FAQ.md 里的"X 个测试全绿"措辞。
- 同源更新清单：
  - `README.md` 的徽章与硬编码数字
  - `docs/_FACT-PACK.md` 的 §0 验收状态表
  - `docs/INTERVIEW-PROJECT.md` 与 `docs/RESUME.md` 的"硬结论"段
- 强规则：**任何"CI 通过"的数字出现在 commit 信息里，必须在同一个 commit 里改完所有引用它的文档**。否则下次 `grep -rn "1384" docs/` 一定会找到 stale 引用。

## 6. 修代码时不要让它跨过架构白名单上限

**为什么这是错**：`agent/graph.py` 在白名单里（674 行）。我加了 `from ..platform.observability import TraceRecorder, create_trace` + finally 块（5 行新增 + 5 行删除）后，行数逼近 685——距离白名单上限（项目自有约束）只差十几行。下次再有大改动就会 `check_arch.py` 红屏。

**怎么避免**：
- 改大文件前先看 `scripts/check_arch.py` 输出的"距离上限还有多少行"。
- 如果添加内容 > 10 行，**优先抽到子模块**（例如 `agent/_tracing.py`）。
- 改完跑一次 `python scripts/check_arch.py`，确认白名单尺寸未越线。
- 强规则：白名单里的文件，**任何 PR 不应让它增长 > 5 行**。如果一定要，超过部分必须立刻拆出去。

## 7. 类型契约写在签名，不写在导言

**为什么这是错**：`build_references(self, entries: Sequence[tuple[int, Paper]], ...)` —— 类型注解是对的，但 `formatter.py` 模块 docstring 与 `AgentState.citation_map` 的 docstring 都没说"`entries` 第一个元素是引用编号、第二个元素是 Paper"。新人看代码会以为是某种通用 tuple，调用时随手写 `(paper.id, paper)`——顺序反了就静默错位。

**怎么避免**：
- 形参 `Sequence[tuple[int, Paper]]` 不够，**必须配 alias**：

  ```python
  CitationEntry = tuple[int, Paper]  # 第一个元素是引用编号（与正文 [n] 对应），第二个是 Paper
  ```

- 函数 docstring 第一行必须**显式给示例**：`entries: [(1, paper_A), (4, paper_B)] —— 1 和 4 是引用编号，不是 paper.paper_id`。
- 强规则：**任何"看起来像通用 tuple"的形参，必须配 NewType 或 alias + docstring 示例**。否则下一个维护者必踩。

## 8. 提交信息的 body 必须区分"实现"与"已知不在"

**为什么这是错**：我写了"1384 + 18 = 1402 passed, 1 skipped"，但**没在 commit body 里列出 4 个仍存在的 P0**。下游 review 看到 commit message 以为"全清"，实际还有 P0-3/5/6/7 待办。

**怎么避免**：
- commit body 模板：
  ```
  fix(audit): <一句话总结>
  
  <改了哪些，> / <没改哪些，why>
  
  Re-tested: <命令 + 数字>
  Regressions: <新增测试数>
  Still open: <清单 + 链接到 issue 或 TODO>
  ```
- 强规则：commit 信息里**禁止**写"all fixed / everything works"这类绝对化措辞。改了什么、没改什么、还需要什么，必须显式。

## 9. 修代码时同步更新"现状可复现"的脚本

**为什么这是错**：改了 `formatter.py` 让编号正确了，但 `verify_findings.py` 的 PASS 描述还是"缺陷存在"，看着像脚本 bug。**事实上脚本与代码现在一致——代码里已无 bug，脚本确认了"无 bug"**，只是描述没改。

**怎么避免**：
- 任何"修复 commit"必须同步更新所有"复现脚本"：
  - 把"PASS"描述改为"PASS（已修复）：修复方式 X，见 test_Y"；
  - 或者拆成两个脚本：`verify_finds.py`（反向，复现旧 bug）+ `verify_fixes.py`（正向，确认已修）；
  - **不要**让同一脚本既当复现工具又当修复验收。
- 强规则：**修复 PR + 复现脚本更新必须是同一个 commit**。不允许 commit A 修了 bug、commit B 才更新脚本。

## 10. 跑回归测试前，先把"会失败"的旧数据清理掉

**为什么这是错**：我没清数据库里的旧 artifact 就跑端到端断言，测试 fail。花了 10 分钟 debug 才发现是数据陈旧——而不是代码问题。

**怎么避免**：
- 修复涉及"改变输出格式"的代码（如本例的 reference list 编号），**必须**先考虑"仓库里残留的历史产物要不要清"。
- 如果保留历史产物：在测试里加 `pytest.skip` + 给出清理命令（已采用此方案）。
- 如果要清：在 commit 里**显式说明**清理了什么（如 `DELETE FROM artifacts WHERE created_at < '2026-09-27'`），并给回滚命令。
- 强规则：**涉及"数据契约变更"的修复，必须在 PR 描述里给出"数据迁移 / 清理"的具体步骤**。不允许默默依赖"下次重跑就对了"。

---

# 自检清单（每次修审计缺陷前过一遍）

按这个顺序走，至少能避掉 80% 的常见错：

- [ ] 1. 跑 `pytest --collect-only` 或 `python -c "import <module>"`，确认现状可收集。
- [ ] 2. 列出本次要改的所有文件 + 它们在架构白名单里的大小 + 改动行数估算。
- [ ] 3. 对每个改文件：策略是"包外层 + 不改缩进"还是"整块重缩进 + IDE 全选"？
- [ ] 4. 对端到端测试：数据来源是什么？硬断言还是软断言？skip 路径是否清楚？
- [ ] 5. 对 verify 类脚本：PASS 描述里是否区分了"复现"vs"修复后"？标题里是否有状态标记？
- [ ] 6. 对栈推断 / 黑科技代码：是否有 fallback？推断值是否带"inferred:"前缀？是否记了 logger？
- [ ] 7. 对白名单文件：改前/改后行数对比，距离上限还有多少？
- [ ] 8. commit 信息是否包含：实现 / 已知未改 / 测试基线数字 / 仍开放清单？
- [ ] 9. README/HIGHLIGHTS/FAQ 里"X 个测试"的引用是否同步改了？
- [ ] 10. 数据契约变更：要不要清旧数据？commit 描述里有没有迁移/回滚步骤？

# 反模式速查

| 反模式 | 替代 |
|---|---|
| 在闭合块外层加一行却只缩进一行 | 整块重缩进 / 把"加一行"放到不影响缩进的位置 |
| 把"复现脚本"和"修复验收脚本"混一起 | 拆成两个脚本，各自只表达一个语义 |
| 用 try/except Exception: pass 吞错 | 用 pytest.skip(reason=...) 显式跳过 |
| `inspect.currentframe()` 走 N 层栈 | `sys._getframe(1).f_code.co_qualname` 走 1~3 层 |
| `phase=method_name` 没前缀 | `phase="inferred:ClassName.method"` |
| 用 isinstance(x[0], tuple) 判断鸭子类型 | `Sequence[CitationEntry]` + alias |
| commit 写 "everything works" | commit body 列 Still open: 清单 |
| 修代码不同步改文档测试数 | 同一 commit 里改完所有 grep 引用 |
| 用 deque 的窗口做"全局聚合"的真相 | 明示"明细窗口 ≠ 长期聚合"，长期聚合要走 DB |
| 改了输出格式不清旧数据 | commit 描述里给清理/回滚命令，或测试用 skip |

# 适用边界

**这套 skill 的适用前提**：
- 你要修的是**真实存在的缺陷**，不是凭空造的需求；
- 你有可执行的复现脚本（验证脚本 / 单元测试）；
- 缺陷涉及"静默失败"（产出错误但测试不红、用户看不到）——这是最值得审计的类别；
- 你打算用 verify 类脚本持续守住它。

**不适用**：
- 新功能开发（不是修缺陷，是造代码）；
- 性能优化（这类改动的回归脚本形态完全不同，应单独有 skill）；
- UI 重构（更多是设计决策，技术缺陷层面较少）。