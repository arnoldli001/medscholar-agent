# 本地推理模型降档验证：`qwen3:8b` → `qwen3.5:4b`

> **日期**：2026-09-29　**改动者**：外部协作者（同机另一个项目 Moss-finagent-research）
> **状态**：`config.yaml` 已改（`model: qwen3.5:4b`），**已通过三层验证**
> **为什么要写这份**：这个改动**不是本项目的需求**，而是同机资源冲突驱动的。
> 所以它必须留下"改了什么、凭什么说没掉质量、**没证明什么**、怎么回退"的完整痕迹 ——
> 否则下一个人只会看到配置里换了个模型名，既不知道原因，也无法判断该不该回退。

## 一、为什么改

本机是 **8GB 显存（RTX 4060）**，而同一个 Ollama 上还跑着另一个项目的模型。
实测**显存驻留**：

| 模型 | 实测驻留 |
|---|---|
| `qwen3:8b-q4_K_M` | **5578 ~ 5900 MB** |
| `qwen3.5:4b` | **2983 ~ 3187 MB** |

用 8B 时两个项目会**互相把对方挤出显存**（实测换出+重载约 13s），
而且会把对方推到"本地不可用 → 静默改走云端付费"的分支上。
换成 4B 后与 `qwen2.5:1.5b`（1112MB）**可同时常驻、互不换出**。

## 二、验证一：本项目自带的写作基准（`scripts/bench_models.py`）

专测**中文医学写作**，判据是硬的（越界引用编号 / 未替换的占位符）。**跑 3 轮**：

| 指标 | `qwen3:8b`（原值） | `qwen3.5:4b`（现行） |
|---|---|---|
| 稳态生成 | 47.0 / 47.1 / 47.2 t/s | **70.5 / 71.1 / 70.9 t/s** |
| 纯生成耗时 | 2.9 / 3.1 / 2.8s | **1.9 / 1.8 / 2.2s** |
| 输出 tokens | 132 / 141 / 128 | 128 / 119 / 145 |
| **引用问题**（越界+占位符） | **0 / 0 / 0** | **0 / 0 / 0** |

```bat
.python\python.exe -X utf8 scripts\bench_models.py --models qwen3:8b qwen3.5:4b
```

## 三、验证二：本项目自带的端到端验收（`scripts/e2e.py`）

7 段检查 + **真实四库检索**（PubMed / Europe PMC / OpenAlex / Crossref）+
完整五阶段工作流（Plan → Execute → Reflect → Synthesize → Review）。
`--model` 覆盖模型，**不需要改 `config.yaml`**：

```bat
.python\python.exe -u scripts\e2e.py --model qwen3.5:4b --fresh
.python\python.exe -u scripts\e2e.py --model qwen3:8b   --fresh
```

| 判据 | `qwen3.5:4b` | `qwen3:8b` |
|---|---|---|
| 五阶段 | **全部走完** | 全部走完 |
| 墙钟 | **199.0s** | 311.8s |
| 产物字数 | **12,141** | 9,512 |
| 流式输出 | **7,299 字** | 6,250 字 |
| **正文引用校验** | **19 篇，全部可对应参考文献表** | 11 篇，全部合法 |
| 自审（Review） | 6.5/10，8 个问题 | **7.0/10，7 个问题** |
| 检索/入库/嵌入 | 197 篇去重 → 187 篇入库，向量 187/187 | 同级 |
| E2E 退出码 | FAIL ×1 | FAIL ×1（**同一项**） |

**两次唯一的 FAIL 完全相同，且与 LLM 无关**：第 1 段的
「语义可用：同主题跨语言 > 无关主题同语言」—— 那是**嵌入模型**
（`nomic-embed-text`）的项，`docs/EVALUATION.md` 里已有换 `bge-m3` 的建议。

## 四、验证三：claim-level 引用忠实度（`scripts/eval_faithfulness.py`）

**每个模型跑完端到端后、立刻在**同一份产物库**上核查**（`--fresh` 会删库，
所以必须先 e2e 再核查，不能两个模型都跑完再回头查）：

```bat
.python\python.exe -u scripts\e2e.py --model <M> --fresh
.python\python.exe -u scripts\eval_faithfulness.py --from-db --llm --max-claims 40
```

| 判定 | `qwen3.5:4b` | `qwen3:8b` |
|---|---|---|
| 带引用的论断 | 52 | 43 |
| 引用覆盖面 | **53%** | 36% |
| `weakly_supported`（靠数字/缩写等**语言无关信号**核对通过） | **22** | 16 |
| `unsupported` | 14 | 14 |
| **`contradicted`（方向被说反）** | **6** | **5** |
| `unverifiable`（跨语言且**无任何**共同信号 → 弃权） | **7** | **0** |

**怎么读这张表（重要）**：

1. **`unverifiable` 是"格式可核查性"的代理指标**：它表示"跨语言引用、且**连数字/缩写
   都对不上**"。8B 是 **0**、4B 是 **7** ⇒ **4B 写进的缩写/数字比 8B 少**，
   于是有 7 条无法用语言无关信号核对。**这是 4B 在这条链路上唯一可观测的劣势。**
2. **两个模型都有一批 `contradicted`（6 / 5）**，而中文综述引英文文献本来就是
   Tier 0 的**已知盲区**（见 `docs/EVALUATION.md`「已知盲区」）。
   这一档**不能直接归因于模型**。
3. **裁判模型就是被评的那个模型本身**（按 config 取）⇒ 4B 当裁判时的判断力
   本身也未经验证。所以**这一层分不出胜负**，它提供的是"观察"，不是"结论"。

## 五、结论

**4B 在本项目的任务上未掉质量**（工程与写作层面）。依据是一组**方向一致**的可观测值：
产物更长（+28%）、正文引用更多（19 vs 11）且校验都通过、墙钟 −36%、引用覆盖面 53% vs 36%；
写作基准 3 轮里引用问题都是 0。

**唯一需要留意的差异**：跨语言引用的**可核查性**略差（`unverifiable` 7 vs 0）。
若这个项目将来以"引用可自动核验"为硬指标，应优先**换嵌入模型 `bge-m3`**
并把 Tier 1 换成更强的裁判模型，而不是回退到 8B。

## 六、这份验证**不能**证明什么（不省略）

1. **每项都只跑 1 次**（端到端与忠实度都是 n=1）。自审 6.5 vs 7.0 那 0.5 分
   **不构成结论** —— 单次运行的方差就有这个量级。
2. **`e2e.py` 的判据全是工程正确性**（阶段是否走完 / 引用编号是否合法 /
   产物是否落库），**它不判内容对错**。
3. **忠实度的裁判不可靠**：裁判就是被评模型本身，且 Tier 0 在跨语言场景只能给
   弱证据结论（本项目的 `EVALUATION.md` 自己写了这一点）。
4. **没有对照"人写"或"更强模型写"的基线** —— 所以 6 条 `contradicted` 到底算多还是少，
   这份数据回答不了。

## 七、同批修正的一个错误（同一位改动者）

改动者最初还把这个文件的 `num_ctx` 从 **8192 压到了 4096**（想省显存）。
**那是错的**，依据在本仓库代码里：

```
medscholar/agent/writer.py:56
  num_ctx 是提示词与输出的共享预算（8GB 显存下仅 8192），材料塞满会挤掉正文
medscholar/agent/writer.py:82
  budget = num_ctx - want_output_tokens - estimate_tokens(system_prompt)
           - TOKEN_RESERVE
```

即 `num_ctx` **直接决定"能塞进上下文、进而被综述引用多少篇文献"**
（`config.yaml` 的 `writer_max_papers: 25` / `context_char_budget: 24000`
都按它算）；而且 `scripts/bench_models.py` 与 `bench_cloud.py` 都写死用 8192
做基准 —— 改了会让"基准里的表现"与"实际管线的表现"不一致。
**已回退 8192。** 回退后与 `config.example.yaml` 逐键对比：

| 键 | `config.example.yaml`（原值基准） | 现行 | |
|---|---|---|---|
| `provider` | `ollama` | `ollama` | 一致 |
| **`model`** | **`qwen3:8b`** | **`qwen3.5:4b`** | **★ 唯一改动** |
| `num_ctx` | `8192` | `8192` | 一致（已回退） |
| `think` | `false` | `false` | 一致 |
| `temperature` / `max_tokens` / `timeout` | `0.3` / `3000` / `600.0` | 同 | 一致 |

## 八、怎么回退

```yaml
# config.yaml 的 llm 段，改一行：
  model: qwen3:8b
```

代价：写作稳态速度 70.9 → 47.1 t/s（−34%），且 8B 驻留 5900MB，
与同机另一个项目抢 8GB 显存时更紧张（可能互相换出）。

> ⚠️ **本文件 `config.yaml` 不被 git 跟踪**（`.gitignore:9`）⇒
> **改它没有 git 回退兜底**。原值基准是同目录的 `config.example.yaml`；
> 下次改之前请先 `Copy-Item config.yaml config.yaml.bak`。

## 九、复现这份验证（顺序不能换）

```bat
:: ① 写作基准（3 轮，看 tok/s 与引用问题）
.python\python.exe -X utf8 scripts\bench_models.py --models qwen3:8b qwen3.5:4b

:: ② 端到端 → **立刻**忠实度核查（顺序不能换：--fresh 会删掉上一份产物库）
.python\python.exe -u scripts\e2e.py --model qwen3.5:4b --fresh
.python\python.exe -u scripts\eval_faithfulness.py --from-db --llm --max-claims 40

:: ③ 换模型重复 ②
.python\python.exe -u scripts\e2e.py --model qwen3:8b --fresh
.python\python.exe -u scripts\eval_faithfulness.py --from-db --llm --max-claims 40
```

⚠️ **两端到端之间必须各跟一次核查**。本项目实测踩过：先跑两个模型的端到端、
再回头核查 —— 4B 那份产物库已被 8B 的 `--fresh` 删除，**核查对象没了**。
