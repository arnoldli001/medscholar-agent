# 本地推理模型降档验证：`qwen3:8b` → `qwen3.5:4b`

> **日期**：2026-09-29　**改动者**：外部协作者（同机另一个项目 Moss-finagent-research）
> **状态**：`config.yaml` 已改（`model: qwen3.5:4b`），**已通过下面两项验证**
> **为什么要写这份**：这个改动**不是本项目的需求**，而是同机资源冲突驱动的。
> 所以它必须留下"改了什么、凭什么说没掉质量、没证明什么"的完整痕迹 ——
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

## 四、结论

**4B 在本项目的任务上未掉质量。** 依据是一组**方向一致**的可观测值：
产物更长（+28%）、正文引用更多（19 vs 11）且校验都通过、墙钟 −36%；
写作基准 3 轮里引用问题都是 0。

## 五、这份验证**不能**证明什么（不省略）

1. **端到端各只跑了 1 次**（n=1）。自审 6.5 vs 7.0 那 0.5 分**不构成结论**
   —— 单次运行的方差就有这个量级。
2. **`e2e.py` 的判据全是工程正确性**（阶段是否走完 / 引用编号是否合法 /
   产物是否落库），**它不判内容对错**。"结论有没有被说反"需要
   Tier 1 的 claim-level 核查。
3. **claim-level faithfulness 未跑**（`scripts/eval_faithfulness.py --from-db --llm`）。
   原因有两条：① 它按 artifact 逐个核查，而 `--fresh` 会删掉上一份产物库
   （4B 那份已被 8B 覆盖）；② Tier 1 会把综述连同**每一条论断**交给裁判模型，
   成本较高。
   **要补跑的正确姿势**：跑完一次 e2e 后**立刻**在同一 `MEDSCHOLAR_HOME` 上执行
   `--from-db --llm`，再换模型重跑。

## 六、同批修正的一个错误（同一位改动者）

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

## 七、怎么回退

```yaml
# config.yaml 的 llm 段，改一行：
  model: qwen3:8b
```

代价：写作稳态速度 70.9 → 47.1 t/s（−34%），且 8B 驻留 5900MB，
与同机另一个项目抢 8GB 显存时更紧张（可能互相换出）。

> ⚠️ **本文件 `config.yaml` 不被 git 跟踪**（`.gitignore:9`）⇒
> **改它没有 git 回退兜底**。原值基准是同目录的 `config.example.yaml`；
> 下次改之前请先 `Copy-Item config.yaml config.yaml.bak`。
