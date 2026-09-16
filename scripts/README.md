# 脚本说明

所有脚本都用项目自带的便携解释器运行：

```bat
.python\python.exe -X utf8 scripts\<脚本名>.py
```

## 日常使用

| 脚本 | 用途 | 是否需要网络 |
|---|---|---|
| `check.py` | 静态自检：语法编译 + 模块导入 + 关键纯函数断言。**改完代码先跑这个** | 否 |
| `check_bat.py` | 校验 `.bat` 为纯 ASCII + CRLF（CI 与 pre-commit 都用它守这条约定） | 否 |
| `smoke_db.py` | 数据层冒烟：建库 → 入库 → BM25/向量/混合检索 → 统计 | 否 |
| `smoke_api.py` | 数据源联调：逐源真实请求并打印诊断（解析结构变了会立刻暴露） | 是 |
| `smoke_http.py` | HTTP 契约测试（71 项）：全部端点 + SSE + 人工审批往返 | 否 |
| `eval_retrieval.py` | **检索质量评测**：recall@k/nDCG/MRR 消融实验 + 阈值门禁。`--embed-provider hashing` 为 CI 确定性模式，`--from-library` 为真实评测 | 视模式而定 |
| `build_golden.py` | 从本地库生成**候选**评测集（人工筛选后使用）。三种模式偏差递减：`known-item` / `title-terms` / `llm-question` | LLM 模式需要 |
| `e2e.py` | 端到端：真实检索 → 嵌入 → 混合检索 → 完整 Agent 工作流 → 成稿 | 是 + 需 Ollama |
| `pack_share.py` | 打包分享包到 `dist/` | 否 |

> 评测的用法与方法论（含"这个评测不能证明什么"）见
> [`docs/EVALUATION.md`](../docs/EVALUATION.md)。

```bat
:: 推荐顺序
.python\python.exe -X utf8 scripts\check.py
.python\python.exe -X utf8 scripts\smoke_http.py
.python\python.exe -m pytest -q
.python\python.exe -X utf8 scripts\e2e.py --fresh
.python\python.exe -X utf8 scripts\pack_share.py
```

`e2e.py` 的常用参数：`--fresh`（清空数据重跑）、`--skip-run`（只跑到检索+嵌入）、
`--model qwen3.5:4b`（换更快的模型）、`--sources pubmed openalex`（限定数据源）。

## 环境搭建

| 脚本 | 用途 |
|---|---|
| `setup_portable_python.py` | 在工作区构建自包含 Python 运行时（`.python/`）。已内置，一般不需要重跑 |

## 故障排查（一次性诊断工具）

这些脚本是为定位具体问题写的，遇到同类问题可以直接复用：

| 脚本 | 排查什么 |
|---|---|
| `probe_endpoints.py` | 各 API 的**原始响应结构**（写/修解析器前先看这个，别猜） |
| `probe_fulltext.py` | 开放获取全文的 URL 形态（Europe PMC 的坑：`/{PMCID}/fullTextXML` 不带 source 段） |
| `probe_chinese.py` | 中文文献通路：CNKI 接口现状 + OpenAlex `language:zh` / PubMed `chinese[la]` 对比 |
| `probe_embed.py` | 嵌入进度与 Ollama 批量嵌入耗时（怀疑嵌入卡住时先跑这个） |
| `probe_hang.py` | 逐端点最小化复现，定位哪个接口会挂住 |

## 注意

* **`.bat` 文件必须保持「纯 ASCII + CRLF」。** 这是硬约束，不是风格偏好：
  cmd.exe 用控制台 OEM 代码页（中文 Windows 是 936/GBK）解析批处理文件，
  UTF-8 中文字节被误当 GBK 解码时，尾字节会吞掉后面的换行，导致下一行命令被
  截断执行（表现为满屏 `"dp0" 不是内部或外部命令`）。`chcp 65001` 也救不了。
  所以三个 `.bat` 只做「找解释器 → 转发参数 → 出错 pause」，
  **所有中文提示都写在 `bootstrap.py` 里**（Python 通过 `WriteConsoleW` 输出 Unicode，
  与控制台代码页无关）。改启动器时请保持这条规则。

* `smoke_http.py` **必须打真实 uvicorn**，不能用 `httpx.ASGITransport`——
  后者会把整个响应体缓冲完再返回，而 SSE 是无限流，在进程内测必然死锁。
* `e2e.py` 会写入 `data/e2e/`，与你的正式知识库（`data/medscholar.db`）互不干扰。
* 打包脚本会自动排除 `data/`、`.cache/`、`.venv/`、`config.yaml`、`.env`
  以及 pytest/ruff 等**开发依赖**（ruff.exe 单个就有 25 MB）。
