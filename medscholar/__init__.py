"""MedScholar Agent — 面向医学研究者的本地化学术智能体。

本地优先的学术研究助手：

* 从免费开放学术 API（PubMed / Europe PMC / Semantic Scholar / OpenAlex / Crossref /
  DOAJ / CORE / arXiv / CNKI）检索文献元数据与开放获取全文；
* 全部数据落在单个本地 SQLite 文件中（FTS5 全文索引 + sqlite-vec 向量索引）；
* 通过「BM25 + 向量 + RRF 融合」构建可语义检索的个人知识库；
* 以 Plan → 审批 → Execute → Reflect → Synthesize 工作流辅助阅读、综述写作与引用格式化；
* 引用编号强一致 + 数字溯源 + claim-level 引用支持性校验（幻觉治理）；
* 可观测性（trace / token 与成本账本）、韧性（重试 / 熔断 / 限流）、
  RAG 提示注入防御、schema 迁移与提示词版本化（生产化）。

分层（依赖方向由 ``scripts/check_arch.py`` 在 CI 里强制）：

``platform``（横切，只依赖标准库）← ``domain``（零 IO）
← ``infrastructure``（db / api / llm / embedding / importers）← ``application``（agent / retrieval / …）
← ``interface``（server / cli / mcp）；``eval`` 独立成层，运行时不得依赖它。

``medscholar.models`` / ``config`` / ``cite`` / ``query`` / ``textutil`` / ``dedupe``
是兼容外壳（真实实现在 ``domain/`` 与 ``platform/`` 下），保留它们是因为旧 import
路径是公开契约；完整性由 ``tests/test_compat_shims.py`` 用身份断言锁定。

本包不下载、不传播任何受版权保护的付费全文；非开放获取文献仅保存元数据与出版商链接。
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
