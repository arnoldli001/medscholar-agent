"""MedScholar Agent — 面向医学研究者的本地化学术智能体。

本地优先的学术研究助手：
  * 从免费开放学术 API（PubMed / Europe PMC / Semantic Scholar / OpenAlex / arXiv / CNKI）
    检索文献元数据与开放获取全文；
  * 全部数据落在单个本地 SQLite 文件中（FTS5 全文索引 + sqlite-vec 向量索引）；
  * 通过「BM25 + 向量 + RRF 融合」的混合检索构建可语义检索的个人知识库；
  * 以 Plan → 审批 → Execute → Reflect → Synthesize 工作流辅助阅读、综述写作与引用格式化。

本包不下载、不传播任何受版权保护的付费全文；非开放获取文献仅保存元数据与出版商链接。
"""

__version__ = "1.0.0"
__all__ = ["__version__"]
