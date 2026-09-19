"""领域层：纯领域模型与规则。

**这一层不依赖任何 IO 与框架**：不导入 httpx / fastapi / sqlite3，也不导入本包的
其他层。它只描述"这个业务里有哪些东西、它们之间有什么规则"：

* :mod:`medscholar.domain.models` —— ``Paper`` / ``SearchFilters`` 等值对象；
* :mod:`medscholar.domain.text` —— 中英文切分、摘要重建等纯文本算法；
* :mod:`medscholar.domain.citation` —— 6 种引用格式与参考文献表排版（纯函数）；
* :mod:`medscholar.domain.dedupe` —— 同一文献的判定与合并规则；
* :mod:`medscholar.domain.quality` —— 证据等级、研究设计识别等医学领域规则。

为什么值得单独一层：这些规则是**最需要被测试、也最容易跨项目复用**的部分，
把它们与 SQLite、HTTP、LLM 隔开之后，测试不需要任何夹具，改动也不会牵动 IO 代码。
"""

from __future__ import annotations

__all__: list[str] = []
