"""文本归一化与 CJK 分词（纯算法）

实现已迁至 :mod:`medscholar.domain.text`，本模块保留为兼容外壳。

为什么保留而不是删掉：``medscholar.textutil`` 是公开契约 —— CLI、HTTP 服务、MCP、
评测脚本、文档示例以及用户自己写的二次脚本都在 import 它。一次性改掉所有调用点
会把"低风险的代码搬迁"变成"高风险的大改造"，而只做重导出的外壳成本几乎为零。

外壳的完整性由 ``tests/test_compat_shims.py`` 锁定：它逐个断言
``getattr(外壳, 名字) is getattr(真实模块, 名字)``。

新代码请直接 import :mod:`medscholar.domain.text`。
"""

from __future__ import annotations

from medscholar.domain.text import *  # noqa: F401,F403
from medscholar.domain.text import __all__ as __all__  # noqa: F401

# 以下名字不在 __all__ 里，但确实被外部引用（测试与脚本在用）。
# 漏掉任何一个都会让 import 直接失败 —— 所以由脚本从 AST 生成，不靠人记。
from medscholar.domain.text import _CJK_RE  # noqa: F401
from medscholar.domain.text import _WS_RE  # noqa: F401
from medscholar.domain.text import _CTRL_RE  # noqa: F401
from medscholar.domain.text import _JATS_TAG_RE  # noqa: F401
from medscholar.domain.text import _NAME_PARTICLES  # noqa: F401
