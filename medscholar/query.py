"""检索式构造（把用户关键词翻译成各数据源的查询语法）。

实现已迁至 :mod:`medscholar.domain.query`，本模块保留为兼容外壳：``medscholar.query``
是公开契约（CLI、HTTP、MCP、评测脚本与用户脚本都在 import），重导出外壳成本近零；
完整性由 ``tests/test_compat_shims.py`` 逐个断言 ``getattr(外壳) is getattr(真实模块)`` 锁定。
新代码请直接 import :mod:`medscholar.domain.query`。
"""

from __future__ import annotations

from medscholar.domain.query import *  # noqa: F401,F403
from medscholar.domain.query import __all__ as __all__  # noqa: F401

# 不在 __all__ 里但被外部引用的名字；由脚本从 AST 生成（漏掉会直接 import 失败），不靠人记。
from medscholar.domain.query import _TOKEN_RE  # noqa: F401
from medscholar.domain.query import _OR_WORDS  # noqa: F401
from medscholar.domain.query import _NOT_WORDS  # noqa: F401
from medscholar.domain.query import _SEPARATORS  # noqa: F401
from medscholar.domain.query import _strip_quotes  # noqa: F401
from medscholar.domain.query import _tokenize  # noqa: F401
from medscholar.domain.query import _dedupe  # noqa: F401
from medscholar.domain.query import _quote_if_needed  # noqa: F401
