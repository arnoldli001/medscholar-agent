"""横切平台层：配置、可观测性、韧性、安全、缓存、提示词注册表。

依赖规则由 ``scripts/check_arch.py`` 强制：本层被所有层使用，只依赖标准库
（及无业务含义的第三方库），绝不导入 ``medscholar`` 的其他层，否则依赖图成环。
"""

from __future__ import annotations

__all__: list[str] = []
