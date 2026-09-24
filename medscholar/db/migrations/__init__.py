"""schema 迁移框架，仅依赖标准库 sqlite3。

base 定义 Migration/注册装饰器与定义期校验；registry 是迁移清单（新增迁移只改它）；
执行器（版本表/事务/备份/dry-run）在 medscholar.db.migrate。不反向依赖上层模块。
"""

from __future__ import annotations

from .base import (
    Migration,
    MigrationError,
    get_migrations,
    migration,
    register,
    registered_migrations,
    reset_registry,
    validate_migrations,
)
from .registry import M0001_BASELINE, M0002_PAPER_SOURCE_ID_INDEX, MIGRATIONS

__all__ = [
    "Migration",
    "MigrationError",
    "migration",
    "register",
    "registered_migrations",
    "validate_migrations",
    "get_migrations",
    "reset_registry",
    "MIGRATIONS",
    "M0001_BASELINE",
    "M0002_PAPER_SOURCE_ID_INDEX",
]
