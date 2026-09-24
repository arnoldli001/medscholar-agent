"""数据库 schema 迁移框架：只依赖标准库 ``sqlite3``，不引入任何新依赖。

包结构（各自职责单一，避免变成一个 god module）：

* :mod:`medscholar.db.migrations.base` —— 迁移的抽象（``Migration`` /
  ``MigrationError`` / ``@migration`` 注册装饰器）与定义期校验；
* :mod:`medscholar.db.migrations.registry` —— 本项目的迁移清单（M0001 基线、
  M0002 …）。新增迁移只改这一个文件；
* :mod:`medscholar.db.migrate` —— 执行器（``MigrationRunner`` / ``apply_migrations``
  / ``migration_status`` / ``plan_migrations``）：版本表、事务、备份、dry-run。

分层上属于 ``infrastructure``：它只依赖标准库与 ``medscholar.db`` 自身，
不反向依赖任何上层模块（由 ``scripts/check_arch.py`` 强制）。

用法::

    from medscholar.db.migrate import apply_migrations, migration_status
    from medscholar.db.migrations import get_migrations

    apply_migrations(db.conn)              # 幂等：已应用的不会重跑
    migration_status(db.conn)              # {current_version, applied, pending, ...}
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
