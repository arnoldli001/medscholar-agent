"""本项目的迁移清单（新增迁移只改这一个文件）。

规则由 validate_migrations 强制：版本号从 1 起连续不重复；name 用小写短横线；
不可逆必须写 irreversible_reason；语句写成幂等形式（IF [NOT] EXISTS），
因为迁移可能在断电/杀进程后被重跑。
"""

from __future__ import annotations

import sqlite3

from .base import Migration, MigrationError, register, registered_migrations

__all__ = [
    "REQUIRED_BASELINE_TABLES",
    "M0001_BASELINE",
    "M0002_PAPER_SOURCE_ID_INDEX",
    "MIGRATIONS",
]


#: 基线对齐要求已存在的关键表（覆盖文献/向量/全文/会话运行/产物主干）。
#: 只查关键表：fulltext_attempts 等后加表在老库里本就可能缺失，
#: 全量检查会把可对齐的老库误判为结构损坏。
REQUIRED_BASELINE_TABLES: tuple[str, ...] = (
    "papers",
    "paper_embeddings",
    "papers_fts",
    "chat_sessions",
    "agent_runs",
    "artifacts",
)

#: M0002 索引名抽成常量：正向建/反向删共用同一字符串，不靠抄写保持一致。
SOURCE_ID_INDEX = "idx_papers_source_id"
_CREATE_SOURCE_ID_INDEX = f"CREATE INDEX IF NOT EXISTS {SOURCE_ID_INDEX} ON papers(source_id)"
_DROP_SOURCE_ID_INDEX = f"DROP INDEX IF EXISTS {SOURCE_ID_INDEX}"


def _existing_tables(conn: sqlite3.Connection) -> set[str]:
    """当前库里已有的表 / 虚拟表名。"""
    rows = conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'").fetchall()
    names: set[str] = set()
    for row in rows:
        # 不依赖 row_factory：调用方可能用默认工厂（tuple），也可能用 sqlite3.Row
        names.add(str(row[0]))
    return names


def _missing_baseline_tables(conn: sqlite3.Connection) -> list[str]:
    existing = _existing_tables(conn)
    return [name for name in REQUIRED_BASELINE_TABLES if name not in existing]


def baseline_align(conn: sqlite3.Connection) -> None:
    """M0001：识别既有结构并补登记迁移历史，不改库的一个字节。

    不能照抄 schema.sql：IF NOT EXISTS 重复执行不报错却会把与 SQL 不一致的
    老库误记为"已应用"。关键表齐全则登记基线；缺失（空库/残缺库）则报错，
    要求走全新 init，而不是猜着补结构。
    """
    missing = _missing_baseline_tables(conn)
    if missing:
        raise MigrationError(
            "基线缺失：库里找不到关键表 "
            + "、".join(missing)
            + "。请用全新初始化建立数据库（python -m medscholar.db.init），"
            "或先修复残缺的库文件再执行迁移 —— 迁移框架不会替你猜一个残缺库该怎么补。"
        )


def _create_source_id_index(conn: sqlite3.Connection) -> None:
    conn.execute(_CREATE_SOURCE_ID_INDEX)


def _drop_source_id_index(conn: sqlite3.Connection) -> None:
    conn.execute(_DROP_SOURCE_ID_INDEX)


M0001_BASELINE = Migration(
    version=1,
    name="baseline",
    description="基线：识别既有表结构并把 M0001 登记为已应用（不改库）",
    python=baseline_align,
    irreversible_reason=(
        "基线登记是迁移历史的起点，本身没有可撤销的结构改动："
        "删掉这一行只会让下一次 apply() 重新对齐到同一个状态，属于无意义操作"
    ),
)

M0002_PAPER_SOURCE_ID_INDEX = Migration(
    version=2,
    name="add-paper-source-id-index",
    description="给 papers.source_id 补索引，加速按外部数据源 ID 定位文献",
    python=_create_source_id_index,
    rollback=(_DROP_SOURCE_ID_INDEX,),
    # 建/删都走 python 钩子共用同一常量，避免索引名抄错导致回滚静默失败。
    statements=(),
)


def _register_builtin() -> None:
    """把内置迁移放进注册表。重复执行安全（例如模块被 reload）。"""
    known = {item.version for item in registered_migrations()}
    for item in (M0001_BASELINE, M0002_PAPER_SOURCE_ID_INDEX):
        if item.version in known:
            continue
        register(item)


_register_builtin()

#: 校验通过的迁移清单；导入本模块即完成校验，坏迁移在启动时就暴露。
MIGRATIONS: tuple[Migration, ...] = tuple(
    sorted(registered_migrations(), key=lambda item: item.version)
)
