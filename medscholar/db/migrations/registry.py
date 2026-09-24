"""本项目的迁移清单（新增迁移只改这一个文件）。

规则（会被 :func:`medscholar.db.migrations.base.validate_migrations` 强制）：

1. 版本号从 1 开始、连续递增、不得重复；
2. ``name`` 用小写短横线英文，会被写进 ``schema_migrations`` 与日志；
3. 不可逆的迁移必须写 ``irreversible_reason``，否则导入本模块时直接报错；
4. 语句尽量写成幂等形式（``IF NOT EXISTS`` / ``IF EXISTS``）——
   迁移会因为断电、进程被杀、用户手抖而在**任意时刻**被重跑，
   幂等写法是"重跑不会造成二次伤害"的最低成本保险。
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


#: 基线对齐时要求必须已经存在的关键表。
#:
#: 它们覆盖了这个库的四条主干：文献本体（papers）、向量索引（paper_embeddings）、
#: 全文检索（papers_fts）、会话与运行记录（chat_sessions / agent_runs）、产物（artifacts）。
#: 只检查关键表而不是全部表：``fulltext_attempts`` / ``subscriptions``
#: 这类后加的表在早期版本的老库里本来就可能没有，把它们算进基线会把
#: "可以做基线对齐"的老库误判成"结构损坏"，逼用户删库重建——那正是本框架
#: 要消灭的事。
REQUIRED_BASELINE_TABLES: tuple[str, ...] = (
    "papers",
    "paper_embeddings",
    "papers_fts",
    "chat_sessions",
    "agent_runs",
    "artifacts",
)

#: M0002 使用的索引名。抽成常量是为了让"正向建 / 反向删"用的是同一个字符串，
#: 不靠人工抄写保持一致。
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
    """M0001：识别既有结构并补登记迁移历史（一个字节都不改库）。

    不抄 ``schema.sql`` 当 M0001：老库里这些表已经存在，一份"建表 SQL"只会
    重复建表；而 ``schema.sql`` 是 ``CREATE TABLE IF NOT EXISTS``，重复建表不会报错。
    于是迁移显示"成功"，但库里的列、索引、触发器可能和这份 SQL 并不一致
    （正是老库的真实状态），历史反而被记错了。

    正确做法是识别：结构在 → 登记 M0001 为已应用，从这里开始版本化；
    结构不在（空库或残缺库）→ 明确报错，让用户走"全新 init"这条路，
    而不是让迁移框架去猜一个残缺库该怎么补。

    本函数不修改任何数据，在事务里执行也完全无害。
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
    # statements 保持为空、逻辑放在 python 钩子里，让建索引与删索引
    # 共用同一个常量：索引名抄错一次，回滚就会找不到要删的东西（静默失败）。
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

#: 校验通过后的迁移清单。导入本模块即完成校验：坏迁移在启动时就会暴露，
#: 不会等到用户点了「开始研究」才发现库升不上去。
MIGRATIONS: tuple[Migration, ...] = tuple(
    sorted(registered_migrations(), key=lambda item: item.version)
)
