"""数据库 schema 迁移框架测试（纯同步、全部在 ``tmp_path`` 里的临时库上）。

覆盖六类真实风险，每类都有一条"实现写错了就会红"的用例：

1. 部分失败 —— 一个迁移里第 2 条 SQL 失败时不能留下半成品；
2. 幂等 —— ``apply()`` 重复调用不重复执行、不报错；
3. checksum 校验 —— 已应用迁移被事后改写必须报错，且有显式逃生口；
4. 备份 —— 用 SQLite backup API，写失败只告警不阻断；
5. 不可逆迁移 —— 必须显式声明原因，``rollback()`` 拒绝并给可读消息；
6. 连接状态 —— 迁移结束后 ``connection.in_transaction is False``。

所有库都是自建的连接（``tmp_path`` 下的文件 / ``:memory:``），
**不依赖 conftest 的全局单例**，也不碰 ``data/`` 与 8760 端口。
"""

from __future__ import annotations

import dataclasses
import logging
import sqlite3
from pathlib import Path

import pytest

from medscholar import db as db_package  # noqa: F401  （仅为确保包可导入）
from medscholar.db.migrate import (
    MigrationRunner,
    apply_migrations,
    main as migrate_main,
    migration_status,
    plan_migrations,
)
from medscholar.db.migrations import MIGRATIONS, Migration, MigrationError, migration
from medscholar.db.migrations import base as migrations_base
from medscholar.db.migrations.base import (
    register,
    registered_migrations,
    validate_migrations,
)

SCHEMA_SQL = Path(__file__).resolve().parent.parent / "medscholar" / "db" / "schema.sql"

#: 基线对齐 M0001 要求存在的关键表（与 registry.REQUIRED_BASELINE_TABLES 对应）
CORE_TABLES = (
    "papers",
    "paper_embeddings",
    "papers_fts",
    "chat_sessions",
    "agent_runs",
    "artifacts",
)


# --------------------------------------------------------------------------- 工具
def _require(condition: object, message: str) -> None:
    """带中文说明的断言。实现出 bug 时能一眼看出**哪一步**不对。"""
    assert condition, message


def make_db(path: Path, *, full_schema: bool = True) -> sqlite3.Connection:
    """建立一个临时库。

    ``full_schema=True`` 走项目真实的 ``schema.sql``（模拟"老库"，只有结构没有版本历史）；
    ``False`` 则是完全空的库。
    """
    conn = sqlite3.connect(str(path), isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    if full_schema:
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        # 项目里 paper_embeddings 由连接层按向量后端创建（vec0 或普通表），
        # 老库里一定存在，这里补一张最小结构即可满足基线识别。
        conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_embeddings ("
            " paper_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, embedding BLOB NOT NULL)"
        )
    conn.commit()
    return conn


def make_memory_db(*, full_schema: bool = True) -> sqlite3.Connection:
    conn = sqlite3.connect(":memory:", isolation_level=None)
    conn.row_factory = sqlite3.Row
    if full_schema:
        conn.executescript(SCHEMA_SQL.read_text(encoding="utf-8"))
        conn.execute(
            "CREATE TABLE IF NOT EXISTS paper_embeddings ("
            " paper_id INTEGER PRIMARY KEY, dim INTEGER NOT NULL, embedding BLOB NOT NULL)"
        )
    conn.commit()
    return conn


def insert_paper(conn: sqlite3.Connection, title: str = "测试文献") -> int:
    """直接插一行 papers（绕过仓储层：本测试只关心迁移本身）。"""
    cur = conn.execute(
        "INSERT INTO papers(title, source, source_id) VALUES (?, 'pubmed', 'S2-1')",
        (title,),
    )
    return int(cur.lastrowid or 0)


def index_names(conn: sqlite3.Connection) -> set[str]:
    return {
        str(row[0])
        for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'index'")
    }


def version_rows(conn: sqlite3.Connection) -> list[sqlite3.Row]:
    return list(
        conn.execute(
            "SELECT version, name, checksum, applied_at, duration_ms, success, error "
            "FROM schema_migrations ORDER BY version"
        )
    )


def pending_migration_item(version: int = 3) -> Migration:
    """一条「待应用的、安全的」迁移，用来制造「有迁移要跑」的场景。

    用显式 ``migrations=`` 注入，不进全局注册表，因此不会污染其它测试。
    全部语句都是 ``IF NOT EXISTS``：这些用例要验证的是"引擎按顺序执行"，
    不是"迁移失败"，所以任何有副作用的语句都不该出现在这里。
    """
    return Migration(
        version=version,
        name=f"test-create-note-index-{version}",
        description="测试用：建一张辅助表与一个索引",
        statements=(
            "CREATE TABLE IF NOT EXISTS test_marker (id INTEGER PRIMARY KEY)",
            "CREATE INDEX IF NOT EXISTS idx_test_marker_id ON test_marker(id)",
        ),
        rollback=(
            "DROP INDEX IF EXISTS idx_test_marker_id",
            "DROP TABLE IF EXISTS test_marker",
        ),
    )


#: 一定失败的 SQL（引用不存在的表）。用它而不是"语法看似有问题的中文列名"：
#: 后者在 SQLite 里有时是**合法**的（中文标识符是允许的），
#: 一旦 SQLite 接受了它，用例就会从"验证失败处理"变成静默通过。
FAILING_SQL = "SELECT * FROM 这张表不存在"


@pytest.fixture()
def db_path(tmp_path: Path, request: pytest.FixtureRequest) -> Path:
    """每个用例一个独立的库文件（用用例名命名，临时库互不干扰）。"""
    safe = "".join(ch for ch in request.node.name if ch.isalnum() or ch in "-_")
    return tmp_path / f"{safe}.db"


@pytest.fixture()
def conn(db_path: Path):
    conn = make_db(db_path, full_schema=True)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def empty_conn(tmp_path: Path):
    conn = make_db(tmp_path / "empty.db", full_schema=False)
    try:
        yield conn
    finally:
        conn.close()


@pytest.fixture()
def fresh_registry():
    """把全局注册表换成给定的迁移集合，测试结束后无条件还原。

    只给需要"整个注册表都是测试数据"的用例使用；其余用例用 ``migrations=``
    显式注入，互不影响。
    """
    original = dict(migrations_base._REGISTRY)

    def _swap(*items: Migration) -> None:
        migrations_base.reset_registry()
        for item in items:
            register(item)

    try:
        yield _swap
    finally:
        migrations_base.reset_registry()
        migrations_base._REGISTRY.update(original)

# =========================================================== 1. 基线对齐 / 注册表
class TestBaseline:
    def test_baseline_aligns_existing_library(self, conn: sqlite3.Connection) -> None:
        """已有完整表结构的老库：M0001 登记成功，且**一个字节都不改库**。"""
        before = index_names(conn)
        result = apply_migrations(conn, backup=False)
        _require(result["failed"] is None, f"基线对齐不应失败：{result['failed']}")
        _require(
            [item["version"] for item in result["applied"]] == [1, 2],
            f"应依次登记 M0001 与执行 M0002，实际 {result['applied']}",
        )
        versions = [int(row[0]) for row in version_rows(conn)]
        _require(versions == [1, 2], f"版本表应有 1、2 两行，实际 {versions}")
        _require(
            "idx_papers_source_id" in index_names(conn),
            "M0002 应真的建出 idx_papers_source_id",
        )
        _require(
            before - {"idx_papers_source_id"} <= index_names(conn),
            "迁移不应删除任何既有索引",
        )

    def test_baseline_keeps_existing_rows(self, conn: sqlite3.Connection) -> None:
        """迁移过程不能动用户的数据（400+ 篇文献的场景）。"""
        paper_id = insert_paper(conn, "用户的真实文献")
        apply_migrations(conn, backup=False)
        row = conn.execute(
            "SELECT COUNT(*) AS n FROM papers WHERE paper_id = ?", (paper_id,)
        ).fetchone()
        _require(int(row["n"]) == 1, "迁移后原有文献必须还在")

    def test_baseline_rejects_empty_database(self, empty_conn: sqlite3.Connection) -> None:
        """空库不能做基线对齐：必须报错并说明缺了哪些表。"""
        with pytest.raises(MigrationError) as excinfo:
            apply_migrations(empty_conn, backup=False)
        message = str(excinfo.value)
        _require("基线缺失" in message, f"消息应说明基线缺失，实际：{message}")
        _require("papers" in message, f"消息应列出缺失的表名，实际：{message}")

    def test_baseline_rejects_partially_broken_database(
        self, tmp_path: Path
    ) -> None:
        """只有 papers 的残缺库同样应被拒绝，而不是"猜着补"。"""
        conn = make_db(tmp_path / "broken.db", full_schema=False)
        try:
            conn.execute("CREATE TABLE papers (paper_id INTEGER PRIMARY KEY)")
            conn.commit()
            with pytest.raises(MigrationError) as excinfo:
                apply_migrations(conn, backup=False)
            message = str(excinfo.value)
            _require("agent_runs" in message, f"缺的表都要列出来，实际：{message}")
            _require("papers、" not in message, "已存在的表不应被报成缺失")
        finally:
            conn.close()

    def test_registry_has_baseline_and_second_migration(self) -> None:
        versions = [item.version for item in MIGRATIONS]
        _require(versions == [1, 2], f"registry 应发布 M0001/M0002，实际 {versions}")
        _require(MIGRATIONS[0].name == "baseline", "M0001 必须叫 baseline")
        _require(MIGRATIONS[0].reversible, "M0001 的「已应用」登记应可撤销")
        _require(
            all(item.description.strip() for item in MIGRATIONS),
            "每个迁移都要有中文说明",
        )

    def test_m0002_index_is_idempotent_and_reversible(self, conn: sqlite3.Connection) -> None:
        """M0002 连续执行两次不报错（幂等写法），且回滚语句能删掉它。"""
        m0002 = MIGRATIONS[1]
        m0002.apply(conn)
        m0002.apply(conn)  # CREATE INDEX IF NOT EXISTS → 第二次不应抛异常
        _require("idx_papers_source_id" in index_names(conn), "索引应已建立")
        m0002.revert(conn)
        _require("idx_papers_source_id" not in index_names(conn), "回滚应删除该索引")


# =========================================================== 2. 正常升级 / 幂等
class TestApply:
    def test_apply_on_fresh_registry_creates_table_and_index(self, db_path: Path) -> None:
        conn = make_db(db_path)
        try:
            runner = MigrationRunner(conn, migrations=[pending_migration_item()], backup=False)
            runner.ensure_table()
            result = runner.apply()
            _require(result["failed"] is None, f"不应失败：{result['failed']}")
            _require(len(result["applied"]) == 1, "应执行 1 个迁移")
            _require(
                "idx_test_marker_id" in index_names(conn), "迁移里的索引应已建立"
            )
            _require(int(result["current_version"]) == 3, "当前版本应更新为 3")
        finally:
            conn.close()

    def test_apply_is_idempotent(self, conn: sqlite3.Connection) -> None:
        """跑两次：第二次不执行任何迁移、不报错、版本表不变。"""
        first = apply_migrations(conn, backup=False)
        before = version_rows(conn)
        second = apply_migrations(conn, backup=False)
        _require(len(first["applied"]) == 2, "第一次应执行 2 个迁移")
        _require(second["applied"] == [], f"第二次不应执行任何迁移，实际 {second['applied']}")
        _require(second["failed"] is None, "第二次不应报错")
        _require(second["current_version"] == first["current_version"], "版本号应保持不变")
        _require(
            [tuple(row) for row in version_rows(conn)] == [tuple(row) for row in before],
            "第二次运行不得改写版本表（否则每次启动都会产生 diff）",
        )

    def test_status_pending_and_current_track_apply(self, db_path: Path) -> None:
        conn = make_db(db_path)
        try:
            state0 = migration_status(conn)
            _require(state0["exists"] is False, "未迁移前 exists 应为 False")
            _require(state0["current_version"] == 0, "未迁移前版本应为 0")
            _require(
                [item["version"] for item in state0["pending"]] == [1, 2],
                f"两个迁移都应待应用，实际 {state0['pending']}",
            )
            apply_migrations(conn, backup=False)
            state1 = migration_status(conn)
            _require(state1["exists"] is True, "迁移后 exists 应为 True")
            _require(state1["current_version"] == 2, "当前版本应为 2")
            _require(state1["pending"] == [], "迁移后不应有待应用项")
            _require(state1["dirty"] is False, "成功迁移后 dirty 应为 False")
            _require(
                [item["version"] for item in state1["applied"]] == [1, 2],
                "applied 应列出 1、2",
            )
            _require(
                all("applied_at" in item and "name" in item for item in state1["applied"]),
                "applied 项应含 name/applied_at",
            )
        finally:
            conn.close()

    def test_status_does_not_create_table(self, db_path: Path) -> None:
        """只读查询不得改库：status/plan 不应凭空建出版本表。"""
        conn = make_db(db_path)
        try:
            migration_status(conn)
            plan_migrations(conn)
            found = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            _require(found is None, "只读接口不应创建 schema_migrations")
        finally:
            conn.close()

    def test_plan_lists_pending_migrations(self, conn: sqlite3.Connection) -> None:
        lines = plan_migrations(conn)
        joined = "\n".join(lines)
        _require(len(lines) >= 3, f"计划应逐条列出迁移与说明，实际 {lines}")
        _require("M0001" in joined and "baseline" in joined, "计划应包含 M0001")
        _require("M0002" in joined and "add-paper-source-id-index" in joined, "计划应包含 M0002")
        _require("可回滚" in joined, "计划应说明迁移可回滚（回滚语句条数）")

    def test_plan_when_nothing_pending(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn, backup=False)
        lines = plan_migrations(conn)
        _require(len(lines) == 1, f"无待应用时应只有一句话，实际 {lines}")
        _require("无待应用" in lines[0], f"提示语应明确，实际 {lines[0]}")

    def test_dry_run_does_not_touch_database(self, db_path: Path) -> None:
        """dry_run=True：不执行、不建版本表、不写备份文件、不留事务。"""
        conn = make_db(db_path)
        try:
            runner = MigrationRunner(conn)
            result = runner.apply(dry_run=True)
            _require(result["dry_run"] is True, "结果应标记 dry_run")
            _require(result["applied"] == [], "dry-run 不应执行任何迁移")
            _require(len(result["plan"]) > 1, "dry-run 应给出计划")
            found = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type='table' AND name='schema_migrations'"
            ).fetchone()
            _require(found is None, "dry-run 不应创建 schema_migrations")
            _require(
                "idx_papers_source_id" not in index_names(conn),
                "dry-run 不应真的建索引",
            )
            _require(conn.in_transaction is False, "dry-run 后不得留在事务里")
            _require(
                list(db_path.parent.glob("*.bak")) == [],
                "dry-run 不应产生备份文件",
            )
        finally:
            conn.close()

    def test_target_limits_scope(self, db_path: Path) -> None:
        """target=1：只到 M0001，M0002 保持待应用。"""
        conn = make_db(db_path)
        try:
            runner = MigrationRunner(conn, backup=False)
            result = runner.apply(target=1)
            _require(
                [item["version"] for item in result["applied"]] == [1],
                f"只应执行 M0001，实际 {result['applied']}",
            )
            _require(result["current_version"] == 1, "当前版本应为 1")
            _require(
                "idx_papers_source_id" not in index_names(conn),
                "M0002 未执行，索引不应存在",
            )
            _require(
                [item["version"] for item in migration_status(conn)["pending"]] == [2],
                "M0002 应仍在待应用列表",
            )
        finally:
            conn.close()

    def test_unknown_target_raises(self, conn: sqlite3.Connection) -> None:
        runner = MigrationRunner(conn, backup=False)
        with pytest.raises(MigrationError) as excinfo:
            runner.apply(target=77)
        _require("77" in str(excinfo.value), f"消息应指出非法 target，实际 {excinfo.value}")
    def test_apply_leaves_connection_out_of_transaction(self, conn: sqlite3.Connection) -> None:
        """风险 6：迁移后连接不能留在事务里（否则后续写入会莫名进不了库）。"""
        apply_migrations(conn, backup=False)
        _require(conn.in_transaction is False, "迁移结束后 in_transaction 必须为 False")
        insert_paper(conn, "迁移之后写入的文献")
        _require(conn.in_transaction is False, "业务写入之后仍不应有悬挂事务")

    def test_prepare_commits_leftover_transaction(self, db_path: Path) -> None:
        """调用方忘了提交时：执行器要处理掉遗留事务而不是让 BEGIN 撞车。

        用真实库文件而不是内存库：遗留事务的提交行为在文件库上才会真正落盘。
        """
        conn = sqlite3.connect(str(db_path))
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")  # 默认模式下这里处于隐式事务中
            _require(conn.in_transaction is True, "前置条件：应处于事务中")
            runner = MigrationRunner(
                conn, migrations=[pending_migration_item()], backup=False
            )
            runner.ensure_table()
            result = runner.apply()
            _require(result["failed"] is None, f"不应失败：{result['failed']}")
            _require(conn.in_transaction is False, "迁移后不应留在事务里")
            _require(
                conn.execute("SELECT COUNT(*) FROM t").fetchone()[0] == 1,
                "调用方未提交的数据应被提交而不是丢弃",
            )
        finally:
            conn.close()


# ======================================================= 3. 部分失败与回滚
class TestPartialFailure:
    def test_second_statement_failure_rolls_back_first(
        self, conn: sqlite3.Connection
    ) -> None:
        """风险 1：一个迁移里第 2 条 SQL 失败 → 第 1 条的改动必须一起回滚。"""
        broken = Migration(
            version=3,
            name="test-broken-two-statements",
            description="测试用：第 1 条能成功、第 2 条必然失败",
            statements=(
                "CREATE TABLE IF NOT EXISTS test_rolled_back (id INTEGER)",
                FAILING_SQL,  # 第 2 条引用不存在的表，必然失败
            ),
            rollback=("DROP TABLE IF EXISTS test_rolled_back",),
        )
        runner = MigrationRunner(conn, migrations=[broken], backup=False)
        runner.ensure_table()
        result = runner.apply()

        _require(result["failed"] is not None, "应当报告失败")
        _require(result["failed"]["version"] == 3, "失败信息应带版本号")
        error = result["failed"]["error"]
        _require("M0003" in error, f"错误信息应带版本号，实际：{error}")
        _require("test-broken-two-statements" in error, "错误信息应带迁移名")
        _require(FAILING_SQL in error, f"错误信息应带失败的那条语句，实际：{error}")
        _require("OperationalError" in error, "错误信息应带原始异常类型")
        _require("已回滚" in error, "错误信息应说明回滚结果")
        _require(result["applied"] == [], "失败的迁移不应算作已应用")
        _require(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='test_rolled_back'"
            ).fetchone()
            is None,
            "第 1 条语句建的辅助表必须被回滚（这是「不能留半成品」的核心断言）",
        )
        _require(conn.in_transaction is False, "失败后连接不得留在事务里")

    def test_failed_record_is_written_with_error(self, conn: sqlite3.Connection) -> None:
        broken = Migration(
            version=4,
            name="test-broken-record",
            description="测试用：写入失败记录",
            statements=(FAILING_SQL,),
            rollback=("SELECT 1",),
        )
        runner = MigrationRunner(conn, migrations=[broken], backup=False)
        runner.ensure_table()
        runner.apply()

        rows = version_rows(conn)
        _require(len(rows) == 1, f"应留下 1 条失败记录，实际 {len(rows)}")
        row = rows[0]
        _require(int(row["version"]) == 4, "失败记录的版本号应为 4")
        _require(str(row["name"]) == "test-broken-record", "应记录迁移名")
        _require(int(row["success"]) == 0, "success 必须为 0")
        _require("不存在" in str(row["error"]), f"应记录错误信息，实际 {row['error']!r}")
        _require(str(row["checksum"]) == "", "失败记录不应带指纹（它没被应用过）")
        _require(str(row["applied_at"]).startswith("20"), "应写入 UTC ISO8601 时间")
        _require(float(row["duration_ms"]) >= 0.0, "应记录耗时")

        state = runner.status()
        _require(state["dirty"] is True, "存在失败记录时 dirty 应为 True")
        _require(
            [item["version"] for item in state["failed"]] == [4],
            "status 应列出失败版本",
        )
        _require(
            [item.version for item in runner.pending()] == [4],
            "失败的迁移应仍然待应用（否则永远不会被修好）",
        )
        _require(
            [item["version"] for item in state["applied"]] == [],
            "失败的迁移不得出现在 applied 里",
        )

    def test_later_migrations_not_attempted_after_failure(
        self, conn: sqlite3.Connection
    ) -> None:
        """失败后立即停止：后续迁移不再执行，避免把「部分升级」扩大。"""
        broken = Migration(
            version=5,
            name="test-broken-stops-chain",
            description="测试用：失败即中断",
            statements=(FAILING_SQL,),
            rollback=("SELECT 1",),
        )
        after = pending_migration_item(version=6)
        runner = MigrationRunner(conn, migrations=[broken, after], backup=False)
        runner.ensure_table()
        result = runner.apply()
        _require(result["failed"]["version"] == 5, "应报告版本 5 失败")
        _require(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='test_marker'"
            ).fetchone()
            is None,
            "失败之后的迁移不应被执行",
        )
        _require(
            [item["version"] for item in result["pending"]] == [5, 6],
            f"失败与未执行的迁移都应留在 pending（可重试），实际 {result['pending']}",
        )

    def test_failure_message_from_apply_or_raise(self, conn: sqlite3.Connection) -> None:
        broken = Migration(
            version=7,
            name="test-broken-or-raise",
            description="测试用：apply_or_raise 抛 MigrationError",
            statements=("ALTER TABLE papers ADD COLUMN",),  # 语句不完整，必然失败
            rollback=("ALTER TABLE papers DROP COLUMN x",),
        )
        runner = MigrationRunner(conn, migrations=[broken], backup=False)
        runner.ensure_table()
        with pytest.raises(MigrationError) as excinfo:
            runner.apply_or_raise()
        message = str(excinfo.value)
        for token in ("M0007", "test-broken-or-raise", "ALTER TABLE", "已回滚", "原始异常"):
            _require(token in message, f"异常消息应包含 {token}，实际：{message}")

    def test_apply_migrations_helper_raises_on_failure(self, conn: sqlite3.Connection) -> None:
        """模块级 apply_migrations 失败时必须抛异常，而不是只体现在返回值里。"""
        broken = Migration(
            version=8,
            name="test-helper-raises",
            description="测试用：便捷函数要抛错",
            statements=(FAILING_SQL,),
            rollback=("SELECT 1",),
        )
        with pytest.raises(MigrationError) as excinfo:
            apply_migrations(conn, backup=False, migrations=[broken])
        _require("test-helper-raises" in str(excinfo.value), "异常应指明是哪个迁移")

    def test_retry_after_fixing_a_failed_migration(self, conn: sqlite3.Connection) -> None:
        """失败记录不能阻止修好之后重跑（版本表里 success 0 → 1）。"""
        broken = Migration(
            version=9,
            name="test-fix-and-retry",
            description="测试用：先失败后修好",
            statements=(FAILING_SQL,),
            rollback=("SELECT 1",),
        )
        runner = MigrationRunner(conn, migrations=[broken], backup=False)
        runner.ensure_table()
        runner.apply()
        _require(migration_status(conn)["dirty"] is True, "第一次应留下失败记录")

        fixed = dataclasses.replace(
            broken, statements=("CREATE TABLE IF NOT EXISTS test_retry (id INTEGER)",)
        )
        second = MigrationRunner(conn, migrations=[fixed], backup=False)
        result = second.apply()
        _require(result["failed"] is None, f"修好后不应失败：{result['failed']}")
        _require(len(result["applied"]) == 1, "修好的迁移应被执行")
        rows = version_rows(conn)
        _require(len(rows) == 1, f"同一版本只应有一行记录，实际 {len(rows)}")
        _require(int(rows[0]["success"]) == 1, "重跑成功后 success 应为 1")
        _require(str(rows[0]["error"]) == "", "成功后错误信息应被清空")
        _require(str(rows[0]["checksum"]) != "", "成功后应写入指纹")
        _require(migration_status(conn)["dirty"] is False, "成功后 dirty 应为 False")


class TestRollback:
    def test_rollback_undoes_migration_and_history(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn, backup=False)
        _require("idx_papers_source_id" in index_names(conn), "前置条件：索引已建立")
        runner = MigrationRunner(conn, backup=False)
        result = runner.rollback(steps=1)
        _require(
            [item["version"] for item in result["rolled_back"]] == [2],
            f"应回滚 M0002，实际 {result['rolled_back']}",
        )
        _require(result["current_version"] == 1, "回滚后当前版本应为 1")
        _require(
            "idx_papers_source_id" not in index_names(conn), "回滚必须真的删掉索引"
        )
        _require(
            [int(row["version"]) for row in version_rows(conn)] == [1],
            "回滚后版本表里不应再有 2",
        )
        _require(conn.in_transaction is False, "回滚后不得留在事务里")

    def test_rollback_irreversible_is_refused_with_readable_message(
        self, conn: sqlite3.Connection
    ) -> None:
        """风险 5：不可逆迁移回滚必须拒绝，并说明作者声明的原因。"""
        irreversible = Migration(
            version=10,
            name="test-drop-legacy-column",
            description="测试用：删除历史列（不可逆）",
            statements=("CREATE TABLE IF NOT EXISTS test_dropped (x INTEGER)",),
            irreversible_reason="列被删除后原数据无法恢复，只能靠迁移前备份还原",
        )
        runner = MigrationRunner(conn, migrations=[irreversible], backup=False)
        runner.ensure_table()
        _require(runner.apply()["failed"] is None, "前置条件：该迁移应能正常应用")
        with pytest.raises(MigrationError) as excinfo:
            runner.rollback(steps=1)
        message = str(excinfo.value)
        _require("不可逆" in message, f"消息应说明不可逆，实际：{message}")
        _require("原数据无法恢复" in message, f"应带作者声明的原因，实际：{message}")
        _require("M0010" in message, "应带版本号")
        _require(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='test_dropped'"
            ).fetchone()
            is not None,
            "被拒绝的回滚不得留下任何改动",
        )
        _require(
            migration_status(conn)["current_version"] == 10,
            "被拒绝的回滚不得改动版本表",
        )

    def test_rollback_with_nothing_applied(self, conn: sqlite3.Connection) -> None:
        runner = MigrationRunner(conn, backup=False)
        runner.ensure_table()
        result = runner.rollback(steps=1)
        _require(result["rolled_back"] == [], "没有已应用迁移时应什么都不做")
        _require("没有已应用的迁移" in result["reason"], f"应说明原因：{result['reason']}")

    def test_rollback_steps_zero_is_rejected(self, conn: sqlite3.Connection) -> None:
        apply_migrations(conn, backup=False)
        runner = MigrationRunner(conn, backup=False)
        with pytest.raises(MigrationError) as excinfo:
            runner.rollback(steps=0)
        _require("steps" in str(excinfo.value), "消息应说明 steps 非法")

    def test_rollback_failure_keeps_database_unchanged(
        self, conn: sqlite3.Connection
    ) -> None:
        """回滚语句本身出错时：整体回滚，不能让库停在一半。"""
        bad_revert = Migration(
            version=11,
            name="test-bad-revert",
            description="测试用：回滚语句必然失败",
            statements=("CREATE TABLE IF NOT EXISTS test_bad_revert (id INTEGER)",),
            rollback=(
                "DROP TABLE IF EXISTS test_bad_revert",
                "DROP TABLE 不存在的表",
            ),
        )
        runner = MigrationRunner(conn, migrations=[bad_revert], backup=False)
        runner.ensure_table()
        _require(runner.apply()["failed"] is None, "前置条件：正向应成功")
        with pytest.raises(MigrationError) as excinfo:
            runner.rollback(steps=1)
        _require("回滚" in str(excinfo.value), f"消息应说明回滚失败：{excinfo.value}")
        _require(
            conn.execute(
                "SELECT 1 FROM sqlite_master WHERE name='test_bad_revert'"
            ).fetchone()
            is not None,
            "回滚失败时辅助表应还在（第 1 条 DROP TABLE 被一起撤销）",
        )
        state = migration_status(conn)
        _require(state["current_version"] == 11, "回滚失败后版本表不应被改")
        _require(conn.in_transaction is False, "回滚失败后不得留在事务里")

    def test_rollback_steps_selects_most_recent(self, conn: sqlite3.Connection) -> None:
        """steps=2 时按版本号从大到小回滚（先回 M0002 再回 M0001）。"""
        apply_migrations(conn, backup=False)
        runner = MigrationRunner(conn, backup=False)
        result = runner.rollback(steps=2)
        _require(
            [item["version"] for item in result["rolled_back"]] == [2, 1],
            f"应从新到旧回滚，实际 {result['rolled_back']}",
        )
        _require(result["current_version"] == 0, "全部回滚后当前版本应为 0")
        _require(version_rows(conn) == [], "回滚后版本表应为空")


# ======================================================= 4. checksum 校验
class TestChecksum:
    def test_checksum_changes_when_statements_change(self) -> None:
        first = pending_migration_item(version=12)
        second = dataclasses.replace(
            first,
            statements=(
                "CREATE TABLE IF NOT EXISTS test_marker (id INTEGER PRIMARY KEY)",
                "CREATE INDEX IF NOT EXISTS idx_test_marker_id ON test_marker(id, 1)",
            ),
        )
        _require(
            first.checksum() != second.checksum(),
            "语句改了指纹必须跟着变，否则校验和形同虚设",
        )
        _require(
            first.checksum() == pending_migration_item(version=12).checksum(),
            "同样的定义两次算出的指纹必须一致",
        )
        _require(len(first.checksum()) == 16, "指纹应为 sha256 前 16 位")
        _require(
            dataclasses.replace(first, name="test-other-name").checksum() != first.checksum(),
            "version|name 参与指纹：改名同样必须被检出",
        )
        _require(
            dataclasses.replace(first, rollback=("SELECT 1",)).checksum()
            != first.checksum(),
            "回滚语句参与指纹：改回滚同样是改写历史",
        )

    def test_rewritten_applied_migration_is_detected(self, db_path: Path) -> None:
        """风险 3：已应用迁移被事后改写 → 下一次 apply 必须报错。"""
        conn = make_db(db_path)
        try:
            migration_item = pending_migration_item(version=13)
            first = MigrationRunner(conn, migrations=[migration_item], backup=False)
            _require(first.apply()["failed"] is None, "前置条件：首次应成功")

            rewritten = dataclasses.replace(
                migration_item,
                statements=(
                    "CREATE TABLE IF NOT EXISTS test_marker (id INTEGER PRIMARY KEY)",
                    "CREATE INDEX IF NOT EXISTS idx_test_marker_id ON test_marker(id, id)",
                ),
            )
            second = MigrationRunner(conn, migrations=[rewritten], backup=False)
            with pytest.raises(MigrationError) as excinfo:
                second.apply()
            message = str(excinfo.value)
            _require("历史被改写" in message, f"消息应指出历史被改写，实际：{message}")
            _require("M0013" in message, "消息应指出是哪个版本")
            _require(
                "allow_checksum_change" in message,
                f"消息应告诉开发者逃生口在哪，实际：{message}",
            )
            _require(
                conn.execute(
                    "SELECT COUNT(*) FROM test_marker"
                ).fetchone()[0]
                == 0,
                "报错时不得改动数据库（校验必须发生在动手之前）",
            )
            _require(conn.in_transaction is False, "报错后不得留在事务里")
        finally:
            conn.close()

    def test_allow_checksum_change_lets_it_through(self, db_path: Path) -> None:
        """逃生口：显式放行后可以继续，并把新指纹写回版本表。"""
        conn = make_db(db_path)
        try:
            item = pending_migration_item(version=14)
            runner = MigrationRunner(conn, migrations=[item], backup=False)
            _require(runner.apply()["failed"] is None, "前置条件：首次应成功")

            # 等价重写：语句换成与结构无关的查询（模拟"缩进/拆分写法变了"）
            rewritten = dataclasses.replace(item, statements=("SELECT 1",))
            other = MigrationRunner(conn, migrations=[rewritten], backup=False)
            with pytest.raises(MigrationError):
                other.apply()  # 未放行时必须报错
            result = other.apply(allow_checksum_change=True)
            _require(result["failed"] is None, f"放行后不应失败：{result['failed']}")
            stored = {
                int(row["version"]): str(row["checksum"]) for row in version_rows(conn)
            }
            _require(
                stored[14] == rewritten.checksum(),
                "放行后应把新指纹写回版本表（否则每次启动都会再报一次）",
            )
            _require(
                MigrationRunner(
                    conn, migrations=[rewritten], backup=False
                ).checksum_mismatches()
                == [],
                "放行之后不应再报告指纹不一致",
            )
        finally:
            conn.close()

    def test_rewritten_stored_checksum_is_detected(self, conn: sqlite3.Connection) -> None:
        """即使库里的指纹是被手工改过的（而不是代码被改），同样要报错。"""
        apply_migrations(conn, backup=False)
        conn.execute(
            "UPDATE schema_migrations SET checksum = 'deadbeefdeadbeef' WHERE version = 2"
        )
        conn.commit()
        runner = MigrationRunner(conn, backup=False)
        mismatches = runner.checksum_mismatches()
        _require(len(mismatches) == 1, f"应检出 1 处不一致，实际 {mismatches}")
        _require(mismatches[0]["stored"] == "deadbeefdeadbeef", "应给出库里的值")
        _require(int(mismatches[0]["version"]) == 2, "应指出是 M0002")
        _require(
            migration_status(conn)["checksum_mismatch"] != [],
            "status 也应呈现指纹异常（可观测性）",
        )

    def test_empty_stored_checksum_is_not_reported_as_mismatch(
        self, conn: sqlite3.Connection
    ) -> None:
        """早期记录/手工插入的行没有指纹：不能误报成「历史被改写」。"""
        apply_migrations(conn, backup=False)
        conn.execute("UPDATE schema_migrations SET checksum = '' WHERE version = 2")
        conn.commit()
        _require(
            MigrationRunner(conn, backup=False).checksum_mismatches() == [],
            "空指纹应视为无法比对，不能报错阻断升级",
        )

    def test_orphan_version_is_reported_not_fatal(self, conn: sqlite3.Connection) -> None:
        """库里登记了代码里已经没有的版本：只报告孤儿，不阻断。"""
        apply_migrations(conn, backup=False)
        conn.execute(
            "INSERT INTO schema_migrations(version, name, checksum, applied_at, duration_ms,"
            " success, error) VALUES (99, 'ghost', 'x', '2025-01-01T00:00:00+00:00', 0, 1, '')"
        )
        conn.commit()
        state = migration_status(conn)
        _require(state["orphan"] == [99], f"应报告孤儿记录，实际 {state['orphan']}")
        _require(state["checksum_mismatch"] == [], "孤儿不是指纹异常")
        _require(
            MigrationRunner(conn, backup=False).apply()["failed"] is None,
            "孤儿记录不应阻断后续升级",
        )

    def test_rollback_also_checks_checksum(self, db_path: Path) -> None:
        """回滚前同样校验：历史被改写时不能盲目回滚。"""
        conn = make_db(db_path)
        try:
            apply_migrations(conn, backup=False)
            rewritten = dataclasses.replace(
                MIGRATIONS[1],
                statements=(
                    "CREATE INDEX IF NOT EXISTS idx_papers_source_id"
                    " ON papers(source_id, title)",
                ),
            )
            runner = MigrationRunner(conn, migrations=[MIGRATIONS[0], rewritten], backup=False)
            with pytest.raises(MigrationError) as excinfo:
                runner.rollback(steps=1)
            _require("指纹" in str(excinfo.value), f"消息应说明指纹问题：{excinfo.value}")
            _require(
                "idx_papers_source_id" in index_names(conn),
                "被拒绝的回滚不得留下改动",
            )
        finally:
            conn.close()


# ======================================================= 5. 备份
class TestBackup:
    def test_apply_creates_backup_file(self, db_path: Path) -> None:
        """默认开启备份：在库同目录生成 <db>.pre-migration-<版本>.bak。"""
        conn = make_db(db_path)
        try:
            insert_paper(conn, "演示备份的文献")
            runner = MigrationRunner(conn)
            runner.ensure_table()
            result = runner.apply()
            _require(result["backup"] is not None, "应返回备份路径")
            backup = Path(result["backup"])
            _require(backup.exists(), f"备份文件应存在：{backup}")
            _require(
                backup.name == f"{db_path.name}.pre-migration-0.bak",
                f"备份命名应为 <db>.pre-migration-<版本>.bak，实际 {backup.name}",
            )
            _require(runner.backup_path == backup, "runner.backup_path 应记录实际备份路径")
            # 备份必须是**能打开的**库，而且包含备份时刻的数据
            copy = sqlite3.connect(str(backup))
            try:
                count = copy.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
                _require(int(count) == 1, "备份里应有备份时刻的那篇文献")
                _require(
                    copy.execute(
                        "SELECT 1 FROM sqlite_master WHERE name='idx_papers_source_id'"
                    ).fetchone()
                    is None,
                    "备份应反映迁移**之前**的结构（还没有新索引）",
                )
            finally:
                copy.close()
            _require(
                "idx_papers_source_id" in index_names(conn),
                "备份不影响真正的迁移执行",
            )
        finally:
            conn.close()

    def test_backup_uses_sqlite_api_and_copies_data(self, db_path: Path) -> None:
        """backup_to 用 SQLite 备份 API：WAL 里未 checkpoint 的数据也要在里面。"""
        conn = sqlite3.connect(str(db_path), isolation_level=None)
        try:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.execute("CREATE TABLE t (x INTEGER)")
            conn.execute("INSERT INTO t VALUES (1)")
            conn.commit()  # 此时数据可能仍在 -wal 文件里
            target = db_path.parent / "manual.bak"
            runner = MigrationRunner(conn, backup=False)
            written = runner.backup_to(target)
            _require(written == target, "应返回传入的备份路径")
            copy = sqlite3.connect(str(target))
            try:
                _require(
                    int(copy.execute("SELECT COUNT(*) FROM t").fetchone()[0]) == 1,
                    "备份必须包含已提交的数据（文件拷贝在 WAL 下会漏掉它）",
                )
            finally:
                copy.close()
        finally:
            conn.close()

    def test_backup_path_is_created_when_missing(self, tmp_path: Path) -> None:
        conn = sqlite3.connect(":memory:", isolation_level=None)
        try:
            conn.execute("CREATE TABLE t (x INTEGER)")
            runner = MigrationRunner(conn, backup=False)
            target = tmp_path / "nested" / "dir" / "copy.bak"
            runner.backup_to(target)
            _require(target.exists(), "缺失的父目录应被自动创建")
        finally:
            conn.close()

    def test_backup_failure_warns_but_does_not_block(
        self, db_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        """风险 4：备份写失败只告警、不阻断迁移，但必须留下「没有备份」的痕迹。"""
        conn = make_db(db_path)
        try:
            runner = MigrationRunner(conn)
            runner.ensure_table()

            def boom(_path: object) -> Path:
                raise OSError("模拟磁盘满")

            runner.backup_to = boom  # type: ignore[method-assign]
            with caplog.at_level(logging.WARNING, logger="medscholar.db.migrate"):
                result = runner.apply()
            _require(result["failed"] is None, "备份失败不应阻断迁移")
            _require(result["backup"] is None, "备份失败时不应谎报备份路径")
            _require(runner.backup_path is None, "没有备份就必须留空，不能假装有")
            _require(
                "idx_papers_source_id" in index_names(conn), "迁移本身应照常执行"
            )
            messages = "\n".join(record.getMessage() for record in caplog.records)
            _require(
                "备份失败" in messages and "没有备份" in messages,
                f"必须留下「没有备份就动手」的告警，实际日志：{messages!r}",
            )
        finally:
            conn.close()

    def test_memory_database_skips_backup_with_warning(
        self, caplog: pytest.LogCaptureFixture
    ) -> None:
        conn = make_memory_db()
        try:
            runner = MigrationRunner(conn)
            runner.ensure_table()
            with caplog.at_level(logging.WARNING, logger="medscholar.db.migrate"):
                result = runner.apply()
            _require(result["backup"] is None, "内存库没有可写盘的备份位置")
            _require(result["failed"] is None, "内存库迁移不应失败")
            _require(
                "备份失败" in "\n".join(record.getMessage() for record in caplog.records),
                "跳过备份同样要告警（否则没人知道这次没有备份）",
            )
        finally:
            conn.close()


# ======================================================= 6. 定义期校验（不可逆等）
class TestDefinitionGuards:
    def test_irreversible_without_reason_is_rejected(self) -> None:
        """风险 5：没有 rollback 又没写原因 → 定义期（校验时）直接报错。"""
        silent = Migration(
            version=20,
            name="test-silent-irreversible",
            description="测试用：既不可逆又不说原因",
            statements=("CREATE TABLE test_silent (x INTEGER)",),
        )
        _require(silent.reversible is False, "没有 rollback 时应判定为不可逆")
        with pytest.raises(MigrationError) as excinfo:
            validate_migrations([silent])
        message = str(excinfo.value)
        _require("irreversible_reason" in message, f"消息应指出缺哪个字段，实际：{message}")
        _require("回不去" in message, f"消息应说明为什么要显式声明，实际：{message}")

    def test_irreversible_with_reason_is_accepted(self) -> None:
        declared = Migration(
            version=21,
            name="test-declared-irreversible",
            description="测试用：显式声明不可逆",
            statements=("CREATE TABLE test_declared (x INTEGER)",),
            irreversible_reason="外键引用被改写后无法可靠恢复原约束",
        )
        ordered = validate_migrations([declared])
        _require(len(ordered) == 1, "显式声明原因后应通过校验")
        _require(ordered[0].reversible is False, "它仍然是不可逆的（只是被承认了）")

    def test_duplicate_versions_are_rejected(self) -> None:
        first = pending_migration_item(version=22)
        second = dataclasses.replace(first, name="test-another-name")
        with pytest.raises(MigrationError) as excinfo:
            validate_migrations([first, second])
        _require("重复" in str(excinfo.value), f"应报重复，实际 {excinfo.value}")

    def test_version_gaps_are_rejected(self) -> None:
        with pytest.raises(MigrationError) as excinfo:
            validate_migrations(
                [
                    pending_migration_item(version=23),
                    pending_migration_item(version=25),
                ]
            )
        _require("连续递增" in str(excinfo.value), f"应报跳号，实际 {excinfo.value}")

    def test_validate_sorts_by_version_without_gap_rule_in_play(self) -> None:
        """排序：``migrations=`` 传入乱序时执行顺序必须是版本号升序。"""
        first = Migration(
            version=40,
            name="test-sort-high",
            description="测试用：乱序输入里的高版本",
            statements=(FAILING_SQL,),
            rollback=("SELECT 1",),
        )
        second = Migration(
            version=41,
            name="test-sort-next",
            description="测试用：紧随其后的版本",
            statements=(FAILING_SQL,),
            rollback=("SELECT 1",),
        )
        ordered = validate_migrations([second, first])
        _require([item.version for item in ordered] == [40, 41], "输出必须按版本号升序")

    def test_register_rejects_non_integer_version(self) -> None:
        with pytest.raises(MigrationError):
            register(
                Migration(
                    version=0,
                    name="test-zero-version",
                    description="测试用：版本号必须 >= 1",
                    statements=("SELECT 1",),
                    rollback=("SELECT 1",),
                )
            )

    def test_bad_name_is_rejected(self) -> None:
        with pytest.raises(MigrationError) as excinfo:
            register(
                Migration(
                    version=26,
                    name="Bad Name",
                    description="测试用：非法名字",
                    statements=("SELECT 1",),
                    rollback=("SELECT 1",),
                )
            )
        _require("短横线" in str(excinfo.value), f"应说明命名规则，实际 {excinfo.value}")

    def test_missing_description_is_rejected(self) -> None:
        with pytest.raises(MigrationError):
            register(
                Migration(
                    version=27,
                    name="test-no-description",
                    description="   ",
                    statements=("SELECT 1",),
                    rollback=("SELECT 1",),
                )
            )

    def test_migration_is_immutable(self) -> None:
        item = pending_migration_item(version=28)
        with pytest.raises(dataclasses.FrozenInstanceError):
            item.version = 999  # type: ignore[misc]

    def test_statements_are_normalized_to_tuple(self) -> None:
        item = Migration(
            version=29,
            name="test-tuple-normalization",
            description="测试用：列表入参要归一化成 tuple",
            statements=["SELECT 1"],  # type: ignore[arg-type]
            rollback=["SELECT 1"],  # type: ignore[arg-type]
        )
        _require(isinstance(item.statements, tuple), "statements 应归一化为 tuple")
        _require(isinstance(item.rollback, tuple), "rollback 应归一化为 tuple")

    def test_checksum_ignores_trailing_whitespace(self) -> None:
        """只忽略行尾空白：缩进/换行的格式变化不该被当成"改写历史"。"""
        base = pending_migration_item(version=30)
        reformatted = dataclasses.replace(
            base, statements=(base.statements[0] + "   \n", base.statements[1])
        )
        _require(
            reformatted.checksum() == base.checksum(),
            "行尾空白差异不应改变指纹（否则跨平台编辑会全员报错）",
        )

    def test_migration_decorator_registers_and_returns_function(self, fresh_registry) -> None:
        """装饰器要能注册进注册表，同时返回原函数（模块级名字仍可直接调用）。"""
        calls: list[int] = []

        @migration(
            31,
            "test-decorator-registration",
            "测试用：装饰器注册",
            rollback=("SELECT 1",),
        )
        def m(conn: sqlite3.Connection) -> None:
            calls.append(1)

        _require(callable(m), "装饰器应返回原函数")
        registered = {item.version: item for item in registered_migrations()}
        _require(31 in registered, f"注册表里应有 31，实际 {sorted(registered)}")
        _require(registered[31].python is m, "注册的 Python 钩子应是同一个函数对象")
        m(None)  # type: ignore[arg-type]
        _require(calls == [1], "原函数应仍可直接调用")

    def test_registry_is_restored_after_previous_test(self) -> None:
        """上一个用例注册的 31 号必须已被清理（否则用例会互相污染）。"""
        versions = {item.version for item in registered_migrations()}
        _require(31 not in versions, f"上一个用例的注册不应泄漏，实际 {sorted(versions)}")
        _require({1, 2} <= versions, "内置迁移应始终在注册表里")


# ======================================================= 7. init.py 集成 / CLI
class TestInitIntegration:
    def test_init_database_registers_baseline(self, tmp_path: Path) -> None:
        """全新库：init_database 之后版本表里就有 M0001/M0002，且连接干净。"""
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        cfg = AppConfig(data_dir=str(tmp_path / "data"), offline=True)
        database = Database(tmp_path / "init.db", config=cfg)
        try:
            state = migration_status(database.conn)
            _require(state["current_version"] == 2, f"应登记到 M0002，实际 {state}")
            _require(state["exists"] is True, "init 之后版本表应存在")
            _require(state["dirty"] is False, "init 之后不应有失败记录")
            _require(
                database.conn.in_transaction is False,
                "init_database 之后连接不得留在事务里",
            )
            # 既有行为不受影响：表能建、meta 能写、重复打开幂等
            _require(
                database.query("SELECT COUNT(*) AS n FROM papers")[0]["n"] == 0,
                "papers 表应可查询",
            )
            _require(
                database.get_meta("schema_version") == "1.0.0",
                "schema_version 元信息应保持原样",
            )
        finally:
            database.close()

        reopened = Database(tmp_path / "init.db", config=cfg)
        try:
            _require(
                migration_status(reopened.conn)["current_version"] == 2,
                "再次打开同一库应保持 M0002，且不重复执行",
            )
        finally:
            reopened.close()

    def test_init_database_on_existing_library_keeps_data(self, tmp_path: Path) -> None:
        """老库（只有 schema.sql 结构 + 数据）被 init 打开后：数据在、基线登记上。"""
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database

        path = tmp_path / "legacy.db"
        legacy = make_db(path, full_schema=True)
        insert_paper(legacy, "老库里的文献")
        legacy.close()

        cfg = AppConfig(data_dir=str(tmp_path / "data"), offline=True)
        database = Database(path, config=cfg)
        try:
            _require(
                database.query("SELECT COUNT(*) AS n FROM papers")[0]["n"] == 1,
                "老库的文献不能在打开时丢失",
            )
            state = migration_status(database.conn)
            _require(state["current_version"] == 2, f"应对齐到 M0002，实际 {state}")
            _require(
                [item["version"] for item in state["applied"]] == [1, 2],
                "applied 应记录 1、2",
            )
        finally:
            database.close()


class TestCli:
    def test_cli_status_plan_apply(self, db_path: Path, capsys: pytest.CaptureFixture) -> None:
        conn = make_db(db_path)
        conn.close()

        code = migrate_main(["--db", str(db_path), "--status"])
        out = capsys.readouterr().out
        _require(code == 0, f"status 应返回 0，实际 {code}")
        _require("当前版本" in out and "M0000" in out, f"应打印当前版本，实际：{out}")

        code = migrate_main(["--db", str(db_path), "--plan"])
        out = capsys.readouterr().out
        _require(code == 0, "plan 应返回 0")
        _require("M0002" in out, f"计划里应有 M0002，实际：{out}")

        code = migrate_main(["--db", str(db_path), "--apply", "--no-backup", "-q"])
        out = capsys.readouterr().out
        _require(code == 0, f"apply 应返回 0，实际 {code}")
        _require("当前版本 M0002" in out, f"应打印升级后的版本，实际：{out}")
        check = sqlite3.connect(str(db_path))
        try:
            _require(
                check.execute(
                    "SELECT COUNT(*) FROM schema_migrations WHERE success = 1"
                ).fetchone()[0]
                == 2,
                "CLI apply 应真的写入 2 条已应用记录",
            )
        finally:
            check.close()

        code = migrate_main(["--db", str(db_path), "--rollback-steps", "1", "-q"])
        out = capsys.readouterr().out
        _require(code == 0, f"rollback 应返回 0，实际 {code}")
        _require("已回滚 M0002" in out, f"应打印回滚结果，实际：{out}")
        check = sqlite3.connect(str(db_path))
        try:
            _require(
                check.execute(
                    "SELECT name FROM sqlite_master WHERE name='idx_papers_source_id'"
                ).fetchone()
                is None,
                "回滚后索引应被删除",
            )
        finally:
            check.close()

    def test_cli_dry_run_keeps_database_untouched(
        self, db_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        conn = make_db(db_path)
        conn.close()
        code = migrate_main(["--db", str(db_path), "--apply", "--dry-run", "-q"])
        capsys.readouterr()
        _require(code == 0, "dry-run 应返回 0")
        check = sqlite3.connect(str(db_path))
        try:
            _require(
                check.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='schema_migrations'"
                ).fetchone()
                is None,
                "dry-run 不应建版本表",
            )
            _require(
                check.execute(
                    "SELECT 1 FROM sqlite_master WHERE name='idx_papers_source_id'"
                ).fetchone()
                is None,
                "dry-run 不应建索引",
            )
        finally:
            check.close()

    def test_cli_missing_database_returns_error(
        self, tmp_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        code = migrate_main(["--db", str(tmp_path / "nope.db"), "--status"])
        err = capsys.readouterr().err
        _require(code == 1, f"库不存在时应返回 1，实际 {code}")
        _require("不存在" in err, f"应说明库不存在，实际：{err}")

    def test_cli_checksum_mismatch_exits_nonzero(
        self, db_path: Path, capsys: pytest.CaptureFixture
    ) -> None:
        """CLI 上「历史被改写」必须是可见的失败（退出码 1），不能静默放过。"""
        conn = make_db(db_path)
        conn.close()
        migrate_main(["--db", str(db_path), "--apply", "--no-backup", "-q"])
        capsys.readouterr()

        raw = sqlite3.connect(str(db_path))
        try:
            raw.execute("UPDATE schema_migrations SET checksum='feedfacefeedface' WHERE version=2")
            raw.commit()
        finally:
            raw.close()

        code = migrate_main(["--db", str(db_path), "--apply", "--no-backup", "-q"])
        err = capsys.readouterr().err
        _require(code == 1, f"指纹不匹配时应返回 1，实际 {code}")
        _require("历史被改写" in err, f"错误信息应说明原因，实际：{err}")
