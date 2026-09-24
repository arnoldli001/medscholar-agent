"""迁移执行器：版本表、事务、备份、dry-run、状态查询。

    from medscholar.db.migrate import apply_migrations, migration_status

    apply_migrations(db.conn)        # 幂等；默认先做一份 SQLite 备份
    migration_status(db.conn)        # 当前版本 / 已应用 / 待应用 / 是否存在失败记录

命令行（不需要起 8760 服务，直接对库文件操作）::

    python -m medscholar.db.migrate --db data/medscholar.db --status
    python -m medscholar.db.migrate --db data/medscholar.db --plan          # dry-run 计划
    python -m medscholar.db.migrate --db data/medscholar.db --apply
    python -m medscholar.db.migrate --db data/medscholar.db --rollback-steps 1

实现上三个不能错的地方：

1. 不能用 ``executescript``：它在执行脚本前会先隐式提交当前事务，
   "一个迁移一个事务"的保证会被撕开。这里一律用 ``conn.execute("BEGIN")``
   + 逐条 ``execute`` + 显式 ``COMMIT`` / ``ROLLBACK``。
2. 连接不能留在事务里：执行器在动手前检查并处理"调用方忘了提交"的情况，
   结束前保证 ``conn.in_transaction is False``。
3. 备份不是文件拷贝：WAL 模式下 ``.db`` 文件里可能缺少尚未 checkpoint 的页面，
   直接 ``shutil.copy`` 会得到打不开的备份。统一用 ``sqlite3.Connection.backup()``。

校验和必须有：``schema_migrations`` 里保存 ``version|name`` 与语句摘要的指纹。
已发布的迁移被人事后改了语句，不同人机器上的库结构会静默分叉，CI 全绿、代码一致，
但线上库少了那一列/那个索引。``apply()`` 在动手之前先做比对，默认直接报错。

``MigrationHistory`` 只管版本表（读记录、算状态、算指纹差异），
``MigrationRunner`` 继承它并负责执行（事务、备份、正反向）。
"""

from __future__ import annotations

import logging
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

from .migrations import Migration, MigrationError, get_migrations, validate_migrations
from .migrations.base import perf_counter_ms

logger = logging.getLogger(__name__)

__all__ = [
    "MigrationRunner",
    "MigrationHistory",
    "MigrationError",
    "apply_migrations",
    "migration_status",
    "plan_migrations",
    "main",
    "SCHEMA_MIGRATIONS_DDL",
    "TABLE",
    "BACKUP_SUFFIX_TEMPLATE",
]

#: 版本表名
TABLE = "schema_migrations"

#: 迁移前备份的文件名后缀模板：``<db>.pre-migration-<版本>.bak``
BACKUP_SUFFIX_TEMPLATE = ".pre-migration-{version}.bak"

#: 存进库里的错误信息上限（字符）。截断是为了不让一条超长 traceback
#: 把版本表撑成大文本；排查时完整的异常仍会随 MigrationError 抛出。
_ERROR_LIMIT = 2000

#: 版本表结构。
#:
#: ``version`` 直接做主键：重复登记同一版本在物理上就不可能发生，
#: 不依赖应用层记得去查重。
#:
#: 时间一律用带时区的 ISO8601（UTC）：用户会跨时区/夏令时使用这个应用，
#: 本地时间字符串在有夏令时的地区会出现两个 02:30，事故时间线直接对不上。
#: 注意与表内其他列（``datetime('now')`` 生成的 UTC "YYYY-MM-DD HH:MM:SS"）的区别：
#: 那些是 sqlite 默认格式，这里是迁移框架自己写入的、显式带时区的格式。
SCHEMA_MIGRATIONS_DDL = f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
    version     INTEGER PRIMARY KEY,
    name        TEXT    NOT NULL,
    checksum    TEXT    NOT NULL DEFAULT '',
    applied_at  TEXT    NOT NULL,
    duration_ms REAL    NOT NULL DEFAULT 0,
    success     INTEGER NOT NULL DEFAULT 1,
    error       TEXT    NOT NULL DEFAULT ''
)
"""


def _utc_now() -> str:
    """带时区的 ISO8601（UTC），例如 ``2025-01-31T09:15:00+00:00``。"""
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is_autocommit(conn: sqlite3.Connection) -> bool:
    """连接是否处于自动提交模式。

    ``isolation_level is None`` 时 sqlite3 不做隐式事务管理，``BEGIN`` 与
    ``COMMIT`` 完全由我们控制。
    默认模式下 sqlite3 会在 DML 前自动开事务，显式 ``BEGIN`` 会撞上
    "cannot start a transaction within a transaction"，
    动手前必须处理（见 :meth:`MigrationRunner._prepare_connection`）。
    """
    return conn.isolation_level is None


class _MigrationFailed(RuntimeError):
    """内部信号：某个迁移失败并已回滚（不对外暴露，apply() 会转成返回值）。"""


class MigrationHistory:
    """``schema_migrations`` 版本表的读写与解读（不做任何迁移执行）。

    独立成类的原因：状态查询必须能在任何连接上安全调用（包括还没建版本表的老库），
    每条语句都必须是只读或幂等的；执行器会 ``BEGIN``/``ROLLBACK``。
    分开之后，不会"只想看一眼状态，结果把库改了"。
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        migrations: Sequence[Migration] | None = None,
    ) -> None:
        self.conn = connection
        self.migrations: tuple[Migration, ...] = (
            validate_migrations(migrations) if migrations is not None else get_migrations()
        )

    # ------------------------------------------------------------ 版本表
    def ensure_table(self) -> None:
        """建 ``schema_migrations`` 表（已存在则什么都不做）。"""
        self.conn.execute(SCHEMA_MIGRATIONS_DDL)
        if _is_autocommit(self.conn):
            return
        # sqlite3 默认模式下 DDL 也会开事务；这里立刻提交，避免把连接留在事务里
        self.conn.commit()

    def table_exists(self) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = ? LIMIT 1",
            (TABLE,),
        ).fetchone()
        return row is not None

    def records(self) -> dict[int, sqlite3.Row]:
        """已登记的行：版本号 → Row（含失败行）。

        表不存在时返回空 dict：``status()`` / ``applied_versions()`` 是只读的，
        不能因为写错库路径就在库里凭空建出版本表。
        """
        if not self.table_exists():
            return {}
        rows = self.conn.execute(
            f"SELECT version, name, checksum, applied_at, duration_ms, success, error "
            f"FROM {TABLE}"
        ).fetchall()
        return {int(row[0]): row for row in rows}

    def applied_versions(self) -> list[int]:
        """成功应用的版本号（升序）。失败记录不算已应用，否则不会重试。"""
        return sorted(
            version for version, row in self.records().items() if int(row[5]) == 1
        )

    def pending(self) -> list[Migration]:
        """待应用的迁移（升序）。"""
        applied = set(self.applied_versions())
        return [item for item in self.migrations if item.version not in applied]

    def checksum_mismatches(self) -> list[dict[str, Any]]:
        """已应用迁移中指纹对不上的那些（空列表 = 历史干净）。"""
        found: list[dict[str, Any]] = []
        for version, row in sorted(self.records().items()):
            if int(row[5]) != 1:
                continue
            item = next((m for m in self.migrations if m.version == version), None)
            stored = str(row[2] or "")
            if item is None or not stored:
                # 代码里已经没有这个版本（孤儿记录）、或早期记录没写指纹：
                # 无法比对，不属于"被改写"，交给 status()['orphan'] 呈现。
                continue
            actual = item.checksum()
            if stored != actual:
                found.append(
                    {
                        "version": version,
                        "name": str(row[1]),
                        "expected_current": actual,
                        "stored": stored,
                        "current_name": item.name,
                    }
                )
        return found

    def status(self) -> dict[str, Any]:
        """当前状态快照。

        * ``exists``：版本表是否存在（False = 这个库从未被迁移框架接管）
        * ``current_version``：已成功应用的最大版本号（没有则为 0）
        * ``dirty``：版本表里是否留有失败行（上一次迁移中途炸了，
          库结构可能既不是旧版也不是新版）
        """
        records = self.records()
        applied = sorted(v for v, row in records.items() if int(row[5]) == 1)
        failed = sorted(v for v, row in records.items() if int(row[5]) != 1)
        by_version = {item.version: item for item in self.migrations}
        return {
            "exists": self.table_exists(),
            "current_version": applied[-1] if applied else 0,
            "applied": [
                {
                    "version": version,
                    "name": str(records[version][1]),
                    "applied_at": str(records[version][3]),
                    "duration_ms": float(records[version][4]),
                }
                for version in applied
            ],
            "pending": [
                {"version": item.version, "name": item.name, "description": item.description}
                for item in self.pending()
            ],
            "failed": [
                {
                    "version": version,
                    "name": str(records[version][1]),
                    "error": str(records[version][6]),
                }
                for version in failed
            ],
            "dirty": bool(failed),
            "checksum_mismatch": self.checksum_mismatches(),
            # 库里登记了、但当前代码里已经没有的版本（回滚过代码 / 删过迁移文件）
            "orphan": sorted(set(applied) - set(by_version)),
        }


class MigrationRunner(MigrationHistory):
    """按版本号顺序执行迁移，并维护 ``schema_migrations`` 版本表。

    参数 ``backup``：是否在 ``apply()`` 真正改动库之前做一份 SQLite 备份。
    默认开启；``init_database`` 里会显式关掉（那时通常是刚建好的空库，
    每次启动都复制一遍文件纯属浪费，而空库也没什么可丢的）。
    """

    def __init__(
        self,
        connection: sqlite3.Connection,
        *,
        migrations: Sequence[Migration] | None = None,
        backup: bool = True,
    ) -> None:
        super().__init__(connection, migrations=migrations)
        self.backup = backup
        #: 最近一次 apply() 的失败信息（供 init_database 这类"不中断启动"的调用方读取）
        self.last_failure: dict[str, Any] | None = None
        #: 最近一次 apply() 实际写出的备份文件路径（None = 没有备份就动手了）
        self.backup_path: Path | None = None

    # ------------------------------------------------------------ 计划
    def plan(self, *, target: int | None = None) -> list[str]:
        """人类可读的执行计划（dry-run 的输出）。

        待应用为空时返回 ``["无待应用的迁移（当前版本 Mxxxx）"]`` ——
        空列表会被误读成"命令没生效"，所以这里必须给一句明确的话。
        """
        todo = self._select(target)
        if not todo:
            return [f"无待应用的迁移（当前版本 M{self.status()['current_version']:04d}）"]
        lines = [f"计划执行 {len(todo)} 个迁移（--plan / --dry-run 的输出）："]
        for item in todo:
            lines.append(f"  M{item.version:04d} {item.name}：{item.description}")
            lines.append(
                f"      正向：{len(item.statements)} 条 SQL"
                + (" + Python 钩子" if item.python_revert is not None else "")
                + (
                    f"；可回滚（{len(item.rollback)} 条反向 SQL）"
                    if item.reversible
                    else "；不可逆"
                )
            )
            if not item.reversible:
                lines.append(f"      不可逆原因：{item.irreversible_reason}")
        return lines

    def _select(self, target: int | None) -> list[Migration]:
        todo = self.pending()
        if target is None:
            return todo
        known = {item.version for item in self.migrations}
        if target not in known:
            raise MigrationError(
                f"target={target} 不是已注册的迁移版本，可用版本：{sorted(known)}"
            )
        return [item for item in todo if item.version <= target]

    # ------------------------------------------------------------ 备份
    def backup_to(self, path: str | Path) -> Path:
        """用 SQLite 的在线备份 API 把当前库备份到 ``path``，返回该路径。

        WAL 模式下最新提交的数据可能还在 ``-wal`` 文件里，只拷 ``.db`` 会得到
        缺数据甚至打不开的"备份"。``Connection.backup()`` 由 SQLite 保证一致性快照，
        且不需要关连接。
        """
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        destination = sqlite3.connect(str(target))
        try:
            self.conn.backup(destination)
            destination.commit()
        finally:
            destination.close()
        return target

    def default_backup_path(self) -> Path | None:
        """``<db>.pre-migration-<版本>.bak``（与库同目录）。

        同目录是为了让备份和库一起被搬走；放系统临时目录会出现"库在、备份被清理掉"。
        带版本号而不是时间戳，方便一眼看出这份备份是升级到哪个版本之前的状态。

        内存库（``:memory:``）返回 None。
        """
        row = self.conn.execute("PRAGMA database_list").fetchone()
        raw = str(row[2]) if row and row[2] else ""
        if not raw:
            return None
        base = Path(raw)
        current = self.status()["current_version"]
        return base.parent / f"{base.name}{BACKUP_SUFFIX_TEMPLATE.format(version=current)}"

    def _make_backup(self) -> Path | None:
        """尽力备份。失败只告警不阻断，但必须留下痕迹。"""
        if not self.backup:
            return None
        try:
            destination = self.default_backup_path()
            if destination is None:
                raise OSError("内存数据库没有可写盘的备份位置")
            written = self.backup_to(destination)
        except (sqlite3.Error, OSError) as exc:
            # 磁盘满 / 只读介质 / 权限不足都可能发生。不阻断，但风险必须留下痕迹。
            logger.warning(
                "迁移前备份失败（%s），将在没有备份的情况下继续执行迁移。"
                "如果迁移中途失败，本次改动无法从备份还原，请手工恢复库文件。",
                exc,
            )
            return None
        self.backup_path = written
        logger.info("迁移前备份完成：%s", written)
        return written

    # ------------------------------------------------------------ 执行
    def apply(
        self,
        *,
        target: int | None = None,
        dry_run: bool = False,
        allow_checksum_change: bool = False,
    ) -> dict[str, Any]:
        """应用迁移。返回结果字典（见下方 Returns）。

        ``allow_checksum_change`` 是逃生口：只在确认改动是等价重写、且目标库
        已是新结构时启用。启用后执行器会记一条 warning 并把新指纹写回版本表。
        不能用来"让报错消失"：真的加了一列时，跑过旧语句的库不会因为放过校验
        而补上这一列。

        Returns:
            成功::

                {
                  "applied": [{"version", "name", "duration_ms"}, ...],
                  "plan": ["M0002 …", ...],
                  "pending": [{"version", "name"}, ...],
                  "current_version": int,
                  "dry_run": bool,
                  "failed": None,
                  "backup": "<备份文件路径或 None>",
                }

            失败（不抛异常，失败信息在返回值里）::

                {"applied": [...成功执行的...], "failed": {"version", "name", "error"},
                 "pending": [...尚未尝试的...], "current_version": int, "backup": ...}

        失败后立即停止后续迁移。
        """
        self.last_failure = None
        self.backup_path = None
        self._prepare_connection()
        if not dry_run:
            # dry-run 必须是**严格只读**的：连版本表都不能建（否则"演示一下"
            # 也会在用户库里留下痕迹），所以建表只发生在真正执行的分支里。
            self.ensure_table()

        plan = self._select(target)
        plan_lines = self.plan(target=target)

        if dry_run:
            return self._result(
                applied=[],
                plan_lines=plan_lines,
                current_version=self.status()["current_version"],
                dry_run=True,
            )

        # 校验和：先比对再动手。已应用的迁移被改写意味着"不同人的库结构已经不同"，
        # 这时候继续往下升级只会把分叉固化；必须在改库之前就拦下来。
        mismatches = self.checksum_mismatches()
        if mismatches and not allow_checksum_change:
            raise MigrationError(self._mismatch_message(mismatches))

        if not plan:
            self._refresh_checksums(mismatches)  # 放行模式下也要把指纹更新掉
            return self._result(
                applied=[],
                plan_lines=[f"无待应用的迁移（当前版本 M{self.status()['current_version']:04d}）"],
                current_version=self.status()["current_version"],
                dry_run=False,
            )

        backup_path = self._make_backup()

        applied: list[dict[str, Any]] = []
        failed: dict[str, Any] | None = None
        for item in plan:
            try:
                duration_ms = self._run_one(item)
            except _MigrationFailed as exc:
                failed = {"version": item.version, "name": item.name, "error": str(exc)}
                self.last_failure = failed
                logger.error("迁移 M%04d %s 失败并已回滚：%s", item.version, item.name, exc)
                break
            applied.append(
                {"version": item.version, "name": item.name, "duration_ms": duration_ms}
            )
            logger.info("迁移 M%04d %s 完成（%.1f ms）", item.version, item.name, duration_ms)

        if mismatches and allow_checksum_change:
            self._refresh_checksums(mismatches)
        applied_versions = self.applied_versions()
        result = self._result(
            applied=applied,
            plan_lines=plan_lines,
            current_version=applied_versions[-1] if applied_versions else 0,
            dry_run=False,
        )
        result["failed"] = failed
        result["backup"] = str(backup_path) if backup_path is not None else None
        return result

    def _result(
        self,
        *,
        applied: list[dict[str, Any]],
        plan_lines: list[str],
        current_version: int,
        dry_run: bool,
    ) -> dict[str, Any]:
        """统一的返回结构（dry-run 与真实执行共用，字段不会两边不一致）。"""
        return {
            "applied": applied,
            "plan": plan_lines,
            "pending": [{"version": m.version, "name": m.name} for m in self.pending()],
            "current_version": current_version,
            "dry_run": dry_run,
            "failed": None,
            "backup": str(self.backup_path) if self.backup_path is not None else None,
        }

    @staticmethod
    def _mismatch_message(mismatches: list[dict[str, Any]]) -> str:
        detail = "；".join(
            f"M{item['version']:04d} 库里登记 {item['stored']}，"
            f"当前代码算出 {item['expected_current']}（{item['current_name']}）"
            for item in mismatches
        )
        return (
            f"检测到已应用迁移的历史被改写：{detail}。"
            "这意味着不同机器上的库结构可能已经不一致（有人直接改了已发布的迁移）。"
            "请把迁移改回原样，或新增一个迁移来承载这次改动；"
            "确认改动是等价重写且目标库已是新结构时，才用 "
            "allow_checksum_change=True 显式放行。"
        )

    def apply_or_raise(self, **kwargs: Any) -> dict[str, Any]:
        """与 :meth:`apply` 相同，但失败时抛 :class:`MigrationError`。

        CLI 用它，所以命令行的退出码与错误信息是可用的（失败不静默）。
        """
        result = self.apply(**kwargs)
        if result["failed"] is not None:
            info = result["failed"]
            raise MigrationError(
                f"迁移 M{info['version']:04d} {info['name']} 失败：{info['error']}"
            )
        return result

    # ------------------------------------------------------------ 回滚
    def rollback(self, *, steps: int = 1) -> dict[str, Any]:
        """回滚最近 ``steps`` 个可逆迁移。

        只回滚可逆的：不可逆的直接抛 :class:`MigrationError`，不跳过继续往下滚。
        跳过会让库停在既非新也非旧的中间态，比明确拒绝更危险。
        """
        self._prepare_connection()
        mismatches = self.checksum_mismatches()
        if mismatches:
            raise MigrationError(
                "回滚前检测到迁移历史被改写（指纹不匹配）："
                + "；".join(f"M{item['version']:04d}" for item in mismatches)
                + "。请先把迁移恢复成已发布的样子，再执行回滚。"
            )
        if steps < 1:
            raise MigrationError(f"steps 必须 >= 1，收到 {steps}")
        self.ensure_table()

        done = set(self.applied_versions())
        applied = [item for item in reversed(self.migrations) if item.version in done]
        if not applied:
            return {
                "rolled_back": [],
                "current_version": self.status()["current_version"],
                "reason": "没有已应用的迁移，无需回滚",
            }

        selected = applied[:steps]
        for item in selected:
            if not item.reversible:
                raise MigrationError(
                    f"迁移 M{item.version:04d} {item.name} 不可逆，拒绝回滚："
                    f"{item.irreversible_reason}。"
                    "不可逆迁移只能靠「迁移前备份」还原来撤销，"
                    "请用 apply() 之前生成的那份 .bak 文件恢复库。"
                )

        undone: list[dict[str, Any]] = []
        for item in selected:
            self._revert_one(item)
            undone.append({"version": item.version, "name": item.name})
            logger.info("已回滚 M%04d %s", item.version, item.name)

        return {
            "rolled_back": undone,
            "current_version": self.status()["current_version"],
            "reason": "",
        }

    def _revert_one(self, item: Migration) -> None:
        self._prepare_connection()
        self.conn.execute("BEGIN")
        try:
            item.revert(self.conn)
        except Exception as exc:  # noqa: BLE001 - 任何异常都必须回滚并原样上报
            self.conn.execute("ROLLBACK")
            raise MigrationError(
                f"回滚 M{item.version:04d} {item.name} 失败（库已回到回滚前状态）："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        try:
            self.conn.execute(f"DELETE FROM {TABLE} WHERE version = ?", (item.version,))
            self.conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001
            self.conn.execute("ROLLBACK")
            raise MigrationError(
                f"回滚 M{item.version:04d} {item.name} 时更新版本表失败：{exc}"
            ) from exc

    def _refresh_checksums(self, mismatches: list[dict[str, Any]]) -> None:
        """把放行后的新指纹写回版本表（只影响 checksum 列）。"""
        by_version = {item.version: item for item in self.migrations}
        for info in mismatches:
            item = by_version[info["version"]]
            logger.warning(
                "已按 allow_checksum_change=True 放行 M%04d 的指纹变更：%s → %s",
                item.version,
                info["stored"],
                info["expected_current"],
            )
            self.conn.execute(
                f"UPDATE {TABLE} SET checksum = ? WHERE version = ?",
                (info["expected_current"], item.version),
            )

    # ------------------------------------------------------------ 单步执行
    def _run_one(self, item: Migration) -> float:
        """在一个事务里执行一个迁移，返回耗时（毫秒）。

        失败时：整体 ``ROLLBACK``（部分失败不留半成品）→ 写一条 ``success=0``
        的失败记录（带错误信息）→ 抛 ``_MigrationFailed``（消息里带
        version / name / 失败语句 / 原始异常）。
        """
        self._prepare_connection()
        started = perf_counter_ms()
        statement = ""
        self.conn.execute("BEGIN")
        try:
            for statement in item.statements:
                self.conn.execute(statement)
            if item.python is not None:
                statement = "<python 钩子>"
                item.python(self.conn)
        except Exception as exc:  # noqa: BLE001 - 迁移里任何异常都要走同一条回滚路径
            self.conn.execute("ROLLBACK")
            duration_ms = perf_counter_ms() - started
            message = (
                f"M{item.version:04d} {item.name} 执行失败，已回滚该迁移的全部改动。"
                f"失败语句：{statement}；原始异常：{type(exc).__name__}: {exc}"
            )
            try:
                self._record(item, duration_ms, success=False, error=message)
            except sqlite3.Error as record_exc:  # pragma: no cover - 版本表不可写
                # 记不上失败原因是"可观测性"损失，不能因此把真正的失败原因盖掉
                logger.warning("失败记录写入 schema_migrations 未成功：%s", record_exc)
            raise _MigrationFailed(message) from exc

        duration_ms = perf_counter_ms() - started
        try:
            self._record(item, duration_ms, success=True, error="")
            self.conn.execute("COMMIT")
        except Exception as exc:  # noqa: BLE001 - 版本表写不进去 = 本次迁移不算数
            self.conn.execute("ROLLBACK")
            raise _MigrationFailed(
                f"M{item.version:04d} {item.name} 的版本记录写入失败，迁移已回滚："
                f"{type(exc).__name__}: {exc}"
            ) from exc
        return duration_ms

    def _record(
        self, item: Migration, duration_ms: float, *, success: bool, error: str
    ) -> None:
        """写入版本表。成功行的指纹保留，失败行的指纹留空。

        失败意味着这个语句组合没被应用过，留指纹会让人误以为历史里有过这一版。
        """
        self.conn.execute(
            f"INSERT INTO {TABLE}"
            "(version, name, checksum, applied_at, duration_ms, success, error) "
            "VALUES (?, ?, ?, ?, ?, ?, ?) "
            "ON CONFLICT(version) DO UPDATE SET name = excluded.name, "
            "checksum = excluded.checksum, applied_at = excluded.applied_at, "
            "duration_ms = excluded.duration_ms, success = excluded.success, "
            "error = excluded.error",
            (
                item.version,
                item.name,
                item.checksum() if success else "",
                _utc_now(),
                round(float(duration_ms), 3),
                1 if success else 0,
                error[:_ERROR_LIMIT],
            ),
        )

    # ------------------------------------------------------------ 连接状态
    def _prepare_connection(self) -> None:
        """保证连接处于自动提交模式，并且**不在**任何事务里。

        两种真实情况：

        * ``isolation_level is None``（本项目 ``Database`` 用的模式）：
          只检查有没有遗留事务，有就提交掉；
        * 默认模式：sqlite3 会在 DML 前偷偷开事务，迁移自己的回滚会撞上
          "cannot rollback - no transaction is active"，所以先切到自动提交。

        为什么"有遗留事务就提交"而不是回滚：调用方走到这里说明它认为自己
        已经做完了（典型的 bug 是忘了 commit），把它未完成的工作悄悄撤销
        会造成更难查的数据丢失；提交则是"按它以为的语义"继续。
        """
        if not _is_autocommit(self.conn):
            if self.conn.in_transaction:
                self.conn.commit()
            self.conn.isolation_level = None
        elif self.conn.in_transaction:
            self.conn.execute("COMMIT")


# ------------------------------------------------------------------ 便捷函数
def apply_migrations(
    connection: sqlite3.Connection,
    *,
    target: int | None = None,
    dry_run: bool = False,
    backup: bool = True,
    allow_checksum_change: bool = False,
    migrations: Sequence[Migration] | None = None,
) -> dict[str, Any]:
    """对给定连接应用迁移（幂等）。

    失败时抛 :class:`MigrationError`（消息里带 version / name / 失败语句 /
    原始异常）—— 迁移失败属于"必须有人知道"的事件，不能只体现在返回值里。
    调用方若需要"失败也不中断"的语义（``medscholar.db.connect.Database`` 就是这种：
    应用打不开比库结构落后更糟），自行 ``except MigrationError`` 并告警。

    ``init_database`` 会传 ``backup=False``：启动路径上不该每次都复制一遍库文件。
    """
    runner = MigrationRunner(connection, migrations=migrations, backup=backup)
    return runner.apply_or_raise(
        target=target, dry_run=dry_run, allow_checksum_change=allow_checksum_change
    )


def migration_status(connection: sqlite3.Connection) -> dict[str, Any]:
    """只看状态、不改库（版本表也不建）。"""
    return MigrationHistory(connection).status()


def plan_migrations(connection: sqlite3.Connection, *, target: int | None = None) -> list[str]:
    """人类可读的待执行计划（dry-run 输出）。"""
    return MigrationRunner(connection, backup=False).plan(target=target)


def main(argv: list[str] | None = None) -> int:
    """命令行入口（实现在 :mod:`medscholar.db.migrations.cli`）。

    这里做一次**函数内**转发并把 :class:`MigrationRunner` 注入进去：
    ``cli`` 不 import ``migrate``，环就不存在（``scripts/check_arch.py`` 会检查），
    同时 ``python -m medscholar.db.migrate`` 仍然可用。
    """
    from .migrations.cli import run

    return run(argv, runner_factory=MigrationRunner)


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    raise SystemExit(main())
