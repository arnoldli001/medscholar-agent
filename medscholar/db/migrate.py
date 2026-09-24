"""迁移执行器：版本表、事务、备份、dry-run、状态查询。

CLI：``python -m medscholar.db.migrate --db <path> [--status|--plan|--apply|--rollback-steps N]``。
三个关键约束：不用 executescript（会隐式提交，破坏一迁移一事务）；备份必须用
``Connection.backup()``（WAL 下直接 copy 文件会缺页打不开）；已应用迁移的语句
指纹存入 schema_migrations，改动已发布迁移默认报错，防止库结构静默分叉。
MigrationHistory 管版本表，MigrationRunner 负责执行。
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

#: 错误信息入库上限：完整异常仍随 MigrationError 抛出，这里截断防止版本表膨胀。
_ERROR_LIMIT = 2000

#: version 作主键物理上杜绝重复登记；applied_at 用显式带时区的 UTC ISO8601
#:（区别于表内其他列 datetime('now') 生成的无时区格式），避免夏令时本地时间歧义。
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
    """``isolation_level is None`` 即自动提交：BEGIN/COMMIT 完全由我们控制；
    默认模式下 sqlite3 会隐式开事务，显式 BEGIN 会报错，动手前必须先处理。"""
    return conn.isolation_level is None


class _MigrationFailed(RuntimeError):
    """内部信号：某个迁移失败并已回滚（不对外暴露，apply() 会转成返回值）。"""


class MigrationHistory:
    """``schema_migrations`` 版本表的只读/幂等读写与状态解读，不执行迁移。"""

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
        """已登记的行：版本号 → Row（含失败行）。表不存在返回空 dict，只读路径不建表。"""
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
        """状态快照：exists（版本表是否存在）、current_version、dirty（留有失败行，
        库结构可能停在新旧之间）、checksum_mismatch、orphan 等。"""
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
    """按版本号顺序执行迁移并维护版本表。

    backup 默认开启（改动库前做 SQLite 备份）；init_database 对空库显式关闭。
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
        """人类可读的执行计划（dry-run 输出）；无待应用时也返回一句明确提示而非空列表。"""
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
        """用在线备份 API 备份到 path。WAL 下最新数据可能还在 -wal 文件，
        直接拷 .db 会缺页打不开，``Connection.backup()`` 保证一致性快照且无需关连接。"""
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
        """``<db>.pre-migration-<版本>.bak``，与库同目录（保证随库一起搬迁）；
        带版本号可一眼看出备份点；内存库返回 None。"""
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
        """应用迁移，返回结果字典（applied/plan/pending/current_version/dry_run/
        failed/backup；失败不抛异常，failed 带 version/name/error，后续迁移立即停止）。

        allow_checksum_change 仅用于确认等价重写且目标库已是新结构的逃生口：
        放过校验不会补上实际缺失的列，只会记 warning 并回写新指纹。
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
        """同 apply()，失败时抛 MigrationError（CLI 用，保证失败不静默、退出码可用）。"""
        result = self.apply(**kwargs)
        if result["failed"] is not None:
            info = result["failed"]
            raise MigrationError(
                f"迁移 M{info['version']:04d} {info['name']} 失败：{info['error']}"
            )
        return result

    # ------------------------------------------------------------ 回滚
    def rollback(self, *, steps: int = 1) -> dict[str, Any]:
        """回滚最近 steps 个可逆迁移；遇不可逆迁移直接报错而不跳过
        （跳过会让库停在非新非旧的中间态；不可逆迁移只能用 .bak 备份还原）。"""
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
        """一个事务执行一个迁移，返回耗时毫秒。失败时整体 ROLLBACK、
        写 success=0 失败记录后抛 _MigrationFailed（消息带版本/语句/原始异常）。"""
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
        """写入版本表；成功行写指纹，失败行指纹留空（未应用成功，留指纹会误导历史）。"""
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
        """保证连接处于自动提交模式且不在事务里：默认 isolation_level 下先
        commit 遗留事务再切 None（隐式事务会让迁移自己的 ROLLBACK 报错）。
        遗留事务选择提交而非回滚：调用方多半是忘了 commit，悄悄撤销其改动更危险。"""
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
    """对给定连接应用迁移（幂等），失败抛 MigrationError（迁移失败必须可见）。

    需要"失败不中断启动"语义的调用方（如 Database 构造）自行捕获告警；
    init_database 传 backup=False，避免每次启动都复制库文件。
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
    """命令行入口，函数内转发到 migrations.cli 并注入 MigrationRunner：
    cli 不反向 import migrate 以避免循环依赖（scripts/check_arch.py 会校验）。"""
    from .migrations.cli import run

    return run(argv, runner_factory=MigrationRunner)


if __name__ == "__main__":  # pragma: no cover - 命令行入口
    raise SystemExit(main())
