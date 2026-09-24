"""迁移命令行（``python -m medscholar.db.migrate``）。

解析参数、解析库路径、建连接都是入口层的事。
把它们和执行器放在一起会让 :class:`~medscholar.db.migrate.MigrationRunner`
所在模块混进"取配置""开连接"这类与迁移语义无关的细节，
也让"执行器有多大"变得难以判断（框架核心应当能被一眼读完）。

本模块只向下依赖 ``migrations.base`` / ``migrations.registry``，
不 import ``medscholar.db.migrate``：入口与执行器互相 import 会形成循环依赖
（``scripts/check_arch.py`` 会把环报红），所以执行入口统一由
:func:`medscholar.db.migrate.main` 提供，它把 :class:`MigrationRunner` 注入进来。

命令的默认动作是只打印状态、不改库：迁移是危险操作，
必须显式 ``--apply`` 才会动手；``--plan`` / ``--dry-run`` 提供零风险的预演。
"""

from __future__ import annotations

import argparse
import logging
import sqlite3
import sys
from pathlib import Path
from typing import Any, Callable

from .base import Migration, MigrationError
from .registry import MIGRATIONS

__all__ = ["build_parser", "run", "main"]

#: 建 runner 的工厂签名（由 :mod:`medscholar.db.migrate` 注入）。
#: 用 Callable 而不是直接 import 具体类，是为了打断"入口 ↔ 执行器"的循环依赖：
#: 入口只要求"给我一个能跑迁移的对象"，不关心它在哪个模块里。
RunnerFactory = Callable[..., Any]


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="medscholar-db-migrate",
        description="MedScholar Agent 数据库 schema 迁移（默认只打印状态，不改库）",
        epilog=(
            "典型用法：先 --plan 看计划，再 --apply 执行；"
            "改动已发布迁移导致指纹不匹配时，用 --allow-checksum-change 显式放行。"
        ),
    )
    parser.add_argument("--db", help="数据库文件路径（默认取配置 data_dir 下的库）")
    parser.add_argument("--status", action="store_true", help="打印迁移状态（默认动作）")
    parser.add_argument("--plan", action="store_true", help="打印待执行计划（不改库）")
    parser.add_argument("--apply", action="store_true", help="执行迁移")
    parser.add_argument("--dry-run", action="store_true", help="只演示，不改动数据库")
    parser.add_argument("--target", type=int, default=None, help="只升级到该版本号")
    parser.add_argument("--no-backup", action="store_true", help="跳过迁移前备份（不推荐）")
    parser.add_argument(
        "--allow-checksum-change",
        action="store_true",
        help="放行已应用迁移的指纹变更（仅限等价重写，且目标库已是新结构）",
    )
    parser.add_argument("--rollback-steps", type=int, default=0, help="回滚最近 N 个可逆迁移")
    parser.add_argument("-q", "--quiet", action="store_true", help="不打印 INFO 日志")
    return parser


def _resolve_path(raw: str | None) -> Path:
    if raw:
        return Path(raw)
    from ...config import get_config

    return get_config().db_path


def _connect(path: Path) -> sqlite3.Connection:
    """CLI 专用连接：与 ``Database`` 一样开 WAL 与外键，但不加载扩展。

    迁移只做 DDL，不需要 sqlite-vec；少一个扩展就少一个"在朋友机器上装不上"的可能。
    """
    conn = sqlite3.connect(str(path), timeout=30.0, isolation_level=None)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA busy_timeout = 15000")
    return conn


def _configure_logging(quiet: bool) -> None:
    """只在本进程尚未配置日志时挂 handler。

    脚本里跑迁移时"用了哪份备份、应用了哪些版本"必须看得见；
    而宿主进程（``init_database``）已经配好日志时不要重复添加 handler。
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.WARNING if quiet else logging.INFO,
            format="%(levelname)s %(name)s: %(message)s",
        )
    elif quiet:
        logging.getLogger().setLevel(logging.WARNING)


def _print_status(runner: Any, path: Path) -> None:
    state = runner.status()
    exists = "存在" if state["exists"] else "不存在（尚未被迁移框架接管）"
    print(f"数据库    : {path}")
    print(f"版本表    : {exists}")
    print(f"当前版本  : M{state['current_version']:04d}")
    print(f"已应用    : {[(i['version'], i['name']) for i in state['applied']]}")
    print(f"待应用    : {[(i['version'], i['name']) for i in state['pending']]}")
    print(f"指纹异常  : {state['checksum_mismatch']}")
    print(f"孤儿记录  : {state['orphan']}")
    print(f"存在失败  : {state['dirty']} {state['failed']}")


def _run_apply(args: argparse.Namespace, runner: Any) -> int:
    result = runner.apply_or_raise(
        target=args.target,
        dry_run=args.dry_run,
        allow_checksum_change=args.allow_checksum_change,
    )
    for line in result["plan"]:
        print(line)
    if result["backup"]:
        print(f"迁移前备份：{result['backup']}")
    print(
        f"本次应用 {len(result['applied'])} 个迁移，"
        f"当前版本 M{result['current_version']:04d}"
    )
    return 0


def _dispatch(args: argparse.Namespace, runner: Any, path: Path) -> int:
    if args.rollback_steps:
        result = runner.rollback(steps=args.rollback_steps)
        for item in result["rolled_back"]:
            print(f"已回滚 M{item['version']:04d} {item['name']}")
        print(f"当前版本：M{result['current_version']:04d}")
        return 0
    if args.apply or args.dry_run:
        return _run_apply(args, runner)
    if args.plan:
        for line in runner.plan(target=args.target):
            print(line)
        return 0
    _print_status(runner, path)
    return 0


def run(
    argv: list[str] | None = None,
    *,
    runner_factory: RunnerFactory,
    migrations: tuple[Migration, ...] = MIGRATIONS,
) -> int:
    """执行一次 CLI 调用，返回进程退出码（0 成功，1 失败）。"""
    args = build_parser().parse_args(argv)
    _configure_logging(args.quiet)

    path = _resolve_path(args.db)
    if not path.exists():
        print(f"数据库不存在：{path}", file=sys.stderr)
        return 1

    conn = _connect(path)
    try:
        # 回滚不需要迁移前备份（它本身就是"撤销"），其余动作按参数决定。
        # CLI 建一次 runner 复用：既省一次 PRAGMA，也让状态与动作看到同一个快照。
        runner = runner_factory(conn, migrations=migrations, backup=not args.no_backup)
        return _dispatch(args, runner, path)
    except Exception as exc:  # noqa: BLE001 - CLI 顶层必须给可读信息 + 退出码 1
        # 迁移失败是 MigrationError；但 OSError（磁盘满）、sqlite3.Error
        # （库损坏/被锁）同样必须变成"退出码 1 + 一句人话"，而不是 traceback。
        if isinstance(exc, MigrationError):
            print(f"迁移失败：{exc}", file=sys.stderr)
        else:
            print(f"迁移失败：{type(exc).__name__}: {exc}", file=sys.stderr)
        return 1
    finally:
        conn.close()


def main(argv: list[str] | None = None, *, runner_factory: RunnerFactory) -> int:
    """CLI 入口别名（实际逻辑见 :func:`run`）。

    这里没有 ``if __name__ == "__main__":``，也不是 ``-m`` 的运行入口。
    原因是依赖方向：``cli`` 需要 ``MigrationRunner``，而 ``migrate`` 需要 ``cli``。
    如果 cli 直接 import migrate（哪怕写在函数体里），
    ``scripts/check_arch.py`` 就会报 ``db.migrate ↔ db.migrations.cli`` 循环依赖。
    这个环是用工厂注入打断的：命令行入口放在 :mod:`medscholar.db.migrate`，
    由它把 ``MigrationRunner`` 传进来。

    所以正确的命令是 ``python -m medscholar.db.migrate``（见 scripts/README.md）。
    踩过的坑：曾按想当然的模块名把文档写成 ``-m medscholar.db.migrations.cli``，
    发现"没输出"后又在这里加了 ``__main__``，
    把刻意打断的环又接了回去，随即被架构校验器抓住。
    """
    return run(argv, runner_factory=runner_factory)
