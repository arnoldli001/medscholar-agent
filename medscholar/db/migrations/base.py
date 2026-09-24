"""数据库迁移的抽象层：迁移对象、注册表与定义期校验。

validate_migrations() 在加载注册表时执行三道闸门：版本号不重复（防止并行分支
撞号）、严格连续递增、不可逆迁移必须显式写 irreversible_reason。
"""

from __future__ import annotations

import hashlib
import inspect
import re
import sqlite3
import time
from dataclasses import dataclass
from typing import Callable, Iterable, Sequence

__all__ = [
    "Migration",
    "MigrationError",
    "migration",
    "register",
    "validate_migrations",
    "registered_migrations",
    "get_migrations",
    "reset_registry",
    "perf_counter_ms",
    "PENDING_IRREVERSIBLE",
    "PENDING_REASON",
    "PENDING_VERSION",
]


class MigrationError(RuntimeError):
    """迁移失败。消息里一律带 version / name / 出错语句 / 原始异常。"""


#: 不可逆且作者没有写明原因时占位的描述（真实描述由 validate_migrations 拒绝）
PENDING_REASON = "（未填写不可逆原因）"
PENDING_VERSION = "（未填写版本号）"
PENDING_IRREVERSIBLE = "（未填写回滚语句）"

#: 迁移名格式：小写短横线 slug，会写进 schema_migrations 与日志，避免空格/中文/大写的引号编码麻烦。
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _is_slug(value: str) -> bool:
    return bool(_NAME_RE.match(value))


@dataclass(frozen=True)
class Migration:
    """不可变迁移定义：先逐条执行 statements 再调 python 钩子，回滚按相反顺序。

    简单 DDL 走 SQL（可 review、可分析），基线对齐/回填类逻辑走 Python 钩子（如 M0001）。
    """

    version: int
    name: str
    description: str
    statements: tuple[str, ...] = ()
    rollback: tuple[str, ...] = ()
    python: Callable[[sqlite3.Connection], None] | None = None
    irreversible_reason: str = ""

    def __post_init__(self) -> None:
        # 统一成 tuple：可变 list 会让同一份定义两次算出的指纹不同。
        object.__setattr__(self, "statements", tuple(self.statements))
        object.__setattr__(self, "rollback", tuple(self.rollback))

    # ------------------------------------------------------------------ 执行
    def apply(self, conn: sqlite3.Connection) -> None:
        """正向执行：先 SQL，再 Python 钩子。事务由执行器负责，
        此处绝不能 executescript（隐式提交会撕开执行器事务）或显式 COMMIT。"""
        for statement in self.statements:
            conn.execute(statement)
        if self.python is not None:
            self.python(conn)

    def revert(self, conn: sqlite3.Connection) -> None:
        """反向执行：先 Python 回滚钩子，再逆序跑回滚 SQL（先撤改动再撤创建）。"""
        if not self.reversible:
            raise MigrationError(
                f"迁移 M{self.version:04d} {self.name} 不可逆，拒绝回滚："
                f"{self.irreversible_reason or '作者未声明原因'}"
            )
        if self.python_revert is not None:
            self.python_revert(conn)
        for statement in reversed(self.rollback):
            conn.execute(statement)

    # -------------------------------------------------------------- 属性
    @property
    def reversible(self) -> bool:
        """是否可以回滚（有回滚 SQL 或回滚钩子即算可逆）。"""
        return bool(self.rollback) or self.python_revert is not None

    @property
    def python_revert(self) -> Callable[[sqlite3.Connection], None] | None:
        """回滚钩子：仅 callable 才算钩子；python 被误填成字符串时返回 None，
        让 revert() 报"不可逆"而不是 TypeError。"""
        return self.python if callable(self.python) else None

    def checksum(self) -> str:
        """迁移定义指纹（sha256 前 16 位），写入 schema_migrations.checksum，
        用于发现已应用迁移的语句被事后改动（库结构静默分叉）。"""
        digest = hashlib.sha256()
        digest.update(f"{self.version}|{self.name}\n".encode("utf-8"))
        for statement in self.statements:
            digest.update(_normalize_sql(statement).encode("utf-8"))
            digest.update(b"\x1e")  # 记录分隔符，避免 'AB','C' 与 'A','BC' 撞车
        for statement in self.rollback:
            digest.update(b"R")
            digest.update(_normalize_sql(statement).encode("utf-8"))
            digest.update(b"\x1e")
        digest.update(b"P")
        digest.update(self._python_fingerprint().encode("utf-8"))
        return digest.hexdigest()[:16]

    def _python_fingerprint(self) -> str:
        """Python 钩子指纹：优先"全限定名:源码哈希"；打包成 exe/zipapp 后
        getsource 失败，退化为"全限定名:字节码哈希"，保证校验和永不为空。"""
        func = self.python
        if func is None:
            return "-"
        if isinstance(func, str):  # 极端误用：直接参与指纹，避免静默通过
            return f"str:{func}"
        qualified = f"{getattr(func, '__module__', '?')}.{getattr(func, '__qualname__', '?')}"
        try:
            return f"{qualified}:{hashlib.sha256(inspect.getsource(func).encode('utf-8')).hexdigest()[:16]}"
        except (OSError, TypeError):  # pragma: no cover - 打包/REPL 环境
            code = getattr(func, "__code__", None)
            if code is None:
                return qualified
            return f"{qualified}:{hashlib.sha256(code.co_code).hexdigest()[:16]}"


def _normalize_sql(statement: str) -> str:
    """指纹比较前对 SQL 做最小归一化（仅统一行尾/首尾空白）：
    缩进差异视为相同，但不做更强重写，以免掩盖真实语义改动。"""
    return "\n".join(line.rstrip() for line in statement.strip().splitlines())


def perf_counter_ms() -> float:
    """单调时钟（毫秒）。用 perf_counter 而非 time.time：后者会因系统对时/
    夏令时跳变甚至倒流，污染 duration_ms 的性能对比。"""
    return time.perf_counter() * 1000.0


# ---------------------------------------------------------------- 注册表
#: 版本号 → 迁移。用 dict 而不是 list，重复注册同一版本时能立刻发现。
_REGISTRY: dict[int, Migration] = {}


def migration(
    version: int,
    name: str,
    description: str,
    *,
    rollback: Sequence[str] = (),
    irreversible_reason: str = "",
) -> Callable[[Callable[[sqlite3.Connection], None]], Callable[[sqlite3.Connection], None]]:
    """把函数注册为迁移的 Python 钩子（接收 Connection，运行在迁移自己的事务里，
    失败整体回滚）。装饰器返回原函数，模块级名字仍可直接调用。"""

    def decorator(
        func: Callable[[sqlite3.Connection], None],
    ) -> Callable[[sqlite3.Connection], None]:
        register(
            Migration(
                version=version,
                name=name,
                description=description,
                rollback=tuple(rollback),
                python=func,
                irreversible_reason=irreversible_reason,
            )
        )
        return func

    return decorator


def register(item: Migration) -> Migration:
    """把一个 :class:`Migration` 放进进程内注册表。"""
    if not isinstance(item.version, int) or isinstance(item.version, bool) or item.version < 1:
        raise MigrationError(f"迁移 version 必须是 >= 1 的整数，收到 {item.version!r}")
    if not isinstance(item.name, str) or not _is_slug(item.name):
        raise MigrationError(
            f"迁移 name 必须是小写短横线格式（如 add-paper-note-index），收到 {item.name!r}"
        )
    if not isinstance(item.description, str) or not item.description.strip():
        raise MigrationError(f"迁移 M{item.version:04d} 缺少 description（中文一句话）")
    if item.version in _REGISTRY:
        other = _REGISTRY[item.version]
        raise MigrationError(
            f"版本号 {item.version} 被重复注册：已有 {other.name}，又来了 {item.name}"
        )
    _REGISTRY[item.version] = item
    return item


def registered_migrations() -> list[Migration]:
    """按版本号升序返回已注册的迁移（不做增删改校验，供诊断使用）。"""
    return [_REGISTRY[v] for v in sorted(_REGISTRY)]


def reset_registry() -> None:
    """清空注册表。**仅供测试**使用（生产代码里清空等于丢失全部迁移）。"""
    _REGISTRY.clear()


def validate_migrations(items: Iterable[Migration]) -> tuple[Migration, ...]:
    """定义期闸门：不碰数据库，校验版本号唯一/连续、不可逆迁移已写原因，
    返回按版本号升序的元组，让错误在导入注册表时就暴露。"""
    ordered = sorted(items, key=lambda m: m.version)

    seen: dict[int, str] = {}
    for item in ordered:
        previous = seen.get(item.version)
        if previous is not None:
            raise MigrationError(
                f"版本号 {item.version} 重复：{previous} 与 {item.name}。"
                "同一版本在两个分支上被分别实现，合并后必须重新编号 —— "
                "否则不同人的库会执行到不同的语句，结构从此分叉。"
            )
        seen[item.version] = item.name

    for item in ordered:
        if not item.reversible and not item.irreversible_reason.strip():
            raise MigrationError(
                f"迁移 M{item.version:04d} {item.name} 既没有 rollback 也没写 "
                "irreversible_reason。请二选一：补上回滚语句，或显式承认"
                "「这一步回不去」并说明原因（用户库里已经写入的数据不会因为"
                "一句 DROP TABLE 就回来）。"
            )

    for previous, current in zip(ordered, ordered[1:]):
        if current.version == previous.version + 1:
            continue
        raise MigrationError(
            f"版本号必须连续递增：M{previous.version:04d} 之后是 "
            f"M{current.version:04d}（跳号会让「版本号越大越新」的排序出现空洞，"
            "并让漏合的分支看起来像正常发布）"
        )
    return tuple(ordered)


def get_migrations() -> tuple[Migration, ...]:
    """返回默认注册表校验通过的迁移元组；每次调用都重新校验（代价极小，避免缓存掩盖坏迁移）。"""
    return validate_migrations(registered_migrations())
