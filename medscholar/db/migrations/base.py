"""数据库迁移的抽象层：迁移对象、注册表与定义期校验。

项目全部数据放在单个 SQLite 文件里，用户的真实库里已有 400+ 篇文献。
此前建表逻辑是 ``schema.sql`` + ``CREATE TABLE IF NOT EXISTS``：
只能保证"表在"，不能保证表结构与代码期望一致。
以后加字段、加索引、改 FTS 触发器时，老库与新代码之间没有机制兜底，
唯一出路是"删库重建"——而用户的库不能丢。

不用 Alembic 的理由：

MedScholar Agent 是免构建的便携应用：拷一个目录、双击 ``run.bat`` 就能跑，
依赖只有 ``requirements.txt`` 里那几项。Alembic 会带进 SQLAlchemy 一整套
ORM 运行时（几十 MB、额外的版本兼容面、额外的打包与分发步骤），
而我们需要的全部能力只是：

* 一张 ``schema_migrations`` 版本表；
* 按版本号顺序、每个迁移一个事务地执行 SQL 或 Python；
* 备份、dry-run、校验和、回滚。

这些用标准库 sqlite3 + 约 500 行代码就能做完，没有任何新增依赖，
``--select F,E9`` 级别的静态检查与便携分发都不受影响。代价是
没有 autogenerate（迁移必须手写），这对本项目可接受：
表结构的手写 SQL 本身就是需要人 review 的产物（见 ``schema.sql`` 的注释密度）。

定义期三道闸门：

``validate_migrations()`` 在加载注册表时（而不是在用户库上执行到一半时）检查：

1. 版本号不重复：两个并行分支各自写了 M0007，合并后是灾难；
2. 版本号严格递增：乱序执行让"第 2 步依赖第 1 步"的假设静默失效；
3. 不可逆必须显式承认：既没有 ``rollback`` 又没写 ``irreversible_reason``
   的迁移直接报错。作者往往默认"这个改动应该能撤"，等线上出事才发现撤不回来。
   把"回不去"变成必须手写的字段，等于强制他当场想一遍退路。
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

#: 迁移名格式：小写英文 + 数字，用短横线连接，例如 "add-paper-note-index"。
#: 为什么限制得这么死：这个名字会被写进 ``schema_migrations`` 并出现在
#: 事故排查的日志里，空格 / 中文 / 大写会带来引号与编码上的无谓麻烦。
_NAME_RE = re.compile(r"^[a-z0-9]+(?:-[a-z0-9]+)*$")


def _is_slug(value: str) -> bool:
    return bool(_NAME_RE.match(value))


@dataclass(frozen=True)
class Migration:
    """一个不可变的迁移定义。

    ``statements`` 与 ``python`` 是同一个迁移的两半，按顺序执行：
    先逐条跑 SQL，再调 Python 钩子；``rollback`` / ``python_revert`` 按相反顺序撤销。

    同时保留 SQL 与 Python 两条路：
    * 纯 SQL 的迁移（加索引、加列）只写 SQL：可读、可 review、可被工具分析；
    * "老库基线对齐""改列并回填数据"这类逻辑用 SQL 表达会又长又脆，
      写成 Python 函数反而清楚。典型是 :mod:`medscholar.db.migrations.registry`
      里的 M0001：它一个字节都不改库，只是识别既有结构并补登记历史。
    """

    version: int
    name: str
    description: str
    statements: tuple[str, ...] = ()
    rollback: tuple[str, ...] = ()
    python: Callable[[sqlite3.Connection], None] | None = None
    irreversible_reason: str = ""

    def __post_init__(self) -> None:
        # 归一化：调用方用 list 传参时统一成 tuple，否则 ``reversible`` /
        # 指纹计算会因为可变对象而出现"同一份定义两次算出不同结果"。
        object.__setattr__(self, "statements", tuple(self.statements))
        object.__setattr__(self, "rollback", tuple(self.rollback))

    # ------------------------------------------------------------------ 执行
    def apply(self, conn: sqlite3.Connection) -> None:
        """正向执行：先 SQL，再 Python 钩子。

        事务由执行器负责（``BEGIN`` / ``COMMIT`` / ``ROLLBACK``），
        这里只负责"把这一条说完"。因此这里绝不能出现 ``executescript``
        或显式 ``COMMIT``：前者会先隐式提交当前事务，把执行器开的事务撕开
        （"第 2 条失败回滚第 1 条"的保证消失，用户库里留下半成品），
        后者会替执行器做决定，让失败分支的 ``ROLLBACK`` 报
        "cannot rollback - no transaction is active"。
        """
        for statement in self.statements:
            conn.execute(statement)
        if self.python is not None:
            self.python(conn)

    def revert(self, conn: sqlite3.Connection) -> None:
        """反向执行：先 Python 回滚钩子，再按相反顺序跑回滚 SQL。

        正向是"先建后改"，回滚自然要先撤改动、再撤创建，
        否则撤销依赖前一条语句的产物时会直接失败。
        """
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
        """回滚钩子：只有真正的函数才会被当成钩子。

        ``Migration`` 是 frozen dataclass，如果有人误把字符串塞进 ``python``，
        这里返回 ``None`` 而不是在 ``revert()`` 里抛 ``TypeError``：
        "不可逆"要给出人类能看懂的理由，而不是类型错误。
        """
        return self.python if callable(self.python) else None

    def checksum(self) -> str:
        """迁移定义的指纹（sha256 前 16 位十六进制）。

        用途见 :meth:`MigrationRunner.apply`：已应用迁移的语句被事后改动时，
        不同人机器上的库结构会静默地不一样，是最阴险的一类事故。
        指纹写进 ``schema_migrations.checksum``，下次 ``apply()`` 就能发现。
        """
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
        """Python 钩子的指纹：优先全限定名，退化为源码哈希。

        源码哈希最准（函数体改了就会变），但打包成单文件 exe / zipapp 后
        ``getsource`` 拿不到源码，此时退化为模块名 + 函数名：虽然漏检函数体改动，
        但至少仍能检出改名与挪位。"拿不到源码"不能变成校验和为空，退化路径必须存在。
        """
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
    """比较语句时忽略纯格式差异（统一行尾、压缩首尾空白）。

    只做这一点点归一化：多行 SQL 的缩进不同算同一条语句，
    但任何更强的"重写"（比如去掉空白）都会掩盖真实的语义改动，
    而校验和存在的唯一目的就是抓语义改动。宁可误报，绝不漏报。
    """
    return "\n".join(line.rstrip() for line in statement.strip().splitlines())


def perf_counter_ms() -> float:
    """单调时钟（毫秒）。用 ``perf_counter`` 而不是 ``time.time``：
    迁移耗时写进 ``schema_migrations.duration_ms`` 供性能对比，
    ``time.time`` 会因系统对时/夏令时跳变（甚至倒流）给出负数或尖峰。
    """
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
    """把一个函数注册成迁移。

    被装饰的函数接收 ``sqlite3.Connection``，在 Python 钩子里可以任意写库
    （它跑在迁移自己的事务里，失败会整体回滚）。装饰器返回原函数本身，
    所以模块级名字仍然可以直接调用，测试里这一点很有用。

    用法::

        @migration(3, "add-paper-source-id-index", "给 papers.source_id 补查询索引",
                   rollback=("DROP INDEX IF EXISTS idx_papers_source_id",))
        def m0003(conn): ...
    """

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
    """校验并归一化一组迁移，返回按版本号升序的元组。

    这是"定义期闸门"的实现：它不碰数据库，错误在导入注册表时就会暴露，
    而不是在用户库上跑到一半才炸。
    """
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
    """返回校验通过的迁移元组（默认注册表）。

    每次调用都重新校验：校验只是几次比较，而缓存会让"测试里注册了坏迁移"
    这类问题逃过检查。
    """
    return validate_migrations(registered_migrations())
