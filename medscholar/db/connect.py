"""SQLite 连接管理与 schema 初始化。

关键点（踩坑记录）：

* ``sqlite_vec.load(conn)`` **不会**自行开启扩展加载权限，必须先调用
  ``conn.enable_load_extension(True)``，否则报 ``OperationalError: not authorized``。
* ``sqlite_vec.loadable_path()`` 返回的是不带 ``.dll`` 后缀的路径，这是正常的
  —— Windows 下 SQLite 会自动补后缀。
* 若扩展无法加载（朋友机器缺少 VC 运行库 / 架构不符 / Python 自带的 SQLite
  未启用扩展），自动退化为**纯 Python 向量检索**，功能不受影响，只是规模上限降低。
"""

from __future__ import annotations

import logging
import sqlite3
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator

from ..config import AppConfig, get_config

logger = logging.getLogger(__name__)

__all__ = [
    "Database",
    "get_db",
    "set_db",
    "close_db",
    "SCHEMA_VERSION",
    "vec_backend_name",
]

SCHEMA_VERSION = "1.0.0"
_SCHEMA_FILE = Path(__file__).with_name("schema.sql")


def vec_backend_name() -> str:
    """返回当前生效的向量后端名（用于状态展示）。"""
    return "sqlite-vec" if get_db().vec_available else "python-fallback"


def _try_load_sqlite_vec(conn: sqlite3.Connection) -> tuple[bool, str]:
    """尝试把 sqlite-vec 扩展载入连接。返回 (是否成功, 说明)。"""
    try:
        import sqlite_vec  # type: ignore
    except ImportError:
        return False, "未安装 sqlite-vec（pip install sqlite-vec）"

    try:
        conn.enable_load_extension(True)
    except (AttributeError, sqlite3.OperationalError) as exc:
        return False, f"当前 Python 的 sqlite3 不支持加载扩展：{exc}"

    try:
        sqlite_vec.load(conn)
        version = conn.execute("SELECT vec_version()").fetchone()[0]
        return True, f"sqlite-vec {version}"
    except (sqlite3.Error, OSError) as exc:
        return False, f"sqlite-vec 加载失败：{exc}"
    finally:
        # 加载完成后立刻关闭，避免留下任意扩展加载能力
        try:
            conn.enable_load_extension(False)
        except sqlite3.Error:  # pragma: no cover
            pass


def _configure(conn: sqlite3.Connection) -> None:
    """设置常用 PRAGMA。"""
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON")
    conn.execute("PRAGMA journal_mode = WAL")
    conn.execute("PRAGMA synchronous = NORMAL")
    conn.execute("PRAGMA temp_store = MEMORY")
    conn.execute("PRAGMA busy_timeout = 15000")
    conn.execute("PRAGMA cache_size = -32000")  # ~32MB page cache


class Database:
    """线程安全的 SQLite 封装（每线程一个连接，WAL 模式下读写并发良好）。"""

    def __init__(self, path: Path | str, *, config: AppConfig | None = None) -> None:
        self.path = Path(path)
        self.config = config or get_config()
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._local = threading.local()
        self._write_lock = threading.RLock()
        self.vec_available = False
        self.vec_note = ""
        self._bootstrap()

    # ------------------------------------------------------------ 连接管理
    @property
    def conn(self) -> sqlite3.Connection:
        conn = getattr(self._local, "conn", None)
        if conn is None:
            conn = sqlite3.connect(
                str(self.path), timeout=30.0, isolation_level=None
            )
            _configure(conn)
            ok, note = _try_load_sqlite_vec(conn)
            if not self.vec_available and ok:
                self.vec_available, self.vec_note = True, note
            elif not ok and not self.vec_note:
                self.vec_note = note
            self._local.conn = conn
        return conn

    def close(self) -> None:
        conn = getattr(self._local, "conn", None)
        if conn is not None:
            try:
                conn.close()
            finally:
                self._local.conn = None

    @contextmanager
    def transaction(self) -> Iterator[sqlite3.Connection]:
        """显式事务；异常回滚。写操作请一律走这里（进程内加锁串行化）。"""
        conn = self.conn
        with self._write_lock:
            conn.execute("BEGIN IMMEDIATE")
            try:
                yield conn
            except BaseException:
                conn.execute("ROLLBACK")
                raise
            else:
                conn.execute("COMMIT")

    # --------------------------------------------------------------- 查询
    def execute(self, sql: str, params: Any = ()) -> sqlite3.Cursor:
        return self.conn.execute(sql, params)

    def query(self, sql: str, params: Any = ()) -> list[sqlite3.Row]:
        return self.conn.execute(sql, params).fetchall()

    def query_one(self, sql: str, params: Any = ()) -> sqlite3.Row | None:
        return self.conn.execute(sql, params).fetchone()

    def scalar(self, sql: str, params: Any = (), default: Any = None) -> Any:
        row = self.conn.execute(sql, params).fetchone()
        if row is None:
            return default
        value = row[0]
        return default if value is None else value

    # --------------------------------------------------------------- 初始化
    def _bootstrap(self) -> None:
        # 先建立一次连接以完成扩展探测
        _ = self.conn
        self._create_base_schema()
        self._ensure_vector_table()
        self._bootstrap_migrations()
        self._set_meta("schema_version", SCHEMA_VERSION)

    def _bootstrap_migrations(self) -> None:
        """建表之后跑一遍 schema 迁移，并把版本历史对齐到 ``schema_migrations``。

        为什么顺序是"先 schema.sql 后迁移"：``schema.sql`` 用
        ``CREATE TABLE IF NOT EXISTS`` 描述的是**当前**结构，它对已有的 400+ 篇
        文献的库是无害的（不会重建已有表）；迁移要解决的是它解决不了的那部分 ——
        老库可能是**旧结构**，需要按版本有序地补索引/补列，并且这个"补"的过程
        必须可重复执行、可回滚、可观测。所以：结构交给 schema.sql 铺底，
        版本化的演进交给迁移。

        为什么这里**不**做迁移前备份（``backup=False``）：启动路径上每次打开库
        都复制一遍文件是纯浪费，而 ``init_database`` 里的迁移通常只是"登记基线"。
        真正会改动结构的迁移由 ``python -m medscholar.db.migrate --apply`` 执行，
        那条路径默认备份。

        为什么迁移失败**只告警、不抛异常**：这个方法在 ``Database()`` 构造里，
        抛异常等于整个应用打不开 —— 用户会因为一次索引没建成而彻底失去工具
        （连自己的 400 篇文献都看不到）。这里的取舍是"功能可用优先"：
        记录 WARNING + 在 ``schema_migrations`` 里留下 ``success=0`` 的失败行，
        让问题可见、可排查、可重试。
        """
        from .migrate import MigrationError, apply_migrations

        try:
            has_data = bool(self.scalar("SELECT COUNT(*) FROM papers", default=0))
        except sqlite3.Error:  # pragma: no cover - papers 缺失时交给迁移的基线检查报错
            has_data = False
        try:
            result = apply_migrations(
                self.conn,
                backup=False,
                # 库里已经有文献时不因为"有人改过已发布的迁移"把应用挡在门外：
                # 用户的库是**不能丢**的资产，"能打开"优先于"立刻报错"。
                # 全新/空库仍然走严格模式，让开发期立刻发现问题。
                allow_checksum_change=has_data,
            )
        except MigrationError:
            logger.exception("数据库迁移失败，已跳过；库结构可能落后于代码，请手工检查")
            return
        if result["applied"]:
            logger.info(
                "数据库迁移完成：%s → M%04d",
                [item["version"] for item in result["applied"]],
                result["current_version"],
            )
        if result["failed"] is not None:
            logger.warning(
                "迁移 M%04d %s 未完成：%s",
                result["failed"]["version"],
                result["failed"]["name"],
                result["failed"]["error"],
            )

    def _create_base_schema(self) -> None:
        sql = _SCHEMA_FILE.read_text(encoding="utf-8")
        with self._write_lock:
            self.conn.executescript(sql)
        if not self.vec_note:
            self.vec_note = "sqlite-vec 不可用，已启用纯 Python 向量检索回退"

    def _ensure_vector_table(self) -> None:
        """按当前配置维度建立向量表；模型/维度变化时自动重建。"""
        dim = int(self.config.embedding.dim)
        model = self.config.embedding.model
        provider = self.config.embedding.provider

        stored_dim = self.get_meta("embedding_dim")
        stored_model = self.get_meta("embedding_model")
        stored_provider = self.get_meta("embedding_provider")

        exists = self._table_exists("paper_embeddings")
        mismatched = exists and (
            (stored_dim and int(stored_dim) != dim)
            or (stored_model and stored_model != model)
            or (stored_provider and stored_provider != provider)
        )

        with self._write_lock:
            if mismatched:
                logger.warning(
                    "嵌入配置已变更（%s/%s/%s → %s/%s/%s），重建向量表",
                    stored_provider, stored_model, stored_dim, provider, model, dim,
                )
                self.conn.execute("DROP TABLE IF EXISTS paper_embeddings")
                exists = False

            if not exists:
                if self.vec_available:
                    try:
                        self.conn.execute(
                            "CREATE VIRTUAL TABLE paper_embeddings USING vec0("
                            "paper_id INTEGER PRIMARY KEY, "
                            f"embedding FLOAT[{dim}]"
                            ")"
                        )
                    except sqlite3.Error as exc:  # pragma: no cover - 极端情况
                        logger.warning("vec0 建表失败（%s），回退纯 Python 后端", exc)
                        self.vec_available = False
                        self.vec_note = f"vec0 建表失败：{exc}"

                if not self.vec_available:
                    self.conn.execute(
                        "CREATE TABLE IF NOT EXISTS paper_embeddings ("
                        "  paper_id  INTEGER PRIMARY KEY REFERENCES papers(paper_id) ON DELETE CASCADE,"
                        "  dim       INTEGER NOT NULL,"
                        "  embedding BLOB NOT NULL,"
                        "  updated_at TEXT NOT NULL DEFAULT (datetime('now'))"
                        ")"
                    )

        self._set_meta("embedding_dim", str(dim))
        self._set_meta("embedding_model", model)
        self._set_meta("embedding_provider", provider)
        self._set_meta("vector_backend", "sqlite-vec" if self.vec_available else "python")

    def _table_exists(self, name: str) -> bool:
        row = self.conn.execute(
            "SELECT 1 FROM sqlite_master WHERE name = ? LIMIT 1", (name,)
        ).fetchone()
        return row is not None

    # ------------------------------------------------------------- meta 读写
    def get_meta(self, key: str, default: str | None = None) -> str | None:
        row = self.conn.execute("SELECT value FROM meta WHERE key = ?", (key,)).fetchone()
        return row["value"] if row else default

    def _set_meta(self, key: str, value: str) -> None:
        with self._write_lock:
            self.conn.execute(
                "INSERT INTO meta(key, value, updated_at) VALUES (?, ?, datetime('now')) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value, "
                "updated_at = datetime('now')",
                (key, str(value)),
            )

    def set_meta(self, key: str, value: str) -> None:
        self._set_meta(key, value)

    def sync_embedding_dim(self, dim: int) -> bool:
        """把向量表维度对齐到嵌入模型的**实际**维度。

        用户换嵌入模型时不必手工删表：探测到实际维度与配置不一致时，
        这里会重建向量表，下次检索自动重新嵌入（原文与元数据不受影响）。

        Returns:
            是否发生了重建（即维度确实变了）。
        """
        dim = int(dim)
        changed = int(self.config.embedding.dim) != dim
        self.config.embedding.dim = dim
        stored = self.get_meta("embedding_dim")
        if stored and int(stored) == dim and not changed:
            return False
        logger.warning(
            "向量维度调整为 %d（原先配置 %s / 表内 %s），重建向量表",
            dim,
            self.config.embedding.dim,
            stored,
        )
        self._ensure_vector_table()
        return True

    # --------------------------------------------------------------- 维护
    def vacuum(self) -> None:
        with self._write_lock:
            self.conn.execute("VACUUM")

    def optimize_fts(self) -> None:
        with self._write_lock:
            for table in ("papers_fts", "fulltext_fts"):
                try:
                    self.conn.execute(
                        f"INSERT INTO {table}({table}) VALUES('optimize')"
                    )
                except sqlite3.Error as exc:  # pragma: no cover
                    logger.debug("FTS optimize %s 跳过：%s", table, exc)

    def stats(self) -> dict[str, Any]:
        """知识库统计信息，供前端左栏展示。"""
        papers = self.scalar("SELECT COUNT(*) FROM papers", default=0)
        embedded = self.scalar("SELECT COUNT(*) FROM paper_embeddings", default=0)
        with_abstract = self.scalar(
            "SELECT COUNT(*) FROM papers WHERE abstract IS NOT NULL AND abstract <> ''",
            default=0,
        )
        fulltext = self.scalar("SELECT COUNT(*) FROM paper_fulltext", default=0)
        citations = self.scalar("SELECT COUNT(*) FROM citations", default=0)
        projects = self.scalar("SELECT COUNT(*) FROM projects", default=0)
        sessions = self.scalar("SELECT COUNT(*) FROM chat_sessions", default=0)
        year_row = self.conn.execute(
            "SELECT MIN(pub_year) AS y0, MAX(pub_year) AS y1 FROM papers WHERE pub_year IS NOT NULL"
        ).fetchone()
        return {
            "papers": papers,
            "embedded": embedded,
            "embedding_coverage": round(embedded / papers, 4) if papers else 0.0,
            "with_abstract": with_abstract,
            "fulltext": fulltext,
            "citations": citations,
            "projects": projects,
            "sessions": sessions,
            "year_min": year_row["y0"] if year_row else None,
            "year_max": year_row["y1"] if year_row else None,
            "vector_backend": "sqlite-vec" if self.vec_available else "python",
            "vector_note": self.vec_note,
            "db_path": str(self.path),
            "db_size_mb": round(self.path.stat().st_size / 1_048_576, 2)
            if self.path.exists()
            else 0.0,
        }


# --------------------------------------------------------------- 全局单例
_DB: Database | None = None
_DB_LOCK = threading.Lock()


def get_db(config: AppConfig | None = None) -> Database:
    """获取全局数据库单例。"""
    global _DB
    if _DB is None:
        with _DB_LOCK:
            if _DB is None:
                cfg = config or get_config()
                cfg.ensure_dirs()
                _DB = Database(cfg.db_path, config=cfg)
                logger.info(
                    "数据库就绪：%s（向量后端 %s）",
                    cfg.db_path,
                    "sqlite-vec" if _DB.vec_available else "python-fallback",
                )
    return _DB


def set_db(db: Database | None) -> None:
    """替换全局单例（测试用）。"""
    global _DB
    _DB = db


def close_db() -> None:
    """关闭当前线程连接。"""
    if _DB is not None:
        _DB.close()
