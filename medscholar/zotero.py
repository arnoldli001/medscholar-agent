"""Zotero 本地库桥接。

为什么这条路径特别合适：你用 Zotero + 学校代理把 PDF 合法收进本地库之后，
MedScholar 只需要读本机数据库就能拿到题录和 PDF 路径，然后对
"你已经合法持有的文件"做全文提取与索引 —— 不需要任何下载或认证。

实现要点（都是踩过的坑）：

* Zotero 运行时锁着 ``zotero.sqlite``，直接连会失败或读到不一致状态，
  因此先复制到临时文件再读（只读，绝不写回）；
* 附件路径形如 ``storage:文件名.pdf``，实际位置是
  ``<data_dir>/storage/<itemKey>/<文件名>``，要按 key 拼出来；
* 只认 ``storage:`` 附件；``attachments:``（链接附件）指向用户自选的目录，
  这里只记录路径不猜测。

本模块只读，不会修改 Zotero 数据。
"""

from __future__ import annotations

import logging
import os
import shutil
import sqlite3
import tempfile
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable

from .models import Paper, coerce_int
from .textutil import normalize_doi

logger = logging.getLogger(__name__)

__all__ = [
    "ZoteroItem",
    "find_zotero_dir",
    "read_zotero_library",
    "to_papers",
]

#: 会导入的 Zotero 条目类型（跳过附件、笔记、标签等）
_ITEM_TYPES = (
    "journalArticle",
    "conferencePaper",
    "thesis",
    "book",
    "bookSection",
    "report",
    "preprint",
    "manuscript",
)

#: Zotero 字段名 → Paper 字段
_FIELD_MAP: dict[str, str] = {
    "title": "title",
    "abstractNote": "abstract",
    "DOI": "doi",
    "date": "pub_year",
    "publicationTitle": "journal",
    "journalAbbreviation": "journal",
    "proceedingsTitle": "journal",
    "bookTitle": "journal",
    "volume": "volume",
    "issue": "issue",
    "pages": "pages",
    "url": "url",
    "language": "language",
    "itemType": "publication_type",
}

_TYPE_MAP: dict[str, str] = {
    "journalArticle": "journal-article",
    "conferencePaper": "conference-paper",
    "thesis": "thesis",
    "book": "book",
    "bookSection": "book-chapter",
    "report": "report",
    "preprint": "preprint",
    "manuscript": "manuscript",
}


@dataclass(slots=True)
class ZoteroItem:
    """从 Zotero 读到的一条条目。"""

    key: str = ""
    item_type: str = ""
    fields: dict[str, str] = field(default_factory=dict)
    authors: list[str] = field(default_factory=list)
    pdf_paths: list[str] = field(default_factory=list)
    tags: list[str] = field(default_factory=list)

    @property
    def title(self) -> str:
        return (self.fields.get("title") or "").strip()

    def to_paper(self) -> Paper | None:
        """转成 :class:`Paper`；缺少标题的条目会被跳过。"""
        title = self.title
        if not title:
            return None
        doi = normalize_doi(self.fields.get("doi") or "") or None
        year = coerce_int(self.fields.get("pub_year") or "")
        urls = [p for p in self.pdf_paths]
        return Paper(
            title=title,
            source="zotero",
            abstract=self.fields.get("abstract") or "",
            authors=list(self.authors),
            journal=self.fields.get("journal") or "",
            pub_year=year,
            doi=doi,
            keywords=list(self.tags),
            volume=self.fields.get("volume") or "",
            issue=self.fields.get("issue") or "",
            pages=self.fields.get("pages") or "",
            language=self.fields.get("language") or "",
            publication_type=_TYPE_MAP.get(self.item_type, ""),
            url=self.fields.get("url") or (f"https://doi.org/{doi}" if doi else ""),
            # 本地有 PDF 就算"能拿到全文"，并记下路径供后续解析
            is_open_access=bool(urls),
            full_text_path=urls[0] if urls else "",
            source_id=self.key,
        )


def find_zotero_dir(explicit: str | Path | None = None) -> Path | None:
    """定位 Zotero 数据目录（含 ``zotero.sqlite`` 的那个目录）。

    优先用显式配置；否则依次试常见位置。找不到返回 ``None``，
    由调用方给出"请在设置里填 Zotero 数据目录"的提示。
    """
    candidates: list[Path] = []
    if explicit:
        candidates.append(Path(explicit))
    home = Path.home()
    candidates.extend(
        [
            home / "Zotero",
            home / "Documents" / "Zotero",
            # Windows 上 Zotero 也可能装在 OneDrive 同步目录下
            home / "OneDrive" / "Zotero",
            home / "OneDrive" / "文档" / "Zotero",
        ]
    )
    env = os.environ.get("ZOTERO_DATA_DIR")
    if env:
        candidates.insert(0, Path(env))

    for path in candidates:
        try:
            if path.is_dir() and (path / "zotero.sqlite").is_file():
                return path
        except OSError:  # pragma: no cover - 权限问题
            continue
    return None


def _open_readonly_copy(db_path: Path) -> tuple[sqlite3.Connection, Path]:
    """把数据库复制到临时文件后以只读方式打开。

    Zotero 正在运行时原库被锁，直接连会 ``database is locked``；
    而且它是 WAL 模式，边写边读可能读到半截状态。复制一份最稳。
    """
    tmp_dir = Path(tempfile.mkdtemp(prefix="medscholar-zotero-"))
    target = tmp_dir / "zotero.sqlite"
    shutil.copy2(db_path, target)
    # WAL/SHM 也一起复制，否则最近写入的内容读不到
    for suffix in ("-wal", "-shm"):
        sidecar = db_path.with_name(db_path.name + suffix)
        if sidecar.exists():
            try:
                shutil.copy2(sidecar, tmp_dir / (target.name + suffix))
            except OSError:  # pragma: no cover
                pass
    conn = sqlite3.connect(f"file:{target}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn, tmp_dir


def read_zotero_library(
    data_dir: str | Path | None = None,
    *,
    limit: int | None = None,
    types: Iterable[str] | None = None,
) -> list[ZoteroItem]:
    """读取 Zotero 本地库的全部条目（只读）。

    Args:
        data_dir: Zotero 数据目录；``None`` 时自动探测。
        limit: 最多返回多少条（按加入时间倒序）。
        types: 只读这些条目类型；默认见 :data:`_ITEM_TYPES`。
    """
    resolved = find_zotero_dir(data_dir)
    if resolved is None:
        raise FileNotFoundError(
            "没有找到 Zotero 数据目录（应包含 zotero.sqlite）。"
            "请在设置里手动填写，例如 C:\\Users\\你\\Zotero。"
        )
    db_path = resolved / "zotero.sqlite"
    wanted = tuple(types or _ITEM_TYPES)

    conn, tmp_dir = _open_readonly_copy(db_path)
    try:
        return _read_items(conn, resolved, wanted, limit)
    finally:
        conn.close()
        shutil.rmtree(tmp_dir, ignore_errors=True)


def _read_items(
    conn: sqlite3.Connection, data_dir: Path, types: tuple[str, ...], limit: int | None
) -> list[ZoteroItem]:
    placeholders = ",".join("?" for _ in types)
    sql = (
        "SELECT i.itemID, i.key, it.typeName "
        "FROM items i JOIN itemTypes it ON it.itemTypeID = i.itemTypeID "
        f"WHERE it.typeName IN ({placeholders}) "
        "AND i.itemID NOT IN (SELECT itemID FROM deletedItems) "
        "ORDER BY i.dateAdded DESC"
    )
    params: list[Any] = list(types)
    if limit:
        sql += " LIMIT ?"
        params.append(int(limit))
    rows = conn.execute(sql, params).fetchall()
    if not rows:
        return []

    item_ids = [r["itemID"] for r in rows]
    id_placeholders = ",".join("?" for _ in item_ids)

    # 字段值（一次查全部，避免逐条查造成 N+1）
    field_values: dict[int, dict[str, str]] = {}
    field_sql = (
        "SELECT idata.itemID AS itemID, f.fieldName AS fieldName, idv.value AS value "
        "FROM itemData idata "
        "JOIN fields f ON f.fieldID = idata.fieldID "
        "JOIN itemDataValues idv ON idv.valueID = idata.valueID "
        f"WHERE idata.itemID IN ({id_placeholders})"
    )
    for row in conn.execute(field_sql, item_ids):
        field_values.setdefault(row["itemID"], {})[row["fieldName"]] = row["value"] or ""

    # 作者
    author_sql = (
        "SELECT ic.itemID AS itemID, c.firstName AS firstName, c.lastName AS lastName, "
        "       c.fieldMode AS fieldMode "
        "FROM itemCreators ic JOIN creators c ON c.creatorID = ic.creatorID "
        f"WHERE ic.itemID IN ({id_placeholders}) "
        "ORDER BY ic.itemID, ic.orderIndex"
    )
    authors: dict[int, list[str]] = {}
    for row in conn.execute(author_sql, item_ids):
        if row["fieldMode"] == 1 or not row["firstName"]:
            name = (row["lastName"] or "").strip()
        else:
            name = f"{row['lastName']}, {row['firstName']}".strip(", ")
        if name:
            authors.setdefault(row["itemID"], []).append(name)

    # 附件（PDF）
    attach_sql = (
        "SELECT ia.parentItemID AS parentItemID, ia.path AS path, i.key AS key, "
        "       ia.contentType AS contentType "
        "FROM itemAttachments ia JOIN items i ON i.itemID = ia.itemID "
        f"WHERE ia.parentItemID IN ({id_placeholders})"
    )
    pdfs: dict[int, list[str]] = {}
    for row in conn.execute(attach_sql, item_ids):
        content_type = (row["contentType"] or "").lower()
        path = (row["path"] or "").strip()
        if path.lower().endswith(".pdf") or content_type == "application/pdf":
            resolved = _resolve_attachment(data_dir, row["key"], path)
            if resolved:
                pdfs.setdefault(row["parentItemID"], []).append(resolved)

    # 标签
    tag_sql = (
        "SELECT it.itemID AS itemID, t.name AS name "
        "FROM itemTags it JOIN tags t ON t.tagID = it.tagID "
        f"WHERE it.itemID IN ({id_placeholders})"
    )
    tags: dict[int, list[str]] = {}
    for row in conn.execute(tag_sql, item_ids):
        name = (row["name"] or "").strip()
        if name:
            tags.setdefault(row["itemID"], []).append(name)

    items: list[ZoteroItem] = []
    for row in rows:
        item_id = row["itemID"]
        raw = field_values.get(item_id, {})
        mapped: dict[str, str] = {}
        for source_field, target in _FIELD_MAP.items():
            value = raw.get(source_field)
            if value and target not in mapped:
                mapped[target] = value
        items.append(
            ZoteroItem(
                key=row["key"] or "",
                item_type=row["typeName"] or "",
                fields=mapped,
                authors=authors.get(item_id, []),
                pdf_paths=pdfs.get(item_id, []),
                tags=tags.get(item_id, []),
            )
        )
    return items


def _resolve_attachment(data_dir: Path, item_key: str, path: str) -> str:
    """把 ``storage:文件名.pdf`` 解析成真实路径；解析不了返回空串。

    ``item_key`` 必须是附件条目自己的 key：Zotero 的
    ``storage/<目录名>`` 用的是附件 key，而不是父文献的 key。
    传父 key 会永远解析失败 —— PDF 明明在库里，却一个也索引不到。
    """
    if not path:
        return ""
    if path.startswith("storage:"):
        filename = path[len("storage:") :]
        candidate = data_dir / "storage" / item_key / filename
        return str(candidate) if candidate.is_file() else ""
    if path.startswith("attachments:"):
        # 链接附件：基准目录由 Zotero 设置决定，这里不猜
        return ""
    candidate = Path(path)
    return str(candidate) if candidate.is_file() else ""


def to_papers(items: Iterable[ZoteroItem]) -> list[Paper]:
    """批量转换，跳过没有标题的条目。"""
    papers: list[Paper] = []
    for item in items:
        paper = item.to_paper()
        if paper is not None:
            papers.append(paper)
    return papers
