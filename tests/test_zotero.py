"""Zotero 本地库桥接。

用合成的 zotero.sqlite 测试（结构与真实库一致的那几张表），
这样不依赖测试机上装没装 Zotero。
"""

from __future__ import annotations

import sqlite3

import pytest

from medscholar.zotero import (
    ZoteroItem,
    find_zotero_dir,
    read_zotero_library,
    to_papers,
)

_SCHEMA = """
CREATE TABLE itemTypes (itemTypeID INTEGER PRIMARY KEY, typeName TEXT);
CREATE TABLE items (itemID INTEGER PRIMARY KEY, itemTypeID INTEGER, key TEXT, dateAdded TEXT);
CREATE TABLE fields (fieldID INTEGER PRIMARY KEY, fieldName TEXT);
CREATE TABLE itemDataValues (valueID INTEGER PRIMARY KEY, value TEXT);
CREATE TABLE itemData (itemID INTEGER, fieldID INTEGER, valueID INTEGER);
CREATE TABLE creators (creatorID INTEGER PRIMARY KEY, firstName TEXT, lastName TEXT, fieldMode INTEGER);
CREATE TABLE itemCreators (itemID INTEGER, creatorID INTEGER, orderIndex INTEGER);
CREATE TABLE itemAttachments (itemID INTEGER, parentItemID INTEGER, contentType TEXT, path TEXT);
CREATE TABLE tags (tagID INTEGER PRIMARY KEY, name TEXT);
CREATE TABLE itemTags (itemID INTEGER, tagID INTEGER);
CREATE TABLE deletedItems (itemID INTEGER PRIMARY KEY);
"""


def build_zotero(tmp_path):
    """造一个最小可用的 Zotero 数据目录。

    含 3 条文献条目，其中 ``KEYDEL01`` 在回收站里（不应被导入），
    另有 1 个附件条目和 0 个笔记条目，用来验证它们不会被当成文献。
    """
    data_dir = tmp_path / "Zotero"
    # 注意：Zotero 的 storage/<目录名> 用的是**附件条目自己**的 key
    # （即 KEYATT01），不是父文献的 key。这里容易搞错，实现与测试都按前者。
    (data_dir / "storage" / "KEYATT01").mkdir(parents=True)
    pdf = data_dir / "storage" / "KEYATT01" / "paper.pdf"
    pdf.write_bytes(b"%PDF-1.4 fake")

    conn = sqlite3.connect(data_dir / "zotero.sqlite")
    conn.executescript(_SCHEMA)
    conn.execute("INSERT INTO itemTypes VALUES (1, 'journalArticle'), (2, 'attachment'), (3, 'note')")
    conn.execute("INSERT INTO fields VALUES (1,'title'),(2,'abstractNote'),(3,'DOI'),(4,'date'),(5,'publicationTitle'),(6,'volume'),(7,'issue'),(8,'pages'),(9,'url'),(10,'language')")
    conn.execute(
        "INSERT INTO items VALUES (1, 1, 'KEYAAA11', '2024-01-02'),"
        "                        (2, 2, 'KEYATT01', '2024-01-02'),"
        "                        (3, 1, 'KEYBBB22', '2024-01-03'),"
        "                        (4, 1, 'KEYDEL01', '2024-01-04')"
    )
    values = [
        (1, "Efficacy of accelerated rTMS for post-stroke depression"),
        (2, "A randomized controlled trial in 120 patients."),
        (3, "10.1016/j.jad.2021.02.045"),
        (4, "2021-05-01"),
        (5, "Journal of Affective Disorders"),
        (6, "285"),
        (7, "3"),
        (8, "112-119"),
        (9, "https://example.org/paper"),
        (10, "en"),
        (11, "Second paper without DOI"),
        (12, "Another abstract."),
        (13, "2020"),
        (14, "Brain Stimulation"),
        (15, "Deleted paper should not appear"),
    ]
    conn.executemany("INSERT INTO itemDataValues VALUES (?, ?)", values)
    conn.executemany(
        "INSERT INTO itemData VALUES (?, ?, ?)",
        [
            (1, 1, 1), (1, 2, 2), (1, 3, 3), (1, 4, 4), (1, 5, 5),
            (1, 6, 6), (1, 7, 7), (1, 8, 8), (1, 9, 9), (1, 10, 10),
            (3, 1, 11), (3, 2, 12), (3, 4, 13), (3, 5, 14),
            (4, 1, 15),
        ],
    )
    conn.executemany(
        "INSERT INTO creators VALUES (?, ?, ?, ?)",
        [(1, "Wei", "Zhang", 0), (2, "Ming", "Li", 0), (3, None, "Consortium", 1)],
    )
    conn.executemany(
        "INSERT INTO itemCreators VALUES (?, ?, ?)",
        [(1, 1, 0), (1, 2, 1), (3, 3, 0)],
    )
    # 附件：指向 storage 下的 PDF（父条目是 1）
    conn.execute("INSERT INTO itemAttachments VALUES (2, 1, 'application/pdf', 'storage:paper.pdf')")
    conn.executemany("INSERT INTO tags VALUES (?, ?)", [(1, "rTMS"), (2, "depression")])
    conn.executemany("INSERT INTO itemTags VALUES (?, ?)", [(1, 1), (1, 2)])
    # 条目 4 在回收站里：绝不能被导入
    conn.execute("INSERT INTO deletedItems VALUES (4)")
    conn.commit()
    conn.close()
    return data_dir, pdf


def by_key(items, key):
    """按 Zotero item key 取条目 —— 不依赖排序（默认是 dateAdded 倒序）。"""
    for item in items:
        if item.key == key:
            return item
    raise AssertionError(f"没有找到 key={key} 的条目，实际有：{[i.key for i in items]}")


class TestFindZoteroDir:
    def test_explicit_path_wins(self, tmp_path):
        data_dir, _ = build_zotero(tmp_path)
        assert find_zotero_dir(data_dir) == data_dir

    def test_missing_returns_none(self, tmp_path):
        assert find_zotero_dir(tmp_path / "not-here") is None

    def test_dir_without_sqlite_is_not_accepted(self, tmp_path):
        empty = tmp_path / "Zotero"
        empty.mkdir()
        assert find_zotero_dir(empty) is None


class TestReadLibrary:
    def test_reads_items_with_fields_authors_and_attachments(self, tmp_path):
        data_dir, pdf = build_zotero(tmp_path)
        items = read_zotero_library(data_dir)
        assert len(items) == 2, "附件条目与回收站条目不应被当成文献"
        first = by_key(items, "KEYAAA11")
        assert first.item_type == "journalArticle"
        assert first.title == "Efficacy of accelerated rTMS for post-stroke depression"
        assert first.authors == ["Zhang, Wei", "Li, Ming"]
        assert first.fields["doi"] == "10.1016/j.jad.2021.02.045"
        assert first.fields["journal"] == "Journal of Affective Disorders"
        # 附件路径要按 itemKey 解析成真实路径
        assert first.pdf_paths == [str(pdf)]
        assert sorted(first.tags) == ["depression", "rTMS"]

    def test_single_field_creator_keeps_last_name(self, tmp_path):
        data_dir, _ = build_zotero(tmp_path)
        items = read_zotero_library(data_dir)
        assert by_key(items, "KEYBBB22").authors == ["Consortium"]

    def test_attachments_and_notes_excluded(self, tmp_path):
        data_dir, _ = build_zotero(tmp_path)
        types = {i.item_type for i in read_zotero_library(data_dir)}
        assert types == {"journalArticle"}

    def test_deleted_items_excluded(self, tmp_path):
        data_dir, _ = build_zotero(tmp_path)
        titles = [i.title for i in read_zotero_library(data_dir)]
        assert not any("Deleted" in t for t in titles)
        assert len(titles) == 2

    def test_limit(self, tmp_path):
        data_dir, _ = build_zotero(tmp_path)
        assert len(read_zotero_library(data_dir, limit=1)) == 1

    def test_missing_dir_raises_actionable_error(self, tmp_path):
        with pytest.raises(FileNotFoundError, match="Zotero 数据目录"):
            read_zotero_library(tmp_path / "nope")

    def test_original_database_not_modified(self, tmp_path):
        """绝不能写用户的 Zotero 库。"""
        data_dir, _ = build_zotero(tmp_path)
        before = (data_dir / "zotero.sqlite").read_bytes()
        read_zotero_library(data_dir)
        assert (data_dir / "zotero.sqlite").read_bytes() == before

    def test_no_temp_files_left_behind(self, tmp_path):
        import tempfile
        from pathlib import Path

        data_dir, _ = build_zotero(tmp_path)
        before = {p.name for p in Path(tempfile.gettempdir()).glob("medscholar-zotero-*")}
        read_zotero_library(data_dir)
        after = {p.name for p in Path(tempfile.gettempdir()).glob("medscholar-zotero-*")}
        assert after <= before, "临时副本没有被清理"


class TestConvertToPapers:
    def test_conversion(self, tmp_path):
        data_dir, pdf = build_zotero(tmp_path)
        papers = to_papers(read_zotero_library(data_dir))
        assert len(papers) == 2
        first = next(p for p in papers if p.title.startswith("Efficacy"))
        assert first.source == "zotero"
        assert first.pub_year == 2021, "日期字段要能取到年份"
        assert first.doi == "10.1016/j.jad.2021.02.045"
        assert first.volume == "285" and first.issue == "3" and first.pages == "112-119"
        assert first.publication_type == "journal-article"
        assert first.is_open_access is True
        assert first.full_text_path == str(pdf)

    def test_paper_without_doi(self, tmp_path):
        data_dir, _ = build_zotero(tmp_path)
        papers = to_papers(read_zotero_library(data_dir))
        second = next(p for p in papers if "Second paper" in p.title)
        assert second.doi is None
        assert second.pub_year == 2020
        assert second.source == "zotero"

    def test_item_without_title_skipped(self):
        item = ZoteroItem(key="k", item_type="journalArticle", fields={"abstract": "x"})
        assert item.to_paper() is None


class TestImportFromZotero:
    async def test_end_to_end_import(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.db.repo import count_papers, list_papers
        from medscholar.importers import import_from_zotero

        data_dir, pdf = build_zotero(tmp_path)
        config = AppConfig(data_dir=str(tmp_path / "ms"), offline=True)
        db = Database(tmp_path / "ms" / "z.db", config=config)

        report = await import_from_zotero(
            data_dir, embed=False, index_pdfs=False, config=config, db=db
        )
        assert report.parsed == 2, report.errors
        assert report.created == 2
        assert count_papers(db=db) == 2
        titles = [p.title for p in list_papers(limit=5, db=db)]
        assert any("accelerated rTMS" in t for t in titles)

    async def test_empty_library_reports_clearly(self, tmp_path):
        from medscholar.config import AppConfig
        from medscholar.db.connect import Database
        from medscholar.importers import import_from_zotero

        data_dir = tmp_path / "Zotero"
        data_dir.mkdir()
        conn = sqlite3.connect(data_dir / "zotero.sqlite")
        conn.executescript(_SCHEMA)
        conn.commit()
        conn.close()

        config = AppConfig(data_dir=str(tmp_path / "ms2"), offline=True)
        db = Database(tmp_path / "ms2" / "z2.db", config=config)
        report = await import_from_zotero(data_dir, embed=False, config=config, db=db)
        assert report.parsed == 0
        assert report.errors and "没有可导入" in report.errors[0]
