"""诊断：向量嵌入进度与 Ollama 批量嵌入耗时。"""

from __future__ import annotations

import asyncio
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

import httpx  # noqa: E402
import sqlite3 as s3  # noqa: E402


def count_embeddings() -> None:
    path = ROOT / "data" / "e2e" / "medscholar.db"
    if not path.exists():
        print("e2e 数据库不存在")
        return
    conn = s3.connect(str(path))
    conn.row_factory = s3.Row
    import sqlite_vec

    conn.enable_load_extension(True)
    sqlite_vec.load(conn)
    conn.enable_load_extension(False)
    papers = conn.execute("SELECT COUNT(*) FROM papers").fetchone()[0]
    embedded = conn.execute("SELECT COUNT(*) FROM paper_embeddings").fetchone()[0]
    # 看有多少条是「已嵌入且非空」
    print(f"papers={papers}  embeddings={embedded}")
    rows = conn.execute(
        "SELECT p.paper_id, length(p.abstract) AS alen FROM papers p "
        "LEFT JOIN paper_embeddings e ON e.paper_id = p.paper_id "
        "WHERE e.paper_id IS NULL LIMIT 5"
    ).fetchall()
    print(f"尚未嵌入示例：{[(r['paper_id'], r['alen']) for r in rows]}")
    sizes = conn.execute(
        "SELECT AVG(length(abstract)) AS avg_len, MAX(length(abstract)) AS max_len FROM papers"
    ).fetchone()
    print(f"摘要长度：平均 {sizes['avg_len']:.0f} 字符，最大 {sizes['max_len']} 字符")
    conn.close()


async def time_embed(batch: int, chars: int) -> None:
    texts = [("加速重复经颅磁刺激治疗卒中后抑郁的临床疗效观察。" * (chars // 22))[:chars] for _ in range(batch)]
    payload = {"model": "nomic-embed-text:latest", "input": texts}
    async with httpx.AsyncClient(base_url="http://127.0.0.1:11434", timeout=600.0) as client:
        started = time.perf_counter()
        response = await client.post("/api/embed", json=payload)
        elapsed = time.perf_counter() - started
        if response.status_code >= 400:
            print(f"batch={batch} chars={chars}: HTTP {response.status_code} {response.text[:120]}")
            return
        vectors = response.json().get("embeddings", [])
        print(f"batch={batch} chars={chars}: {elapsed:.1f}s  返回 {len(vectors)} 条  "
              f"单条均摊 {elapsed / max(1, batch):.2f}s")


async def main() -> int:
    print("=" * 70)
    print("1) 当前嵌入进度")
    print("=" * 70)
    count_embeddings()

    print()
    print("=" * 70)
    print("2) Ollama 批量嵌入耗时实测")
    print("=" * 70)
    for batch, chars in ((1, 2500), (8, 2500), (16, 2500), (16, 6000)):
        await time_embed(batch, chars)
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
