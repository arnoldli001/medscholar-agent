"""数据层冒烟测试：建库 → 入库 → 三路检索 → 统计。

直接用便携解释器运行::

    .python\\python.exe scripts\\smoke_db.py
"""

from __future__ import annotations

import json
import math
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:  # Windows 控制台默认 GBK，中文输出会乱码
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    sys.stderr.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):  # pragma: no cover
    pass

from medscholar.db import (  # noqa: E402
    hybrid_search,
    init,
    insert_papers,
    search_fts,
    search_vector,
    store_embedding,
)
from medscholar.models import Paper  # noqa: E402

SMOKE_HOME = ROOT / "data" / "_smoke"


def fake_vec(seed: int, dim: int = 768) -> list[float]:
    """确定性伪向量，仅用于贯通链路，不代表真实语义。"""
    return [math.sin(seed * 0.37 + i * 0.017) for i in range(dim)]


def main() -> int:
    if SMOKE_HOME.exists():
        shutil.rmtree(SMOKE_HOME, ignore_errors=True)
    SMOKE_HOME.mkdir(parents=True, exist_ok=True)

    from medscholar.config import AppConfig

    cfg = AppConfig(data_dir=str(SMOKE_HOME))
    cfg.ensure_dirs()

    db = init(config=cfg)
    print(f"数据库     : {db.path}")
    print(f"向量后端   : {'sqlite-vec' if db.vec_available else 'python-fallback'} | {db.vec_note}")
    print()

    papers = [
        Paper(
            title="Accelerated rTMS for post-stroke depression: a randomized trial",
            abstract=(
                "Repetitive transcranial magnetic stimulation with an accelerated "
                "protocol improved Hamilton Depression Rating Scale scores in "
                "patients with post-stroke depression."
            ),
            authors=["Zhang Wei", "Li Ming", "Chen Hua"],
            journal="Brain Stimulation",
            pub_year=2023,
            source="pubmed",
            pmid="37123456",
            doi="10.1016/j.brs.2023.001",
            mesh_terms=["Depression", "Stroke", "Transcranial Magnetic Stimulation"],
            cited_by_count=42,
            is_open_access=True,
        ),
        Paper(
            title="加速rTMS治疗卒中后抑郁的临床疗效观察",
            abstract=(
                "目的：探讨加速重复经颅磁刺激治疗卒中后抑郁的临床疗效。"
                "方法：将60例卒中后抑郁患者随机分为治疗组与对照组。"
                "结果：治疗组HAMD评分显著降低。结论：加速rTMS疗效确切。"
            ),
            authors=["王伟", "李静"],
            journal="中国康复医学杂志",
            pub_year=2022,
            source="cnki",
            keywords=["卒中后抑郁", "重复经颅磁刺激", "康复"],
            cited_by_count=8,
        ),
        Paper(
            title="Deep brain stimulation for treatment-resistant depression",
            abstract="DBS of the subcallosal cingulate showed a sustained antidepressant response.",
            authors=["Malone DA"],
            journal="Biological Psychiatry",
            pub_year=2019,
            source="europepmc",
            doi="10.1016/j.biopsych.2019.05.011",
            cited_by_count=310,
        ),
    ]

    result = insert_papers(papers, db=db)
    print(f"入库       : 新增 {result['new']} 篇 / 更新 {result['updated']} 篇")
    ids = result["ids"]
    for pid in ids:
        store_embedding(pid, fake_vec(pid), db=db)
    print(f"向量写入   : {len(ids)} 条")
    print()

    def show(label: str, hits: list[tuple[int, float]]) -> None:
        print(f"--- {label} ---")
        if not hits:
            print("   （无结果）")
        for pid, score in hits[:5]:
            print(f"   id={pid:<3} score={score:>10.4f}")

    show("BM25 英文：accelerated rTMS depression", search_fts("accelerated rTMS depression", db=db))
    show("BM25 中文整句：加速rTMS治疗卒中后抑郁", search_fts("加速rTMS治疗卒中后抑郁", db=db))
    show("BM25 中文片段：卒中后抑郁", search_fts("卒中后抑郁", db=db))
    show("BM25 中文片段：经颅磁刺激", search_fts("经颅磁刺激", db=db))
    show("向量 KNN（以第 1 篇为查询）", search_vector(fake_vec(ids[0]), db=db))

    print("--- 混合检索（RRF k=60）：rTMS 抑郁 ---")
    fused = hybrid_search("rTMS 抑郁", embedding=fake_vec(ids[0]), top_k=5, db=db)
    for sp in fused:
        print(
            f"   score={sp.score:.6f}  matched={sp.matched_by:<12} "
            f"fts_rank={sp.fts_rank} vec_rank={sp.vector_rank}  {sp.paper.title[:42]}"
        )
    print()

    print("--- 去重校验：重复插入同一篇 ---")
    dup = insert_papers([papers[0]], db=db)
    print(f"   新增 {dup['new']} / 更新 {dup['updated']}（期望 0 / 1）")
    print()

    print("--- 统计 ---")
    print(json.dumps(db.stats(), ensure_ascii=False, indent=2))

    ok = result["new"] == 3 and dup["new"] == 0 and bool(fused)
    print()
    print("SMOKE:", "PASS" if ok else "FAIL")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
