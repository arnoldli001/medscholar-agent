"""诊断：为什么「补齐全文」会大量失败。

逐篇尝试抓取剩余的开放获取候选，把失败原因**归类汇总**，
而不是只报一句"失败 100 条"。

    .python\\python.exe -X utf8 scripts\\diagnose_remaining.py
    .python\\python.exe -X utf8 scripts\\diagnose_remaining.py --limit 40
"""

from __future__ import annotations

import argparse
import asyncio
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except (AttributeError, OSError):
    pass

from medscholar.db import repo  # noqa: E402
from medscholar.db.connect import get_db  # noqa: E402

#: 把错误文本归入可读的类别（顺序敏感：先匹配更具体的）
REASON_RULES: tuple[tuple[str, str], ...] = (
    (r"没有提供 JATS 正文", "文献本身没有正文（会议摘要 / 勘误 / 社论等）"),
    (r"出版商拒绝了自动下载|HTTP 403", "出版商风控拒绝自动下载（403）"),
    (r"不支持直接下载|HTTP 405", "链接是落地页而非 PDF（405）"),
    (r"链接指向网页而非 PDF", "链接指向网页而非 PDF"),
    (r"PDF 无可提取文本", "PDF 是扫描件，无法提取文本"),
    (r"PDF 解析失败", "PDF 解析失败"),
    (r"PDF 体积过大", "PDF 体积过大，已跳过"),
    (r"非开放获取", "非开放获取（合规跳过）"),
    (r"既没有 PMCID", "标称 OA 但缺 PMCID 与链接"),
    (r"失败：HTTP 5\d\d", "对方服务端错误（5xx，可稍后重试）"),
    (r"网络错误|ConnectError|Timeout|timed out", "网络超时/连接失败（可重试）"),
    (r"PDF 下载失败", "PDF 下载失败（其他）"),
)


def classify(message: str) -> str:
    for pattern, label in REASON_RULES:
        if re.search(pattern, message or "", re.IGNORECASE):
            return label
    return "其他（原样保留在下方样本中）"


async def main() -> int:
    parser = argparse.ArgumentParser(description="补齐全文失败原因归类")
    parser.add_argument("--limit", type=int, default=200)
    args = parser.parse_args()

    from medscholar.agent.reader import PDF_AVAILABLE, PDF_NOTE, ReaderAgent

    db = get_db()
    total = db.scalar("SELECT COUNT(*) FROM papers", default=0)
    have = db.scalar("SELECT COUNT(*) FROM paper_fulltext", default=0)
    print("=" * 78)
    print("知识库全文状况")
    print("=" * 78)
    print(f"  总文献 {total} 篇 | 已入库全文 {have} 篇 | 尚未入库 {total - have} 篇")
    print(f"  PDF 解析后端：{'可用' if PDF_AVAILABLE else '不可用 — ' + PDF_NOTE.splitlines()[0]}")

    candidates = repo.fulltext_candidates(limit=args.limit, db=db)
    print(f"  本轮候选：{len(candidates)} 篇（有 PMCID 的优先）")
    if not candidates:
        print("\n  没有更多候选了 —— 能抓的都已经抓过。")
        return 0

    print()
    print("=" * 78)
    print("逐篇尝试并归类失败原因（这需要几分钟，请耐心等待）")
    print("=" * 78)

    reader = ReaderAgent(config=None, db=db)
    reasons: Counter[str] = Counter()
    samples: dict[str, str] = {}
    fetched = 0
    try:
        for index, paper in enumerate(candidates, start=1):
            try:
                result = await reader.fetch_fulltext(paper)
            except Exception as exc:  # noqa: BLE001
                label = classify(f"{type(exc).__name__}: {exc}")
                reasons[label] += 1
                samples.setdefault(label, f"#{paper.paper_id} {type(exc).__name__}: {exc}")
                continue
            if result.ok:
                fetched += 1
            else:
                label = classify(result.error)
                reasons[label] += 1
                samples.setdefault(label, f"#{paper.paper_id} {result.error}")
            if index % 10 == 0:
                print(f"  已处理 {index}/{len(candidates)}（成功 {fetched}）…", flush=True)
    finally:
        await reader.close()

    failed = len(candidates) - fetched
    print()
    print("=" * 78)
    print(f"结果：成功 {fetched} 篇 / 失败 {failed} 篇")
    print("=" * 78)
    if failed:
        print(f"  {'失败原因':<40}{'条数':>6}{'占比':>8}")
        print("  " + "-" * 54)
        for label, count in reasons.most_common():
            print(f"  {label:<40}{count:>6}{count / failed * 100:>7.0f}%")
        print()
        print("  各类原因的真实样本：")
        for label, sample in samples.items():
            print(f"    · {label}")
            print(f"        {sample[:150]}")
    print()
    print("  说明：")
    print("  · 「文献本身没有正文」是**正常现象**，不是程序缺陷 —— 会议摘要、勘误、")
    print("    社论在 PMC 里就没有 JATS 正文，任何工具都取不到。")
    print("  · 「出版商风控拒绝」是对方策略，程序不会（也不应该）绕过其访问控制；")
    print("    需要全文时请点文献卡片上的链接手动获取。")
    return 0


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
