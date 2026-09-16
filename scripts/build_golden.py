"""从本地库构造**候选**黄金数据集，供人工筛选后固化。

为什么需要"人工筛选"这一步：自动生成的查询一定有偏差，直接把自动结果当基准
等于自欺。本脚本负责把标注工作**变便宜**（把 90% 的体力活做掉），
但保留"人过一遍"的决定权。

三种生成模式，偏差从大到小：

* ``known-item``  —— 查询就是标题。词面重叠极高，**会显著高估 BM25**。
  适合快速验证管道，不适合对外宣称质量。
* ``title-terms`` —— 只用标题里的实词。泄漏少一些。
* ``llm-question``—— 让模型读摘要写一个可回答的研究问题。最接近真实使用，
  但需要 LLM，且受模型措辞影响。

用法::

    :: 生成候选（不写库，只出文件）
    .python\\python.exe scripts\\build_golden.py --mode llm-question --limit 60 --out eval/draft

    :: 生成后请打开 eval/draft.golden.jsonl 逐条过一遍：
    ::   * 删掉 query 有歧义的（可能命中多篇却只标了一篇）
    ::   * 删掉摘要本身信息不足的
    ::   * 把 source 改成 "manual" 表示你确认过
    :: 然后重命名为正式数据集，并同步更新 corpus 文件。
"""

from __future__ import annotations

import argparse
import asyncio
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

try:
    sys.stdout.reconfigure(encoding="utf-8", errors="ignore")
except (AttributeError, OSError):  # pragma: no cover
    pass

from medscholar.config import get_config  # noqa: E402
from medscholar.eval.dataset import EvalCase, EvalDataset, corpus_from_papers, save_dataset  # noqa: E402

#: 让模型写"研究问题"的提示词。要求可被摘要回答、且不要照抄标题用词 ——
#: 照抄会让词面重叠回到 known-item 的水平，白白浪费了 LLM 这一步。
_QUESTION_SYSTEM = (
    "你是医学文献检索专家。给定一篇文献的标题与摘要，写一个**研究者可能真实输入**的检索问题。"
    "要求：\n"
    "1. 问题必须能由该摘要回答；\n"
    "2. **不要照抄标题里的连续词组**，用同义表达或换一种问法；\n"
    "3. 长度 10~25 字（中文）或 6~15 个词（英文）；\n"
    "4. 只输出问题本身，不要编号、不要解释、不要引号。"
)


def build_known_item(papers) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for index, paper in enumerate(papers, start=1):
        if not (paper.title or "").strip():
            continue
        cases.append(
            EvalCase(
                query=paper.title.strip(),
                relevant={str(index): 1.0},
                source="known-item",
                notes="查询=标题；词面重叠高，指标会偏高",
                tags=["draft", "known-item"],
            )
        )
    return cases


def build_title_terms(papers) -> list[EvalCase]:
    cases: list[EvalCase] = []
    for index, paper in enumerate(papers, start=1):
        words = [
            w for w in (paper.title or "").replace(":", " ").replace("—", " ").split()
            if len(w) > 3
        ]
        query = " ".join(words[:6]).strip()
        if not query:
            continue
        cases.append(
            EvalCase(
                query=query,
                relevant={str(index): 1.0},
                source="known-item",
                notes="由标题实词拼成；泄漏少于 known-item",
                tags=["draft", "title-terms"],
            )
        )
    return cases


async def build_llm_questions(papers, *, config) -> list[EvalCase]:
    from medscholar.llm.client import LLMError, get_llm

    client = get_llm(config)
    await client.start()
    cases: list[EvalCase] = []
    for index, paper in enumerate(papers, start=1):
        abstract = (paper.abstract or "").strip()
        if len(abstract) < 120:
            # 摘要太短，写不出有信息量的问题
            continue
        prompt = f"标题：{paper.title}\n\n摘要：{abstract[:1800]}"
        try:
            question = await client.chat(
                [{"role": "user", "content": prompt}],
                system=_QUESTION_SYSTEM,
                temperature=0.4,
                max_tokens=120,
            )
        except LLMError as exc:
            print(f"  第 {index} 篇生成失败：{exc}")
            continue
        question = (question or "").strip().strip('"').strip("「」").split("\n")[0].strip()
        if len(question) < 6:
            continue
        cases.append(
            EvalCase(
                query=question,
                relevant={str(index): 1.0},
                source="llm-question",
                notes=f"模型依据摘要生成（源文献：{paper.title[:40]}）",
                tags=["draft", "llm-question"],
            )
        )
        print(f"  [{index}] {question}")
    return cases


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="build-golden", description="从本地库生成候选评测集（供人工筛选）"
    )
    parser.add_argument("--mode", choices=["known-item", "title-terms", "llm-question"],
                        default="llm-question")
    parser.add_argument("--limit", type=int, default=60, help="取多少篇文献")
    parser.add_argument("--out", default="eval/draft", help="输出路径前缀（不含扩展名）")
    parser.add_argument("--year-from", type=int, default=None, help="只用该年之后的文献")
    parser.add_argument("--min-cited", type=int, default=0, help="只要被引次数 >= 该值的文献")
    args = parser.parse_args(argv)

    from medscholar.db.repo import list_papers

    papers = list_papers(limit=max(args.limit * 3, 120), order_by="cited_desc")
    if args.year_from:
        papers = [p for p in papers if (p.pub_year or 0) >= args.year_from]
    if args.min_cited:
        papers = [p for p in papers if p.cited_by_count >= args.min_cited]
    # 只保留有摘要的（没有摘要的文献写不出像样的问题，也不适合做检索评测）
    papers = [p for p in papers if (p.abstract or "").strip()][: args.limit]

    if not papers:
        print(
            "本地库里没有符合条件的文献（需要非空摘要）。\n"
            "先检索入库或用 `medscholar import` 导入题录。",
            file=sys.stderr,
        )
        return 1

    print(f"从本地库取到 {len(papers)} 篇（模式：{args.mode}）")
    config = get_config()

    if args.mode == "known-item":
        cases = build_known_item(papers)
    elif args.mode == "title-terms":
        cases = build_title_terms(papers)
    else:
        cases = asyncio.run(build_llm_questions(papers, config=config))

    if not cases:
        print("没有生成任何候选查询。", file=sys.stderr)
        return 1

    out = Path(args.out)
    dataset = EvalDataset(
        name=out.stem,
        description=f"候选评测集（{args.mode}，{len(cases)} 条）—— **需人工筛选后才可用**",
        provenance=(
            f"由 scripts/build_golden.py 以 {args.mode} 模式从本地库自动生成。"
            "自动生成的查询存在标注偏差，**必须人工过一遍**：删掉有歧义的、"
            "确认 relevant 正确的，并把 source 改为 manual 表示已确认。"
        ),
        corpus=corpus_from_papers(papers),
        cases=cases,
    )

    golden, corpus = save_dataset(
        dataset,
        golden_path=out.with_suffix(".golden.jsonl"),
        corpus_path=out.with_suffix(".corpus.jsonl"),
    )
    print()
    print("=" * 68)
    print("  候选评测集已生成（**尚未通过校验，请先人工筛选**）")
    print("=" * 68)
    print(f"  查询 : {golden}")
    print(f"  语料 : {corpus}")
    print(f"  条数 : {len(cases)} 条查询 / {len(papers)} 篇语料")
    print()
    print("  下一步：")
    print(f"    1. 打开 {golden}，逐条确认 query 是否清晰、relevant 是否正确；")
    print("    2. 删掉有歧义的条目（可能命中多篇却只标了一篇）；")
    print("    3. 把确认过的条目 source 改成 \"manual\"；")
    print("    4. 跑一次校验：scripts/eval_retrieval.py --dataset <你的文件> --k 10")
    print()
    print("  校验会拒绝「没有标注相关文献」的条目 —— 若某条是故意的阴性对照，")
    print("  请显式加 \"expect_no_relevant\": true。")
    return 0


if __name__ == "__main__":
    sys.exit(main())
