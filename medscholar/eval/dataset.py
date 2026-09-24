"""黄金数据集：读写、校验与偏差说明。

数据集是 JSONL（每行一条），刻意做成纯文本以便在 PR 里逐行 review——
评测集的改动应该像代码一样被审查，而不是悄悄改掉数字。

一行查询的格式::

    {
      "query": "加速 rTMS 治疗卒中后抑郁的疗效",
      "relevant": {"12": 1, "45": 1},        // id -> 相关性（二元用 1，分级用 2/3）
      "notes": "标注口径：只要报告了 HAMD 变化即算相关",
      "source": "llm-question",              // known-item / llm-question / manual
      "tags": ["zh", "rct"]
    }

关于标注偏差（写在数据文件里，不只是文档里）：不同来源的查询难度差别很大，
所以每条查询都带 ``source`` 字段，报告里可以按来源分组看指标：

* ``known-item``：查询就是文献标题。词面重叠极高，会显著高估 BM25，
  适合验证"管道通不通"，不适合用来吹检索质量。
* ``llm-question``：由模型根据摘要写一个可回答的研究问题，词面重叠低得多，
  更接近真实使用，但受模型措辞影响。
* ``manual``：人工撰写，最可信，成本最高。

语料（被检索的库）与数据集分开：语料是"临时装进 SQLite 的文献元数据"，
数据集是"查询 + 应命中的文献 id"。id 用语料内的序号而不是真实 paper_id，
这样换机器、换库都能复现。
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from ..models import Paper

logger = logging.getLogger(__name__)

__all__ = [
    "EvalCase",
    "EvalDataset",
    "load_dataset",
    "save_dataset",
    "SOURCES",
    "DATASET_DIR",
]

#: 允许的查询来源（用于分组看指标与提示偏差）
SOURCES = ("known-item", "llm-question", "manual")

#: 内置数据集目录
DATASET_DIR = Path(__file__).resolve().parent / "datasets"


@dataclass(slots=True)
class EvalCase:
    """一条评测查询。"""

    query: str
    #: doc_id（语料内的字符串序号）→ 相关性。二元用 1.0，分级可用 2.0/3.0
    relevant: dict[str, float] = field(default_factory=dict)
    notes: str = ""
    source: str = ""
    tags: list[str] = field(default_factory=list)
    #: 显式声明"这条查询本来就没有相关文献"（阴性对照）。
    #: 有了它，校验器才能区分"故意留空"与"忘了标注"——
    #: 后者会让指标虚高，必须拦下；前者是有价值的对照实验。
    expect_no_relevant: bool = False

    def to_dict(self) -> dict[str, Any]:
        out: dict[str, Any] = {
            "query": self.query,
            "relevant": dict(self.relevant),
            "notes": self.notes,
            "source": self.source,
            "tags": list(self.tags),
        }
        if self.expect_no_relevant:
            out["expect_no_relevant"] = True
        return out

    @classmethod
    def from_dict(cls, raw: Mapping[str, Any]) -> "EvalCase":
        query = str(raw.get("query") or "").strip()
        if not query:
            raise ValueError("评测用例缺少 query")
        raw_relevant = raw.get("relevant") or {}
        relevant: dict[str, float] = {}
        if isinstance(raw_relevant, Mapping):
            for key, value in raw_relevant.items():
                try:
                    relevant[str(key)] = float(value)
                except (TypeError, ValueError):
                    relevant[str(key)] = 1.0
        elif isinstance(raw_relevant, Sequence):
            for key in raw_relevant:
                relevant[str(key)] = 1.0
        tags = raw.get("tags") or []
        return cls(
            query=query,
            relevant=relevant,
            notes=str(raw.get("notes") or ""),
            source=str(raw.get("source") or ""),
            tags=[str(t) for t in tags] if isinstance(tags, Sequence) else [],
            expect_no_relevant=bool(raw.get("expect_no_relevant")),
        )


@dataclass(slots=True)
class EvalDataset:
    """一个数据集 = 语料 + 查询集。"""

    name: str = ""
    description: str = ""
    #: 语料：每条是一个 Paper 的字典形式（metadata）。doc_id = 1-based 序号
    corpus: list[dict[str, Any]] = field(default_factory=list)
    cases: list[EvalCase] = field(default_factory=list)
    #: 采集方式与已知偏差，随数据集一起保存
    provenance: str = ""

    # ------------------------------------------------------------------ 校验
    def validate(self) -> list[str]:
        """返回问题列表（空表示通过）。评测集本身出错比检索出错更隐蔽。"""
        problems: list[str] = []

        if not self.corpus:
            problems.append("语料为空")
        if not self.cases:
            problems.append("没有评测查询")

        valid_ids = {str(i) for i in range(1, len(self.corpus) + 1)}

        for index, case in enumerate(self.cases, start=1):
            if not case.relevant:
                if case.expect_no_relevant:
                    continue  # 显式声明的阴性对照，合法
                problems.append(
                    f"第 {index} 条「{case.query[:24]}」没有标注相关文献 —— "
                    "它会被跳过而虚高整体指标。请补标注；"
                    "若这是故意为之的阴性对照，请显式加 \"expect_no_relevant\": true"
                )
                continue
            unknown = sorted(set(case.relevant) - valid_ids)
            if unknown:
                problems.append(
                    f"第 {index} 条「{case.query[:24]}」引用了不存在的语料 id：{unknown}"
                )
            if case.source and case.source not in SOURCES:
                problems.append(
                    f"第 {index} 条的 source「{case.source}」不在 {SOURCES} 中"
                )

        # 语料里必须有标题，否则 Paper 构造会失败
        for position, item in enumerate(self.corpus, start=1):
            if not str(item.get("title") or "").strip():
                problems.append(f"语料第 {position} 条没有 title")

        if not self.provenance.strip():
            problems.append("缺少 provenance（数据来源与已知偏差说明）")

        return problems

    def papers(self) -> list[Paper]:
        """把语料转成 :class:`Paper`，``source_id`` 固定为语料内序号。"""
        out: list[Paper] = []
        for index, item in enumerate(self.corpus, start=1):
            data = dict(item)
            data.setdefault("source", "eval")
            data["source_id"] = str(index)
            out.append(Paper(**data))
        return out

    def to_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "description": self.description,
            "provenance": self.provenance,
            "corpus": list(self.corpus),
            "cases": [c.to_dict() for c in self.cases],
        }

    def by_source(self) -> dict[str, list[EvalCase]]:
        grouped: dict[str, list[EvalCase]] = {}
        for case in self.cases:
            grouped.setdefault(case.source or "(未标注来源)", []).append(case)
        return grouped


def load_dataset(path: str | Path) -> EvalDataset:
    """读取数据集。支持两种形式：

    * ``.jsonl``：每行一条查询，语料另存为同名 ``.corpus.jsonl``；
    * ``.json``：一个对象，含 ``corpus`` 与 ``cases`` 两个键（便于小数据集单文件携带）。
    """
    target = Path(path)
    if not target.is_file():
        raise FileNotFoundError(f"数据集不存在：{target}")

    if target.suffix.lower() == ".json":
        raw = json.loads(target.read_text(encoding="utf-8"))
        dataset = EvalDataset(
            name=str(raw.get("name") or target.stem),
            description=str(raw.get("description") or ""),
            provenance=str(raw.get("provenance") or ""),
            corpus=list(raw.get("corpus") or []),
            cases=[EvalCase.from_dict(item) for item in (raw.get("cases") or [])],
        )
        return dataset

    # JSONL：查询集
    cases: list[EvalCase] = []
    for line_no, line in enumerate(target.read_text(encoding="utf-8").splitlines(), start=1):
        text = line.strip()
        if not text or text.startswith("#"):
            continue
        try:
            cases.append(EvalCase.from_dict(json.loads(text)))
        except json.JSONDecodeError as exc:
            raise ValueError(f"{target.name} 第 {line_no} 行不是合法 JSON：{exc}") from exc

    # 语料：同名 .corpus.jsonl
    corpus_path = target.with_suffix("").with_suffix(".corpus.jsonl")
    if not corpus_path.is_file():
        corpus_path = target.parent / f"{target.stem.replace('.golden','')}.corpus.jsonl"
    corpus: list[dict[str, Any]] = []
    if corpus_path.is_file():
        for line in corpus_path.read_text(encoding="utf-8").splitlines():
            text = line.strip()
            if text and not text.startswith("#"):
                corpus.append(json.loads(text))
    else:
        logger.warning("没有找到配套语料文件 %s", corpus_path.name)

    return EvalDataset(
        name=target.stem,
        description=f"来自 {target.name}",
        provenance=f"从 {target.name} 读取（provenance 见同目录 README）",
        corpus=corpus,
        cases=cases,
    )


def save_dataset(
    dataset: EvalDataset,
    *,
    golden_path: str | Path,
    corpus_path: str | Path | None = None,
) -> tuple[Path, Path | None]:
    """写出数据集为 JSONL（查询 + 语料两个文件）。返回实际写出的路径。"""
    golden = Path(golden_path)
    golden.parent.mkdir(parents=True, exist_ok=True)
    with golden.open("w", encoding="utf-8") as handle:
        for case in dataset.cases:
            handle.write(json.dumps(case.to_dict(), ensure_ascii=False) + "\n")

    written_corpus: Path | None = None
    if corpus_path is not None:
        written_corpus = Path(corpus_path)
        written_corpus.parent.mkdir(parents=True, exist_ok=True)
        with written_corpus.open("w", encoding="utf-8") as handle:
            for item in dataset.corpus:
                handle.write(json.dumps(item, ensure_ascii=False) + "\n")
    return golden, written_corpus


def corpus_from_papers(papers: Iterable[Paper]) -> list[dict[str, Any]]:
    """把 Paper 列表转成可序列化的语料条目（只保留检索会用到的字段）。"""
    keep = (
        "title", "abstract", "authors", "journal", "pub_year", "doi", "pmid",
        "pmcid", "keywords", "mesh_terms", "publication_type", "language",
        "is_open_access", "cited_by_count",
    )
    out: list[dict[str, Any]] = []
    for paper in papers:
        data = paper.to_dict()
        out.append({key: data[key] for key in keep if data.get(key) not in (None, "", [], 0)})
    return out
