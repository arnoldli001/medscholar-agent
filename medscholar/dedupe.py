"""跨数据源去重与记录合并。

同一篇文献常同时出现在 PubMed、Europe PMC、OpenAlex、Semantic Scholar 中。
去重键优先级：**DOI → PMID → 标题指纹**（见 :attr:`Paper.dedup_key`）。

合并策略是「取长补短」而非简单去重：保留信息最全的一条作为骨架，
再逐字段补空（摘要取更长的、被引取更大值、开放获取取逻辑或）。
"""

from __future__ import annotations

from typing import Iterable, Sequence

from .models import Paper

__all__ = ["merge_papers", "group_by_key", "pick_richest"]


def _richness(paper: Paper) -> tuple[int, int, int, int]:
    """给记录打分：摘要长度 → 作者数 → MeSH/关键词数 → 被引数。"""
    return (
        len(paper.abstract or ""),
        len(paper.authors),
        len(paper.mesh_terms) + len(paper.keywords),
        paper.cited_by_count,
    )


def pick_richest(papers: Sequence[Paper]) -> Paper:
    """从同一文献的多条记录中选出信息量最大的一条。"""
    return max(papers, key=_richness)


def group_by_key(papers: Iterable[Paper]) -> dict[str, list[Paper]]:
    """按去重键分组，保持首次出现的顺序。"""
    groups: dict[str, list[Paper]] = {}
    for paper in papers:
        groups.setdefault(paper.dedup_key, []).append(paper)
    return groups


def _union(*lists: Iterable[str]) -> list[str]:
    out: list[str] = []
    seen: set[str] = set()
    for items in lists:
        for item in items:
            text = str(item).strip()
            if not text:
                continue
            key = text.lower()
            if key in seen:
                continue
            seen.add(key)
            out.append(text)
    return out


def _merge_group(group: Sequence[Paper]) -> Paper:
    """把一个去重组内的多条记录合并成一条。"""
    if len(group) == 1:
        paper = group[0]
        if not paper.found_in:
            paper.found_in = [paper.source]
        return paper

    base = pick_richest(group)

    # 摘要取最长的那条（Europe PMC 常带结构化标签，PubMed 更干净）
    best_abstract = max((p.abstract or "" for p in group), key=len, default="")

    merged = Paper(
        title=max((p.title or "" for p in group), key=len),
        source=base.source,
        abstract=best_abstract,
        authors=_union(*(p.authors for p in group)),
        journal=next((p.journal for p in group if p.journal), ""),
        pub_year=base.pub_year
        or min((p.pub_year for p in group if p.pub_year), default=None),
        pmid=next((p.pmid for p in group if p.pmid), None),
        pmcid=next((p.pmcid for p in group if p.pmcid), None),
        doi=next((p.doi for p in group if p.doi), None),
        mesh_terms=_union(*(p.mesh_terms for p in group)),
        keywords=_union(*(p.keywords for p in group)),
        cited_by_count=max((p.cited_by_count for p in group), default=0),
        is_open_access=any(p.is_open_access for p in group),
        full_text_url=next((p.full_text_url for p in group if p.full_text_url), ""),
        full_text_path=next((p.full_text_path for p in group if p.full_text_path), ""),
        url=next((p.url for p in group if p.url), ""),
        source_id=next((p.source_id for p in group if p.source_id), ""),
        volume=next((p.volume for p in group if p.volume), ""),
        issue=next((p.issue for p in group if p.issue), ""),
        pages=next((p.pages for p in group if p.pages), ""),
        publication_type=next((p.publication_type for p in group if p.publication_type), ""),
        language=next((p.language for p in group if p.language), ""),
        found_in=_union(*(p.found_in or [p.source] for p in group)),
    )
    return merged


def merge_papers(papers: Iterable[Paper]) -> list[Paper]:
    """合并去重，返回顺序稳定的文献列表（首次出现的顺序优先）。"""
    groups = group_by_key(papers)
    return [_merge_group(group) for group in groups.values()]
