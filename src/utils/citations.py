"""Stable citation numbering across the research DAG.

Every retrieved source gets one global number the first time any node cites it. Node answers are stored
with global numbers, so a parent sees its children's citations in the same numbering as the list of
available citations it is given, and the final report's bibliography uses that numbering too.

Before this module, each leaf numbered its own sources [1]..[k] and each parent built a fresh merged list,
while the children's answer text kept the leaf-local numbers. A parent copying "[1]" from its second child
therefore pointed at its first child's first source, and citations collapsed onto the first few entries
of the bibliography.
"""
import re
from collections.abc import Iterable

CITE = re.compile(r"\[(\d+)\]")
NODE_REF = re.compile(r"\(\s*(?:see\s+)?((?:node_[A-Za-z0-9_]+)(?:\s*[;,]\s*(?:and\s+)?node_[A-Za-z0-9_]+)*)\s*\)")


class CitationIndex:
    """Assigns each URL one number, in order of first appearance."""

    def __init__(self):
        self._num: dict[str, int] = {}
        self._docs: list = []

    def add(self, docs: Iterable) -> None:
        for d in docs:
            if d.url not in self._num:
                self._docs.append(d)
                self._num[d.url] = len(self._docs)

    def number(self, url: str) -> int | None:
        return self._num.get(url)

    @property
    def documents(self) -> list:
        return list(self._docs)


def to_global(text: str, local_docs: list, index: CitationIndex) -> str:
    """Rewrite citations numbered against `local_docs` (1-based) into global numbers.
    Numbers outside the local list are removed rather than left pointing at an unrelated source."""
    index.add(local_docs)

    def sub(m):
        k = int(m.group(1))
        if 1 <= k <= len(local_docs):
            return f"[{index.number(local_docs[k - 1].url)}]"
        return ""

    return CITE.sub(sub, text)


def citations_in(text: str) -> list[int]:
    return sorted({int(x) for x in CITE.findall(text)})


def expand_node_refs(text: str, node_citations: dict[str, list[int]]) -> str:
    """Replace node references such as "(node_3)" or "(node_2; node_5)" with the global citations of those nodes."""

    def sub(m):
        nums = []
        for nid in re.split(r"\s*[;,]\s*(?:and\s+)?", m.group(1)):
            nums += node_citations.get(nid.strip(), [])
        nums = sorted(set(nums))
        return "".join(f"[{n}]" for n in nums) if nums else ""

    return NODE_REF.sub(sub, text)
