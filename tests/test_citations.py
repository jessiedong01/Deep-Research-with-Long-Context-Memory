"""Global citation numbering and the attribution audit."""
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from utils.citations import CitationIndex, citations_in, expand_node_refs, to_global

ROOT = Path(__file__).resolve().parent.parent


def doc(u):
    return SimpleNamespace(url=u, title=u)


def test_leaf_numbers_become_global():
    idx = CitationIndex()
    a = to_global("x [1][2]. y [2].", [doc("a"), doc("b")], idx)
    b = to_global("z [1]. w [2].", [doc("c"), doc("a")], idx)
    assert a == "x [1][2]. y [2]."
    assert b == "z [3]. w [1]."           # second leaf's [2] is the same source as the first leaf's [1]
    assert [d.url for d in idx.documents] == ["a", "b", "c"]


def test_out_of_range_numbers_are_dropped():
    idx = CitationIndex()
    assert to_global("claim [1][7].", [doc("a")], idx) == "claim [1]."


def test_node_references_expand_to_child_citations():
    out = expand_node_refs("first (node_3). second (node_3; node_4). third (see node_9).", {"node_3": [1, 2], "node_4": [5]})
    assert out == "first [1][2]. second [1][2][5]. third ."
    assert citations_in(out) == [1, 2, 5]


@pytest.mark.skipif(not (ROOT / "output" / "results.json").exists(), reason="logged run not present")
def test_replay_logged_run_keeps_leaf_attribution():
    """Leaf answers of the logged run, converted to global numbers, still cite the sources they were written from."""
    from utils.attribution_audit import audit
    d = json.load(open(ROOT / "output" / "results.json")); nodes = d["graph"]["nodes"]
    idx = CitationIndex()
    texts = {}
    for n in nodes.values():
        for c in n.get("cited_documents") or []:
            texts.setdefault(c["url"], " ".join(c.get("excerpts", [])))
    merged = []
    for nid, n in nodes.items():
        if n.get("children"):
            continue
        docs = [doc(c["url"]) for c in n["cited_documents"]]
        merged.append(to_global(n["metadata"]["answer"], docs, idx))
    sources = {idx.number(u): t for u, t in texts.items() if idx.number(u)}
    result = audit("\n".join(merged), sources)
    assert result["best_cited"] > 0.8
