"""Lexical attribution audit for cited claims.

For each sentence with bracketed citations, rank every available source by TF-IDF cosine similarity to the
sentence and report where the cited sources land. On answers whose citations are known to be correct the
cited source is the top match for nearly every claim; on scrambled citations its rank is uniform. The audit
needs no model calls, so it can run on every report to flag citations that do not match their claims.
"""
import re

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer


def cited_claims(text: str, min_words: int = 6):
    text = re.sub(r"\s+", " ", text)
    out = []
    for p in re.split(r"(?<=[.!?\]])\s+(?=[A-Z*\-•])", text):
        nums = sorted({int(x) for x in re.findall(r"\[(\d+)\]", p)})
        clean = re.sub(r"\[\d+\]", " ", p).strip(" *-•")
        if nums and len(clean.split()) >= min_words:
            out.append((clean, nums))
    return out


def audit(text: str, sources: dict[int, str], flag_percentile: float = 0.5) -> dict:
    """sources: citation number -> source text. Returns per-claim results and summary statistics.
    A claim is flagged when none of its cited sources ranks in the better half of the candidates."""
    claims = cited_claims(text)
    ids = sorted(sources)
    if len(ids) < 2 or not claims:
        return {"claims": [], "median_percentile": None, "best_cited": None, "flagged": 0}
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=0.9)
    vec.fit([sources[i] for i in ids] + [c for c, _ in claims])
    S = (vec.transform([c for c, _ in claims]) @ vec.transform([sources[i] for i in ids]).T).toarray()
    rows, pct = [], []
    for k, (claim, cited) in enumerate(claims):
        order = list(np.argsort(-S[k])); rank = {ids[j]: r / (len(ids) - 1) for r, j in enumerate(order)}
        cited_known = [c for c in cited if c in rank]
        if not cited_known:
            continue
        best = min(rank[c] for c in cited_known)
        pct += [rank[c] for c in cited_known]
        rows.append({"claim": claim, "cited": cited_known, "best_match": ids[order[0]], "best_cited_percentile": best, "flagged": best > flag_percentile})
    return {"claims": rows, "median_percentile": float(np.median(pct)) if pct else None,
            "best_cited": float(np.mean([r["best_match"] in r["cited"] for r in rows])) if rows else None,
            "flagged": sum(r["flagged"] for r in rows)}
