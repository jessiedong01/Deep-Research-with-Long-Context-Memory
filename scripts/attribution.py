"""Citation attribution test.

For every sentence with bracketed citations, rank all sources of the same document by TF-IDF cosine
similarity between the sentence and the source text, and record where the cited sources land.
If citations point to the sources a claim came from, cited sources rank near the top; if the numbers
are scrambled, their rank is uniform (median percentile 0.5).

Settings
  validation  leaf answers of the logged run, against the run's 26 retrieved excerpts (citations known correct)
  logged      final report of the logged run (pipeline with global citation numbers)
  evaluation  the ten evaluated De(ep)Composition reports, against the fetched text of their bibliographies

Usage: python scripts/attribution.py
"""
import json
import re
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer

ROOT = Path(__file__).resolve().parent.parent
PAPER = ROOT / "paper"
MIN_CHARS = 2000


def claims(text: str):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?\]])\s+(?=[A-Z*\-•])", text)
    out = []
    for p in parts:
        nums = [int(x) for x in re.findall(r"\[(\d+)\]", p)]
        clean = re.sub(r"\[\d+\]", " ", p).strip(" *-•")
        if nums and len(clean.split()) >= 6:
            out.append((clean, sorted(set(nums))))
    return out


def ranks(claim_list, docs: dict[int, str]):
    """docs: citation number -> text. Returns per (claim, cited) pairs: percentile rank, top-3 hit; per claim: best-in-cited."""
    ids = sorted(docs)
    if len(ids) < 4 or not claim_list:
        return [], []
    vec = TfidfVectorizer(stop_words="english", sublinear_tf=True, max_df=0.9, min_df=1)
    vec.fit([docs[i] for i in ids] + [c for c, _ in claim_list])
    D = vec.transform([docs[i] for i in ids]); C = vec.transform([c for c, _ in claim_list])
    S = (C @ D.T).toarray()
    pairs, per_claim = [], []
    n = len(ids)
    for k, (_, cited) in enumerate(claim_list):
        order = np.argsort(-S[k]); rank_of = {ids[j]: r for r, j in enumerate(order)}
        cited_in = [c for c in cited if c in docs]
        for c in cited_in:
            r = rank_of[c]; pairs.append({"pct": r / (n - 1), "top3": r < 3, "n": n})
        if cited_in:
            per_claim.append({"best_cited": ids[order[0]] in cited_in, "n": n, "k": len(cited_in)})
    return pairs, per_claim


def summarize(name, pairs, per_claim):
    if not pairs:
        return {"setting": name, "pairs": 0}
    pct = np.array([p["pct"] for p in pairs]); top3 = np.mean([p["top3"] for p in pairs])
    exp_top3 = np.mean([min(3, p["n"]) / p["n"] for p in pairs])
    best = np.mean([c["best_cited"] for c in per_claim]); exp_best = np.mean([c["k"] / c["n"] for c in per_claim])
    rng = np.random.default_rng(0)
    boots = [np.median(pct[rng.integers(0, len(pct), len(pct))]) for _ in range(4000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    return {"setting": name, "pairs": len(pairs), "claims": len(per_claim), "median_pct": float(np.median(pct)), "pct_lo": float(lo), "pct_hi": float(hi),
            "top3": float(top3), "top3_chance": float(exp_top3), "best_cited": float(best), "best_chance": float(exp_best),
            "mean_candidates": float(np.mean([p["n"] for p in pairs]))}


def main():
    results, macros = [], {}
    # validation + logged run
    d = json.load(open(ROOT / "output" / "results.json")); nodes = d["graph"]["nodes"]
    pool = {}
    for c in d["cited_documents"]:
        pool.setdefault(c["url"], " ".join(c.get("excerpts", [])))
    url_list = [c["url"] for c in d["cited_documents"]]
    docs_all = {i + 1: pool[u] for i, u in enumerate(url_list)}
    vp, vc = [], []
    for nid, n in nodes.items():
        if n.get("children"):
            continue
        local = [c["url"] for c in n.get("cited_documents") or []]
        ans = (n.get("metadata") or {}).get("answer", "")
        # candidates: every source of the run; the leaf's own numbering maps [i] -> local[i-1]
        docs = {i + 1: pool[u] for i, u in enumerate(url_list)}
        remap = {i + 1: url_list.index(u) + 1 for i, u in enumerate(local) if u in url_list}
        cl = [(c, [remap[x] for x in nums if x in remap]) for c, nums in claims(ans)]
        cl = [x for x in cl if x[1]]
        p, q = ranks(cl, docs); vp += p; vc += q
    results.append(summarize("validation", vp, vc))
    # the same leaf claims against full page text instead of retrieved excerpts (the setting used for the evaluated reports)
    idx = json.load(open(ROOT / "data" / "source_text" / "index.json"))
    page = {}
    for i, u in enumerate(url_list):
        f = ROOT / "data" / "source_text" / f"{idx.get(u, '')}.txt"
        if f.exists() and len(f.read_text()) >= MIN_CHARS:
            page[i + 1] = f.read_text()[:60000]
    wp, wc = [], []
    for nid, n in nodes.items():
        if n.get("children"):
            continue
        local = [c["url"] for c in n.get("cited_documents") or []]
        remap = {i + 1: url_list.index(u) + 1 for i, u in enumerate(local) if u in url_list}
        cl = [(c, [remap[x] for x in nums if x in remap and remap[x] in page]) for c, nums in claims((n.get("metadata") or {}).get("answer", ""))]
        p_, q_ = ranks([x for x in cl if x[1]], page); wp += p_; wc += q_
    results.append(summarize("validation_pages", wp, wc))
    # replay: what each parent saw. Old pipeline: children's local numbers read against the parent's merged list.
    op, oc, fp, fc = [], [], [], []
    for nid, n in nodes.items():
        kids = n.get("children") or []
        if len(kids) < 2:
            continue
        merged = []
        for k in kids:
            for c in nodes[k].get("cited_documents") or []:
                if c["url"] not in merged:
                    merged.append(c["url"])
        for k in kids:
            child = nodes[k]
            if child.get("children"):
                continue                       # only leaf children carry known-correct local numbering
            local = [c["url"] for c in child.get("cited_documents") or []]
            cl = claims((child.get("metadata") or {}).get("answer", ""))
            old_cl = [(c, [url_list.index(merged[x - 1]) + 1 for x in nums if x <= len(merged) and merged[x - 1] in url_list]) for c, nums in cl]
            new_cl = [(c, [url_list.index(local[x - 1]) + 1 for x in nums if x <= len(local) and local[x - 1] in url_list]) for c, nums in cl]
            p_, q_ = ranks([x for x in old_cl if x[1]], docs_all); op += p_; oc += q_
            p_, q_ = ranks([x for x in new_cl if x[1]], docs_all); fp += p_; fc += q_
    results.append(summarize("replay_old", op, oc)); results.append(summarize("replay_fixed", fp, fc))
    docs = {i + 1: pool[u] for i, u in enumerate(url_list)}
    p, q = ranks(claims(d["writeup"]), docs); results.append(summarize("logged", p, q))
    # evaluation reports
    idx = json.load(open(ROOT / "data" / "source_text" / "index.json"))
    ep, ec, conc = [], [], []
    cp, cc = [], []
    prompts = json.load(open(ROOT / "data" / "reports" / "prompts.json"))
    for key in prompts:
        t = (ROOT / "data" / "reports" / key / "deepcomposition.md").read_text(encoding="utf-8-sig")
        bib = {int(m.group(1)): m.group(3) for m in re.finditer(r"(?m)^\s*\[(\d+)\]\s(.*?)\s-\s(https?://\S+)\s*$", t)}
        body = "\n".join(l for l in t.splitlines() if not re.match(r"^\s*\[\d+\]\s.*https?://", l))
        texts = {}
        for n_, u in bib.items():
            f = ROOT / "data" / "source_text" / f"{idx.get(u, '')}.txt"
            if f.exists():
                s = f.read_text()
                if len(s) >= MIN_CHARS:
                    texts[n_] = s[:60000]
        p, q = ranks(claims(body), texts); ep += p; ec += q
        cited_nums = set(int(x) for x in re.findall(r"\[(\d+)\]", body))
        p, q = ranks(claims(body), {k_: v for k_, v in texts.items() if k_ in cited_nums}); cp += p; cc += q
        cites = [int(x) for x in re.findall(r"\[(\d+)\]", body)]
        conc.append({"prompt": key, "bib": len(bib), "used": len(set(cites)), "share_first5": sum(c <= 5 for c in cites) / max(len(cites), 1)})
    results.append(summarize("evaluation", ep, ec)); results.append(summarize("evaluation_cited", cp, cc))
    json.dump({"results": results, "concentration": conc}, open(PAPER / "attribution.json", "w"), indent=1)
    names = {"validation": "Val", "logged": "Log", "evaluation": "Eval", "replay_old": "Old", "replay_fixed": "Fix", "evaluation_cited": "EvalCited", "validation_pages": "ValPages"}
    for r in results:
        k = names[r["setting"]]
        for f, nd in (("pairs", 0), ("claims", 0), ("median_pct", 2), ("pct_lo", 2), ("pct_hi", 2), ("top3", 2), ("top3_chance", 2), ("best_cited", 2), ("best_chance", 2), ("mean_candidates", 0)):
            if f in r:
                macros[f"at{k}{''.join(w.capitalize() for w in f.split('_')).replace('3', 'Three')}"] = f"{r[f]:.{nd}f}" if nd else f"{int(round(r[f]))}"
    import statistics as st
    macros["atBibMed"] = f"{st.median(c['bib'] for c in conc):.0f}"; macros["atUsedMed"] = f"{st.median(c['used'] for c in conc):.0f}"
    macros["atUsedMin"] = min(c["used"] for c in conc); macros["atUsedMax"] = max(c["used"] for c in conc)
    macros["atFirstFiveShare"] = f"{100 * st.median(c['share_first5'] for c in conc):.0f}"
    macros["atUsedShare"] = f"{100 * st.median(c['used'] / c['bib'] for c in conc):.0f}"
    lines = ["\\begin{tabular}{lrrrrr}", "\\toprule", "Setting & Claim--source pairs & Median rank percentile & Top-3 hit rate & Chance & Best match cited \\\\", "\\midrule"]
    labels = {"validation_pages": "Leaf answers, own numbering, full pages", "validation": "Leaf answers, own numbering", "replay_old": "Leaf answers as a parent read them, original pipeline", "replay_fixed": "Leaf answers as a parent reads them, global numbering", "logged": "Final report, logged run", "evaluation": "Evaluated reports, all listed sources", "evaluation_cited": "Evaluated reports, cited sources only"}
    for r in sorted(results, key=lambda r: ["validation", "validation_pages", "replay_fixed", "replay_old", "logged", "evaluation", "evaluation_cited"].index(r["setting"])):
        lines.append(f"{labels[r['setting']]} & {r['pairs']} & {r['median_pct']:.2f} [{r['pct_lo']:.2f}, {r['pct_hi']:.2f}] & {r['top3']:.2f} & {r['top3_chance']:.2f} & {r['best_cited']:.2f} (chance {r['best_chance']:.2f}) \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tables" / "attribution.tex").write_text("\n".join(lines) + "\n")
    (PAPER / "attribution_numbers.tex").write_text("% Generated by scripts/attribution.py. Do not edit by hand.\n" + "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    # figure: distribution of the cited sources' rank percentiles
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 3, figsize=(8.2, 2.3), sharey=True)
    for ax, (pairs, title, color) in zip(axes, [(vp, "leaf answers (correct citations)", "#2a78d6"), (op, "leaf answers read by a parent,\noriginal numbering", "#eda100"),
                                                (ep, "ten evaluated reports", "#eb6834")]):
        x = [p_["pct"] for p_ in pairs]
        ax.hist(x, bins=10, range=(0, 1), weights=[1 / len(x)] * len(x), color=color, edgecolor="white")
        ax.axhline(0.1, color="#52514e", ls="--", lw=0.8)
        ax.set_title(title, fontsize=8.5); ax.set_xlabel("rank percentile of cited source"); ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
    axes[0].set_ylabel("share of claim-source pairs")
    fig.tight_layout(); fig.savefig(PAPER / "figs" / "attribution.pdf"); fig.savefig(PAPER / "figs" / "attribution.png", dpi=170)
    for r in results:
        print(r)
    print(conc)


if __name__ == "__main__":
    main()
