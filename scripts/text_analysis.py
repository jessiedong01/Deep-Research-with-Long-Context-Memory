"""Measurements on the evaluation reports and the logged DAG runs, written to paper/analysis_numbers.tex,
paper/tables/*.tex and paper/figs/*.pdf.

Report measures (per report, then paired across the ten motions):
  - citation density: bracketed citations per 100 words of body text
  - cited-sentence share: fraction of sentences carrying at least one citation
  - distinct sources and citations per source
  - hedge, contrast, commitment, and causal-connective rates per 1,000 words, from fixed word lists
  - claim-style title: the first line states a claim rather than a topic or question
Source measures (De(ep)Composition only; the STORM exports carry no URLs):
  - source type from the domain, and whether each URL still resolves
DAG measures (the two logged runs):
  - sources per node, how many of the root's sources reach it through each child, and how many came from refinement

Usage: python scripts/text_analysis.py [--check-links]
"""
import argparse
import json
import re
import subprocess
import statistics
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from urllib.parse import urlparse

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "data" / "reports"
PAPER = ROOT / "paper"
SYSTEMS = ("deepcomposition", "storm")

HEDGES = ["may", "might", "could", "possibly", "perhaps", "arguably", "potentially", "appears", "appear", "seems", "seem",
          "suggests", "suggest", "likely", "unlikely", "uncertain", "unclear", "debated", "contested", "controversial", "depends",
          "depending", "can be", "to some extent", "in some cases", "remains open"]
CONTRAST = ["however", "on the other hand", "conversely", "while", "whereas", "although", "though", "nevertheless",
            "nonetheless", "critics", "opponents", "proponents", "detractors", "both sides"]
COMMIT = ["must", "should", "clearly", "demonstrates", "demonstrate", "shows", "show", "proves", "establishes", "therefore",
          "thus", "decisively", "undeniably", "strongly", "essential", "necessary", "outweigh", "outweighs", "superior", "preferable"]
CAUSAL = ["because", "therefore", "thus", "hence", "as a result", "consequently", "leads to", "lead to", "led to", "results in",
          "result in", "resulting in", "drives", "drive", "causes", "cause", "due to", "which means", "so that", "enables", "enable",
          "reduces", "increases", "raises", "lowers"]

macros = {}


def macro(k, v):
    macros[k] = v


def body(text: str, system: str) -> str:
    """Report text without the bibliography and markdown markup."""
    lines = text.splitlines()
    out = []
    for ln in lines:
        if re.match(r"^\s*\[\d+\]\s.*https?://", ln):          # De(ep)Composition bibliography entry
            continue
        if re.match(r"^\s*(\*\*)?(references|bibliography|sources)(\*\*)?\s*$", ln.strip(), re.I):
            continue
        out.append(ln)
    t = "\n".join(out)
    t = re.sub(r"(?m)^#+\s*", "", t)
    t = t.replace("**", "")
    return t


def rate(text_lower: str, words: int, terms) -> float:
    n = 0
    for term in terms:
        n += len(re.findall(r"\b" + re.escape(term) + r"\b", text_lower))
    return 1000.0 * n / max(words, 1)


def sentences(text: str):
    text = re.sub(r"\s+", " ", text)
    parts = re.split(r"(?<=[.!?])(?:\[\d+\])*\s+(?=[A-Z\"“(*-])", text)
    return [p for p in parts if len(p.split()) >= 4]


def measure(text: str, system: str) -> dict:
    b = body(text, system)
    words = len(b.split())
    cites = re.findall(r"\[(\d+)\]", b)
    sents = sentences(b)
    cited = sum(1 for s in sents if re.search(r"\[\d+\]", s))
    low = b.lower()
    first = next((ln.strip() for ln in text.splitlines() if ln.strip()), "")
    claim_title = not (first.lower().startswith("# summary") or first.endswith("?") or first.lower().startswith("summary"))
    headings = len(re.findall(r"(?m)^#+ ", text)) if system == "storm" else len(re.findall(r"(?m)^[A-Z][^\n\[]{3,90}$", text))
    return {"words": words, "citations": len(cites), "sources": len(set(cites)), "cite_per_100w": 100.0 * len(cites) / max(words, 1),
            "cited_sentence_share": cited / max(len(sents), 1), "n_sentences": len(sents), "cites_per_source": len(cites) / max(len(set(cites)), 1),
            "hedge_rate": rate(low, words, HEDGES), "contrast_rate": rate(low, words, CONTRAST), "commit_rate": rate(low, words, COMMIT),
            "causal_rate": rate(low, words, CAUSAL), "claim_title": claim_title, "headings": headings, "title": first[:120]}


def paired(rows, field):
    prompts = sorted({r["prompt"] for r in rows})
    a = np.array([next(r[field] for r in rows if r["prompt"] == p and r["system"] == "deepcomposition") for p in prompts], float)
    b = np.array([next(r[field] for r in rows if r["prompt"] == p and r["system"] == "storm") for p in prompts], float)
    d = a - b
    rng = np.random.default_rng(0)
    boots = [np.median(d[rng.integers(0, len(d), len(d))]) for _ in range(5000)]
    lo, hi = np.percentile(boots, [2.5, 97.5])
    p = wilcoxon(a, b).pvalue if np.any(d != 0) else 1.0
    return {"dc_med": float(np.median(a)), "st_med": float(np.median(b)), "diff_med": float(np.median(d)), "lo": float(lo), "hi": float(hi),
            "p": float(p), "dc_wins": int((d > 0).sum()), "n": len(d)}


DOMAIN_TYPES = [
    ("government", r"\.gov(\.[a-z]{2})?$|\.gov\.|europa\.eu$|\.int$|imf\.org$|worldbank\.org$|oecd\.org$|un\.org$|federalreserve|bis\.org$|ecb\.europa"),
    ("academic", r"\.edu$|\.ac\.[a-z]{2}$|doi\.org$|nber\.org$|ssrn\.com$|arxiv\.org$|jstor\.org$|springer|sciencedirect|wiley|tandfonline|nature\.com$|sagepub|cambridge\.org$|oup\.com$|academic\.oup|pmc\.ncbi|ncbi\.nlm|plato\.stanford|iep\.utm|researchgate|mdpi|frontiersin|plos"),
    ("think tank or NGO", r"brookings|cbpp|urban\.org|rand\.org|cfr\.org|csis\.org|piie|aei\.org|heritage\.org|cato\.org|pew|kff\.org|carnegie|chathamhouse|atlanticcouncil|wilsoncenter|ifs\.org|\.org$"),
    ("news or magazine", r"nytimes|wsj|ft\.com|bloomberg|reuters|apnews|cnbc|cnn|bbc|economist|theguardian|washingtonpost|forbes|fortune|axios|politico|npr\.org|time\.com|theatlantic|vox\.com|businessinsider|marketwatch|techcrunch|wired|nbcnews|cbsnews|abcnews|aljazeera|timesofisrael|haaretz|jpost"),
    ("encyclopedia", r"wikipedia\.org$|britannica\.com$"),
]


def domain_type(url: str) -> str:
    host = urlparse(url).netloc.lower().removeprefix("www.")
    for name, pat in DOMAIN_TYPES:
        if re.search(pat, host):
            return name
    return "company, blog, or other"


def check(url: str) -> int:
    try:
        r = subprocess.run(["curl", "-s", "-o", "/dev/null", "-L", "-m", "20", "-A", "Mozilla/5.0 (Macintosh) research-link-check",
                            "-w", "%{http_code}", url], capture_output=True, text=True, timeout=30)
        return int(r.stdout.strip() or 0)
    except Exception:
        return 0


def sources(check_links: bool):
    rows = []
    for p in sorted(REPORTS.glob("*/deepcomposition.md")):
        for ln in p.read_text(encoding="utf-8-sig").splitlines():
            m = re.match(r"^\s*\[(\d+)\]\s(.*?)\s-\s(https?://\S+)\s*$", ln)
            if m:
                rows.append({"prompt": p.parent.name, "n": int(m.group(1)), "title": m.group(2), "url": m.group(3), "type": domain_type(m.group(3))})
    cache = ROOT / "data" / "link_check.json"
    status = json.loads(cache.read_text()) if cache.exists() else {}
    if check_links:
        todo = sorted({r["url"] for r in rows} - set(status))
        with ThreadPoolExecutor(12) as ex:
            for u, s in zip(todo, ex.map(check, todo)):
                status[u] = s
        cache.write_text(json.dumps(status, indent=0))
    for r in rows:
        r["status"] = status.get(r["url"])
    return rows


def dag_runs():
    runs = []
    src = ROOT / "output" / "results.json"
    if src.exists():
        d = json.load(open(src)); runs.append(("poverty", d["graph"], d))
    out = []
    for name, g, d in runs:
        nodes = g["nodes"]; root = g["root_id"]
        urls = {nid: {c["url"] for c in (n.get("cited_documents") or [])} for nid, n in nodes.items()}
        leaves = [nid for nid, n in nodes.items() if not n.get("children")]
        refined = [nid for nid in nodes if "gap" in nid]
        leaf_union = set().union(*(urls[l] for l in leaves))
        refine_urls = set().union(*(urls[n] for n in refined)) if refined else set()
        orig_leaf_urls = set().union(*(urls[l] for l in leaves if l not in refined))
        out.append({"run": name, "nodes": len(nodes), "leaves": len(leaves), "parents": len(nodes) - len(leaves), "max_depth": max(int(n["depth"]) for n in nodes.values()),
                    "refined_nodes": len(refined), "root_sources": len(urls[root]), "leaf_sources": len(leaf_union),
                    "root_from_refinement": len(urls[root] & (refine_urls - orig_leaf_urls)), "leaf_sources_mean": statistics.mean(len(urls[l]) for l in leaves),
                    "root_answer_words": len((nodes[root].get("metadata") or {}).get("answer", "").split()), "report_words": len((d.get("writeup") or "").split()),
                    "report_citations": len(re.findall(r"\[(\d+)\]", d.get("writeup") or "")), "report_sources": len(set(re.findall(r"\[(\d+)\]", d.get("writeup") or "")))})
    return out


def fmt(x, nd=1):
    return f"{x:.{nd}f}"


def main():
    ap = argparse.ArgumentParser(); ap.add_argument("--check-links", action="store_true"); args = ap.parse_args()
    prompts = json.load(open(REPORTS / "prompts.json"))
    rows = []
    for key in prompts:
        for s in SYSTEMS:
            rows.append({"prompt": key, "system": s, **measure((REPORTS / key / f"{s}.md").read_text(encoding="utf-8-sig"), s)})
    json.dump(rows, open(PAPER / "text_measures.json", "w"), indent=1)

    # paired comparison table
    spec = [("cite_per_100w", "Citations per 100 words", 1), ("cited_sentence_share", "Sentences with a citation", 2),
            ("sources", "Distinct sources", 0), ("cites_per_source", "Citations per source", 1),
            ("hedge_rate", "Hedges per 1,000 words", 1), ("contrast_rate", "Contrast markers per 1,000 words", 1),
            ("commit_rate", "Commitment markers per 1,000 words", 1), ("causal_rate", "Causal connectives per 1,000 words", 1), ("words", "Words", 0)]
    lines = ["\\begin{tabular}{lrrrrc}", "\\toprule", "Measure & \\sys{} & STORM & Difference & 95\\% CI & Wins \\\\", "\\midrule"]
    keys = {"cite_per_100w": "CiteDensity", "cited_sentence_share": "CitedShare", "sources": "Sources", "cites_per_source": "CitesPerSource",
            "hedge_rate": "Hedge", "contrast_rate": "Contrast", "commit_rate": "Commit", "causal_rate": "Causal", "words": "Words"}
    for field, label, nd in spec:
        r = paired(rows, field); k = keys[field]
        for name, v in (("Dc", r["dc_med"]), ("St", r["st_med"]), ("Diff", r["diff_med"]), ("Lo", r["lo"]), ("Hi", r["hi"])):
            macro(f"tx{k}{name}", fmt(v, nd) if name in ("Dc", "St") else (("+" if v > 0 else "") + fmt(v, nd)).replace("+-", "-"))
        macro(f"tx{k}Wins", r["dc_wins"]); macro(f"tx{k}P", f"{r['p']:.3f}" if r["p"] >= 0.001 else "<0.001")
        sign = lambda v: ("$+$" if v > 0 else ("$-$" if v < 0 else "")) + fmt(abs(v), nd)
        lines.append(f"{label} & {fmt(r['dc_med'], nd)} & {fmt(r['st_med'], nd)} & {sign(r['diff_med'])} & [{sign(r['lo'])}, {sign(r['hi'])}] & {r['dc_wins']}/{r['n']} \\\\")
    lines += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tables").mkdir(exist_ok=True)
    (PAPER / "tables" / "text_measures.tex").write_text("\n".join(lines) + "\n")
    labels = {k: v for k, v in json.load(open(REPORTS / "stance_labels.json")).items() if not k.startswith("_")}
    for field, values in (("title", ("claim", "topic")), ("ending", ("firm", "conditional", "qualified", "compromise")), ("side", ("for", "against", "neither"))):
        for v in values:
            macro(f"lab{field.capitalize()}{v.capitalize()}", sum(1 for x in labels.values() if x[field] == v))
    sl = ["\\begin{tabular}{p{5.2cm}lll}", "\\toprule", "Motion & Side & Title & Final paragraph \\\\", "\\midrule"]
    for key, text in prompts.items():
        t = text.replace("&", "\\&").replace("%", "\\%")
        if len(t) > 70:
            t = t[:67].rsplit(" ", 1)[0] + "\\,\\ldots"
        lab = labels[key]
        sl.append(f"{t} & {lab['side']} & {lab['title']} & {lab['ending']} \\\\")
    sl += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tables" / "stance_labels.tex").write_text("\n".join(sl) + "\n")

    # sources
    src = sources(args.check_links)
    if src:
        n = len(src); macro("srcN", n); macro("srcUnique", len({r["url"] for r in src}))
        types = {}
        for r in src:
            types[r["type"]] = types.get(r["type"], 0) + 1
        order = ["government", "academic", "think tank or NGO", "news or magazine", "encyclopedia", "company, blog, or other"]
        tl = ["\\begin{tabular}{lrr}", "\\toprule", "Source type & Sources & Share \\\\", "\\midrule"]
        for t in order:
            c = types.get(t, 0); tl.append(f"{t[0].upper() + t[1:]} & {c} & {100 * c / n:.0f}\\% \\\\")
            macro("srcShare" + "".join(w.capitalize() for w in re.findall(r"[a-z]+", t))[:20], f"{100 * c / n:.0f}")
        tl += ["\\bottomrule", "\\end{tabular}"]
        (PAPER / "tables" / "source_types.tex").write_text("\n".join(tl) + "\n")
        checked = [r for r in src if r["status"] is not None]
        if checked:
            ok = sum(1 for r in checked if 200 <= r["status"] < 400); blocked = sum(1 for r in checked if r["status"] in (401, 403, 429))
            dead = sum(1 for r in checked if r["status"] in (404, 410) or r["status"] == 0 or r["status"] >= 500)
            macro("linkChecked", len(checked)); macro("linkOk", ok); macro("linkOkPct", f"{100 * ok / len(checked):.0f}")
            macro("linkBlocked", blocked); macro("linkBlockedPct", f"{100 * blocked / len(checked):.0f}")
            macro("linkDead", dead); macro("linkDeadPct", f"{100 * dead / len(checked):.0f}")
        json.dump(src, open(PAPER / "sources.json", "w"), indent=1)

    # DAG runs
    for r in dag_runs():
        for k, v in r.items():
            if k != "run":
                macro("dag" + "".join(w.capitalize() for w in k.split("_")), fmt(v, 1) if isinstance(v, float) else v)

    # figure: paired per-motion comparison for four measures
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "serif", "font.size": 8.5, "axes.spines.top": False, "axes.spines.right": False})
    fig, axes = plt.subplots(1, 4, figsize=(8.6, 2.5))
    for ax, (field, label) in zip(axes, [("cite_per_100w", "citations per 100 words"), ("cited_sentence_share", "sentences with a citation"),
                                         ("hedge_rate", "hedges per 1k words"), ("causal_rate", "causal connectives per 1k words")]):
        for p in prompts:
            a = next(r[field] for r in rows if r["prompt"] == p and r["system"] == "deepcomposition")
            b = next(r[field] for r in rows if r["prompt"] == p and r["system"] == "storm")
            ax.plot([0, 1], [b, a], color="#b8b6b0", lw=0.8, zorder=1)
            ax.scatter([0], [b], color="#eb6834", s=14, zorder=2); ax.scatter([1], [a], color="#2a78d6", s=14, zorder=2)
        ax.set_xticks([0, 1], ["STORM", "De(ep)Comp."]); ax.set_xlim(-0.35, 1.35); ax.set_title(label, fontsize=8.5)
        ax.grid(axis="y", color="#e4e3df", lw=0.5); ax.set_axisbelow(True)
    fig.tight_layout(); fig.savefig(PAPER / "figs" / "paired_measures.pdf"); fig.savefig(PAPER / "figs" / "paired_measures.png", dpi=170)

    # per-motion table
    bibsize = {}
    for key in prompts:
        t = (REPORTS / key / "deepcomposition.md").read_text(encoding="utf-8-sig")
        bibsize[key] = len(re.findall(r"(?m)^\s*\[\d+\]\s.*https?://", t))
    tl = ["\\begin{tabular}{p{6.4cm}rrrrr}", "\\toprule",
          "Motion & \\multicolumn{3}{c}{\\sys{}} & \\multicolumn{2}{c}{STORM} \\\\",
          "\\cmidrule(lr){2-4}\\cmidrule(lr){5-6}", " & words & cited & listed & words & cited \\\\", "\\midrule"]
    for key, text in prompts.items():
        a = next(r for r in rows if r["prompt"] == key and r["system"] == "deepcomposition")
        b = next(r for r in rows if r["prompt"] == key and r["system"] == "storm")
        t = text.replace("&", "\\&").replace("%", "\\%")
        if len(t) > 95:
            t = t[:92].rsplit(" ", 1)[0] + "\\,\\ldots"
        w = lambda n: f"{n:,}".replace(",", "{,}")
        tl.append(f"{t} & {w(a['words'])} & {a['sources']} & {bibsize[key]} & {w(b['words'])} & {b['sources']} \\\\")
    tl += ["\\bottomrule", "\\end{tabular}"]
    (PAPER / "tables" / "reports.tex").write_text("\n".join(tl) + "\n")
    macro("nPrompts", len(prompts)); macro("nReports", len(rows))
    macro("bibMedDc", f"{statistics.median(bibsize.values()):.0f}"); macro("bibMinDc", min(bibsize.values())); macro("bibMaxDc", max(bibsize.values()))
    (PAPER / "analysis_numbers.tex").write_text("% Generated by scripts/text_analysis.py. Do not edit by hand.\n" +
        "\n".join(rf"\newcommand{{\{k}}}{{{v}}}" for k, v in sorted(macros.items())) + "\n")
    for r in rows:
        print(f"{r['prompt']:18} {r['system']:16} w={r['words']:5d} c/100w={r['cite_per_100w']:.1f} cited={r['cited_sentence_share']:.2f} "
              f"hedge={r['hedge_rate']:.1f} contrast={r['contrast_rate']:.1f} commit={r['commit_rate']:.1f} causal={r['causal_rate']:.1f} claim={r['claim_title']} | {r['title'][:60]}")
    print(f"wrote {len(macros)} macros")


if __name__ == "__main__":
    main()
