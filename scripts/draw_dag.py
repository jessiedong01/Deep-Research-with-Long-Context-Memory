"""Draw a research DAG from a saved run as a figure.

Usage: python scripts/draw_dag.py output/results.json paper/figs/example_dag.pdf
"""
import json
import sys
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch


def main(src, dst):
    d = json.load(open(src))
    g = d.get("graph") or d.get("data", {}); nodes = g["nodes"]; root = g.get("root_id", "node_1")
    # positions: depth -> column, order within depth -> row
    by_depth = {}
    order = []
    def walk(nid):
        order.append(nid)
        for c in nodes[nid].get("children", []):
            walk(c)
    walk(root)
    for nid in order:
        by_depth.setdefault(nodes[nid]["depth"], []).append(nid)
    maxd = max(by_depth)
    pos = {}
    for depth, ids in by_depth.items():
        n = len(ids)
        for i, nid in enumerate(ids):
            pos[nid] = (depth * 3.5, -(i - (n - 1) / 2) * 1.6)
    fig, ax = plt.subplots(figsize=(9.2, 6.6))
    for nid in order:
        x, y = pos[nid]
        for c in nodes[nid].get("children", []):
            cx, cy = pos[c]
            ax.annotate("", xy=(cx - 1.6, cy), xytext=(x + 1.6, y), arrowprops=dict(arrowstyle="-|>", color="#52514e", lw=0.9, shrinkA=0, shrinkB=0))
    for nid in order:
        x, y = pos[nid]; n = nodes[nid]
        refined = "gap" in nid
        fc = "#dcd9ff" if refined else ("#e8f4ee" if n["depth"] == 0 else "#ffffff")
        ax.add_patch(FancyBboxPatch((x - 1.6, y - 0.68), 3.2, 1.36, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec="#52514e", lw=0.8))
        q = textwrap.shorten(n["question"], 96, placeholder=" …")
        ax.text(x, y + 0.5, ("refinement " + nid.split("_")[-1]) if "gap" in nid else nid.replace("_", " "), ha="center", va="center", fontsize=9.5, color="#52514e")
        ax.text(x, y - 0.07, "\n".join(textwrap.wrap(q, 30)[:4]), ha="center", va="center", fontsize=9.5)
    ax.set_xlim(-1.75, maxd * 3.5 + 1.75); ax.set_ylim(min(p[1] for p in pos.values()) - 0.9, max(p[1] for p in pos.values()) + 0.9)
    for depth in range(maxd + 1):
        ax.text(depth * 3.5, max(p[1] for p in pos.values()) + 0.95, f"depth {depth}", ha="center", fontsize=10, color="#52514e")
    ax.axis("off"); fig.tight_layout(); fig.savefig(dst); fig.savefig(dst.replace(".pdf", ".png"), dpi=170)
    print("saved", dst, "nodes", len(order))


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
