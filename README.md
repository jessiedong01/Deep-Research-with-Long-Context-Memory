# De(ep)Composition: Deep Research for Argumentation by Query Decomposition

Deep research pipelines summarize. Debate needs a side. De(ep)Composition is a deep research pipeline for argumentation: it decomposes a research question into a directed acyclic graph of sub-questions, answers the leaves by web search, composes each parent from its children under generated composition instructions, refines nodes whose answers have gaps, and writes a report that argues the root answer. The paper compares it with OpenAI Deep Research and STORM on debate-style prompts: [`paper/main.pdf`](paper/main.pdf).

## Setup

```bash
python -m venv .venv && .venv/bin/pip install -e .
cp .env.example .env   # add an OpenAI (or Azure OpenAI) key and a Serper key
```

## Usage

```bash
.venv/bin/python src/deepcomposition/main.py
```

The pipeline asks for a research question and writes the DAG, every node's answer, and the final report to `output/`. Every source gets one citation number for the whole run (`src/utils/citations.py`), and `src/utils/attribution_audit.py` checks whether each cited sentence matches the source it cites.

## Reproducing the paper's analysis

```bash
.venv/bin/python scripts/fetch_sources.py          # text of every source in the evaluated bibliographies
.venv/bin/python scripts/text_analysis.py --check-links
.venv/bin/python scripts/attribution.py
.venv/bin/python scripts/draw_dag.py output/results.json paper/figs/example_dag.pdf
cd paper && tectonic -X compile main.tex
```

## Data

`data/reports/` holds the 20 retained evaluation reports (De(ep)Composition and STORM, 10 motions), the motions, and the stance labels. `output/results.json` and `examples/` hold the logs of two runs. `data/link_check.json` records the link check.

## Authors

Justin Blumencranz and Jessie Dong, with Yucheng Jiang and Monica Lam.
