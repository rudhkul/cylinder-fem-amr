#!/bin/bash
# Regenerate every table and figure of the manuscript from the run directories, then build the PDF.
#   scripts/make_paper.sh [python]      (python with numpy + matplotlib; default ~/.venv/bin/python)
# Runs needed: runs/v14/timing (U3-U5), runs/v17/timing (A1-A3), runs/v17/d3 (A2, A3), runs/v17_campaign
# (symlinks), runs/paper (scripts/paper_queue.sh). Nothing here runs the solver.
set -eu
cd "$(dirname "$0")/.."
PY=${1:-$HOME/.venv/bin/python}
"$PY" scripts/paper_tables.py
"$PY" scripts/paper_figures.py
"$PY" scripts/plot_efficiency.py paper/figures/dfg2d2_efficiency.pdf runs/v17_campaign --split
[ -f paper/manuscript.tex ] || { echo "tables and figures written to paper/ (no manuscript.tex here)"; exit 0; }
cd paper && latexmk -pdf -interaction=nonstopmode -halt-on-error manuscript.tex > latexmk.log 2>&1 \
    || { tail -40 latexmk.log; exit 1; }
grep -c 'TBD' manuscript.tex | xargs -I{} echo "manuscript.pdf built; {} TBD markers left in manuscript.tex"
