#!/usr/bin/env python3
"""
Error against mesh nodes (top row) and against wall time (bottom row) for the uniform (U*) and
adaptive (A*) runs of the DFG 2D-2 timing campaign (scripts/timing.sh, scripts/efficiency_table.py).

    plot_efficiency.py <out.png> <timing_dir> [--uniform U3 U4 U5] [--amr A1 A2 A3] [--split]

--split writes two 2x2 figures instead, <out>_nodes.<ext> and <out>_wall.<ext>, with larger type and
no caption text (for the manuscript, where the caption is in LaTeX).

Error is |q / q_FEATFLOW,lv6 - 1| in percent. The grey band is the FEATFLOW level-5 against
level-6 difference: errors inside it are below the uncertainty of the reference. A point below the
plotted range is drawn on the lower edge as a triangle. Wall time is the mean over the repetitions
that were not overlapped by other GPU jobs (see CONTAMINATED files).

Needs numpy and matplotlib (python3 -m venv venv && venv/bin/pip install numpy matplotlib).
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FixedLocator, FuncFormatter, NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))


def load_module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


ct, et = load_module("convergence_table"), load_module("efficiency_table")

# reference palette, light mode: categorical slots 1-2, text and grid tokens (as plot_convergence.py)
SERIES = {"uniform": "#2a78d6", "adaptive": "#eb6834"}
MARKER = {"uniform": "o", "adaptive": "s"}   # second encoding besides colour
TEXT, MUTED, GRID, BAND = "#0b0b0b", "#52514e", "#e4e3df", "#ecebe7"
# label positions (points), chosen to stay clear of the lines
LABEL_OFFSET = {"U3": (8, 4), "U4": (-8, 9), "U5": (8, 6), "A1": (8, 6), "A2": (-6, -16), "A3": (-6, 10)}
LABEL_OFFSET_WALL = {"U3": (8, 4), "U4": (-22, -14), "U5": (8, 6), "A1": (8, 6), "A2": (8, 6), "A3": (-6, 10)}   # bottom row
PANELS = [("cd_max", "C$_{D,\\max}$"), ("cl_max", "C$_{L,\\max}$"), ("St", "St"), ("dp_half", "$\\Delta p$")]


def main(argv):
    out, root = argv[0], argv[1]
    groups = {"uniform": ["U3", "U4", "U5"], "adaptive": ["A1", "A2", "A3"]}
    for flag, g in (("--uniform", "uniform"), ("--amr", "adaptive")):
        if flag in argv:
            i = argv.index(flag) + 1
            groups[g] = []
            while i < len(argv) and not argv[i].startswith("--"):
                groups[g].append(argv[i])
                i += 1
    rows = et.collect(root, [c for g in groups.values() for c in g])

    split = "--split" in argv
    fs = 1.35 if split else 1.0   # type scale
    def draw(axes_for, rows_x):
        for col, (q, label) in enumerate(PANELS):
            floor = abs(ct.FEATFLOW_LV5[q] / ct.FEATFLOW[q] - 1) * 100
            ylo, yhi = floor / 5.0, 3.0
            yclamp = ylo * 1.5   # points below the range sit just above the lower edge
            for row, (xkey, xlabel) in rows_x:
                ax = axes_for(row, col)
                ax.axhspan(ylo, floor, color=BAND, lw=0, zorder=0)
                for g, cases in groups.items():
                    pts = []
                    for c in cases:
                        r = rows.get(c)
                        if r is None:
                            continue
                        x = r["nodes"] if xkey == "nodes" else (sum(r["walls"]) / len(r["walls"]) if r["walls"] else None)
                        if x is None:
                            continue
                        pts.append((c, x, abs(r["err"][q])))
                    xs = [p[1] for p in pts]
                    ys = [max(p[2], yclamp) for p in pts]
                    ax.loglog(xs, ys, "-", color=SERIES[g], lw=2, zorder=2)
                    for c, x, e in pts:
                        clipped = e < yclamp
                        ax.loglog([x], [max(e, yclamp)], "v" if clipped else MARKER[g], color=SERIES[g], ms=8,
                                  mec="white", mew=1.5, zorder=3, label="_nolegend_")
                        if col == 0:   # direct labels on the first panel only
                            ax.annotate(c, (x, max(e, yclamp)), textcoords="offset points",
                                        xytext=(LABEL_OFFSET if xkey == "nodes" else LABEL_OFFSET_WALL)[c],
                                        color=MUTED, fontsize=8 * fs)
                    ax.loglog([], [], "-" + MARKER[g], color=SERIES[g], lw=2, ms=8, mec="white", mew=1.5, label=g)
                if xkey == "nodes":   # slope -1 guide (second order in h), anchored at the coarsest uniform point
                    ux = [rows[c]["nodes"] for c in groups["uniform"] if c in rows]
                    uy = [abs(rows[c]["err"][q]) for c in groups["uniform"] if c in rows]
                    if ux:
                        x0, x1 = ux[0], rows[groups["adaptive"][-1]]["nodes"] if groups["adaptive"][-1] in rows else ux[-1]
                        ax.loglog([x0, x1], [uy[0], uy[0] * x0 / x1], ":", color=MUTED, lw=1.2, zorder=1)
                ax.set_ylim(ylo, yhi)
                ax.set_title(label if row == 0 else "", color=TEXT, fontsize=12 * fs)
                ax.grid(True, which="major", color=GRID, lw=0.8)
                ax.tick_params(colors=MUTED, labelsize=9 * fs)
                ticks = [3e4, 1e5, 3e5] if xkey == "nodes" else [50, 100, 300, 1000]
                ax.xaxis.set_major_locator(FixedLocator(ticks))
                ax.xaxis.set_major_formatter(FuncFormatter(lambda v, _: f"{v:g}"))
                ax.xaxis.set_minor_formatter(NullFormatter())
                ax.yaxis.set_minor_formatter(NullFormatter())
                for s in ax.spines.values():
                    s.set_color(GRID)
                ax.set_xlabel(xlabel, color=MUTED, fontsize=10 * fs)
                if col % (2 if split else 4) == 0:
                    ax.set_ylabel("relative error vs FEATFLOW lv6 [%]", color=MUTED, fontsize=10 * fs)

    xrows = (("nodes", "mesh nodes"), ("wall", "wall time [s]"))
    if split:
        stem, ext = os.path.splitext(out)
        for xr in xrows:
            fig, axes = plt.subplots(2, 2, figsize=(9, 7.4))
            draw(lambda row, col: axes[col // 2, col % 2], [(0, xr)])
            axes[0, 0].legend(frameon=False, fontsize=11, labelcolor=TEXT, loc="lower left")
            if xr[0] == "nodes":
                axes[1, 1].text(0.98, 0.04, "dotted: slope $-1$ ($h^2$)", transform=axes[1, 1].transAxes,
                                ha="right", va="bottom", color=MUTED, fontsize=10)
            for ax in axes.flat:
                ax.set_title(ax.get_title(), color=TEXT, fontsize=14)
            fig.tight_layout()
            name = f"{stem}_{xr[0]}{ext}"
            fig.savefig(name, dpi=160, facecolor="white" if ext == ".pdf" else "#fcfcfb")
            print("wrote", name)
        return
    fig, axes = plt.subplots(2, 4, figsize=(13, 6.6))
    draw(lambda row, col: axes[row, col], list(enumerate(xrows)))
    axes[0, 0].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left")
    axes[0, 3].text(0.98, 0.04, "dotted: slope $-1$ ($h^2$)", transform=axes[0, 3].transAxes,
                    ha="right", va="bottom", color=MUTED, fontsize=8)
    # caption from the data: number of valid (not CONTAMINATED) repetitions per case
    nrep = {c: len(r["walls"]) for c, r in rows.items()}
    common = max(set(nrep.values()), key=list(nrep.values()).count)
    odd = ", ".join(f"{c}: {n}" for c, n in nrep.items() if n != common)
    reps = "one run per case" if common == 1 else f"mean of {common} repetitions"
    fig.text(0.5, 0.004,
             "DFG 2D-2, $\\Delta t$ = 1 ms, statistics over $t\\geq 6.5$ s. Grey band: FEATFLOW level 5 vs level 6 (reference uncertainty).\n"
             "Triangle: error below the plotted range. C$_{L,\\max}$ includes the ~0.2% time-step error.\n"
             f"Wall times: idle GPU, {reps}" + (f" ({odd} runs)." if odd else "."),
             ha="center", va="bottom", color=MUTED, fontsize=8)
    fig.tight_layout(rect=(0, 0.07, 1, 1))
    fig.savefig(out, dpi=160, facecolor="white" if out.endswith(".pdf") else "#fcfcfb")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
