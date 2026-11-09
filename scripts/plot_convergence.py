#!/usr/bin/env python3
"""
Relative error against the FEATFLOW level-6 reference versus number of mesh nodes, for the
uniform (U*) and adaptive (A*) runs of the spatial convergence study.

    plot_convergence.py <out.png> --uniform U3 U4 U5 --amr A1 A2 A3 [--t-start 6.5] [--ref featflow|extrapolated]

--ref featflow (default): error against FEATFLOW level 6 (includes our time-step error).
--ref extrapolated: error against the Richardson extrapolation of the uniform series at the same
time step (spatial self-convergence; falls back to the finest uniform run if non-monotone).

Needs numpy and matplotlib. Small multiples, one panel per quantity, one shared log-log
scale per panel; slope-2 guide (error ~ h^2 ~ N^-1).
"""
import importlib.util
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("convergence_table", os.path.join(HERE, "convergence_table.py"))
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

# reference palette, light mode: categorical slots 1-2, text and grid tokens
SERIES = {"uniform": "#2a78d6", "adaptive": "#eb6834"}
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
REF_MODE = "featflow"
PANELS = [("cd_max", "C$_{D,\\max}$"), ("cl_max", "C$_{L,\\max}$"), ("St", "St"), ("dp_half", "$\\Delta p$")]


def parse(argv):
    out, groups, cur, t_start = argv[0], {"uniform": [], "adaptive": []}, None, 6.5
    global REF_MODE
    i = 1
    while i < len(argv):
        a = argv[i]
        if a == "--uniform":
            cur = "uniform"
        elif a == "--amr":
            cur = "adaptive"
        elif a == "--ref":
            REF_MODE = argv[i + 1]
            i += 1
        elif a == "--t-start":
            t_start = float(argv[i + 1])
            i += 1
        else:
            groups[cur].append(a)
        i += 1
    return out, groups, t_start


def main(argv):
    out, groups, t_start = parse(argv)
    data = {g: [ct.load(d, t_start) for d in dirs] for g, dirs in groups.items() if dirs}
    fig, axes = plt.subplots(1, 4, figsize=(13, 3.4), sharex=True)
    for ax, (q, label) in zip(axes, PANELS):
        ref = ct.FEATFLOW[q]
        if REF_MODE == "extrapolated":
            u = data["uniform"]
            ref = u[-1][q]
            if len(u) >= 3:
                a, b, c = u[-3:]
                r12 = (b["nodes"] / a["nodes"]) ** 0.5
                r23 = (c["nodes"] / b["nodes"]) ** 0.5
                p = ct.observed_order(a[q], b[q], c[q], r12, r23)
                if p is not None:
                    ref = c[q] + (c[q] - b[q]) / (r23 ** p - 1.0)
        for g, runs in data.items():
            n = [r["nodes"] for r in runs]
            e = [abs(r[q] - ref) / abs(ref) for r in runs]
            ax.loglog(n, e, "-o", color=SERIES[g], lw=2, ms=8, mec="white", mew=1.5, label=g)
        # slope -1 guide (second order in h)
        allx = [r["nodes"] for runs in data.values() for r in runs]
        alle = [abs(r[q] - ref) / abs(ref) for runs in data.values() for r in runs]
        x0, x1 = min(allx), max(allx)
        y0 = max(alle)
        ax.loglog([x0, x1], [y0, y0 * x0 / x1], ":", color=MUTED, lw=1.2)
        ax.set_title(label, color=TEXT, fontsize=12)
        ax.grid(True, which="major", color=GRID, lw=0.8)
        ax.tick_params(colors=MUTED, labelsize=9)
        ax.xaxis.set_minor_formatter(NullFormatter())
        for s in ax.spines.values():
            s.set_color(GRID)
        ax.set_xlabel("mesh nodes", color=MUTED, fontsize=10)
    axes[0].set_ylabel("relative error vs " + ("FEATFLOW" if REF_MODE == "featflow" else "extrapolated uniform"),
                       color=MUTED, fontsize=10)
    axes[0].legend(frameon=False, fontsize=9, labelcolor=TEXT, loc="lower left")
    axes[-1].text(0.98, 0.04, "dotted: slope $-1$ ($h^2$)", transform=axes[-1].transAxes,
                  ha="right", va="bottom", color=MUTED, fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160, facecolor="#fcfcfb")
    print("wrote", out)


if __name__ == "__main__":
    main(sys.argv[1:])
