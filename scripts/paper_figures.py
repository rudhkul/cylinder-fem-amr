#!/usr/bin/env python3
"""
Figures of the manuscript that the older scripts do not make (paper/figures/*.pdf):

  fig_mesh_d3       DFG 2D-3, A2: AMR level and vorticity contours at t = 2, 4, 6, 8 s
  fig_mesh_zoom     DFG 2D-2, A2 at t = 8 s: mesh near the cylinder
  fig_d3_history    DFG 2D-3: C_D(t), C_L(t) on U4, A2, A3 against FEATFLOW, zooms at the maxima
  fig_cp            DFG 2D-2 at t = 8 s: wall pressure on A1 (OSS 0.3 and 0.03) and A3
  fig_remesh        DFG 2D-2, A1: drag around remeshes for the re-projection/keep-radius ablations

    paper_figures.py [--runs runs] [--out paper/figures] [name ...]

A figure whose input runs are missing is skipped with a message. Needs numpy and matplotlib.
"""
import argparse
import csv
import glob
import importlib.util
import math
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import BoundaryNorm, ListedColormap

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)
spec = importlib.util.spec_from_file_location("render_vtk", os.path.join(HERE, "render_vtk.py"))
rv = importlib.util.module_from_spec(spec)
spec.loader.exec_module(rv)
spec = importlib.util.spec_from_file_location("surface_pressure", os.path.join(HERE, "surface_pressure.py"))
sp = importlib.util.module_from_spec(spec)
spec.loader.exec_module(sp)

# reference palette, light mode: categorical slots 1-4, text and grid tokens, blue ordinal ramp
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
TEXT, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
RAMP = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]

plt.rcParams.update({
    "font.size": 8, "axes.titlesize": 8, "axes.labelsize": 8, "legend.fontsize": 7,
    "xtick.labelsize": 7, "ytick.labelsize": 7, "axes.edgecolor": MUTED, "axes.labelcolor": TEXT,
    "xtick.color": MUTED, "ytick.color": MUTED, "text.color": TEXT, "axes.grid": True,
    "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False, "axes.spines.right": False,
    "lines.linewidth": 1.2, "savefig.bbox": "tight", "savefig.dpi": 300, "legend.frameon": False,
})
COL_W, PAGE_W = 3.5, 7.2   # inches (Computers & Fluids single / double column)


def forces(run):
    f = os.path.join(run, "cylinder_fem_amr_forces.csv")
    if not os.path.exists(f):
        return None
    with open(f) as fh:
        rows = list(csv.DictReader(fh))
    return {k: np.array([float(r[k]) for r in rows]) for k in ("time", "C_D", "C_L", "nodes")}


# John (2004) reference maxima for DFG 2D-3: (time, value)
JOHN_2D3 = {"C_D": (3.93625, 2.950921575), "C_L": (5.693125, 0.47795)}


def need(*paths):
    miss = [p for p in paths if not os.path.exists(p)]
    if miss:
        print("  skipped, missing:", ", ".join(miss))
        return False
    return True


def snapshot(run, t, dt=1e-3):
    step = int(round(t / dt))
    c = sorted(glob.glob(os.path.join(run, f"*_{step}.vtk")) + glob.glob(os.path.join(run, f"*_{step:06d}.vtk")))
    if not c and abs(t - 8.0) < 1e-9:
        c = glob.glob(os.path.join(run, "*_final.vtk"))
    return c[0] if c else os.path.join(run, f"<snapshot t={t}>")


def level_plot(ax, d, xlim=(0, 2.2), ylim=(0, 0.41), edges=False, lmin=2, lmax=5):
    P, T = d["P"], d["T"]
    tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
    levels = list(range(lmin, lmax + 1))
    cmap = ListedColormap(RAMP[1:1 + len(levels)])
    norm = BoundaryNorm([l - 0.5 for l in levels] + [lmax + 0.5], cmap.N)
    pc = ax.tripcolor(tri, facecolors=np.clip(d["amr_level"], lmin, lmax), cmap=cmap, norm=norm,
                      edgecolors="none", rasterized=True)
    if edges:
        ax.triplot(tri, lw=0.12, color=TEXT, alpha=0.55, rasterized=True)
    w = d["vorticity"]
    ax.tricontour(tri, w, levels=[-20, -5], colors=[SERIES[1]], linewidths=0.5, linestyles="solid")
    ax.tricontour(tri, w, levels=[5, 20], colors=[TEXT], linewidths=0.5, linestyles="solid")
    ax.add_patch(plt.Circle((0.2, 0.2), 0.05, color="white", ec=MUTED, lw=0.6, zorder=5))
    ax.set_xlim(*xlim); ax.set_ylim(*ylim); ax.set_aspect("equal")
    ax.grid(False)
    return pc, levels


def fig_mesh_d3(runs, out):
    run = os.path.join(runs, "paper/snap_d3_A2")
    times = [2.0, 4.0, 6.0, 8.0]
    files = [snapshot(run, t) for t in times]
    if not need(*files):
        return
    fig, axs = plt.subplots(len(times), 1, figsize=(PAGE_W, 1.45 * len(times)), sharex=True)
    for ax, t, f in zip(axs, times, files):
        d = rv.read_vtk(f)
        pc, levels = level_plot(ax, d)
        ax.set_title(f"t = {t:g} s, inflow {1.5 * math.sin(math.pi * t / 8):.2f} m/s, "
                     f"{len(d['P']) / 1e3:.0f}k nodes", loc="left")
        ax.set_ylabel("y [m]")
    axs[-1].set_xlabel("x [m]")
    cb = fig.colorbar(pc, ax=axs, ticks=levels, shrink=0.6, pad=0.01)
    cb.set_label("AMR level"); cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, "fig_mesh_d3.pdf"))
    plt.close(fig)


def fig_mesh_zoom(runs, out):
    f = snapshot(os.path.join(runs, "paper/snap_d2_A2"), 8.0)
    if not need(f):
        return
    d = rv.read_vtk(f)
    box = (0.200, 0.215, 0.245, 0.256)   # top of the cylinder: the finest elements become visible
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W, 2.0), gridspec_kw={"width_ratios": [2.4, 1]})
    level_plot(axs[0], d, xlim=(0.0, 0.9))
    pc, levels = level_plot(axs[1], d, xlim=box[:2], ylim=box[2:], edges=True)
    for ln in axs[1].lines:
        ln.set_linewidth(0.25)
    axs[0].add_patch(plt.Rectangle((box[0], box[2]), box[1] - box[0], box[3] - box[2], fill=False, ec=TEXT, lw=0.8, zorder=6))
    axs[0].set_title(f"DFG 2D-2, A2, t = 8 s ({len(d['P']) / 1e3:.0f}k nodes)", loc="left")
    axs[1].set_title("mesh in the box", loc="left")
    for a in axs:
        a.set_xlabel("x [m]")
    axs[0].set_ylabel("y [m]")
    axs[1].set_xticks([0.200, 0.210])
    cb = fig.colorbar(pc, ax=axs, ticks=levels, shrink=0.8, pad=0.01)
    cb.set_label("AMR level"); cb.outline.set_visible(False)
    fig.savefig(os.path.join(out, "fig_mesh_zoom.pdf"))
    plt.close(fig)


def fig_d3_history(runs, out):
    cases = [("U4", os.path.join(runs, "paper/d3_U4")), ("A2", os.path.join(runs, "v17/d3/A2")),
             ("A3", os.path.join(runs, "v17/d3/A3"))]
    if not need(*[os.path.join(r, "cylinder_fem_amr_forces.csv") for _, r in cases]):
        return
    fig, axs = plt.subplots(2, 2, figsize=(PAGE_W, 4.2))
    zoom = {"C_D": (3.80, 4.08, 2.935, 2.955), "C_L": (5.665, 5.725, 0.462, 0.482)}
    for col, (key, lab) in enumerate([("C_D", "$C_D$"), ("C_L", "$C_L$")]):
        top, bot = axs[0, col], axs[1, col]
        for k, (name, run) in enumerate(cases):
            f = forces(run)
            top.plot(f["time"], f[key], color=SERIES[k], lw=0.9, label=name)
            bot.plot(f["time"], f[key], color=SERIES[k], lw=1.2, label=name)
        t_ref, v_ref = JOHN_2D3[key]
        bot.plot([t_ref], [v_ref], ls="none", marker="x", ms=6, mew=1.2, color=TEXT, label="John (2004)")
        x0, x1, y0, y1 = zoom[key]
        top.set_xlim(0, 8); top.set_ylim(-0.6 if key == "C_L" else -0.2, 3.2 if key == "C_D" else 0.6)
        bot.set_xlim(x0, x1); bot.set_ylim(y0, y1)
        top.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=TEXT, lw=0.6))
        top.set_ylabel(lab); bot.set_ylabel(lab); bot.set_xlabel("t [s]"); top.set_xlabel("t [s]")
        top.set_title(f"{lab} over the run", loc="left")
        bot.set_title(f"{lab} near its maximum (box above)", loc="left")
    axs[1, 0].legend(loc="lower center", ncol=2)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "fig_d3_history.pdf"))
    plt.close(fig)


def wall_pressure(vtk):
    pts, p = sp.read(vtk)   # full VTK, or the cylinder-wall extract next to it
    P, p = np.array(pts), np.array(p)
    r = np.hypot(P[:, 0] - 0.2, P[:, 1] - 0.2)
    on = np.abs(r - 0.05) < 1e-6
    th = np.degrees(np.arctan2(P[on, 1] - 0.2, -(P[on, 0] - 0.2)))   # 0 at the front stagnation point
    th = np.mod(th, 360.0)
    o = np.argsort(th)
    return th[o], p[on][o]


def fig_cp(runs, out):
    cases = [("A1, OSS factor 0.3", os.path.join(runs, "v17/timing/A1_r1/cylinder_fem_amr_final.vtk")),
             ("A1, OSS factor 0.03", os.path.join(runs, "paper/abl_oss_003/cylinder_fem_amr_final.vtk")),
             ("A3, OSS factor 0.3", os.path.join(runs, "v17/timing/A3_r1/cylinder_fem_amr_final.vtk"))]
    if not need(*[f if os.path.exists(f) else os.path.join(os.path.dirname(f), sp.WALL_CSV) for _, f in cases]):
        return
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W, 2.7), gridspec_kw={"width_ratios": [1.6, 1]})
    for k, (name, f) in enumerate(cases):
        th, p = wall_pressure(f)
        cp = 2.0 * p   # kinematic pressure, mean inflow velocity 1 m/s; p = 0 at the outlet
        for ax in axs:
            ax.plot(th, cp, color=SERIES[k], lw=1.0 if k != 1 else 0.8, label=name,
                    marker="o" if ax is axs[1] else None, ms=2.2, mew=0)
    axs[0].set_xlim(0, 360); axs[0].set_xticks(range(0, 361, 60))
    axs[1].set_xlim(100, 140)
    th1, p1 = wall_pressure(cases[1][1])
    th0, p0 = wall_pressure(cases[0][1])
    m = (th1 > 100) & (th1 < 140)
    m0 = (th0 > 100) & (th0 < 140)
    lo = 2 * min(p1[m].min(), p0[m0].min()); hi = 2 * max(p1[m].max(), p0[m0].max())
    axs[1].set_ylim(lo - 0.05 * (hi - lo), hi + 0.05 * (hi - lo))
    for ax in axs:
        ax.set_xlabel(r"$\theta$ [deg] from the front stagnation point")
    axs[0].set_ylabel(r"$C_p = 2p/\bar U^2$")
    axs[0].set_title("Wall pressure at t = 8 s", loc="left")
    axs[1].set_title("Zoom at the upper separation region", loc="left")
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    h, l = axs[0].get_legend_handles_labels()
    leg = fig.legend(h, l, loc="lower center", ncol=3, frameon=False)   # below the axes, clear of the curves
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    fig.savefig(os.path.join(out, "fig_cp.pdf"))
    plt.close(fig)


def fig_remesh(runs, out):
    cases = [("A1 (scheme re-projection, keep radius 0.25 m)", os.path.join(runs, "v17/timing/A1_r1")),
             ("no keep radius", os.path.join(runs, "paper/abl_keep_radius_0")),
             ("u$^n$ only, unstabilised projection", os.path.join(runs, "paper/abl_projection_current"))]
    if not need(*[os.path.join(r, "cylinder_fem_amr_forces.csv") for _, r in cases]):
        return
    fig, axs = plt.subplots(1, 2, figsize=(PAGE_W, 2.9), gridspec_kw={"width_ratios": [1.5, 1]})
    for k in reversed(range(len(cases))):   # reference run drawn last, on top
        name, run = cases[k]
        f = forces(run)
        m = (f["time"] >= 7.0) & (f["time"] <= 7.5)
        axs[0].plot(f["time"][m], f["C_D"][m], color=SERIES[k], lw=0.9, label=name)
        m = (f["time"] >= 7.100) & (f["time"] <= 7.135)
        axs[1].plot(f["time"][m], f["C_D"][m], color=SERIES[k], lw=0.8, marker="o", ms=2.5, mew=0, label=name)
    axs[0].set_xlabel("t [s]"); axs[0].set_ylabel("$C_D$")
    axs[0].set_title("Drag, DFG 2D-2, A1, $\\Delta t$ = 1 ms, remesh every 10 ms", loc="left")
    axs[1].set_xlabel("t [s]")
    axs[1].set_title("Zoom: one marker per time step", loc="left")
    h, l = axs[0].get_legend_handles_labels()
    leg = fig.legend(h[::-1], l[::-1], loc="lower center", ncol=3, fontsize=7, frameon=False)   # below the axes
    for line in leg.get_lines():
        line.set_linewidth(2.0)
    fig.tight_layout(rect=(0, 0.1, 1, 1))
    fig.savefig(os.path.join(out, "fig_remesh.pdf"))
    plt.close(fig)


FIGS = {"fig_mesh_d3": fig_mesh_d3, "fig_mesh_zoom": fig_mesh_zoom, "fig_d3_history": fig_d3_history,
        "fig_cp": fig_cp, "fig_remesh": fig_remesh}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper/figures"))
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in a.names or FIGS:
        print(n)
        FIGS[n](a.runs, a.out)


if __name__ == "__main__":
    sys.exit(main())
