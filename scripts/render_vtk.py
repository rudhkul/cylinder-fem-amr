#!/usr/bin/env python3
"""Render a cylinder_fem_amr VTK snapshot to a 4-panel PNG:
vorticity | AMR level + vorticity contours | log10 refinement score | mesh zoom near the cylinder.

Usage: python3 -I render_vtk.py <file.vtk> <out.png> [title]
(-I avoids a user-site NumPy 2.x that breaks the system matplotlib on the original machine)
"""
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.tri as mtri


def read_vtk(fn):
    """Minimal reader for the ASCII legacy VTK written by write_vtk_unstructured()."""
    lines = open(fn).read().split("\n")
    d, i = {}, 0
    while i < len(lines):
        s = lines[i].split()
        if not s:
            i += 1
            continue
        if s[0] == "POINTS":
            n = int(s[1])
            d["P"] = np.array([list(map(float, lines[i + 1 + k].split()))[:2] for k in range(n)])
            i += n + 1
            continue
        if s[0] == "CELLS":
            m = int(s[1])
            d["T"] = np.array([list(map(int, lines[i + 1 + k].split()))[1:] for k in range(m)])
            i += m + 1
            continue
        if s[0] == "VECTORS":
            n = len(d["P"])
            d[s[1]] = np.array([list(map(float, lines[i + 1 + k].split()))[:2] for k in range(n)])
            i += n + 1
            continue
        if s[0] == "SCALARS":
            name, vals = s[1], []
            i += 2
            while i < len(lines) and lines[i].strip() and not lines[i][0].isalpha():
                vals.append(float(lines[i]))
                i += 1
            d[name] = np.array(vals)
            continue
        i += 1
    return d


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    d = read_vtk(sys.argv[1])
    P, T = d["P"], d["T"]
    tri = mtri.Triangulation(P[:, 0], P[:, 1], T)
    lv = d["amr_level"]
    fig, ax = plt.subplots(4, 1, figsize=(18, 16), gridspec_kw={"height_ratios": [1, 1, 1, 1.6]})
    for a in ax[:3]:
        a.set_aspect("equal"); a.set_xlim(0, 2.2); a.set_ylim(0, 0.41)
    ax[0].tripcolor(tri, d["vorticity"], shading="gouraud", cmap="RdBu_r", vmin=-30, vmax=30)
    ax[0].set_title("vorticity")
    ax[1].tripcolor(tri, facecolors=lv, cmap="viridis", vmin=0, vmax=max(4, lv.max()))
    ax[1].tricontour(tri, d["vorticity"], levels=[-20, -8, -3, 3, 8, 20], colors="r", linewidths=0.6)
    ax[1].set_title("AMR level + vorticity contours")
    ax[2].tripcolor(tri, facecolors=np.log10(d["error_indicator"] + 1e-9), cmap="magma", vmin=-2, vmax=1)
    ax[2].set_title("log10 refinement score  (>0 refine, < log10(0.35) coarsen)")
    ax[3].tripcolor(tri, facecolors=lv, cmap="viridis", vmin=0, vmax=max(4, lv.max()), alpha=0.6)
    ax[3].triplot(tri, lw=0.15, color="k")
    ax[3].tricontour(tri, d["vorticity"], levels=[-20, -8, 8, 20], colors="r", linewidths=0.8)
    ax[3].set_xlim(0.05, 0.8); ax[3].set_ylim(0.0, 0.41); ax[3].set_aspect("equal")
    ax[3].set_title("mesh zoom")
    if len(sys.argv) > 3:
        fig.suptitle(sys.argv[3], fontsize=16, y=0.995)
    plt.savefig(sys.argv[2], dpi=100, bbox_inches="tight")


if __name__ == "__main__":
    main()
