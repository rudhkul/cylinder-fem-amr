#!/usr/bin/env python3
"""
LaTeX tables of the manuscript (paper/tables/*.tex) from the run directories:

  tab_mms       manufactured solution, runs/paper/mms_L1..5
  tab_2d2       DFG 2D-2 errors, nodes, wall: U3-U5 runs/v14/timing, A1-A3 runs/v17/timing
  tab_dt        DFG 2D-2 time step on A2 (runs/paper/d2_A2_*, runs/v17/timing/A2_r1) and U4
                (runs/review/d2_U4_*, runs/v14/timing/U4_r1)
  tab_2d3       DFG 2D-3 against John (2004): runs/paper/d3_*, runs/v17/d3/A2, A3, runs/review/d3_*_outdir*
  tab_ablation  ablations on A1: runs/v17/timing/A1_r1 and runs/paper/abl_*
  tab_reproj    re-projection after a remesh, 2x2 factorial on A1: runs/review/abl_projection_*
  tab_outlet    2D-2 with the parabola prescribed at the outlet: runs/review/d2_*_outdir
  tab_fixed     re-projection every 10 steps on the unchanged U3 mesh: runs/review2/U3_*
  tab_cost      wall-time breakdown of A3 and U5

    paper_tables.py [--runs runs] [--out paper/tables]

Also prints the numbers quoted in the text (orders, extrapolations, iteration counts).
No third-party packages are needed.
"""
import argparse
import csv
import importlib.util
import math
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(HERE)


def module(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(HERE, name + ".py"))
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


ct = module("convergence_table")
T_START = 6.5
JOHN = {"t_cd_max": 3.93625, "cd_max": 2.950921575, "t_cl_max": 5.693125, "cl_max": 0.47795, "dp_8": -0.1116}


def pct(v, ref):
    return 100.0 * (v - ref) / abs(ref)


def sci(x, nd=2):
    m, e = f"{x:.{nd}e}".split("e")
    return f"${m}\\times10^{{{int(e)}}}$"


def fmt_pct(x, nd=3):
    return f"{x:+.{nd}f}".replace("-", "$-$")


def wall(run):
    with open(os.path.join(run, "run.log"), errors="replace") as f:
        txt = f.read()
    m = re.search(r"\[Timing\] ([0-9.]+) s wall, ([0-9.]+) s in AMR", txt)
    return (float(m.group(1)), float(m.group(2))) if m else (float("nan"), float("nan"))


def timing_block(run):
    with open(os.path.join(run, "run.log"), errors="replace") as f:
        txt = f.read()
    out = {}
    m = re.search(r"step phases: momentum ([0-9.]+) s.*?\| pressure ([0-9.]+) s", txt)
    if m:
        out["momentum"], out["pressure"] = float(m.group(1)), float(m.group(2))
    for name, val in re.findall(r"^\s{4}([a-zA-Z:/+\- ]+?)\s{2,}([0-9.]+) s\s", txt, re.M):
        out[name.strip()] = float(val)
    return out


def mean_iters(run, t_start=T_START):
    its = []
    with open(os.path.join(run, "run.log"), errors="replace") as f:
        for line in f:
            m = re.match(r"\[Step\s+\d+ \| t = ([0-9.]+) s\].*Iters\(u,v,p\): \((\d+),(\d+),(\d+)\)", line)
            if m and float(m.group(1)) >= t_start:
                its.append(tuple(int(m.group(k)) for k in (2, 3, 4)))
    n = max(len(its), 1)
    return tuple(sum(i[k] for i in its) / n for k in range(3))


def stats_csv(run):
    with open(os.path.join(run, "cylinder_fem_amr_stats.csv")) as f:
        return {r["quantity"]: float(r["value"]) for r in csv.DictReader(f)}


def exists(*runs):
    miss = [r for r in runs if not os.path.exists(os.path.join(r, "cylinder_fem_amr_forces.csv"))
            and not os.path.exists(os.path.join(r, "done"))]
    if miss:
        print("  skipped, missing:", ", ".join(miss))
    return not miss


def write(out, name, body):
    with open(os.path.join(out, name + ".tex"), "w") as f:
        f.write(body)


# ------------------------------------------------------------------------------------------
def tab_mms(runs, out):
    dirs = [os.path.join(runs, f"paper/mms_L{L}") for L in range(1, 6)]
    files = [os.path.join(d, "cylinder_fem_amr_mms.csv") for d in dirs]
    if not all(os.path.exists(f) for f in files):
        print("  skipped, missing MMS runs")
        return
    rows = [list(csv.DictReader(open(f)))[0] for f in files]
    keys = ["err_u_L2", "err_u_H1", "err_p_L2", "err_p_cyl_max"]
    lines = [r"\begin{tabular}{rrr" + "r" * len(keys) + "}", r"\toprule",
             r"level & nodes & $\Delta t$ [s] & $\|\mathbf{u}-\mathbf{u}_h\|_{L^2}$ & "
             r"$|\mathbf{u}-\mathbf{u}_h|_{H^1}$ & $\|p-p_h\|_{L^2}$ & $\max_\Gamma|p-p_h|$ \\", r"\midrule"]
    for i, r in enumerate(rows):
        cells = [str(i + 1), f"{int(float(r['nodes'])):,}".replace(",", r"\,"), sci(float(r["dt"]))]
        for k in keys:
            e = float(r[k])
            c = sci(e)
            if i:
                e0, h0, h = float(rows[i - 1][k]), float(rows[i - 1]["h_max"]), float(r["h_max"])
                if h0 <= h:
                    raise SystemExit(f"MMS level {i + 1} is not finer than level {i} (h {h0} -> {h})")
                c += f" ({math.log(e0 / e) / math.log(h0 / h):.2f})"
            cells.append(c)
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_mms", "\n".join(lines))


def tab_2d2(runs, out):
    cases = [("U3", "v14/timing/U3_r1"), ("U4", "v14/timing/U4_r1"), ("U5", "v14/timing/U5_r1"),
             ("A1", "v17/timing/A1_r1"), ("A2", "v17/timing/A2_r1"), ("A3", "v17/timing/A3_r1")]
    dirs = [os.path.join(runs, d) for _, d in cases]
    if not exists(*dirs):
        return
    keys = ["cd_max", "cd_min", "cl_max", "cl_min", "St", "dp_half"]
    lines = [r"\begin{tabular}{lrrrrrrrr}", r"\toprule",
             r"mesh & nodes & $C_{D,\max}$ & $C_{D,\min}$ & $C_{L,\max}$ & $C_{L,\min}$ & St & $\Delta p$ & wall [s] \\",
             r"\midrule",
             r"FEATFLOW lv6 & & " + " & ".join(f"{ct.FEATFLOW[k]:.5f}" for k in keys) + r" & \\",
             r"\midrule"]
    loaded = {}
    for (name, _), d in zip(cases, dirs):
        st = ct.load(d, T_START)
        loaded[name] = st
        w, _ = wall(d)
        cells = [name, f"{int(round(st['nodes'])):,}".replace(",", r"\,")]
        cells += [fmt_pct(pct(st[k], ct.FEATFLOW[k])) for k in keys]
        cells.append(f"{w:.0f}")
        lines.append(" & ".join(cells) + r" \\")
    lines.append(r"\midrule")
    lines.append(r"FEATFLOW lv5 & & " + " & ".join(fmt_pct(pct(ct.FEATFLOW_LV5[k], ct.FEATFLOW[k])) for k in keys) + r" & \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_2d2", "\n".join(lines))
    for series in (["U3", "U4", "U5"], ["A1", "A2", "A3"]):
        s = [loaded[n] for n in series]
        r12 = math.sqrt(s[1]["nodes"] / s[0]["nodes"])
        r23 = math.sqrt(s[2]["nodes"] / s[1]["nodes"])
        for k in ("cd_max", "cl_max", "St"):
            q = [x[k] for x in s]
            p = ct.observed_order(*q, r12, r23)
            if p is None or (isinstance(p, float) and math.isnan(p)):
                print(f"  {series[0][0]} {k}: non-monotone {q}")
                continue
            ext = q[2] + (q[2] - q[1]) / (r23 ** p - 1)
            print(f"  {series[0][0]} {k}: order {p:.2f}, extrapolated {ext:.5f} ({pct(ext, ct.FEATFLOW[k]):+.3f}% vs FEATFLOW)")


def tab_dt(runs, out):
    series = [("A2", [("2", "paper/d2_A2_dt2"), ("1", "v17/timing/A2_r1"), ("0.5", "paper/d2_A2_dt05")]),
              ("U4", [("2", "review/d2_U4_dt2"), ("1", "v14/timing/U4_r1"), ("0.5", "review/d2_U4_dt05")])]
    keys = [("cd_max", r"$C_{D,\max}$"), ("cl_max", r"$C_{L,\max}$"), ("St", "St"), ("dp_half", r"$\Delta p$")]
    lines = [r"\begin{tabular}{ll" + "r" * len(keys) + "}", r"\toprule",
             r"mesh & $\Delta t$ [ms] & " + " & ".join(l for _, l in keys) + r" \\", r"\midrule"]
    for mesh, cases in series:
        dirs = [os.path.join(runs, d) for _, d in cases]
        if not exists(*dirs):
            continue
        st = [ct.load(d, T_START) for d in dirs]
        for (dtl, _), x in zip(cases, st):
            lines.append(f"{mesh} & {dtl} & " + " & ".join(f"{x[k]:.5f}" for k, _ in keys) + r" \\")
        d1 = {k: st[1][k] - st[0][k] for k, _ in keys}
        d2 = {k: st[2][k] - st[1][k] for k, _ in keys}
        lines.append(r" & difference 2 $\to$ 1 & " + " & ".join(sci(d1[k]) for k, _ in keys) + r" \\")
        lines.append(r" & difference 1 $\to$ 0.5 & " + " & ".join(sci(d2[k]) for k, _ in keys) + r" \\")
        ratio = {k: (d1[k] / d2[k] if d2[k] else float("nan")) for k, _ in keys}
        lines.append(r" & ratio & " + " & ".join(f"{ratio[k]:.2f}" for k, _ in keys) + r" \\")
        lines.append(r"\midrule")
        for k, _ in keys:
            r = ratio[k]
            p = math.log2(r) if r > 0 else float("nan")
            # time error at 1 ms from the two finer levels, assuming order p
            err1 = (st[1][k] - st[2][k]) * (2 ** p) / (2 ** p - 1) if r > 0 else float("nan")
            err2 = (st[1][k] - st[2][k]) * 4 / 3   # assuming order 2
            print(f"  {mesh} dt {k}: ratio {r:.2f} order {p:.2f}; time error at 1 ms {err1:+.3e} "
                  f"({100 * err1 / ct.FEATFLOW[k]:+.3f}% of FEATFLOW; order 2: {100 * err2 / ct.FEATFLOW[k]:+.3f}%); "
                  f"value at 0.5 ms {pct(st[2][k], ct.FEATFLOW[k]):+.3f}%")
    lines[-1] = r"\bottomrule"
    lines += [r"\end{tabular}", ""]
    write(out, "tab_dt", "\n".join(lines))


def tab_2d3(runs, out):
    cases = [("U3", "1", "paper/d3_U3"), ("U4", "1", "paper/d3_U4"), ("A2", "1", "v17/d3/A2"), ("A3", "1", "v17/d3/A3"),
             ("U4", "0.5", "paper/d3_U4_dt05"), ("U4", "0.25", "paper/d3_U4_dt025"),
             ("A2", "0.5", "paper/d3_A2_dt05"), ("A2", "0.25", "paper/d3_A2_dt025"),
             ("U4*", "1", "review/d3_U4_outdir"), ("U4*", "0.5", "review/d3_U4_outdir_dt05"),
             ("U4*", "0.25", "review/d3_U4_outdir_dt025"), ("A2*", "1", "review/d3_A2_outdir")]
    present = [(n, dtl, os.path.join(runs, d)) for n, dtl, d in cases
               if os.path.exists(os.path.join(runs, d, "cylinder_fem_amr_stats.csv"))]
    missing = [d for _, _, d in cases if not os.path.exists(os.path.join(runs, d, "cylinder_fem_amr_stats.csv"))]
    if missing:
        print("  missing (rows left out):", ", ".join(missing))
    keys = [("cd_max", r"$C_{D,\max}$"), ("t_cd_max", r"$t(C_{D,\max})$"), ("cl_max", r"$C_{L,\max}$"),
            ("t_cl_max", r"$t(C_{L,\max})$"), ("dp_8", r"$\Delta p(8\,\mathrm{s})$")]
    lines = [r"\begin{tabular}{lr" + "r" * len(keys) + "r}", r"\toprule",
             r"mesh & $\Delta t$ [ms] & " + " & ".join(l for _, l in keys) + r" & wall [s] \\", r"\midrule",
             r"John (2004) & & " + " & ".join(f"{JOHN[k]:.6g}".replace("-", "$-$") for k, _ in keys) + r" & \\", r"\midrule"]
    for i, (n, dtl, d) in enumerate(present):
        s = stats_csv(d)
        w, _ = wall(d)
        if n.endswith("*") and not present[i - 1][0].endswith("*"):
            lines.append(r"\midrule")
        lines.append(f"{n} & {dtl} & " + " & ".join(fmt_pct(pct(s[k], JOHN[k])) for k, _ in keys) + f" & {w:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_2d3", "\n".join(lines))
    for n in ("U4", "A2", "U4*"):
        v = {dtl: stats_csv(d)["cl_max"] for m, dtl, d in present if m == n}
        if len(v) == 3:
            a, b, c = v["1"], v["0.5"], v["0.25"]
            r = (b - a) / (c - b)
            ext = c + (c - b) / 3.0
            print(f"  2D-3 {n} C_L,max: {a:.6f} {b:.6f} {c:.6f}, ratio {r:.2f}, "
                  f"second-order extrapolation {ext:.5f} ({pct(ext, JOHN['cl_max']):+.3f}%)")


def glitch(run, t_start=T_START):
    rows = list(csv.DictReader(open(os.path.join(run, "cylinder_fem_amr_forces.csv"))))
    t = [float(r["time"]) for r in rows]
    y = [float(r["C_D"]) for r in rows]
    return max(abs(y[n + 1] - 2 * y[n] + y[n - 1]) for n in range(1, len(y) - 1) if t[n] >= t_start)


def glitch_split(run, t_start=T_START, after=2):
    """G split into steps near a remesh (centre n with m - 1 <= n <= m + after, m the first
    step on a new mesh, detected from the node/element counts) and all other steps."""
    rows = list(csv.DictReader(open(os.path.join(run, "cylinder_fem_amr_forces.csv"))))
    t = [float(r["time"]) for r in rows]
    y = [float(r["C_D"]) for r in rows]
    mesh = [(r["nodes"], r["elements"]) for r in rows]
    near = set()
    for m in range(1, len(rows)):
        if mesh[m] != mesh[m - 1]:
            near.update(range(m - 1, m + after + 1))
    g_r = g_o = 0.0
    for n in range(1, len(y) - 1):
        if t[n] < t_start:
            continue
        d = abs(y[n + 1] - 2 * y[n] + y[n - 1])
        if n in near:
            g_r = max(g_r, d)
        else:
            g_o = max(g_o, d)
    return g_r, g_o


def wall_osc(run):
    sp = module("surface_pressure")
    pts, p = sp.read(os.path.join(run, "cylinder_fem_amr_final.vtk"))
    cyl = sorted((math.atan2(y - 0.2, x - 0.2), p[k]) for k, (x, y) in enumerate(pts)
                 if abs(math.hypot(x - 0.2, y - 0.2) - 0.05) < 1e-6)
    m = len(cyl)
    return sum(abs(cyl[i][1] - 0.5 * (cyl[i - 1][1] + cyl[(i + 1) % m][1])) for i in range(m)) / m


def tab_ablation(runs, out):
    cases = [(r"\textbf{A1 (all choices)}", "v17/timing/A1_r1"),
             ("no keep radius", "paper/abl_keep_radius_0"),
             ("no minimum level", "paper/abl_min_level_0"),
             ("OSS factor 0.03", "paper/abl_oss_003"),
             ("rotational term from $\\mathbf{u}^*$", "paper/abl_rotvel"),
             ("SUPG $T = 0.5$ ms", "paper/abl_supg_T05"),
             ("SUPG $T = 2$ ms", "paper/abl_supg_T2"),
             ("no overlap ($K = 0$)", "paper/abl_overlap_0")]
    present = [(n, os.path.join(runs, d)) for n, d in cases if os.path.exists(os.path.join(runs, d, "done"))]
    missing = [d for _, d in cases if not os.path.exists(os.path.join(runs, d, "done"))]
    if missing:
        print("  missing (rows left out):", ", ".join(missing))
    keys = [("cd_max", r"$C_{D,\max}$"), ("cd_min", r"$C_{D,\min}$"), ("cl_max", r"$C_{L,\max}$"),
            ("St", "St"), ("dp_half", r"$\Delta p$")]
    lines = [r"\begin{tabular}{l" + "r" * (len(keys) + 3) + "}", r"\toprule",
             "variant & " + " & ".join(l for _, l in keys) + r" & $\mathcal{G}$ & wall osc. & wall [s] \\", r"\midrule"]
    for n, d in present:
        st = ct.load(d, T_START)
        g, o = glitch(d), wall_osc(d)
        w, _ = wall(d)
        wt = "--" if os.path.exists(os.path.join(d, "CONTAMINATED")) else f"{w:.0f}"   # shared GPU: no time
        lines.append(n + " & " + " & ".join(fmt_pct(pct(st[k], ct.FEATFLOW[k]), 2) for k, _ in keys)
                     + f" & {sci(g, 1)} & {o:.3f} & {wt}" + r" \\")
        print(f"  {d}: glitch {g:.2e}, wall osc {o:.4f}")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_ablation", "\n".join(lines))


def tab_cost(runs, out):
    a3, u5 = os.path.join(runs, "v17/timing/A3_r1"), os.path.join(runs, "v14/timing/U5_r1")
    if not exists(a3, u5):
        return
    ta, tu = timing_block(a3), timing_block(u5)
    wa, ama = wall(a3)
    wu, _ = wall(u5)
    ia, iu = mean_iters(a3), mean_iters(u5)
    exposed = ["indicators + download", "upload", "ILU colouring", "re-projection AMG setup",
               "re-projection solve", "overlap: transfer", "overlap: join wait"]
    hidden = ["adapt: look-ahead", "adapt: marking", "adapt: neighbours/buffer", "adapt: coarsening",
              "adapt: refinement", "extract", "CSR build", "validate_mesh"]
    rows = [("momentum predictor (assembly, ILU(0), BiCGSTAB)", ta["momentum"], tu["momentum"]),
            ("pressure (assembly, AMG setup, CG)", ta["pressure"], tu["pressure"]),
            ("AMR on the critical path", ama, 0.0)]
    rows += [(r"\quad " + k.replace("_", r"\_"), ta.get(k, 0.0), None) for k in exposed if k in ta]
    other_a = wa - ta["momentum"] - ta["pressure"] - ama
    other_u = wu - tu["momentum"] - tu["pressure"]
    rows.append(("other (forces, statistics, output)", other_a, other_u))
    rows.append((r"\textbf{wall time}", wa, wu))
    hid = sum(ta.get(k, 0.0) for k in hidden)
    lines = [r"\begin{tabular}{lrr}", r"\toprule", r"phase & A3 & U5 \\", r"\midrule"]
    for n, a, u in rows:
        lines.append(f"{n} & {a:.0f} & " + ("" if u is None else f"{u:.0f}") + r" \\")
    lines.append(r"\midrule")
    lines.append(r"host remesh in the worker thread (hidden) & " + f"{hid:.0f}" + r" & \\")
    lines.append(r"mean BiCGSTAB / CG iterations per step & " + f"{ia[0]:.0f} / {ia[2]:.0f} & {iu[0]:.0f} / {iu[2]:.0f}" + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_cost", "\n".join(lines))
    print(f"  A3 iters (u,v,p) {ia[0]:.1f} {ia[1]:.1f} {ia[2]:.1f}; U5 {iu[0]:.1f} {iu[1]:.1f} {iu[2]:.1f}; "
          f"A3 AMR share {100 * ama / wa:.0f}%; hidden worker {hid:.0f} s")


def tab_reproj(runs, out):
    cases = [("none", "--", "review/abl_projection_none"),
             ("$\\mathbf{u}^n$", "unstabilised", "paper/abl_projection_current"),
             ("$\\mathbf{u}^n$, $\\mathbf{u}^{n-1}$", "unstabilised", "review/abl_projection_both"),
             ("$\\mathbf{u}^n$", "scheme step", "review/abl_projection_scheme_current"),
             (r"\textbf{$\mathbf{u}^n$, $\mathbf{u}^{n-1}$ (A1)}", r"\textbf{scheme step}", "v17/timing/A1_r1"),
             ("$\\mathbf{u}^n$, $\\mathbf{u}^{n-1}$", "scheme step + next-step term", "review2/abl_reproject_shift")]
    present = [(a, b, os.path.join(runs, d)) for a, b, d in cases if os.path.exists(os.path.join(runs, d, "done"))]
    missing = [d for _, _, d in cases if not os.path.exists(os.path.join(runs, d, "done"))]
    if missing:
        print("  missing (rows left out):", ", ".join(missing))
    keys = [("cd_max", r"$C_{D,\max}$"), ("cd_min", r"$C_{D,\min}$"), ("cl_max", r"$C_{L,\max}$"),
            ("St", "St"), ("dp_half", r"$\Delta p$")]
    lines = [r"\begin{tabular}{ll" + "r" * (len(keys) + 2) + "}", r"\toprule",
             "re-projected & with & " + " & ".join(l for _, l in keys)
             + r" & $\mathcal{G}_{\rm r}$ & $\mathcal{G}_{\rm o}$ \\", r"\midrule"]
    for a, b, d in present:
        st = ct.load(d, T_START)
        g, (g_r, g_o) = glitch(d), glitch_split(d)
        lines.append(f"{a} & {b} & " + " & ".join(fmt_pct(pct(st[k], ct.FEATFLOW[k]), 2) for k, _ in keys)
                     + f" & {sci(g_r, 1)} & {sci(g_o, 1)}" + r" \\")
        print(f"  {d}: glitch {g:.2e} (near remesh {g_r:.2e}, elsewhere {g_o:.2e}); " + ", ".join(f"{k} {pct(st[k], ct.FEATFLOW[k]):+.3f}%" for k, _ in keys))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_reproj", "\n".join(lines))


def tab_outlet(runs, out):
    cases = [("U4", "$p$", "v14/timing/U4_r1"), ("U4", "parabola", "review/d2_U4_outdir"),
             ("A1", "$p$", "v17/timing/A1_r1"), ("A1", "parabola", "review/d2_A1_outdir")]
    present = [(a, b, os.path.join(runs, d)) for a, b, d in cases if os.path.exists(os.path.join(runs, d, "done"))]
    keys = [("cd_max", r"$C_{D,\max}$"), ("cd_min", r"$C_{D,\min}$"), ("cl_max", r"$C_{L,\max}$"),
            ("cl_min", r"$C_{L,\min}$"), ("St", "St"), ("dp_half", r"$\Delta p$")]
    lines = [r"\begin{tabular}{ll" + "r" * len(keys) + "}", r"\toprule",
             "mesh & outlet & " + " & ".join(l for _, l in keys) + r" \\", r"\midrule"]
    prev = None
    for a, b, d in present:
        st = ct.load(d, T_START)
        lines.append(f"{a} & {b} & " + " & ".join(fmt_pct(pct(st[k], ct.FEATFLOW[k])) for k, _ in keys) + r" \\")
        if prev and prev[0] == a:
            print(f"  2D-2 {a} outlet change [pp]: " + ", ".join(
                f"{k} {pct(st[k], ct.FEATFLOW[k]) - pct(prev[1][k], ct.FEATFLOW[k]):+.4f}" for k, _ in keys))
        prev = (a, st)
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_outlet", "\n".join(lines))


def tab_fixed(runs, out):
    cases = [("none (U3)", "--", "review2/U3_base"),
             ("$\\mathbf{u}^n$, $\\mathbf{u}^{n-1}$", "scheme step", "review2/U3_rt_scheme"),
             ("$\\mathbf{u}^n$, $\\mathbf{u}^{n-1}$", "scheme step + next-step term", "review2/U3_rt_scheme_shift"),
             ("$\\mathbf{u}^n$", "scheme step", "review2/U3_rt_scheme_cur"),
             ("$\\mathbf{u}^n$", "unstabilised", "review2/U3_rt_zero_cur"),
             ("$\\mathbf{u}^n$, $\\mathbf{u}^{n-1}$", "unstabilised", "review2/U3_rt_zero_both")]
    present = [(a, b, os.path.join(runs, d)) for a, b, d in cases if os.path.exists(os.path.join(runs, d, "done"))]
    if len(present) < len(cases):
        print("  missing runs")
    base = ct.load(present[0][2], T_START)
    keys = [("cd_max", r"$C_{D,\max}$"), ("cd_min", r"$C_{D,\min}$"), ("cl_max", r"$C_{L,\max}$"),
            ("St", "St"), ("dp_half", r"$\Delta p$")]
    lines = [r"\begin{tabular}{ll" + "r" * (len(keys) + 2) + "}", r"\toprule",
             r"re-projected & with & $\max|\mathbf{u}_{\rm new} - \mathbf{u}|$ & $\mathcal{G}$ & "
             + " & ".join(l for _, l in keys) + r" \\", r"\midrule"]
    for a, b, d in present:
        st = ct.load(d, T_START)
        du = [float(m.group(1)) for m in re.finditer(r"max\|u_new - u\| = ([0-9.e+-]+)",
                                                       open(os.path.join(d, "run.log"), errors="replace").read())]
        late = du[int(T_START / 0.01) - 1:]   # re-projections at t >= 6.5 s (every 10 steps of 1 ms)
        dus = sci(sum(late) / len(late), 1) if late else "--"
        lines.append(f"{a} & {b} & {dus} & {sci(glitch(d), 1)} & "
                     + " & ".join(fmt_pct(pct(st[k], ct.FEATFLOW[k]), 3) for k, _ in keys) + r" \\")
        print(f"  {d}: du {dus}, glitch {glitch(d):.2e}, max change vs U3 "
              f"{max(abs(pct(st[k], ct.FEATFLOW[k]) - pct(base[k], ct.FEATFLOW[k])) for k, _ in keys):.4f} pp")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    write(out, "tab_fixed", "\n".join(lines))


TABS = {"tab_mms": tab_mms, "tab_2d2": tab_2d2, "tab_dt": tab_dt, "tab_2d3": tab_2d3,
        "tab_ablation": tab_ablation, "tab_cost": tab_cost,
        "tab_reproj": tab_reproj, "tab_outlet": tab_outlet, "tab_fixed": tab_fixed}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", default=os.path.join(ROOT, "runs"))
    ap.add_argument("--out", default=os.path.join(ROOT, "paper/tables"))
    ap.add_argument("names", nargs="*")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    for n in a.names or TABS:
        print(n)
        TABS[n](a.runs, a.out)


if __name__ == "__main__":
    sys.exit(main())
