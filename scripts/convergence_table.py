#!/usr/bin/env python3
"""
Convergence table and Richardson extrapolation for DFG 2D-2 quantities.

Each run directory must contain <prefix>_forces.csv. The quantities are evaluated over
t >= t_start (default 6.5 s, the periodic state of an 8 s run) by benchmark_stats.py; the mesh
size is the mean number of nodes over the same window. Runs are listed coarse to fine.

    convergence_table.py [--t-start S] dir1 dir2 dir3 [dir4 ...]

For three consecutive levels with values q1, q2, q3 and refinement ratios r12 = h1/h2,
r23 = h2/h3 (h ~ N^(-1/2) from the mean number of nodes after t = 2 s, so it also applies to
adaptive meshes), q_k = q* + C h_k^p gives
    (q2 - q1) / (q3 - q2) = (r12^p - 1) / (1 - r23^(-p)),
which is solved for the observed order p; the extrapolated value is
    q* = q3 + (q3 - q2) / (r23^p - 1).
Non-monotone sequences get no order.
No third-party packages are needed.
"""
import csv
import math
import os
import sys

QUANTITIES = ["cd_max", "cd_min", "cl_max", "cl_min", "St", "dp_half"]
LABELS = {"cd_max": "C_D,max", "cd_min": "C_D,min", "cl_max": "C_L,max", "cl_min": "C_L,min",
          "St": "St", "dp_half": "dp(t0+1/2f)"}
# FEATFLOW Q2/P1disc level 6, dt = 1/1600 (2014 reference data, evaluated with the same
# definitions by scripts/benchmark_stats.py)
FEATFLOW = {"cd_max": 3.22739, "cd_min": 3.16426, "cl_max": 0.98658, "cl_min": -1.02129,
            "St": 0.30184, "dp_half": 2.48520}
# FEATFLOW level 5 (same time step), to show how converged the reference itself is
FEATFLOW_LV5 = {"cd_max": 3.22662, "cd_min": 3.16351, "cl_max": 0.98620, "cl_min": -1.02093,
                "St": 0.30186, "dp_half": 2.48540}


def find(dirpath, suffix):
    for name in sorted(os.listdir(dirpath)):
        if name.endswith(suffix):
            return os.path.join(dirpath, name)
    raise SystemExit(f"no *{suffix} in {dirpath}")


def _stats_module():
    import importlib.util
    path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "benchmark_stats.py")
    spec = importlib.util.spec_from_file_location("benchmark_stats", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def load(dirpath, t_start):
    """DFG quantities over [t_start, end] from the force history, plus the mean mesh size."""
    bs = _stats_module()
    forces = find(dirpath, "_forces.csv")
    stats = bs.stats(*bs.read_solver_csv(forces), t_start=t_start)
    nodes, elems, n = 0.0, 0.0, 0
    with open(forces) as f:
        for row in csv.DictReader(f):
            if float(row["time"]) >= t_start:
                nodes += float(row["nodes"])
                elems += float(row["elements"])
                n += 1
    stats["nodes"] = nodes / max(n, 1)
    stats["elements"] = elems / max(n, 1)
    return stats


def observed_order(q1, q2, q3, r12, r23):
    d12, d23 = q2 - q1, q3 - q2
    if r12 <= 1.0 + 1e-9 or r23 <= 1.0 + 1e-9:
        return None   # not a refinement sequence
    if d23 == 0.0 or d12 / d23 <= 0.0:
        return None
    target = d12 / d23
    g = lambda p: (r12 ** p - 1.0) / (1.0 - r23 ** (-p)) - target
    lo, hi = 1e-3, 12.0
    if g(lo) * g(hi) > 0:
        return None
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if g(lo) * g(mid) <= 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def main(argv):
    t_start = 6.5
    if "--t-start" in argv:
        k = argv.index("--t-start")
        t_start = float(argv[k + 1])
        argv = argv[:k] + argv[k + 2:]
    dirs = [a for a in argv if not a.startswith("--")]
    runs = [load(d, t_start) for d in dirs]
    print(f"statistics over t >= {t_start} s (periodic state)")
    names = [os.path.basename(os.path.normpath(d)) for d in dirs]
    w = max(len(n) for n in names) + 2
    print(f"{'run':<{w}}{'nodes':>9}{'elems':>9}  " + "".join(f"{LABELS[q]:>13}" for q in QUANTITIES))
    for n, r in zip(names, runs):
        print(f"{n:<{w}}{r['nodes']:9.0f}{r['elements']:9.0f}  " + "".join(f"{r[q]:13.5f}" for q in QUANTITIES))
    print(f"{'FEATFLOW lv5':<{w}}{'':>9}{'':>9}  " + "".join(f"{FEATFLOW_LV5[q]:13.5f}" for q in QUANTITIES))
    print(f"{'FEATFLOW lv6':<{w}}{'':>9}{'':>9}  " + "".join(f"{FEATFLOW[q]:13.5f}" for q in QUANTITIES))
    print()
    print("relative error vs FEATFLOW lv6 [%]")
    for n, r in zip(names, runs):
        print(f"{n:<{w}}{'':>18}  " + "".join(f"{100 * (r[q] - FEATFLOW[q]) / abs(FEATFLOW[q]):13.3f}" for q in QUANTITIES))
    if len(runs) >= 3:
        print()
        print("observed order p and Richardson extrapolation (last three runs)")
        a, b, c = runs[-3:]
        r12 = math.sqrt(b["nodes"] / a["nodes"])
        r23 = math.sqrt(c["nodes"] / b["nodes"])
        print(f"  effective ratios r12 = {r12:.3f}, r23 = {r23:.3f}")
        for q in QUANTITIES:
            p = observed_order(a[q], b[q], c[q], r12, r23)
            if p is None:
                print(f"  {LABELS[q]:12s} non-monotone: {a[q]:.5f} {b[q]:.5f} {c[q]:.5f}")
                continue
            ext = c[q] + (c[q] - b[q]) / (r23 ** p - 1.0)
            print(f"  {LABELS[q]:12s} p = {p:5.2f}   extrapolated {ext:.5f}   "
                  f"(FEATFLOW {FEATFLOW[q]:.5f}, diff {100 * (ext - FEATFLOW[q]) / abs(FEATFLOW[q]):+.3f} %)")


if __name__ == "__main__":
    main(sys.argv[1:])
