#!/usr/bin/env python3
"""Self-convergence table for a dt sweep (see dt_sweep.sh).

Usage: python3 -I dt_analyze.py <sweep_dir> <scheme> [dt1,dt2,...]
Reads <sweep_dir>/<scheme>_<dt>/cylinder_fem_amr_forces.csv, takes C_D,max, C_L,max and the
Strouhal number (upward zero crossings of C_L - mean) over t in [4, 6] s, and reports errors
against the smallest dt with observed orders log2(e(2dt) / e(dt)). The order in the last row
before the reference is biased upward (the reference is only 2x finer).
"""
import csv
import math
import sys


def stats(path, t0=4.0, diameter=0.1, u_mean=1.0):
    rows = list(csv.DictReader(open(path)))
    t = [float(r["time"]) for r in rows]
    cd = [float(r["C_D"]) for r in rows]
    cl = [float(r["C_L"]) for r in rows]
    i0 = next(i for i, tt in enumerate(t) if tt >= t0)
    T, CD, CL = t[i0:], cd[i0:], cl[i0:]
    mean = sum(CL) / len(CL)
    crossings = []
    for i in range(1, len(T)):
        a, b = CL[i - 1] - mean, CL[i] - mean
        if a < 0 <= b:
            crossings.append(T[i - 1] + (T[i] - T[i - 1]) * (-a) / (b - a))
    st = ((len(crossings) - 1) / (crossings[-1] - crossings[0]) * diameter / u_mean
          if len(crossings) > 2 else float("nan"))
    return {"CDmax": max(CD), "CLmax": max(CL), "St": st}


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    root, scheme = sys.argv[1], sys.argv[2]
    dts = ([float(x) for x in sys.argv[3].split(",")] if len(sys.argv) > 3
           else [0.004, 0.002, 0.001, 0.0005, 0.00025])
    res = {dt: stats(f"{root}/{scheme}_{dt}/cylinder_fem_amr_forces.csv") for dt in dts}
    ref = res[dts[-1]]
    keys = ["CDmax", "CLmax", "St"]
    print(f"{scheme.upper()}  (reference: dt = {dts[-1]})")
    print(f"{'dt':>8} " + " ".join(f"{k:>10} {'err':>9} {'order':>5}" for k in keys))
    prev = {}
    for dt in dts:
        row = f"{dt:8.5f} "
        for k in keys:
            v = res[dt][k]
            e = abs(v - ref[k]) if dt != dts[-1] else float("nan")
            o = math.log2(prev[k] / e) if k in prev and e > 0 and prev[k] > 0 else float("nan")
            row += f"{v:10.5f} {e:9.2e} {o:5.2f} "
            prev[k] = e
        print(row)


if __name__ == "__main__":
    main()
