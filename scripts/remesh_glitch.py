#!/usr/bin/env python3
"""
Size of the remesh-induced glitches in a force history: the second difference
|y[n+1] - 2 y[n] + y[n-1]| of C_D, C_L and delta_p, split by the position of step n relative
to the last remesh (k = n mod interval: k = 1 is the first step on a new mesh).

Usage: remesh_glitch.py <forces.csv> [--interval 10] [--t-start S]
"""
import csv
import sys

path = sys.argv[1]
interval, t_start = 10, 1.0
for i, a in enumerate(sys.argv):
    if a == "--interval":
        interval = int(sys.argv[i + 1])
    if a == "--t-start":
        t_start = float(sys.argv[i + 1])
rows = list(csv.DictReader(open(path)))
step = [int(r["step"]) for r in rows]
t = [float(r["time"]) for r in rows]
cols = {"C_D": [float(r["C_D"]) for r in rows], "C_L": [float(r["C_L"]) for r in rows],
        "delta_p": [float(r["delta_p"]) for r in rows]}
print(f"{'k':>3} " + " ".join(f"{c + ' max|d2|':>16}" for c in cols))
for k in range(interval):
    out = []
    for c, y in cols.items():
        m = 0.0
        for n in range(1, len(y) - 1):
            if t[n] < t_start or step[n] % interval != k:
                continue
            m = max(m, abs(y[n + 1] - 2.0 * y[n] + y[n - 1]))
        out.append(m)
    print(f"{k:>3} " + " ".join(f"{v:16.2e}" for v in out))
