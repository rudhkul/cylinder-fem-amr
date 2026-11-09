#!/usr/bin/env python3
"""
Error table and observed orders for manufactured-solution runs (solver option --mms).

    mms_table.py run1_mms.csv run2_mms.csv ...      (coarse to fine)

Orders are log(e_k-1 / e_k) / log(h_k-1 / h_k) with h the maximum edge length.
No third-party packages are needed.
"""
import csv
import math
import sys

COLS = [("err_u_L2", "||u-u_h||_L2"), ("err_u_H1", "|u-u_h|_H1"), ("err_p_L2", "||p-p_h||_L2"),
        ("err_p_cyl_max", "max|p-p_h| cyl"), ("err_p_probe_max", "|p-p_h| probes")]

rows = [list(csv.DictReader(open(path)))[0] for path in sys.argv[1:]]
head = f"{'nodes':>9} {'h_max':>9} {'dt':>9}" + "".join(f" {label:>15} {'p':>5}" for _, label in COLS)
print(head)
for i, r in enumerate(rows):
    line = f"{int(float(r['nodes'])):9d} {float(r['h_max']):9.2e} {float(r['dt']):9.2e}"
    for key, _ in COLS:
        if key not in r:
            line += f" {'-':>15} {'':>5}"
            continue
        e = float(r[key])
        if i > 0 and key in rows[i - 1]:
            ratio = float(rows[i - 1]["h_max"]) / float(r["h_max"])
            p = math.log(float(rows[i - 1][key]) / e) / math.log(ratio)
            line += f" {e:15.3e} {p:5.2f}"
        else:
            line += f" {e:15.3e} {'':>5}"
    print(line)
