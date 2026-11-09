#!/usr/bin/env python3
"""
Error against wall time for the DFG 2D-2 timing campaign (scripts/timing.sh).

    efficiency_table.py <timing_dir> U3 U4 U5 A1 A2 A3

For every case, all repetitions <case>_r<k> that finished are read. A repetition with a
CONTAMINATED file (another job shared the GPU) contributes its errors but not its wall time.
Errors are relative to FEATFLOW level 6, evaluated over t >= 6.5 s by convergence_table.load.
No third-party packages are needed.
"""
import glob
import importlib.util
import os
import re
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
spec = importlib.util.spec_from_file_location("convergence_table", os.path.join(HERE, "convergence_table.py"))
ct = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ct)

KEYS = ["cd_max", "cl_max", "St", "dp_half"]


def collect(root, cases):
    """{case: dict(nodes, elems, walls [s], amr [%], err {key: signed % vs FEATFLOW lv6})}"""
    out = {}
    for c in cases:
        dirs = sorted(d for d in glob.glob(os.path.join(root, f"{c}_r*")) if os.path.exists(d + "/done"))
        if not dirs:
            continue
        st = ct.load(dirs[0], 6.5)
        walls, amr = [], []
        for d in dirs:
            if os.path.exists(d + "/CONTAMINATED"):
                continue
            m = re.search(r"\[Timing\] ([0-9.]+) s wall, ([0-9.]+) s in AMR", open(d + "/run.log", errors="replace").read())
            if m:
                walls.append(float(m.group(1)))
                amr.append(float(m.group(2)) / float(m.group(1)) * 100)
        out[c] = dict(nodes=st["nodes"], elems=st["elements"], walls=walls, amr=amr,
                      err={k: (st[k] / ct.FEATFLOW[k] - 1) * 100 for k in KEYS})
    return out


def main(argv):
    root, cases = argv[0], argv[1:]
    print(f"{'case':5s} {'nodes':>8s} {'elems':>8s} {'wall [s]':>16s} {'AMR %':>6s} | "
          + " ".join(f"{ct.LABELS[k] + ' [%]':>14s}" for k in KEYS))
    for c, r in collect(root, cases).items():
        w = "not measured" if not r["walls"] else "/".join(f"{x:.0f}" for x in r["walls"])
        a = "-" if not r["amr"] or r["amr"][0] == 0 else f"{sum(r['amr']) / len(r['amr']):.0f}"
        print(f"{c:5s} {r['nodes']:8.0f} {r['elems']:8.0f} {w:>16s} {a:>6s} | "
              + " ".join(f"{r['err'][k]:+14.3f}" for k in KEYS))


if __name__ == "__main__":
    main(sys.argv[1:])
