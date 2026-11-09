#!/usr/bin/env python3
"""Compare two forces CSVs: meshes must be identical (nodes, elements per step); report the largest
force differences.   compare_runs.py <run_a_dir> <run_b_dir>"""
import csv, sys, os
def load(d):
    f = [x for x in os.listdir(d) if x.endswith('_forces.csv')][0]
    with open(os.path.join(d, f)) as fh:
        return list(csv.DictReader(fh))
a, b = load(sys.argv[1]), load(sys.argv[2])
n = min(len(a), len(b))
mesh_diff = [i for i in range(n) if (a[i]['nodes'], a[i]['elements']) != (b[i]['nodes'], b[i]['elements'])] \
    if 'nodes' in a[0] else None
print(f"steps {len(a)} / {len(b)}")
keys = [k for k in a[0] if k.lower().startswith(('c_d', 'c_l', 'cd', 'cl', 'delta'))] or list(a[0])[2:5]
if mesh_diff is None:
    ks = list(a[0].keys())
    print("columns:", ks)
else:
    print("mesh differs at", len(mesh_diff), "steps", ("(first %d)" % (mesh_diff[0] + 1)) if mesh_diff else "")
for k in keys:
    print(f"max |d {k}| = {max(abs(float(a[i][k]) - float(b[i][k])) for i in range(n)):.3e}")
