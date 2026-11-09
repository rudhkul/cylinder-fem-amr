#!/usr/bin/env python3
"""
Sensitivity of the DFG pressure difference dp = p(0.15, 0.20) - p(0.25, 0.20) to how the two
surface pressures are read from the nodal field of a solver VTK (post-processing only).

    dp_probe_variants.py <file.vtk> [forces.csv]

Variants (surface nodes ordered by polar angle; front theta = pi, back theta = 0):
  edge   linear interpolation on the boundary edge (what the solver does)
  q<k>   least-squares quadratic in theta over the nearest 2k+1 surface nodes, evaluated at theta
The dp stored in forces.csv at the last step is printed for comparison. No third-party packages.
"""
import math
import sys


def read(path):
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("POINTS"))
    n = int(lines[i].split()[1])
    pts = [tuple(map(float, lines[i + 1 + k].split()[:2])) for k in range(n)]
    j = next(k for k, l in enumerate(lines) if l.startswith("SCALARS pressure")) + 2
    return pts, [float(lines[j + k]) for k in range(n)]


def solve3(A, b):
    M = [A[r][:] + [b[r]] for r in range(3)]
    for c in range(3):
        piv = max(range(c, 3), key=lambda r: abs(M[r][c]))
        M[c], M[piv] = M[piv], M[c]
        for r in range(3):
            if r != c:
                f = M[r][c] / M[c][c]
                M[r] = [M[r][k] - f * M[c][k] for k in range(4)]
    return [M[r][3] / M[r][r] for r in range(3)]


def trace(pts, p):
    return sorted((math.atan2(y - 0.2, x - 0.2), p[k]) for k, (x, y) in enumerate(pts)
                  if abs(math.hypot(x - 0.2, y - 0.2) - 0.05) < 1e-6)


def at(cyl, theta, k):
    m = len(cyl)
    # signed angular offset of every node from theta, wrapped to (-pi, pi]
    off = [((t - theta + math.pi) % (2 * math.pi)) - math.pi for t, _ in cyl]
    order = sorted(range(m), key=lambda i: abs(off[i]))
    if k == 0:   # edge interpolation between the nearest node on each side
        lo = max((i for i in range(m) if off[i] <= 0), key=lambda i: off[i])
        hi = min((i for i in range(m) if off[i] > 0), key=lambda i: off[i], default=lo)
        if off[lo] == 0:
            return cyl[lo][1]
        w = -off[lo] / (off[hi] - off[lo])
        return (1 - w) * cyl[lo][1] + w * cyl[hi][1]
    sel = order[:2 * k + 1]
    S = [[sum(off[i] ** (a + b) for i in sel) for b in range(3)] for a in range(3)]
    r = [sum(off[i] ** a * cyl[i][1] for i in sel) for a in range(3)]
    return solve3(S, r)[0]


def main(argv):
    pts, p = read(argv[0])
    cyl = trace(pts, p)
    print(f"{argv[0]}: {len(cyl)} surface nodes")
    for name, k in (("edge", 0), ("q2", 2), ("q4", 4), ("q8", 8)):
        pf, pb = at(cyl, math.pi, k), at(cyl, 0.0, k)
        print(f"  {name:5s} p_front {pf:+.6f}  p_back {pb:+.6f}  dp {pf - pb:.6f}")
    if len(argv) > 1:
        last = open(argv[1]).read().strip().split("\n")[-1].split(",")
        print(f"  csv   step {last[0]} t={last[1]}  dp {float(last[4]):.6f}")


if __name__ == "__main__":
    main(sys.argv[1:])
