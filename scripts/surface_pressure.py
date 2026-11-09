#!/usr/bin/env python3
"""
Pressure along the cylinder surface from a legacy-ASCII solver VTK, and a node-to-node
oscillation measure: max and mean |p_i - (p_{i-1} + p_{i+1}) / 2| over surface nodes ordered
by angle (a smooth P1 trace gives O(h^2) values; a spurious pressure mode gives O(1)).

    surface_pressure.py <file.vtk> [--dump out.csv] [--extract]   (--extract: write the cylinder
    nodes to cylinder_wall_pressure.csv next to the VTK; read() falls back to it without the VTK)
No third-party packages are needed.
"""
import math
import os
import sys


WALL_CSV = "cylinder_wall_pressure.csv"   # x, y, p of the cylinder nodes, written by --extract


def read(path):
    """Points and pressure from a legacy VTK file; if it is absent (the public data set omits the
    VTK files), the cylinder-wall extract next to it, which holds only the surface nodes."""
    if not os.path.exists(path):
        with open(os.path.join(os.path.dirname(path), WALL_CSV)) as f:
            rows = [l.split(",") for l in f.read().split("\n")[1:] if l]
        return [(float(r[0]), float(r[1])) for r in rows], [float(r[2]) for r in rows]
    with open(path) as f:
        lines = f.read().split("\n")
    i = next(k for k, l in enumerate(lines) if l.startswith("POINTS"))
    n = int(lines[i].split()[1])
    pts = [tuple(map(float, lines[i + 1 + k].split()[:2])) for k in range(n)]
    j = next(k for k, l in enumerate(lines) if l.startswith("SCALARS pressure")) + 2
    p = [float(lines[j + k]) for k in range(n)]
    return pts, p


def main(argv):
    pts, p = read(argv[0])
    if "--extract" in argv:
        with open(os.path.join(os.path.dirname(argv[0]), WALL_CSV), "w") as f:
            f.write("x,y,p\n")
            for (x, y), v in zip(pts, p):
                if abs(math.hypot(x - 0.2, y - 0.2) - 0.05) < 1e-6:
                    f.write(f"{x!r},{y!r},{v!r}\n")
    cyl = sorted((math.atan2(y - 0.2, x - 0.2), p[k]) for k, (x, y) in enumerate(pts)
                 if abs(math.hypot(x - 0.2, y - 0.2) - 0.05) < 1e-6)
    m = len(cyl)
    osc = [abs(cyl[i][1] - 0.5 * (cyl[i - 1][1] + cyl[(i + 1) % m][1])) for i in range(m)]
    print(f"{m} surface nodes | oscillation max {max(osc):.4f} mean {sum(osc) / m:.4f} | "
          f"p range [{min(c[1] for c in cyl):.4f}, {max(c[1] for c in cyl):.4f}]")
    if "--dump" in argv:
        with open(argv[argv.index("--dump") + 1], "w") as f:
            f.write("theta,p\n")
            for t, v in cyl:
                f.write(f"{t:.8f},{v:.8f}\n")


if __name__ == "__main__":
    main(sys.argv[1:])
