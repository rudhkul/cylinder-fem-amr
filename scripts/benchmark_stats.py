#!/usr/bin/env python3
"""
DFG 2D-2 benchmark quantities from a force history, following Schaefer & Turek (1996):
C_D,max, C_L,max (and the minima), St = D f / U_mean with f from C_L, and
delta_p(t0 + 1/(2f)) with t0 the time of maximum lift.

Lift periods are delimited by upward crossings of the mean lift. Peaks are located with a parabola through the three samples around each discrete extremum,
so the result does not depend on where the time steps fall. Each quantity is evaluated for
every complete period in the window; the mean is reported, and the spread (max - min over
periods) shows how periodic the solution is.

Usage:
  benchmark_stats.py <forces.csv> [--t-start S] [--t-end S]          solver CSV (time,C_D,C_L,delta_p)
  benchmark_stats.py --featflow <bdforces_file> <pointvalues_file>     FEATFLOW reference logs
  add --json to print one JSON object instead of the table
No third-party packages are needed.
"""
import csv
import json
import sys

D, U_MEAN = 0.1, 1.0


def read_solver_csv(path):
    t, cd, cl, dp = [], [], [], []
    with open(path) as f:
        for row in csv.DictReader(f):
            t.append(float(row["time"]))
            cd.append(float(row["C_D"]))
            cl.append(float(row["C_L"]))
            dp.append(float(row["delta_p"]))
    return t, cd, cl, dp


def read_featflow(bdforces, pointvalues):
    t, cd, cl, dp = [], [], [], []
    with open(bdforces) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            t.append(float(v[1]))
            cd.append(float(v[3]))
            cl.append(float(v[4]))
    with open(pointvalues) as f:
        for line in f:
            if line.startswith("#") or not line.strip():
                continue
            v = line.split()
            # blocks of (x, y, type, deriv, value); points a1 = (0.15, 0.2), a2 = (0.25, 0.2)
            p1, p2 = float(v[6]), float(v[11])
            dp.append(p1 - p2)
    return t, cd, cl, dp


def refine(t, y, i):
    a, b, c = y[i - 1], y[i], y[i + 1]
    denom = a - 2.0 * b + c
    s = 0.5 * (a - c) / denom if denom != 0.0 else 0.0
    s = max(-0.5, min(0.5, s))
    return (t[i] + s * (t[i + 1] - t[i]), b - 0.25 * (a - c) * s)


def peaks(t, y, sign=+1):
    """(time, value) of interior local maxima of sign*y, refined by a parabola."""
    out = []
    for i in range(1, len(y) - 1):
        a, b, c = sign * y[i - 1], sign * y[i], sign * y[i + 1]
        if b >= a and b > c:
            denom = a - 2.0 * b + c
            s = 0.5 * (a - c) / denom if denom != 0.0 else 0.0   # offset in steps, |s| <= 0.5
            h = t[i + 1] - t[i]
            out.append((t[i] + s * h, sign * (b - 0.25 * (a - c) * s)))
    return out


def interp(t, y, tq):
    lo, hi = 0, len(t) - 1
    if tq <= t[0]:
        return y[0]
    if tq >= t[-1]:
        return y[-1]
    while hi - lo > 1:
        mid = (lo + hi) // 2
        if t[mid] <= tq:
            lo = mid
        else:
            hi = mid
    w = (tq - t[lo]) / (t[hi] - t[lo])
    return (1.0 - w) * y[lo] + w * y[hi]


def stats(t, cd, cl, dp, t_start=None, t_end=None):
    if t_start is not None or t_end is not None:
        a = t_start if t_start is not None else -1e300
        b = t_end if t_end is not None else 1e300
        idx = [i for i, ti in enumerate(t) if a <= ti <= b]
        t, cd, cl, dp = [t[i] for i in idx], [cd[i] for i in idx], [cl[i] for i in idx], [dp[i] for i in idx]

    # lift periods are delimited by upward crossings of the mean lift (robust to glitches);
    # the lift maximum inside each is refined by a parabola
    mean = sum(cl) / len(cl)
    up = [i for i in range(1, len(cl)) if cl[i - 1] < mean <= cl[i]]
    cl_max = []
    for k in range(len(up) - 1):
        best = max(range(up[k], up[k + 1]), key=lambda i: cl[i])
        if 0 < best < len(cl) - 1:
            cl_max.append(refine(t, cl, best))
    if len(cl_max) < 2:
        raise SystemExit("fewer than two lift maxima in the window")
    periods = [cl_max[k + 1][0] - cl_max[k][0] for k in range(len(cl_max) - 1)]
    res = {"periods": len(periods), "t_window": [t[0], t[-1]]}

    # per period [t0, t1] between successive lift maxima
    per = {"cd_max": [], "cd_min": [], "cl_max": [], "cl_min": [], "St": [], "dp_half": []}
    cd_max_pk, cd_min_pk = peaks(t, cd, +1), peaks(t, cd, -1)
    cl_min_pk = peaks(t, cl, -1)
    for k in range(len(periods)):
        t0, t1 = cl_max[k][0], cl_max[k + 1][0]
        f = 1.0 / (t1 - t0)
        inside = lambda pk: [v for (tp, v) in pk if t0 <= tp < t1]
        # C_D has two maxima per lift period (it oscillates at 2f): take the larger
        dmax, dmin, lmin = inside(cd_max_pk), inside(cd_min_pk), inside(cl_min_pk)
        if not dmax or not dmin or not lmin:
            continue
        per["cd_max"].append(max(dmax))
        per["cd_min"].append(min(dmin))
        per["cl_max"].append(cl_max[k][1])
        per["cl_min"].append(min(lmin))
        per["St"].append(D * f / U_MEAN)
        per["dp_half"].append(interp(t, dp, t0 + 0.5 / f))
    for key, vals in per.items():
        if vals:
            res[key] = sum(vals) / len(vals)
            res[key + "_spread"] = max(vals) - min(vals)
    res["cd_mean"] = 0.5 * (res["cd_max"] + res["cd_min"])
    res["cl_mean"] = 0.5 * (res["cl_max"] + res["cl_min"])
    return res


def main(argv):
    as_json = "--json" in argv
    argv = [a for a in argv if a != "--json"]
    if argv and argv[0] == "--featflow":
        data = read_featflow(argv[1], argv[2])
        rest = argv[3:]
    else:
        data = read_solver_csv(argv[0])
        rest = argv[1:]
    t_start = t_end = None
    for i, a in enumerate(rest):
        if a == "--t-start":
            t_start = float(rest[i + 1])
        if a == "--t-end":
            t_end = float(rest[i + 1])
    r = stats(*data, t_start=t_start, t_end=t_end)
    if as_json:
        print(json.dumps(r))
        return
    print(f"window [{r['t_window'][0]:.3f}, {r['t_window'][1]:.3f}] s, {r['periods']} lift periods")
    for key, label in [("cd_max", "C_D,max"), ("cd_min", "C_D,min"), ("cl_max", "C_L,max"),
                       ("cl_min", "C_L,min"), ("St", "St"), ("dp_half", "dp(t0+1/2f)")]:
        if key in r:
            print(f"  {label:12s} {r[key]:.5f}   (spread over periods {r[key + '_spread']:.1e})")


if __name__ == "__main__":
    main(sys.argv[1:])
