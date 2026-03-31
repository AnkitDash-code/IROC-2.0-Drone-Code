#!/usr/bin/env python3
"""
Compare standalone vs fusion Phase2 route outputs.

Default inputs:
  - my_drone_route.csv
  - my_drone_route_fusion.csv

Produces objective route-quality metrics and percent deltas.
"""

import argparse
import csv
import math
from pathlib import Path


def load_points(path: Path):
    rows = list(csv.DictReader(path.open("r", newline="")))
    if not rows:
        return []

    if "x_m" in rows[0] and "z_m" in rows[0]:
        scale = 1.0
        xk, zk = "x_m", "z_m"
    elif "x_cm" in rows[0] and "z_cm" in rows[0]:
        scale = 0.01
        xk, zk = "x_cm", "z_cm"
    else:
        raise ValueError(f"Unsupported CSV columns in {path}")

    pts = []
    for r in rows:
        pts.append((float(r[xk]) * scale, float(r[zk]) * scale))
    return pts


def metrics(pts):
    if len(pts) < 2:
        return {
            "n": len(pts),
            "path_len_m": 0.0,
            "net_disp_m": 0.0,
            "path_over_net": math.inf,
            "step_p95_m": 0.0,
            "step_max_m": 0.0,
            "tiny_step_frac": 0.0,
            "big_step_frac": 0.0,
        }

    steps = []
    path_len = 0.0
    step_max = 0.0
    for i in range(1, len(pts)):
        dx = pts[i][0] - pts[i - 1][0]
        dz = pts[i][1] - pts[i - 1][1]
        d = math.hypot(dx, dz)
        steps.append(d)
        path_len += d
        step_max = max(step_max, d)

    net_disp = math.hypot(pts[-1][0] - pts[0][0], pts[-1][1] - pts[0][1])
    p95 = sorted(steps)[max(0, min(len(steps) - 1, int(math.ceil(0.95 * len(steps)) - 1)))]
    tiny_frac = sum(1 for d in steps if d < 0.002) / len(steps)
    big_frac = sum(1 for d in steps if d > 0.12) / len(steps)

    return {
        "n": len(pts),
        "path_len_m": path_len,
        "net_disp_m": net_disp,
        "path_over_net": path_len / max(net_disp, 1e-9),
        "step_p95_m": p95,
        "step_max_m": step_max,
        "tiny_step_frac": tiny_frac,
        "big_step_frac": big_frac,
    }


def pct_delta(old, new):
    if abs(old) < 1e-12:
        return math.nan
    return 100.0 * (new - old) / old


def main():
    parser = argparse.ArgumentParser(description="Compare standalone and fusion route outputs")
    parser.add_argument("--standalone", default="my_drone_route.csv")
    parser.add_argument("--fusion", default="my_drone_route_fusion.csv")
    args = parser.parse_args()

    s_path = Path(args.standalone)
    f_path = Path(args.fusion)

    if not s_path.exists():
        raise FileNotFoundError(f"Standalone route CSV not found: {s_path}")
    if not f_path.exists():
        raise FileNotFoundError(f"Fusion route CSV not found: {f_path}")

    s = metrics(load_points(s_path))
    f = metrics(load_points(f_path))

    print("Standalone:", s_path)
    print("Fusion:", f_path)
    print("")

    keys = [
        "n",
        "path_len_m",
        "net_disp_m",
        "path_over_net",
        "step_p95_m",
        "step_max_m",
        "tiny_step_frac",
        "big_step_frac",
    ]

    for k in keys:
        sv = s[k]
        fv = f[k]
        delta = pct_delta(sv, fv)
        if math.isnan(delta):
            dtext = "n/a"
        else:
            dtext = f"{delta:+.2f}%"
        print(f"{k:16s} standalone={sv:.6f} fusion={fv:.6f} delta={dtext}")

    print("\nBetter-if-lower metrics: path_over_net, step_p95_m, step_max_m, big_step_frac")
    print("Better-if-higher metric: net_disp_m (if commanded motion was similar)")


if __name__ == "__main__":
    main()
