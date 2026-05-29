#!/usr/bin/env python3
"""Plot deep-test CSV from d455_ipm_lk_deep_test_monitor.py."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load_csv(path: Path):
    rows = list(csv.DictReader(path.open("r", newline="", encoding="utf-8")))
    if not rows:
        raise RuntimeError(f"No data rows in {path}")

    t = np.array([float(r["t"]) for r in rows], dtype=float)
    t = t - t[0]

    data = {
        "t": t,
        "alt": np.array([float(r["alt"]) for r in rows], dtype=float),
        "pitch_deg": np.array([float(r["pitch_deg"]) for r in rows], dtype=float),
        "pitch_rate_dps": np.array([float(r["pitch_rate_dps"]) for r in rows], dtype=float),
        "v_forward": np.array([float(r["v_forward"]) for r in rows], dtype=float),
        "v_right": np.array([float(r["v_right"]) for r in rows], dtype=float),
        "quality": np.array([int(r["quality"]) for r in rows], dtype=int),
        "tracked": np.array([int(r["tracked"]) for r in rows], dtype=int),
        "med_mag": np.array([float(r["med_mag"]) for r in rows], dtype=float),
        "angle_std_deg": np.array([float(r["angle_std_deg"]) for r in rows], dtype=float),
    }
    return data


def quality_color(q: int) -> str:
    if q >= 2:
        return "#2ca02c"
    if q == 1:
        return "#ffbf00"
    return "#d62728"


def main() -> None:
    ap = argparse.ArgumentParser(description="Plot deep-test metrics")
    ap.add_argument("--csv", default="d455_ipm_lk_deep_test.csv")
    ap.add_argument("--out", default="d455_ipm_lk_deep_test_plot.png")
    args = ap.parse_args()

    csv_path = Path(args.csv)
    out_path = Path(args.out)

    d = load_csv(csv_path)

    fig, axes = plt.subplots(4, 1, figsize=(14, 12), sharex=True)

    axes[0].plot(d["t"], d["v_forward"], label="V_forward (m/s)", color="#1f77b4", linewidth=1.6)
    axes[0].plot(d["t"], d["v_right"], label="V_right (m/s)", color="#17becf", linewidth=1.2)
    axes[0].set_ylabel("Velocity (m/s)")
    axes[0].grid(True, alpha=0.25)
    axes[0].legend(loc="upper right")

    axes[1].plot(d["t"], d["pitch_deg"], label="Pitch (deg)", color="#ff7f0e", linewidth=1.6)
    axes[1].plot(d["t"], d["pitch_rate_dps"], label="Pitch rate (deg/s)", color="#9467bd", linewidth=1.2)
    axes[1].set_ylabel("Pitch")
    axes[1].grid(True, alpha=0.25)
    axes[1].legend(loc="upper right")

    axes[2].plot(d["t"], d["tracked"], label="Tracked points", color="#2ca02c", linewidth=1.4)
    axes[2].plot(d["t"], d["med_mag"], label="Median flow magnitude (px)", color="#8c564b", linewidth=1.2)
    axes[2].plot(d["t"], d["angle_std_deg"], label="Flow angle std (deg)", color="#e377c2", linewidth=1.0)
    axes[2].set_ylabel("Flow health")
    axes[2].grid(True, alpha=0.25)
    axes[2].legend(loc="upper right")

    # Quality heat strip.
    q = d["quality"]
    for i in range(len(q) - 1):
        axes[3].axvspan(d["t"][i], d["t"][i + 1], color=quality_color(int(q[i])), alpha=0.9)
    axes[3].set_ylim(0, 1)
    axes[3].set_yticks([])
    axes[3].set_ylabel("Quality")
    axes[3].set_xlabel("Time (s)")
    axes[3].grid(False)

    # Correlation readout.
    corr = 0.0
    if np.std(d["v_forward"]) > 1e-9 and np.std(d["pitch_deg"]) > 1e-9:
        corr = float(np.corrcoef(d["v_forward"], d["pitch_deg"])[0, 1])
    fig.suptitle(f"D455 IPM-LK Deep Test | corr(V_forward, Pitch)={corr:+.3f}")

    fig.tight_layout(rect=[0, 0, 1, 0.97])
    fig.savefig(out_path, dpi=170)
    print(f"Saved plot: {out_path.resolve()}")


if __name__ == "__main__":
    main()
