#!/usr/bin/env python3
import csv
import math
import sys
from pathlib import Path


def load_csv(path: Path):
    rows = []
    if not path.exists():
        return rows
    with path.open("r", newline="") as f:
        for r in csv.DictReader(f):
            rows.append(r)
    return rows


def f(row, key, default=0.0):
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def summarize(rows):
    if not rows:
        return {
            "n": 0,
            "q2_ratio": 0.0,
            "vf_abs_mean": math.inf,
            "vr_abs_mean": math.inf,
            "vf_abs_max": 0.0,
            "vf_abs_p95": 0.0,
            "vr_abs_p95": 0.0,
            "pitch_rate_abs_mean": 0.0,
            "pitch_rate_abs_p95": 0.0,
            "med_mag_median": 0.0,
        }

    n = len(rows)
    q2 = 0
    vf_abs = []
    vr_abs = []
    pr_abs = []
    med_mag = []

    for r in rows:
        q = int(f(r, "quality", 0.0))
        if q == 2:
            q2 += 1
        vf_abs.append(abs(f(r, "v_forward", 0.0)))
        vr_abs.append(abs(f(r, "v_right", 0.0)))
        pr_abs.append(abs(f(r, "pitch_rate_dps", 0.0)))
        med_mag.append(f(r, "med_mag", 0.0))

    med_mag_sorted = sorted(med_mag)
    med_mag_median = med_mag_sorted[len(med_mag_sorted) // 2]

    vf_sorted = sorted(vf_abs)
    vr_sorted = sorted(vr_abs)
    pr_sorted = sorted(pr_abs)
    p95_idx = min(n - 1, max(0, int(0.95 * n) - 1))
    vf_abs_p95 = vf_sorted[p95_idx]
    vr_abs_p95 = vr_sorted[p95_idx]
    pitch_rate_abs_p95 = pr_sorted[p95_idx]

    return {
        "n": n,
        "q2_ratio": q2 / n,
        "vf_abs_mean": sum(vf_abs) / n,
        "vr_abs_mean": sum(vr_abs) / n,
        "vf_abs_max": max(vf_abs),
        "vf_abs_p95": vf_abs_p95,
        "vr_abs_p95": vr_abs_p95,
        "pitch_rate_abs_mean": sum(pr_abs) / n,
        "pitch_rate_abs_p95": pitch_rate_abs_p95,
        "med_mag_median": med_mag_median,
    }


def pass_fail(phase, s):
    if s["n"] < 30:
        return "FAIL", "insufficient samples"

    if phase == "phase1_static":
        ok = (
            s["vf_abs_mean"] < 0.02
            and s["vr_abs_mean"] < 0.02
            and s["vf_abs_p95"] < 0.05
            and s["q2_ratio"] > 0.50
        )
        why = (
            f"static jitter vf={s['vf_abs_mean']:.4f}, vr={s['vr_abs_mean']:.4f}, "
            f"vf_p95={s['vf_abs_p95']:.4f}, q2={s['q2_ratio']:.2f}"
        )
        return ("PASS" if ok else "FAIL"), why

    if phase == "phase2_translate":
        # Expect robust forward response with quality support, not just spikes.
        forward_dominant = s["vf_abs_mean"] > (1.5 * max(s["vr_abs_mean"], 1e-6))
        outlier_heavy = s["vf_abs_max"] > (6.0 * max(s["vf_abs_p95"], 1e-6))
        ok = (
            s["q2_ratio"] >= 0.60
            and s["vf_abs_mean"] >= 0.02
            and s["vf_abs_p95"] >= 0.04
            and forward_dominant
            and not outlier_heavy
        )
        why = (
            f"translation response vf_mean={s['vf_abs_mean']:.4f}, "
            f"vf_p95={s['vf_abs_p95']:.4f}, vf_max={s['vf_abs_max']:.4f}, "
            f"vr_mean={s['vr_abs_mean']:.4f}, q2={s['q2_ratio']:.2f}"
        )
        return ("PASS" if ok else "FAIL"), why

    if phase == "phase3_tilt":
        # During tilt-in-place, forward velocity should remain low and IMU must be active.
        imu_active = s["pitch_rate_abs_p95"] >= 1.0
        if not imu_active:
            why = (
                f"tilt IMU inactive pitch_rate_p95={s['pitch_rate_abs_p95']:.2f} dps; "
                f"cannot verify tilt leakage reliably"
            )
            return "FAIL", why

        ok = s["vf_abs_mean"] < 0.03 and s["vf_abs_p95"] < 0.08
        why = (
            f"tilt leakage vf_mean={s['vf_abs_mean']:.4f}, "
            f"vf_p95={s['vf_abs_p95']:.4f}, pitch_rate_p95={s['pitch_rate_abs_p95']:.2f} dps"
        )
        return ("PASS" if ok else "FAIL"), why

    return "FAIL", "unknown phase"


def main():
    if len(sys.argv) != 2:
        print("usage: score_d455_real_motion_protocol.py <protocol_output_dir>")
        sys.exit(2)

    out_dir = Path(sys.argv[1])
    phases = ["phase1_static", "phase2_translate", "phase3_tilt"]

    print(f"Protocol scoring: {out_dir}")
    overall = True
    for phase in phases:
        rows = load_csv(out_dir / f"{phase}.csv")
        s = summarize(rows)
        verdict, detail = pass_fail(phase, s)
        if verdict != "PASS":
            overall = False
        print(
            f"- {phase}: {verdict} | n={s['n']} | q2={s['q2_ratio']:.2f} "
            f"| vf_mean={s['vf_abs_mean']:.4f} | vr_mean={s['vr_abs_mean']:.4f} "
            f"| vf_p95={s['vf_abs_p95']:.4f} | {detail}"
        )

    print(f"Overall: {'PASS' if overall else 'FAIL'}")


if __name__ == "__main__":
    main()
