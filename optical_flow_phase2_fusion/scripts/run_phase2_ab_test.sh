#!/usr/bin/env bash
set -euo pipefail

# Interactive A/B helper for Phase2 standalone vs fusion.
# This script does not launch long-running nodes itself; it guides the operator,
# archives outputs, and runs the comparator with explicit file paths.

ROOT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$ROOT_DIR"

MATCH_POINTS="${MATCH_POINTS:-1}"  # 1=compare equal sample counts, 0=raw only
ALLOW_STALE_INPUT="${ALLOW_STALE_INPUT:-0}"  # 1=allow copying pre-existing route CSVs

TS="$(date +%Y%m%d_%H%M%S)"
OUT_DIR="$ROOT_DIR/ab_phase2_${TS}"
mkdir -p "$OUT_DIR"
SCRIPT_START_EPOCH="$(date +%s)"

echo "============================================================"
echo "Phase2 A/B Test Helper"
echo "Output folder: $OUT_DIR"
echo "============================================================"
echo ""
echo "Prerequisites:"
echo "1) RealSense stream running"
echo "2) ROS sourced in the terminal where you run Phase2"
echo "3) Use /usr/bin/python3 for ROS scripts"
echo ""

wait_for_file_copy() {
  local src="$1"
  local dst="$2"
  local label="$3"
  local src_mtime=""

  get_mtime_epoch() {
    local p="$1"
    if stat -c %Y "$p" >/dev/null 2>&1; then
      stat -c %Y "$p"
    else
      stat -f %m "$p"
    fi
  }

  if [[ ! -f "$src" ]]; then
    echo "ERROR: Expected $label file not found: $src"
    echo "Run the phase and ensure it exits cleanly so route CSV is written."
    exit 1
  fi

  src_mtime="$(get_mtime_epoch "$src")"
  if [[ "$ALLOW_STALE_INPUT" != "1" && "$src_mtime" -lt "$SCRIPT_START_EPOCH" ]]; then
    echo "ERROR: Detected stale $label route file: $src"
    echo "       file mtime epoch=$src_mtime is older than test start epoch=$SCRIPT_START_EPOCH"
    echo "       Re-run the $label phase now, then press Enter again."
    echo "       (If intentional, run with ALLOW_STALE_INPUT=1 to bypass this safety check.)"
    exit 1
  fi

  cp -f "$src" "$dst"
  echo "Saved $label route: $dst"
}

echo "Step A: Run STANDALONE in another terminal:"
echo "  source /opt/ros/humble/setup.bash"
echo "  /usr/bin/python3 phase2_standalone.py"
echo "After finishing movement, press Ctrl+C and return here."
read -r -p "Press Enter when standalone run is complete... " _

wait_for_file_copy "$ROOT_DIR/my_drone_route.csv" "$OUT_DIR/standalone_route.csv" "standalone"

echo ""
echo "Step B: Run FUSION in another terminal:"
echo "  source /opt/ros/humble/setup.bash"
echo "  /usr/bin/python3 phase2_fusion_standalone.py"
echo "After finishing movement, press Ctrl+C and return here."
read -r -p "Press Enter when fusion run is complete... " _

wait_for_file_copy "$ROOT_DIR/my_drone_route_fusion.csv" "$OUT_DIR/fusion_route.csv" "fusion"

echo ""
echo "Running comparator..."
/usr/bin/python3 "$ROOT_DIR/compare_phase2_standalone_vs_fusion.py" \
  --standalone "$OUT_DIR/standalone_route.csv" \
  --fusion "$OUT_DIR/fusion_route.csv" | tee "$OUT_DIR/compare_report.txt"

if [[ "$MATCH_POINTS" == "1" ]]; then
  echo ""
  echo "Running matched-sample comparator (equal row counts)..."
  export OUT_DIR
  /usr/bin/python3 - <<'PY'
import csv
import os
import math
from pathlib import Path

out_dir = Path(os.environ["OUT_DIR"])
standalone = out_dir / "standalone_route.csv"
fusion = out_dir / "fusion_route.csv"
status_file = out_dir / "matched_status.txt"

def read_rows(path: Path):
    with path.open("r", newline="") as f:
        rows = list(csv.DictReader(f))
    return rows

def write_rows(path: Path, fieldnames, rows):
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        for r in rows:
            w.writerow(r)

def row_xy(row):
    if "x_m" in row and "z_m" in row:
      return float(row["x_m"]), float(row["z_m"])
    if "x_cm" in row and "z_cm" in row:
      return float(row["x_cm"]) * 0.01, float(row["z_cm"]) * 0.01
    return None

def first_motion_index(rows):
    if not rows:
      return 0
    p0 = row_xy(rows[0])
    if p0 is None:
      return 0
    x0, z0 = p0
    for i, r in enumerate(rows):
      p = row_xy(r)
      if p is None:
        continue
      if math.hypot(p[0] - x0, p[1] - z0) > 1e-6:
        return i
    return 0

s_rows = read_rows(standalone)
f_rows = read_rows(fusion)
if not s_rows or not f_rows:
  status_file.write_text("MATCHED_SKIP one route CSV is empty\n")
  raise SystemExit(0)

s0 = first_motion_index(s_rows)
f0 = first_motion_index(f_rows)
s_rows = s_rows[s0:]
f_rows = f_rows[f0:]

n = min(len(s_rows), len(f_rows))
if n < 2:
    status_file.write_text(
        f"MATCHED_SKIP not enough active points after motion alignment n={n} (standalone={len(s_rows)}, fusion={len(f_rows)})\n"
    )
    raise SystemExit(0)

write_rows(out_dir / "standalone_route_matched.csv", list(s_rows[0].keys()), s_rows[:n])
write_rows(out_dir / "fusion_route_matched.csv", list(f_rows[0].keys()), f_rows[:n])
status_file.write_text(
    f"MATCHED_OK n={n} start_offsets standalone={s0} fusion={f0}\n"
)
print(f"Matched rows written with n={n} (start offsets: standalone={s0}, fusion={f0})")
PY

  if [[ -f "$OUT_DIR/standalone_route_matched.csv" && -f "$OUT_DIR/fusion_route_matched.csv" ]]; then
    /usr/bin/python3 "$ROOT_DIR/compare_phase2_standalone_vs_fusion.py" \
      --standalone "$OUT_DIR/standalone_route_matched.csv" \
      --fusion "$OUT_DIR/fusion_route_matched.csv" | tee "$OUT_DIR/compare_report_matched.txt"
  else
    echo "Matched comparison skipped. See status:" | tee "$OUT_DIR/compare_report_matched.txt"
    if [[ -f "$OUT_DIR/matched_status.txt" ]]; then
      cat "$OUT_DIR/matched_status.txt" | tee -a "$OUT_DIR/compare_report_matched.txt"
    else
      echo "MATCHED_SKIP unknown reason" | tee -a "$OUT_DIR/compare_report_matched.txt"
    fi
  fi
fi

echo ""
echo "Done. Results:"
echo "- $OUT_DIR/standalone_route.csv"
echo "- $OUT_DIR/fusion_route.csv"
echo "- $OUT_DIR/compare_report.txt"
if [[ "$MATCH_POINTS" == "1" ]]; then
  echo "- $OUT_DIR/matched_status.txt"
  echo "- $OUT_DIR/standalone_route_matched.csv"
  echo "- $OUT_DIR/fusion_route_matched.csv"
  echo "- $OUT_DIR/compare_report_matched.txt"
fi
