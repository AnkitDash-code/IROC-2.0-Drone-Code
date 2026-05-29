#!/usr/bin/env python3
"""
Phase 1 + Phase 2 protocol validator with optional web GUI.

This tool evaluates protocol CSV outputs using both baseline gates and additional
error estimates so operators can quickly verify reliability.

Default expected files inside a protocol directory:
  - phase1_static.csv
  - phase2_translate.csv
"""

import argparse
import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple


@dataclass
class PhaseStats:
    n: int
    q2_ratio: float
    vf_abs_mean: float
    vr_abs_mean: float
    vf_abs_max: float
    vf_abs_p50: float
    vf_abs_p95: float
    vf_abs_p99: float
    vr_abs_p95: float
    vf_rms: float
    vr_rms: float
    vf_sigma_robust: float


@dataclass
class TestCaseResult:
    name: str
    status: str
    value: float
    threshold: str
    note: str


def _f(row: dict, key: str, default: float = 0.0) -> float:
    try:
        return float(row.get(key, default))
    except Exception:
        return default


def _load_csv(path: Path) -> List[dict]:
    if not path.exists():
        return []
    with path.open("r", newline="") as f:
        return list(csv.DictReader(f))


def _pct(values: List[float], q: float) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    idx = int(max(0, min(len(ordered) - 1, math.ceil(q * len(ordered)) - 1)))
    return float(ordered[idx])


def _robust_sigma(values: List[float]) -> float:
    if not values:
        return 0.0
    med = _pct(values, 0.5)
    mad = _pct([abs(v - med) for v in values], 0.5)
    return 1.4826 * mad


def summarize(rows: List[dict]) -> PhaseStats:
    if not rows:
        return PhaseStats(
            n=0,
            q2_ratio=0.0,
            vf_abs_mean=math.inf,
            vr_abs_mean=math.inf,
            vf_abs_max=0.0,
            vf_abs_p50=0.0,
            vf_abs_p95=0.0,
            vf_abs_p99=0.0,
            vr_abs_p95=0.0,
            vf_rms=0.0,
            vr_rms=0.0,
            vf_sigma_robust=0.0,
        )

    n = len(rows)
    vf_abs: List[float] = []
    vr_abs: List[float] = []
    q2 = 0

    for r in rows:
        q = int(_f(r, "quality", 0.0))
        if q == 2:
            q2 += 1
        vf_abs.append(abs(_f(r, "v_forward", 0.0)))
        vr_abs.append(abs(_f(r, "v_right", 0.0)))

    vf_mean = float(sum(vf_abs) / n)
    vr_mean = float(sum(vr_abs) / n)

    return PhaseStats(
        n=n,
        q2_ratio=float(q2) / float(n),
        vf_abs_mean=vf_mean,
        vr_abs_mean=vr_mean,
        vf_abs_max=max(vf_abs),
        vf_abs_p50=_pct(vf_abs, 0.50),
        vf_abs_p95=_pct(vf_abs, 0.95),
        vf_abs_p99=_pct(vf_abs, 0.99),
        vr_abs_p95=_pct(vr_abs, 0.95),
        vf_rms=math.sqrt(sum(v * v for v in vf_abs) / n),
        vr_rms=math.sqrt(sum(v * v for v in vr_abs) / n),
        vf_sigma_robust=_robust_sigma(vf_abs),
    )


def evaluate_phase1_static(s: PhaseStats) -> Tuple[str, List[TestCaseResult], Dict[str, float]]:
    tests: List[TestCaseResult] = []

    def add(name: str, ok: bool, value: float, threshold: str, note: str) -> None:
        tests.append(TestCaseResult(name=name, status="PASS" if ok else "FAIL", value=value, threshold=threshold, note=note))

    add("minimum_samples", s.n >= 30, float(s.n), ">= 30", "Need enough samples for stable statistics")
    add("q2_ratio", s.q2_ratio > 0.50, s.q2_ratio, "> 0.50", "High-quality tracking support")
    add("vf_mean_drift", s.vf_abs_mean < 0.02, s.vf_abs_mean, "< 0.02 m/s", "Forward drift during static")
    add("vr_mean_drift", s.vr_abs_mean < 0.02, s.vr_abs_mean, "< 0.02 m/s", "Lateral drift during static")
    add("vf_p95_drift", s.vf_abs_p95 < 0.05, s.vf_abs_p95, "< 0.05 m/s", "Tail drift behavior")
    add("vf_p99_guard", s.vf_abs_p99 < 0.08, s.vf_abs_p99, "< 0.08 m/s", "Extra spike guard")

    estimated = {
        "static_drift_mps": s.vf_abs_mean,
        "static_drift_pct_of_1mps": 100.0 * s.vf_abs_mean,
        "static_noise_rms_mps": s.vf_rms,
        "static_noise_sigma_robust_mps": s.vf_sigma_robust,
    }

    verdict = "PASS" if all(t.status == "PASS" for t in tests) else "FAIL"
    return verdict, tests, estimated


def evaluate_phase2_translate(s: PhaseStats) -> Tuple[str, List[TestCaseResult], Dict[str, float]]:
    tests: List[TestCaseResult] = []

    cross_axis_pct = 100.0 * s.vr_abs_mean / max(s.vf_abs_mean, 1e-9)
    forward_dominant = s.vf_abs_mean > (1.5 * max(s.vr_abs_mean, 1e-9))
    outlier_heavy = s.vf_abs_max > (6.0 * max(s.vf_abs_p95, 1e-9))

    def add(name: str, ok: bool, value: float, threshold: str, note: str) -> None:
        tests.append(TestCaseResult(name=name, status="PASS" if ok else "FAIL", value=value, threshold=threshold, note=note))

    add("minimum_samples", s.n >= 30, float(s.n), ">= 30", "Need enough samples for stable statistics")
    add("q2_ratio", s.q2_ratio >= 0.60, s.q2_ratio, ">= 0.60", "High-quality tracking support")
    add("vf_mean_response", s.vf_abs_mean >= 0.02, s.vf_abs_mean, ">= 0.02 m/s", "Mean forward response")
    add("vf_p95_response", s.vf_abs_p95 >= 0.04, s.vf_abs_p95, ">= 0.04 m/s", "P95 forward response")
    add("forward_dominance", forward_dominant, s.vf_abs_mean / max(s.vr_abs_mean, 1e-9), "> 1.5", "Forward axis should dominate lateral")
    add("outlier_heavy_guard", not outlier_heavy, s.vf_abs_max / max(s.vf_abs_p95, 1e-9), "<= 6.0", "Reject spike-only behavior")
    add("cross_axis_leakage", cross_axis_pct <= 50.0, cross_axis_pct, "<= 50%", "Estimated lateral leakage during translation")

    estimated = {
        "translate_cross_axis_error_pct": cross_axis_pct,
        "translate_forward_rms_mps": s.vf_rms,
        "translate_vf_p95_mps": s.vf_abs_p95,
        "translate_vf_p99_mps": s.vf_abs_p99,
    }

    verdict = "PASS" if all(t.status == "PASS" for t in tests) else "FAIL"
    return verdict, tests, estimated


def evaluate_protocol(protocol_dir: Path) -> dict:
    p1_rows = _load_csv(protocol_dir / "phase1_static.csv")
    p2_rows = _load_csv(protocol_dir / "phase2_translate.csv")

    p1 = summarize(p1_rows)
    p2 = summarize(p2_rows)

    p1_verdict, p1_tests, p1_err = evaluate_phase1_static(p1)
    p2_verdict, p2_tests, p2_err = evaluate_phase2_translate(p2)

    overall = "PASS" if (p1_verdict == "PASS" and p2_verdict == "PASS") else "FAIL"

    return {
        "protocol_dir": str(protocol_dir),
        "phase1_static": {
            "verdict": p1_verdict,
            "stats": p1.__dict__,
            "tests": [t.__dict__ for t in p1_tests],
            "estimated_error": p1_err,
        },
        "phase2_translate": {
            "verdict": p2_verdict,
            "stats": p2.__dict__,
            "tests": [t.__dict__ for t in p2_tests],
            "estimated_error": p2_err,
        },
        "overall": overall,
    }


def _latest_protocol_dir(base_dir: Path) -> Path:
    candidates = sorted([
        p for p in base_dir.iterdir()
        if p.is_dir() and (p.name.startswith("protocol_") or p.name.startswith("manual_phase12_"))
    ])
    if not candidates:
        raise FileNotFoundError(f"No protocol_* or manual_phase12_* directory found under {base_dir}")
    return candidates[-1]


def write_reports(result: dict, out_prefix: str = "phase12_report") -> Tuple[Path, Path]:
    protocol_dir = Path(result["protocol_dir"])
    json_path = protocol_dir / f"{out_prefix}.json"
    txt_path = protocol_dir / f"{out_prefix}.txt"

    with json_path.open("w") as f:
        json.dump(result, f, indent=2)

    lines: List[str] = []
    lines.append(f"Phase1+Phase2 Test Report: {protocol_dir}")
    lines.append(f"Overall: {result['overall']}")

    for phase in ("phase1_static", "phase2_translate"):
        section = result[phase]
        lines.append("")
        lines.append(f"[{phase}] verdict={section['verdict']}")
        st = section["stats"]
        lines.append(
            "stats: "
            f"n={st['n']} q2={st['q2_ratio']:.3f} "
            f"vf_mean={st['vf_abs_mean']:.4f} vr_mean={st['vr_abs_mean']:.4f} "
            f"vf_p95={st['vf_abs_p95']:.4f} vf_p99={st['vf_abs_p99']:.4f}"
        )
        lines.append("tests:")
        for t in section["tests"]:
            lines.append(
                f"- {t['name']}: {t['status']} | value={t['value']:.4f} "
                f"| threshold={t['threshold']} | {t['note']}"
            )
        lines.append("estimated_error:")
        for k, v in section["estimated_error"].items():
            lines.append(f"- {k}={v:.6f}")

    with txt_path.open("w") as f:
        f.write("\n".join(lines) + "\n")

    return txt_path, json_path


def print_console_report(result: dict) -> None:
    print(f"Protocol: {result['protocol_dir']}")
    print(f"Overall: {result['overall']}")

    for phase in ("phase1_static", "phase2_translate"):
        section = result[phase]
        st = section["stats"]
        print(
            f"- {phase}: {section['verdict']} | n={st['n']} | q2={st['q2_ratio']:.2f} "
            f"| vf_mean={st['vf_abs_mean']:.4f} | vr_mean={st['vr_abs_mean']:.4f} "
            f"| vf_p95={st['vf_abs_p95']:.4f}"
        )
        for t in section["tests"]:
            print(f"  * {t['name']}: {t['status']} ({t['value']:.4f} vs {t['threshold']})")


def run_gui(base_dir: Path, default_protocol: Path, host: str, port: int) -> None:
    try:
        from flask import Flask, jsonify, render_template_string, request
    except Exception as exc:
        raise RuntimeError(f"Flask is required for --gui mode: {exc}")

    app = Flask(__name__)

    PAGE = """
<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Phase1+Phase2 Test Dashboard</title>
  <style>
    :root { --bg:#f4f1ea; --panel:#fffdf8; --ok:#226c42; --bad:#9b1c1c; --muted:#6f6a61; --ink:#1f1d1a; --accent:#c46a1a; }
    body { margin:0; font-family: 'Trebuchet MS', 'Segoe UI', sans-serif; background: radial-gradient(circle at 10% 0%, #efe7d8 0%, var(--bg) 35%, #ece8df 100%); color: var(--ink); }
    .wrap { max-width: 1100px; margin: 18px auto; padding: 0 14px 24px; }
    .head { display:flex; gap:10px; align-items:center; flex-wrap:wrap; }
    .title { font-size: 1.4rem; font-weight: 700; letter-spacing: 0.3px; }
    .badge { padding: 6px 10px; border-radius: 999px; background: #efe3cd; color: #6f4a1f; font-weight: 700; }
    .badge.pass { background:#dff2e6; color:var(--ok); }
    .badge.fail { background:#f7dede; color:var(--bad); }
    .controls { margin-top: 10px; display:flex; gap:8px; flex-wrap: wrap; }
    select, button { padding: 8px 10px; border-radius: 10px; border: 1px solid #c7bda9; background: #fff; color:#26221c; }
    button { cursor:pointer; background:#f6eee2; font-weight: 700; }
    .grid { margin-top: 14px; display:grid; grid-template-columns: repeat(auto-fit,minmax(330px,1fr)); gap: 12px; }
    .card { background: var(--panel); border:1px solid #ddd3bf; border-radius: 14px; padding: 12px; box-shadow: 0 4px 14px rgba(80,60,30,0.08); }
    .card h3 { margin: 0 0 10px; font-size: 1.02rem; }
    table { width:100%; border-collapse: collapse; font-size: 0.92rem; }
    th, td { text-align:left; border-bottom: 1px solid #eee3cf; padding: 6px 4px; }
    .ok { color: var(--ok); font-weight:700; }
    .fail { color: var(--bad); font-weight:700; }
    .muted { color: var(--muted); font-size: 0.9rem; }
    .errline { margin: 5px 0; }
  </style>
</head>
<body>
  <div class="wrap">
    <div class="head">
      <div class="title">Phase1 + Phase2 Validation Dashboard</div>
      <div id="overallBadge" class="badge">loading...</div>
    </div>
    <div class="controls">
      <select id="protocolSelect"></select>
      <button onclick="refreshNow()">Refresh</button>
      <span class="muted" id="runPath"></span>
    </div>

    <div class="grid">
      <div class="card">
        <h3>Phase1 Static</h3>
        <div id="p1stats" class="muted"></div>
        <table id="p1tests"></table>
        <div id="p1err"></div>
      </div>
      <div class="card">
        <h3>Phase2 Translate</h3>
        <div id="p2stats" class="muted"></div>
        <table id="p2tests"></table>
        <div id="p2err"></div>
      </div>
    </div>
  </div>

<script>
let selected = "";

function fmt(v, d=4){ return Number(v).toFixed(d); }

function statusClass(s){ return s === "PASS" ? "ok" : "fail"; }

function testsTable(el, tests){
  el.innerHTML = "<tr><th>Test</th><th>Status</th><th>Value</th><th>Threshold</th></tr>" +
    tests.map(t => `<tr><td>${t.name}</td><td class='${statusClass(t.status)}'>${t.status}</td><td>${fmt(t.value,4)}</td><td>${t.threshold}</td></tr>`).join('');
}

function errorBlock(el, err){
  el.innerHTML = Object.entries(err).map(([k,v]) => `<div class='errline'><b>${k}</b>: ${fmt(v,6)}</div>`).join('');
}

function fillData(data){
  document.getElementById('runPath').textContent = data.protocol_dir;
  const badge = document.getElementById('overallBadge');
  badge.textContent = `Overall ${data.overall}`;
  badge.className = `badge ${data.overall === 'PASS' ? 'pass' : 'fail'}`;

  const p1 = data.phase1_static;
  const p2 = data.phase2_translate;

  document.getElementById('p1stats').textContent = `verdict=${p1.verdict} | n=${p1.stats.n} q2=${fmt(p1.stats.q2_ratio,3)} vf_mean=${fmt(p1.stats.vf_abs_mean)} vr_mean=${fmt(p1.stats.vr_abs_mean)} vf_p95=${fmt(p1.stats.vf_abs_p95)}`;
  document.getElementById('p2stats').textContent = `verdict=${p2.verdict} | n=${p2.stats.n} q2=${fmt(p2.stats.q2_ratio,3)} vf_mean=${fmt(p2.stats.vf_abs_mean)} vr_mean=${fmt(p2.stats.vr_abs_mean)} vf_p95=${fmt(p2.stats.vf_abs_p95)}`;

  testsTable(document.getElementById('p1tests'), p1.tests);
  testsTable(document.getElementById('p2tests'), p2.tests);
  errorBlock(document.getElementById('p1err'), p1.estimated_error);
  errorBlock(document.getElementById('p2err'), p2.estimated_error);
}

async function listProtocols(){
  const r = await fetch('/api/protocols');
  const data = await r.json();
  const sel = document.getElementById('protocolSelect');
  sel.innerHTML = data.protocols.map(p => `<option value='${p}'>${p}</option>`).join('');
  if (!selected) { selected = data.default_protocol; }
  if (selected && data.protocols.includes(selected)) {
    sel.value = selected;
  }
  sel.onchange = () => { selected = sel.value; refreshNow(); };
}

async function refreshNow(){
  const q = selected ? `?protocol=${encodeURIComponent(selected)}` : '';
  const r = await fetch('/api/report' + q);
  const data = await r.json();
  fillData(data);
}

async function boot(){
  await listProtocols();
  await refreshNow();
  setInterval(refreshNow, 2000);
}

boot();
</script>
</body>
</html>
"""

    @app.route("/")
    def index():
        return render_template_string(PAGE)

    @app.route("/api/protocols")
    def api_protocols():
        dirs = sorted([
            p.name for p in base_dir.iterdir()
            if p.is_dir() and (p.name.startswith("protocol_") or p.name.startswith("manual_phase12_"))
        ])
        return jsonify({
            "protocols": dirs,
            "default_protocol": default_protocol.name,
        })

    @app.route("/api/report")
    def api_report():
        rel = request.args.get("protocol", default_protocol.name)
        candidate = (base_dir / rel).resolve()

        # Keep selection within base directory to avoid path traversal.
        if base_dir.resolve() not in candidate.parents and candidate != base_dir.resolve():
            return jsonify({"error": "invalid protocol path"}), 400

        if not candidate.exists() or not candidate.is_dir():
            return jsonify({"error": f"protocol dir not found: {rel}"}), 404

        return jsonify(evaluate_protocol(candidate))

    print(f"Dashboard: http://{host}:{port}")
    app.run(host=host, port=port, debug=False, threaded=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate Phase1+Phase2 protocol outputs")
    parser.add_argument("--protocol-dir", type=str, default="", help="Path to protocol_YYYYMMDD_HHMMSS directory. Defaults to latest protocol_* in current directory.")
    parser.add_argument("--base-dir", type=str, default=".", help="Base directory containing protocol_* folders")
    parser.add_argument("--out-prefix", type=str, default="phase12_report", help="Output report file prefix")
    parser.add_argument("--gui", action="store_true", help="Launch web GUI dashboard")
    parser.add_argument("--host", type=str, default="0.0.0.0", help="GUI host")
    parser.add_argument("--port", type=int, default=5060, help="GUI port")
    args = parser.parse_args()

    base_dir = Path(args.base_dir).resolve()

    if args.protocol_dir:
        protocol_dir = Path(args.protocol_dir).resolve()
    else:
        protocol_dir = _latest_protocol_dir(base_dir)

    if not protocol_dir.exists() or not protocol_dir.is_dir():
        raise FileNotFoundError(f"Protocol directory not found: {protocol_dir}")

    result = evaluate_protocol(protocol_dir)
    txt_path, json_path = write_reports(result, out_prefix=args.out_prefix)
    print_console_report(result)
    print(f"Saved report text: {txt_path}")
    print(f"Saved report json: {json_path}")

    if args.gui:
        run_gui(base_dir=base_dir, default_protocol=protocol_dir, host=args.host, port=args.port)


if __name__ == "__main__":
    main()
