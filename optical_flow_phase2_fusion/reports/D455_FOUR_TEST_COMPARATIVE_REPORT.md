# D455 IPM+LK Comparative Test Report (4 Runs)

Date: 2026-03-29

## Scope
This report consolidates four protocol runs:
- protocol_20260329_033806
- protocol_20260329_034651
- protocol_20260329_035417
- protocol_20260329_035829

Primary focus:
- Static and translation performance for tracking quality.
- Tilt is included for completeness but handled separately by operator request.

## Executive Summary
- Run 033806 was invalid due monitor/runtime failure and produced no usable CSV samples.
- Runs 034651, 035417, and 035829 all passed static and translation phases.
- All runs failed tilt for the same root reason: IMU-derived pitch-rate remained inactive (pitch_rate_p95=0.00), so tilt isolation could not be validated by the scorer.
- Best non-tilt quality was seen in run 034651 and 035417 (higher quality ratio, lower static jitter).
- Run 035829 is still usable for non-tilt tracking analysis; its quality ratio dropped compared to 034651/035417.

## Per-Run Score Snapshot

| Run | Phase1 Static | Phase2 Translate | Phase3 Tilt | Overall |
|---|---|---|---|---|
| 033806 | FAIL (n=0) | FAIL (n=0) | FAIL (n=0) | FAIL |
| 034651 | PASS (n=201) | PASS (n=209) | FAIL (n=192) | FAIL |
| 035417 | PASS (n=201) | PASS (n=199) | FAIL (n=205) | FAIL |
| 035829 | PASS (n=200) | PASS (n=207) | FAIL (n=197) | FAIL |

## Detailed Metrics (from score.txt)

### Run 033806
- static: q2=0.00, vf_mean=inf, vr_mean=inf, vf_p95=0.0000, n=0
- translate: q2=0.00, vf_mean=inf, vr_mean=inf, vf_p95=0.0000, n=0
- tilt: q2=0.00, vf_mean=inf, vr_mean=inf, vf_p95=0.0000, n=0
- verdict: invalid data collection (insufficient samples)

### Run 034651
- static: q2=1.00, vf_mean=0.0012, vr_mean=0.0005, vf_p95=0.0029, n=201
- translate: q2=0.99, vf_mean=0.4329, vr_mean=0.0315, vf_p95=1.0940, n=209
- tilt: q2=0.94, vf_mean=0.0503, vr_mean=0.0326, vf_p95=0.1656, n=192
- tilt failure reason: IMU inactive for scorer (pitch_rate_p95=0.00)

### Run 035417
- static: q2=0.99, vf_mean=0.0008, vr_mean=0.0004, vf_p95=0.0035, n=201
- translate: q2=0.99, vf_mean=0.3468, vr_mean=0.0587, vf_p95=0.6957, n=199
- tilt: q2=0.99, vf_mean=0.6538, vr_mean=0.1825, vf_p95=1.5033, n=205
- tilt failure reason: IMU inactive for scorer (pitch_rate_p95=0.00)

### Run 035829
- static: q2=0.81, vf_mean=0.0021, vr_mean=0.0012, vf_p95=0.0110, n=200
- translate: q2=0.80, vf_mean=0.3897, vr_mean=0.0390, vf_p95=0.7135, n=207
- tilt: q2=0.77, vf_mean=0.9527, vr_mean=0.3198, vf_p95=3.3735, n=197
- tilt failure reason: IMU inactive for scorer (pitch_rate_p95=0.00)

## Non-Tilt Tracking Accuracy (Run 035829)
Computed from phase1_static vs phase2_translate using static p95 threshold on |v_forward|.

- threshold (static p95): 0.010969 m/s
- confusion matrix: TP=165, FN=42, FP=9, TN=191
- precision: 0.9483
- recall: 0.7971
- F1: 0.8661
- overall classification accuracy: 0.8747
- distribution separability AUC (|v_forward|): 0.8367

Interpretation:
- Good non-tilt movement discrimination.
- High precision (low false positives), with moderate miss rate during translate windows.

## Graph References
For run 035829 non-tilt comparison:
- protocol_20260329_035829/static_translate_accuracy_comparison.png

Per-phase generated plots:
- protocol_20260329_034651/phase1_static_plot.png
- protocol_20260329_034651/phase2_translate_plot.png
- protocol_20260329_035417/phase1_static_plot.png
- protocol_20260329_035417/phase2_translate_plot.png
- protocol_20260329_035829/phase1_static_plot.png
- protocol_20260329_035829/phase2_translate_plot.png

## Root Causes and Reliability Notes
- Common blocker across valid runs: tilt scoring depends on IMU pitch-rate activity, but recorded pitch_rate_dps remained at zero for scoring.
- Range source was dummy in these runs (no live range topic continuity test).
- Minor shutdown RCLError traces are post-stop artifacts and did not prevent static/translate scoring in valid runs.

## Conclusion
- For non-tilt (static + translation), the pipeline is repeatably functional and accurate enough for practical relative-motion tracking.
- Best quality profile was observed in runs 034651 and 035417.
- Run 035829 confirms usable non-tilt performance with quantified accuracy (87.47% classification accuracy), but quality margin is lower than 034651/035417.
- Final flight-readiness still requires a dedicated tilt campaign with valid IMU orientation dynamics and live range continuity.
