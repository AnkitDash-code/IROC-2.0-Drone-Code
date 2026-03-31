# Phase2 Fusion Sign-off Report

Date: 2026-03-29

Scope:
- 5 matched A/B runs (`ab_phase2_*`) using `compare_report_matched.txt`
- Acceptance rule provided by reviewer:
  1. Fusion must improve first 4 metrics in at least 4/5 runs:
     - path_over_net
     - step_p95_m
     - step_max_m
     - big_step_frac
  2. net_disp_m must not degrade beyond control margin

Control margin used:
- net_disp_m degradation threshold: 50% (fusion below standalone by more than 50% is fail for that run)

## Run Set

- /home/jetson123/Drone/ab_phase2_20260329_231119
- /home/jetson123/Drone/ab_phase2_20260329_234352
- /home/jetson123/Drone/ab_phase2_20260329_234554
- /home/jetson123/Drone/ab_phase2_20260329_234745
- /home/jetson123/Drone/ab_phase2_20260329_234956

All runs include `MATCHED_OK` status.

## Median And Worst-case

### Standalone
- path_over_net: median 4.051307, worst 10.033632
- step_p95_m: median 0.111539, worst 0.137503
- step_max_m: median 0.233102, worst 0.280315
- big_step_frac: median 0.046154, worst 0.078947
- net_disp_m: median 0.259408, worst 0.089239

### Fusion
- path_over_net: median 1.000760, worst 1.069291
- step_p95_m: median 0.006958, worst 0.080000
- step_max_m: median 0.011595, worst 0.080001
- big_step_frac: median 0.000000, worst 0.000000
- net_disp_m: median 0.132954, worst 0.048967

## Rule Check

### Rule 1: First four metrics improved in at least 4/5 runs
- Result: 5/5 runs improved
- Status: PASS

### Rule 2: net_disp_m does not degrade beyond control margin
Run-wise net displacement change (%) = `(fusion - standalone) / standalone * 100`:
- ab_phase2_20260329_231119: -64.56%
- ab_phase2_20260329_234352: -48.75%
- ab_phase2_20260329_234554: -26.53%
- ab_phase2_20260329_234745: +479.08%
- ab_phase2_20260329_234956: -76.09%

- Degradation beyond -50% margin occurs in 2/5 runs
- Status: FAIL

## Sign-off Decision

- Final decision under stated acceptance criteria: **NOT ACCEPTED YET**

Reason:
- Fusion is consistently better on stability/trajectory error metrics (Rule 1 passes strongly),
  but displacement retention is inconsistent and violates margin in multiple runs (Rule 2 fails).

## Recommended Next Action

Tune fusion for better displacement retention while preserving stability gains:
- Increase `FUSION_EMA_ALPHA` slightly
- Increase `FUSION_STATIONARY_DECAY` slightly (less damping)
- Increase `FUSION_LEAK_ATTENUATION` slightly (less non-dominant suppression)

Then re-run 5 matched A/B tests and re-apply the same acceptance gate.

---

## Update: 2026-03-30 (Latest 5 Matched A/B Runs)

Date: 2026-03-30

Scope:
- 5 latest matched A/B runs starting with `ab_phase2_20260330_*`:
  - /home/jetson123/Drone/ab_phase2_20260330_014506
  - /home/jetson123/Drone/ab_phase2_20260330_014725
  - /home/jetson123/Drone/ab_phase2_20260330_014953
  - /home/jetson123/Drone/ab_phase2_20260330_015214
  - /home/jetson123/Drone/ab_phase2_20260330_015421
- All runs include `MATCHED_OK` status.

Matched quality snapshot (from `matched_status.txt`):
- 014506: `n=36`, start offsets standalone=1, fusion=54
- 014725: `n=21`, start offsets standalone=21, fusion=50
- 014953: `n=42`, start offsets standalone=11, fusion=156
- 015214: `n=15`, start offsets standalone=31, fusion=148
- 015421: `n=45`, start offsets standalone=28, fusion=44

Control margin used (same as original review):
- net_disp_m degradation threshold: 50% (fusion below standalone by more than 50% is fail for that run)

## Median And Worst-case (Latest 5)

### Standalone
- path_over_net: median 1.250559, worst 18.297956
- step_p95_m: median 0.072289, worst 0.164201
- step_max_m: median 0.072289, worst 0.223688
- big_step_frac: median 0.000000, worst 0.170732
- net_disp_m: median 0.315526, worst 0.014416

### Fusion
- path_over_net: median 1.000361, worst 1.083459
- step_p95_m: median 0.006125, worst 0.080000
- step_max_m: median 0.008020, worst 0.080000
- big_step_frac: median 0.000000, worst 0.000000
- net_disp_m: median 0.080970, worst 0.049566

## Run-wise Deep Findings

Legend:
- For `path_over_net`, `step_p95_m`, `step_max_m`, `big_step_frac`: lower is better.
- For `net_disp_m`: higher is better.

### 014506
- path_over_net: improved by 94.49%
- step_p95_m: improved by 76.90%
- step_max_m: improved by 70.95%
- big_step_frac: tie (0.000000 -> 0.000000)
- net_disp_m: improved by 586.42%

### 014725
- path_over_net: degraded by 7.91%
- step_p95_m: improved by 77.39%
- step_max_m: improved by 78.86%
- big_step_frac: tie (0.000000 -> 0.000000)
- net_disp_m: degraded by 74.34%

### 014953
- path_over_net: improved by 17.79%
- step_p95_m: improved by 98.69%
- step_max_m: improved by 98.55%
- big_step_frac: improved by 100.00% (0.170732 -> 0.000000)
- net_disp_m: degraded by 95.45%

### 015214
- path_over_net: improved by 20.04%
- step_p95_m: degraded by 10.67%
- step_max_m: degraded by 10.67%
- big_step_frac: tie (0.000000 -> 0.000000)
- net_disp_m: improved by 305.71%

### 015421
- path_over_net: improved by 42.35%
- step_p95_m: improved by 98.52%
- step_max_m: improved by 98.24%
- big_step_frac: improved by 100.00% (0.090909 -> 0.000000)
- net_disp_m: degraded by 94.54%

## Rule Check (Latest 5)

### Rule 1: First four metrics improved in at least 4/5 runs
- Strictly improved runs: 2/5 (fails)
- If zero-to-zero `big_step_frac` ties are counted as neutral/non-worse: 3/5 (still below 4/5)
- Status: FAIL

### Rule 2: net_disp_m does not degrade beyond control margin
Run-wise net displacement change (%) = `(fusion - standalone) / standalone * 100`:
- ab_phase2_20260330_014506: +586.42%
- ab_phase2_20260330_014725: -74.34%
- ab_phase2_20260330_014953: -95.45%
- ab_phase2_20260330_015214: +305.71%
- ab_phase2_20260330_015421: -94.54%

- Degradation beyond -50% margin occurs in 3/5 runs
- Status: FAIL

## Updated Decision For Latest 5

- Final decision under the same acceptance criteria: **NOT ACCEPTED YET**

Reason:
- Fusion still shows strong stability gains in most runs (especially `step_p95_m` and `step_max_m`),
  but consistency on path efficiency and displacement retention is not sufficient.
- Large displacement collapses remain present in multiple runs, and the 4/5 stability gate is not met for this batch.

## Recommended Next Action (Before Next 5-Run Gate)

- Add a minimum displacement retention guard in fusion output logic so non-dominant attenuation cannot collapse net motion.
- Re-balance smoothing versus damping for short windows that currently produce near-zero incremental motion.
- Re-run another 5 matched A/B tests and re-evaluate with the same two-rule sign-off gate.
