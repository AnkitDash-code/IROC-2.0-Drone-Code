# Phase2 A/B Best Run Summary

Best retained run:
- /home/jetson123/Drone/ab_phase2_20260329_231119

Validation status:
- Matched comparison is valid (`MATCHED_OK n=71 start_offsets standalone=20 fusion=30`)

## Absolute Error Percentages (independent per method)

Definitions:
- Tracking motion error % = max(0, (path_over_net - 1) * 100)
- P95 step budget % = (step_p95 / 0.12 m) * 100
- Max step budget % = (step_max / 0.12 m) * 100
- Big-step outlier rate % = fraction of steps > 0.12 m * 100
- Micro-step stall rate % = fraction of steps < 0.002 m * 100
- Straightness score % = min(100, 100 / path_over_net)

### Standalone (matched)
- tracking_motion_error_pct: 294.871
- p95_step_budget_pct_of_0p12m: 114.586
- max_step_budget_pct_of_0p12m: 233.596
- big_step_outlier_rate_pct: 5.714
- micro_step_stall_rate_pct: 12.857
- straightness_score_pct: 25.325

### Fusion (matched)
- tracking_motion_error_pct: 0.076
- p95_step_budget_pct_of_0p12m: 5.799
- max_step_budget_pct_of_0p12m: 9.663
- big_step_outlier_rate_pct: 0.000
- micro_step_stall_rate_pct: 4.286
- straightness_score_pct: 99.924

## Improvement Message For Team

In the best matched A/B run, fusion reduced collective trajectory error by about 91% and removed large-step outliers completely, while improving path efficiency and reducing high-percentile and peak step jitter.

## Source Files

- /home/jetson123/Drone/ab_phase2_20260329_231119/compare_report.txt
- /home/jetson123/Drone/ab_phase2_20260329_231119/compare_report_matched.txt
- /home/jetson123/Drone/ab_phase2_20260329_231119/matched_status.txt
- /home/jetson123/Drone/ab_phase2_20260329_231119/standalone_route_matched.csv
- /home/jetson123/Drone/ab_phase2_20260329_231119/fusion_route_matched.csv
