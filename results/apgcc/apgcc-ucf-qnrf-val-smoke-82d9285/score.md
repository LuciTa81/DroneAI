# model-evaluation: Model-dataset evaluation: apgcc-ucf-qnrf-validation-smoke

- Status: **PASS_RESEARCH_ONLY**
- Decision scope: **research_only**
- Score: **80.0/100**
- Pass threshold: **80.0**
- Generated at: `2026-07-18T06:35:38.681874+00:00`

| Result | Category | Check | Points | Blocker | Observed |
|---|---|---|---:|:---:|---|
| PASS | data and split integrity | Frozen split and no leakage | 15.0/15.0 | Y | split=True; leakage_free=True |
| PASS | evaluation coverage | Every expected sample has a success or explicit failure record | 10.0/10.0 | Y | records=36; successes+failures=36 |
| FAIL | count quality | MAE, RMSE, and absolute bias meet frozen targets | 0.0/20.0 |  | mae=266.19444444444446/250.0; rmse=431.2962052541926/400.0; bias=93.63888888888889/250.0 |
| PASS | spatial output quality | Family-specific spatial targets pass | 15.0/15.0 |  | mean=0.6831052027940661 |
| PASS | condition robustness | Density and condition targets pass | 10.0/10.0 |  | {"band_coverage": {"high": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "low": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "medium": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}}, "band_signed_bias": {"high": -261.6666666666667, "low": -11.75, "medium": -7.5}, "condition_coverage": {"density_band": {"high": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "low": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "medium": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}}, "source_partition": {"official_validation": {"coverage": 1.0, "expected_samples": 36, "explicit_failures": 0, "recorded_samples": 36, "successful_samples": 36}}}, "condition_signed_bias": {"density_band": {"high": -261.6666666666667, "low": -11.75, "medium": -7.5}, "source_partition": {"official_validation": -93.63888888888889}}} |
| PASS | runtime and resources | Latency and peak VRAM meet frozen targets | 10.0/10.0 |  | latency=67.66282950047753/2000.0; vram=5258.68603515625/24000.0 |
| PASS | reproducibility and provenance | Code, data, checkpoint, config, and environment are verified | 10.0/10.0 | Y | True |
| PASS | review artifacts | Approved model brief and verified budgeted review bundle exist | 10.0/10.0 | Y | brief=True; artifacts=True; budget=True |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.
