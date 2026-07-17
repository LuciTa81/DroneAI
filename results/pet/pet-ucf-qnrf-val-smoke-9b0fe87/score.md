# model-evaluation: Model-dataset evaluation: pet-ucf-qnrf-validation-smoke

- Status: **PASS_RESEARCH_ONLY**
- Decision scope: **research_only**
- Score: **100.0/100**
- Pass threshold: **80.0**
- Generated at: `2026-07-17T19:28:38.560902+00:00`

| Result | Category | Check | Points | Blocker | Observed |
|---|---|---|---:|:---:|---|
| PASS | data and split integrity | Frozen split and no leakage | 15.0/15.0 | Y | split=True; leakage_free=True |
| PASS | evaluation coverage | Every expected sample has a success or explicit failure record | 10.0/10.0 | Y | records=36; successes+failures=36 |
| PASS | count quality | MAE, RMSE, and absolute bias meet frozen targets | 20.0/20.0 |  | mae=76.44444444444444/250.0; rmse=129.80177194476198/400.0; bias=15.666666666666666/250.0 |
| PASS | spatial output quality | Family-specific spatial targets pass | 15.0/15.0 |  | mean=0.5121234229790129 |
| PASS | condition robustness | Density and condition targets pass | 10.0/10.0 |  | {"band_coverage": {"high": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "low": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "medium": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}}, "band_signed_bias": {"high": 7.5, "low": 5.0, "medium": 34.5}, "condition_coverage": {"density_band": {"high": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "low": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}, "medium": {"coverage": 1.0, "expected_samples": 12, "explicit_failures": 0, "recorded_samples": 12, "successful_samples": 12}}, "source_partition": {"official_validation": {"coverage": 1.0, "expected_samples": 36, "explicit_failures": 0, "recorded_samples": 36, "successful_samples": 36}}}, "condition_signed_bias": {"density_band": {"high": 7.5, "low": 5.0, "medium": 34.5}, "source_partition": {"official_validation": 15.666666666666666}}} |
| PASS | runtime and resources | Latency and peak VRAM meet frozen targets | 10.0/10.0 |  | latency=57.99415199908253/2000.0; vram=4291.91455078125/24000.0 |
| PASS | reproducibility and provenance | Code, data, checkpoint, config, and environment are verified | 10.0/10.0 | Y | True |
| PASS | review artifacts | Approved model brief and verified budgeted review bundle exist | 10.0/10.0 | Y | brief=True; artifacts=True; budget=True |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.
