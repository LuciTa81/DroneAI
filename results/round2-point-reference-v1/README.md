# Round 2 Point-Model Reference Result

## Result identity

- Models: STEERER, PET, APGCC
- Frozen dataset lanes:
  - UCF-QNRF official Test: 334 images/model
  - JHU-CROWD++ official Validation: 500 images/model
  - UP-COUNT official validation/test selection: 166 images/model
- Total: 1,000 images/model, 3,000 model-image inferences
- Training: false
- Fine-tuning: false
- Calibration: false
- Shared sample-manifest SHA-256:
  `37adb8e058611de79d6f97c3181106f6606f5d1bc1a9b611720e4a699395cc04`
- Point-config identity SHA-256:
  `b71a11572441a65969d6f2ae4f93a22f13ea9002a9a20f29f6f64174097caa27`
- Git commit:
  `14e786e6a79f7104ddc8e91f731b630b9d19ffcc`

The point-config identity differs from the historical Round 2 config identity,
while the sample-row identity is exactly the same. This is why the completed
STEERER runs can be reused without rerunning them.

## Compact comparison

The authoritative compact outputs in this directory are:

- `comparison.json`
  - SHA-256:
    `a2dcb7727c0923a5fea1ef7d4795f5d9ef42e04d851a0b05e5c33881d22d9f58`
- `comparison.md`
  - SHA-256:
    `c2bd6a3aaf2896251241e932065fdb780ce389e28bbc525e376cb89ff899bd7e`

Full predictions, progress ledgers, environment records, rights decisions,
native-output metadata, and 12 curated panels per lane remain on the home5090
SSD:

```text
/workspace/data/results/round2-reference-v1/{steerer,pet,apgcc}/{dataset_id}
```

The new shortlist manifest and comparison remain at:

```text
/workspace/data/results/round2-point-reference-v1/
```

## Headline results

| Dataset | Model | MAE | RMSE | MAPE ref. | Spatial metric | Median ms | FPS | Peak VRAM |
|---|---|---:|---:|---:|---|---:|---:|---:|
| UCF Test 334 | STEERER | 86.67 | 152.57 | 11.61% | GAME(1) 95.994 | 141.15 | 7.08 | 4.01 GB |
| UCF Test 334 | PET | 82.28 | 152.91 | 11.46% | localization F1 0.448 | 57.49 | 17.40 | 5.13 GB |
| UCF Test 334 | APGCC | 125.86 | 228.05 | 19.73% | localization F1 0.650 | 67.75 | 14.76 | 6.23 GB |
| JHU Val 500 | STEERER | 94.99 | 243.78 | 40.70% | GAME(1) 95.017 | 21.98 | 45.49 | 3.37 GB |
| JHU Val 500 | PET | 89.04 | 229.25 | 44.18% | localization F1 0.282 | 31.66 | 31.59 | 5.13 GB |
| JHU Val 500 | APGCC | 89.46 | 259.29 | 46.49% | localization F1 0.640 | 9.83 | 101.74 | 5.61 GB |
| UP-COUNT 166 | STEERER | 17.72 | 43.76 | 58.40% | GAME(1) 19.487 | 119.74 | 8.35 | 2.59 GB |
| UP-COUNT 166 | PET | 19.22 | 46.82 | 68.53% | localization F1 0.047 | 55.01 | 18.18 | 3.45 GB |
| UP-COUNT 166 | APGCC | 69.65 | 207.54 | 1214.41% | localization F1 0.258 | 57.90 | 17.27 | 3.96 GB |

GAME(1), a density-grid error, and localization F1, a point-matching metric,
are not numerically comparable. PET and APGCC do not produce a native density
map; any rendered point heatmap is only a derived point visualization.

STEERER also emits auxiliary localization points. Their common 16-pixel
localization F1 values are shown separately from its native density metric:

| Dataset | STEERER auxiliary F1 | PET native F1 | APGCC native F1 |
|---|---:|---:|---:|
| UCF Test 334 | 0.688 | 0.448 | 0.650 |
| JHU Val 500 | 0.702 | 0.282 | 0.640 |
| UP-COUNT 166 | 0.197 | 0.047 | 0.258 |

MAPE is included as a reference value only. It becomes unstable when
ground-truth counts are small, which is especially visible in the UP-COUNT
APGCC lane. MAE/RMSE, signed bias, per-density-band results, and spatial output
should be reviewed together.

## New full-run wall time

Wall time is measured from `runner.log` file birth to final modification and
includes manifest verification, model load, inference, aggregation, panel
rendering, scoring, and artifact writing.

| Model | UCF 334 | JHU 500 | UP-COUNT 166 | Total |
|---|---:|---:|---:|---:|
| PET | 244 s | 303 s | 31 s | 578 s |
| APGCC | 199 s | 268 s | 40 s | 507 s |

## Scope-separated harness score

| Gate | Score | Result |
|---|---:|---|
| Technical reproducibility | 100/100 | PASS |
| Dataset/split integrity | 100/100 | PASS |
| Artifact/evidence integrity | 100/100 | PASS |
| Runtime completeness | 100/100 | PASS |
| Commercial rights readiness | 35/100 | BLOCKED |

The commercial score is intentionally separate and is not averaged into the
technical score:

- STEERER checkpoint terms remain research-only pending clearance.
- PET code and official checkpoint are academic-use only.
- APGCC code is MIT, but the published checkpoint has no separate commercial
  terms; UCF remains a commercial candidate, not production-approved.
- JHU-CROWD++ and UP-COUNT remain research-only dataset lanes.

Therefore the combined result is `PASS_RESEARCH_ONLY`, not a production model
selection.

## Interpretation

- PET has the best count MAE on UCF and JHU among the three frozen checkpoints.
- STEERER has the highest auxiliary localization F1 on UCF and JHU; APGCC has
  the highest localization F1 on UP-COUNT and is notably fast on JHU, but its
  UP-COUNT count error is poor.
- STEERER remains the only evaluated model here with a native density map, so
  it is still the direct fit for density heatmaps and density-grid operations.
- No single winner should be declared by pooling these datasets or spatial
  metrics. A commercial shortlist must use commercially cleared data and
  weights, then repeat this frozen protocol before any fine-tuning decision.
