# DroneAI review roadmap

The project does not select a model from one leaderboard. It proves three
claims in order:

1. Published code is reproducible.
2. The result generalizes to festival CCTV and drone imagery.
3. The result remains useful after latency, failures and operator workflow are included.

Each stage is scored out of 100. Passing requires both the threshold and all
blockers. Future numeric targets must be frozen before the corresponding test
set is opened.

## Gate sequence

| Stage | Main score groups | Threshold | Blockers |
|---|---|---:|---|
| 0. Foundation | Runtime 35, storage 25, reproducibility 25, automation 15 | 85 | CUDA smoke, writable profiled storage, commit SHA, tests |
| 1. Data | License/access 20, annotation integrity 25, split integrity 25, condition coverage 20, manifest 10 | 85 | Usage rights, corrupt files, train/test leakage |
| 2. CSRNet smoke | Pipeline correctness 30, count preservation 25, tiny-set overfit 20, determinism 10, artifacts 15 | 80 | Density sum preservation, valid evaluation |
| 3. Official reproduction | Paper gap 35, seed stability 15, evaluation integrity 20, traceability 15, runtime 15 | 85 | Test leakage, wrong split/metric; median paper gap must be within 5% |
| 4. Drone generalization | Count error 25, high-density undercount 25, domain drop 20, condition robustness 20, latency 10 | 80 | High-density undercount target, sealed test integrity |
| 5. Controlled field | Count 20, area calibration 20, alert recall 25, latency 15, failure signaling 10, operator review 10 | 85 | Missed critical event, failure reported as zero people |
| 6. Shadow mode | Event recall 25, false alarms 15, lead time 20, uptime/recovery 15, operator acceptance 15, audit completeness 10 | 85 | Critical-event recall and audit log completeness |

## Model and data order

1. Stage 0 uses no training dataset. It only verifies the experiment foundation.
2. Stage 1 uses UP-COUNT as the first non-commercial UAV research benchmark.
   DroneCrowd remains blocked until explicit dataset terms are obtained. Public
   datasets never become production-training assets merely because they can be downloaded.
3. Stage 2 uses CSRNet only as a pipeline smoke baseline.
4. Stage 3 reproduces DM-Count first, then P2PNet. MPCount is evaluated after the basic reproductions are stable.
5. Stage 4 starts with UP-COUNT for true-UAV generalization. DroneCrowd can be
   added after rights verification, DLR-ACD is an optional non-commercial
   extreme-density stress set, and MovingDroneCrowd is reserved for the later
   moving-drone individual-counting track.
6. Local festival data is used for fine-tuning and a sealed gold test, split by camera, date, zone or flight session rather than adjacent frame.

CSRNet is therefore not the proposed production winner. It is the cheapest way
to prove that loading, annotation, density integration, evaluation and artifact
logging all agree before more complex official repositories are introduced.

## User review points

At the end of every stage, review these four artifacts before proceeding:

1. `score.md`: readable pass/fail table and blockers.
2. `score.json`: exact values used by automation.
3. Stage configuration snapshot: thresholds, seeds and dataset/model references.
4. Evidence files: environment, predictions, plots, logs and checksums.

No test threshold may be relaxed after seeing the sealed test result without a
new versioned gate and an explicit review note.

New runs use `home5090_docker` by default; `colab_drive` is retained for
historical reproduction and fallback. Runtime success remains separate from
Stage 3C code, dataset, weight, and commercial-fit approval.
