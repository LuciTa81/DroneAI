# Stage 1 — Dataset readiness and split integrity

## Decision

The first data import is the **606 MB official UP-COUNT sequence-sample pack**,
restricted to non-commercial research experiments. The full image ZIP is
28.4 GB, so it is a later capacity-reviewed import rather than the first gate.
The commercial product path must use
locally collected data with explicit collection, privacy and model-training
rights.

| Dataset | Technical role | Rights decision | Current action |
|---|---|---|---|
| UP-COUNT samples | Stage 1 integrity pilot across the official sequences | CC BY-NC 4.0; commercial use is not allowed without separate permission | Import 606 MB pack to Drive and run gate |
| UP-COUNT full | Primary modern UAV research benchmark | Same non-commercial restriction; image ZIP is 28.4 GB | Defer until capacity review |
| DroneCrowd | Desirable true-drone video benchmark | Official repository has no `LICENSE`; README only states citation | Keep as a due-diligence candidate; block download/use until terms are confirmed |
| DLR-ACD | Optional extreme-density stress evaluation | CC BY-NC-ND 3.0; non-commercial and no-derivatives restrictions | Keep out of production training |
| Owned festival data | Production fine-tuning and sealed field evaluation | Must be documented by the project | Required before production claim |

This is not a claim that UP-COUNT represents total festival attendance. Its
point labels measure visible people in frames. Attendance, repeat visitors and
people hidden by occlusion are separate operational quantities.

## 100-point gate

| Group | Points | What is proven |
|---|---:|---|
| License/access | 20 | Official provenance, verified rights evidence and allowed intended use |
| Annotation integrity | 25 | Every image/annotation pair is intact; counts and point coordinates are valid |
| Split integrity | 25 | Splits are group-based; no sequence/session or exact-image leakage |
| Condition coverage | 20 | Train/validation/test exist and required operating tags are represented |
| Manifest | 10 | Every file is SHA-256 pinned and the exact manifest is snapshotted |

Pass threshold: **85/100**, with all blockers passing. A 99 cannot hide a
license failure, corrupt annotation, adjacent-frame leakage or missing hash.

## Normalized data contract

The dataset root contains `inventory.jsonl`. Each line represents one frame:

```json
{
  "sample_id": "0015/0000042",
  "split": "train",
  "group_id": "0015",
  "image_path": "images/0015/0000042.jpg",
  "annotation_path": "annotations/0015/0000042.json",
  "image_sha256": "...",
  "annotation_sha256": "...",
  "width": 3840,
  "height": 2160,
  "point_count": 713,
  "condition_tags": {
    "altitude_band": "mid",
    "density_band": "high"
  }
}
```

Normalized annotations are deliberately simple:

```json
{"points": [[120.5, 80.0], [410.0, 300.25]]}
```

The original files remain untouched. A dataset-specific adapter creates the
normalized annotations and inventory next to them so later model code does not
silently reinterpret coordinates or counts. UP-COUNT's official loader casts
coordinates to integers and clips negative or edge-overflow values to the image
bounds. The adapter reproduces that pinned behavior and records the number of
corrected points in every annotation and inventory row; it never edits the raw
label. The reference implementation is pinned to commit `a6d3664` in
`src/datamodule/dataset/upcount_dataset.py`.

## Run command

```bash
python scripts/run_stage1.py \
  --manifest configs/datasets/up_count.sample.research.json \
  --dataset-root /workspace/data/datasets/up-count-sequence-samples-v1 \
  --output-dir /workspace/data/results/stage-1
```

The historical Colab lane may use its recorded `/content/drive/MyDrive/DroneAI`
paths; new runs default to the home5090 profile above.

Until the dataset is downloaded and the adapter writes a non-empty inventory,
the command is expected to return exit code 2 and `BLOCKED`. That is intentional:
metadata alone cannot earn integrity points.

## Review checkpoint

Before Stage 2, inspect:

1. `score.md` and the failed-blocker list.
2. `manifest.snapshot.json`, especially `intended_use` and `allowed_uses`.
3. Split counts and any group or SHA-256 overlap.
4. At least three original image/normalized annotation pairs from low, medium
   and high density sequences.
5. Confirmation that no non-commercial dataset is used for production weights.
