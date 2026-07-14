# Stage 3C — rights scope and production suitability

Stage 3C prevents a successful research result from being read as commercial
permission. It evaluates code, dataset, pretrained weights, newly derived
weights, and deployment approval as separate components.

## Outcomes

| Outcome | Interpretation | Allowed next action |
|---|---|---|
| `PASS_RESEARCH_ONLY` | Every component permits the declared research lane; commercial use is restricted. | Research reproduction only |
| `PASS_COMMERCIAL_CANDIDATE` | No explicit prohibition is recorded, but one or more permissions are silent, pending, or unverified. | Catalog and due diligence; synthetic smoke only when code rights are verified |
| `PRODUCTION_APPROVED` | All five components have verified evidence for commercial use. | Commercial training/deployment may proceed subject to the recorded scope |
| `BLOCKED` | Explicit prohibition, metadata contradiction, or missing required provenance/component. | Review and replace or obtain permission |

Candidate does not mean allowed. In particular, it omits ambiguous asset
download, training, pretrained-weight reuse, derived-weight use, and deployment
from its machine-readable `allowed_actions`. If even the code rights are
ambiguous, synthetic smoke is omitted too.

## DM-Count and ShanghaiTech current classification

The DM-Count implementation is pinned to an MIT-licensed commit. The current
ShanghaiTech Part A evidence does not establish dataset terms, and the upstream
checkpoint is not assumed to inherit the code license. Derived-weight and
deployment approvals are also pending. The bundle is therefore retained as
`PASS_COMMERCIAL_CANDIDATE` for rights investigation while its historical
research reproduction remains isolated. It is not a product-training asset.

## Run

```bash
python scripts/run_stage3c.py \
  --manifest configs/candidates/dm_count_shanghaitech.candidate.json \
  --output-dir /workspace/data/results/stage-3c/dm-count-shanghaitech
```

Review `rights-decision.json` first, then `score.md`, `score.json`, and the
manifest snapshot. This engineering classification is not legal advice; a
commercial or deployment decision should use the recorded terms and appropriate
project/legal approval.
