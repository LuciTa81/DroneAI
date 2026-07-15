# Stage 3C — rights scope and production suitability

Stage 3C prevents a successful research result from being read as commercial
permission. It evaluates code, dataset, pretrained weights, newly derived
weights, and deployment approval as separate components.

## Outcomes

| Outcome | Interpretation | Allowed next action |
|---|---|---|
| `PASS_RESEARCH_ONLY` | Every component permits the declared research lane; commercial use is restricted. | Research reproduction only |
| `PASS_COMMERCIAL_CANDIDATE` | No explicit prohibition is recorded, but one or more downstream permissions are silent, pending, or unverified. | Catalog and due diligence; synthetic smoke when code is verified; frozen evaluation when code, dataset, and pretrained weights are all verified |
| `PRODUCTION_APPROVED` | All five components have verified evidence for commercial use. | Commercial training/deployment may proceed subject to the recorded scope |
| `BLOCKED` | Explicit prohibition, metadata contradiction, or missing required provenance/component. | Review and replace or obtain permission |

Candidate is not blanket permission. Its machine-readable actions are computed
from the verified component chain. Verified code permits a synthetic smoke;
verified code, dataset, and pretrained weights permit asset download and
frozen-checkpoint evaluation. Commercial training, derived-weight use,
product weight reuse, and deployment remain absent until their required
components are verified.

`PRODUCTION_APPROVED` retains all lower-scope actions, including synthetic
compatibility smoke, asset download, and frozen-checkpoint evaluation. Raising
the approval scope must never remove an already verified lower-scope action.

## DM-Count and ShanghaiTech current classification

The DM-Count implementation is pinned to an MIT-licensed commit. The current
ShanghaiTech Part A evidence does not establish dataset terms, and the upstream
checkpoint is not assumed to inherit the code license. Derived-weight and
deployment approvals are also pending. The bundle is therefore retained as
`PASS_COMMERCIAL_CANDIDATE` for rights investigation while its historical
research reproduction remains isolated. It is not a product-training asset.

## DM-Count and UCF-QNRF approved benchmark classification

For the UCF-QNRF lane, the project owner accepts the Apache-2.0 label on the
recorded Kaggle mirror as dataset permission. The manifest retains the mirror
uploader's statement that they are not the dataset owner as a visible risk
note. DM-Count code is MIT, and project policy treats the official checkpoint
tracked in that repository as covered by the repository license. These three
verified inputs authorize download and frozen-checkpoint evaluation. Derived
weights and product deployment remain pending, so the bundle stays
`PASS_COMMERCIAL_CANDIDATE` rather than `PRODUCTION_APPROVED`.

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
