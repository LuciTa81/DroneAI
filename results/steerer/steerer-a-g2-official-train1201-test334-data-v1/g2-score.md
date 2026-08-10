# steerer-a-g2-official-train-test-data: STEERER A full official Train/Test preparation

- Status: **PASS_COMMERCIAL_CANDIDATE**
- Decision scope: **commercial_candidate**
- Score: **100.0/100**
- Pass threshold: **100.0**
- Generated at: `2026-08-10T05:33:46.118644+00:00`

| Result | Category | Check | Points | Blocker | Observed |
|---|---|---|---:|:---:|---|
| PASS | split | Train1201, validation0, Test334 are exact | 20.0/20.0 | Y | 1201/0/334 |
| PASS | split | Train and Test identities/content are disjoint | 20.0/20.0 | Y | [] |
| PASS | integrity | Every prepared image and annotation hash was recalculated | 20.0/20.0 | Y | True |
| PASS | integrity | Dataset and manifest SHA-256 identities are valid | 20.0/20.0 | Y | dataset=ac55c9fa3e381a5511dd7f63206e1e9883006562f02364bb1ec49dda975d4092; manifest=57c5b26205f8146a5593dfa98a17a007aaa7582cb91c85c4938f88f444706a35 |
| PASS | rights | MIT code, user-accepted dataset basis, ImageNet-only initialization, and no production approval | 20.0/20.0 | Y | PASS_COMMERCIAL_CANDIDATE; production=False; official checkpoint forbidden |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.
