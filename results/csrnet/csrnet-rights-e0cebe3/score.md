# stage-3c: Commercial rights and deployment suitability

- Status: **PASS_COMMERCIAL_CANDIDATE**
- Decision scope: **commercial_candidate**
- Score: **88.0/100**
- Pass threshold: **80.0**
- Generated at: `2026-07-18T07:02:54.020651+00:00`

| Result | Category | Check | Points | Blocker | Observed |
|---|---|---|---:|:---:|---|
| PASS | manifest | Candidate identity and intended commercial scope are explicit | 10.0/10.0 | Y | {'schema_version': 1, 'candidate_id': 'csrnet-independent-shha-to-ucf-qnrf-research-comparison', 'intended_use': 'commercial_product_rnd'} |
| PASS | rights inventory | All five rights components are present exactly once | 15.0/15.0 | Y | code, dataset, deployment, derived_weights, pretrained_weights |
| PASS | rights inventory | Rights values and bases are internally consistent | 20.0/20.0 | Y | consistent |
| PASS | traceability | Every component has stable identity and source provenance | 15.0/15.0 | Y | recorded |
| PASS | decision | The bundle has one safe, non-contradictory scope | 20.0/20.0 | Y | PASS_COMMERCIAL_CANDIDATE: No explicit commercial prohibition is recorded, but at least one required permission is absent, silent, pending, or unverified. |
| PASS | production rights | code has verified commercial evidence | 4.0/4.0 |  | True |
| PASS | production rights | dataset has verified commercial evidence | 4.0/4.0 |  | True |
| FAIL | production rights | deployment has verified commercial evidence | 0.0/4.0 |  | False |
| FAIL | production rights | derived_weights has verified commercial evidence | 0.0/4.0 |  | False |
| FAIL | production rights | pretrained_weights has verified commercial evidence | 0.0/4.0 |  | False |

## Decision rule

A stage passes only when the score reaches the threshold and every blocker passes.
