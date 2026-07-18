# Round 1 Dual-Scenario Share Report Design

## Purpose

Revise the Round 1 fixed-CCTV share report so every evaluated model is shown on
the same two UCF-QNRF validation inputs:

- `img_0775`, an illustrative moderate-count scene with ground truth 195;
- `img_0221`, an illustrative high-density scene with ground truth 1,762.

The figures are sample-level illustrations. They do not replace the frozen
36-image aggregate metrics and must not be described as an official ranking or
as representative accuracy for the complete dataset.

## Scope and boundaries

The change covers report-only inference, curated figure packaging, the share-PDF
input contract, and the report layout. It does not train or fine-tune a model,
access a test split, alter the 36-image benchmark, advance the evaluation queue,
or change any rights decision.

All twelve report panels are regenerated from the existing rights-gated model
environments and checkpoints. Existing benchmark result directories remain
immutable. New native outputs and logs stay under `/workspace/data/results`;
only the twelve curated panels, compact manifests, code, tests, and report
metadata may be committed, subject to the existing 25 MiB Git budget.

## Alternatives considered

### Recommended: two pages per model

Use one moderate-scene page and one high-density page for each model. This keeps
the four-part composite output readable and preserves the larger text sizes the
share report already enforces. The report grows from 12 to 18 pages.

### Rejected: two figures on one model page

This preserves the 12-page length, but both wide composite panels become too
small and force the structure, rights, and operational notes into cramped cards.

### Rejected: separate cross-model galleries only

Two gallery pages would compare scenarios efficiently, but the figures would be
too small and would separate each model's outputs from its architecture and
operational interpretation.

## Page contract

The PDF has exactly 18 A4 portrait pages:

1. cover;
2. summary and boundaries;
3. evaluation data and conditions;
4-5. STEERER moderate and high-density pages;
6-7. DM-Count moderate and high-density pages;
8-9. PET moderate and high-density pages;
10-11. MPCount moderate and high-density pages;
12-13. APGCC moderate and high-density pages;
14-15. CSRNet moderate and high-density pages;
16. frozen 36-image performance comparison;
17. rights and fixed-CCTV application;
18. conclusion and roadmap.

Each moderate page contains the aggregate metric strip, the `img_0775` output
panel, GT/prediction/absolute error/error-rate caption, native output and count
method, and concise model structure. Each high-density page contains the
`img_0221` panel, the same sample metrics, the applicable density/zone or
localization metric, operational interpretation, limitation, and rights scope.

The report uses neutral scenario labels such as `적정 인원 공통 장면` and
`고밀도 공통 장면`. Sample captions and narrative must not contain promotional
selection language such as `강점 사례`, `best case`, `winner`, or an assertion
that a sample-level figure proves the aggregate ranking.

## Scenario asset contract

The PDF input manifest advances to a scenario-based comparison structure:

```json
{
  "schema_version": 2,
  "scenarios": {
    "moderate": {
      "sample_id": "img_0775",
      "ground_truth_count": 195,
      "models": {"steerer": {}, "dm-count": {}, "pet": {}, "mpcount": {}, "apgcc": {}, "csrnet": {}}
    },
    "high_density": {
      "sample_id": "img_0221",
      "ground_truth_count": 1762,
      "models": {"steerer": {}, "dm-count": {}, "pet": {}, "mpcount": {}, "apgcc": {}, "csrnet": {}}
    }
  }
}
```

Every model entry records the source and packaged panel paths and SHA-256 values,
model/output type, sample ID, GT and predicted counts, signed and normalized
errors, applicable spatial metric, source Git/upstream/checkpoint identity, and
the existing rights scope. The package builder fails unless each scenario has
all six models, every model uses the frozen scenario sample, the sample image and
annotation identities agree, and every packaged asset hash recomputes exactly.

## Report-only inference flow

1. Verify Windows and home5090 repositories are clean and at the same commit.
2. Verify the active queue remains at CSRNet `review_and_advance`, validation 36,
   with `fine_tuning=false`.
3. For each of the six pinned model environments, run exactly `img_0775` and
   `img_0221` from the frozen validation split.
4. Write each run to a new report-scenario directory without overwriting accepted
   benchmark evidence.
5. Validate finite counts, native output contracts, point bounds or density mass,
   sample identity, checkpoint identity, and panel hashes.
6. Package the twelve accepted panels and render the 18-page report.

No result is accepted merely because an image file exists. Its manifest binding
and SHA-256 must pass before the PDF renderer opens it.

## Failure handling

- Missing model/scenario output blocks report generation.
- A changed panel, comparison JSON, sample identity, checkpoint, or split hash
  blocks packaging.
- Non-finite counts, negative operational density, out-of-bounds points, or a
  mismatched GT count blocks that model/scenario result.
- A single inference failure is reported with its model, sample, command, and
  log path; accepted benchmark runs are never rerun or replaced silently.
- Temporary PDF files are removed only by the existing atomic-render path; user
  output directories and SSD data are never recursively deleted.

## Verification

Development follows TDD. Tests first freeze:

- the exact 18-page contract and two pages per model;
- the two scenario IDs and GT values;
- all six model entries in both scenarios;
- at least one embedded output image on every model/scenario page;
- presence of GT, prediction, absolute error, error rate, spatial metric, rights
  scope, and the frozen 36-image aggregate metrics;
- absence of promotional selection terms;
- A4 portrait dimensions, readable font-size contracts, and no truncated text;
- all source and packaged SHA-256 values;
- no change to the evaluation queue, split role, or fine-tuning flag.

After focused tests and the full suite pass, render all 18 pages to PNG, inspect
each page and contact sheets, then recompute the final PDF SHA-256. The resulting
report remains `PASS_RESEARCH_ONLY`; technical and rights conclusions remain
independent.
