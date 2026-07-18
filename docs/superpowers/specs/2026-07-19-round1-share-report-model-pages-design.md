# Round 1 Share Report Model Pages Design

## Goal

Reorganize the Round 1 share report so every evaluated model has its own page
where the model description and that model's actual accepted output image appear
together. Preserve the 12-page A4 portrait format and remove the `RTX 5090`
reference from the cover.

## Exact Page Structure

1. Cover
2. Executive summary and claim boundaries
3. Evaluation purpose, dataset, split, and metric definitions
4. STEERER model detail
5. DM-Count model detail
6. PET model detail
7. MPCount model detail
8. APGCC model detail
9. CSRNet model detail
10. Six-model performance comparison
11. Component rights and fixed-CCTV application
12. Recommendation, evidence identities, and roadmap

The former two model-table pages and four output-gallery pages are replaced by
six model-detail pages. No new pages are added.

## Cover Change

- Remove `RTX 5090` from the cover.
- Replace the execution-environment line with a neutral statement such as
  `batch 1 / fine-tuning 없음`.
- Keep the dataset, comparison scope, date, research-only boundary, and
  non-ranking disclaimer.

## Model Detail Page Contract

Every model page contains all of the following in one view:

1. Model name, family, backbone, and paper or architecture label.
2. Input preprocessing, native output, and count calculation.
3. Technical score, MAE, RMSE, relevant spatial metric, FPS, and peak VRAM.
4. At least one actual packaged output image generated during the accepted
   Round 1 run.
5. The image caption with sample ID, density band, GT, prediction, and spatial
   metric.
6. A short explanation of what the output means and how it maps to CCTV zones.
7. A concise strengths/limitations statement.
8. The independent rights scope and deployment caution.

The actual output image is not a decorative thumbnail. It must remain large
enough to distinguish input, ground truth, model prediction, and operator-zone
regions. The model name and output explanation must be on the same page as the
image.

## Model-to-Image Mapping

Use only already verified packaged assets from the accepted asset manifest:

- STEERER: accepted hybrid/density output panel, with density and localization
  behavior explained in the page text.
- DM-Count: accepted density comparison panel.
- PET: accepted point-prediction comparison panel.
- MPCount: accepted density comparison panel.
- APGCC: accepted point-prediction comparison panel.
- CSRNet: accepted operational-density comparison panel.

STEERER may include a second compact panel only if the primary full-width panel
and all text remain legible. One verified full-width image is the required
minimum for every model.

## Evidence and Claim Boundaries

- Dataset remains UCF-QNRF validation 36 images with low, medium, and high bands.
- The result remains a cross-domain compatibility smoke, not an official model
  ranking.
- Fine-tuning remains absent.
- PET remains `PASS_RESEARCH_ONLY` and is never presented as a product candidate.
- `PASS_COMMERCIAL_CANDIDATE` is not production approval.
- Existing metrics, scores, split identity, comparison hash, and asset hashes
  are not changed.
- Do not run inference, training, fine-tuning, test-split access, or queue
  advancement.

## Readability

- Keep body text at 10.5pt or larger and 15pt or larger leading.
- Keep table text at 8.5pt or larger.
- Keep figure captions at 8.5pt or larger and 11pt or larger leading.
- Use a full-content-width image panel on every model page.
- Avoid clipped captions, oversized table headers, excessive empty space,
  overlapping content, and unreadable image internals.

## Verification

Automated tests must require the exact 12-page model-page contract, the absence
of `RTX 5090` from extracted cover text, and the presence of all six model names,
model-specific output notes, metrics, and image panels. Generate a new untracked
PDF filename, render all 12 pages with Poppler, inspect each model page at
original detail, recompute all 19 packaged asset hashes, and confirm six model
pages contain at least six embedded images in total.

The revised PDF remains on the feature branch for user review. Do not merge to
`main` until the user approves the model-page layout.
