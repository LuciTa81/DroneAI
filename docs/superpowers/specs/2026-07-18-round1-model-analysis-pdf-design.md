# Round 1 Crowd-Counting Model Analysis PDF Design

## Purpose and audience

Create a Korean, 12-15 page personal-review PDF that the AI researcher can
inspect before deciding what to show company leadership. The report explains
the six-model Round 1 result, shows real curated outputs, separates technical
performance from commercialization rights, and ends with a decision and next
validation roadmap.

The report is not a marketing brochure and must not present the 36-image
cross-domain screen as an official benchmark ranking or deployment approval.

## Evidence sources

The report must derive numeric and rights claims from these frozen artifacts:

- `results/round1-cctv-comparison-34ad450/comparison.json`
- `results/round1-cctv-comparison-34ad450/comparison.csv`
- `results/round1-cctv-comparison-34ad450/evidence-manifest.json`
- `configs/evaluation/round1_cctv_scorecard.json`
- the six accepted 36-image runs on home5090 under
  `/workspace/data/results`
- each model's repository model brief and pinned source metadata for
  architecture descriptions and references

All displayed values must be copied or computed directly from these sources.
The report must retain the frozen dataset and split identity:

- dataset: `ucf-qnrf-kaggle-apache`
- split: `dm-count-upstream-qnrf-val-smoke-36-v1`
- role: `validation`
- samples: 36, with 12 low, 12 medium, and 12 high density samples
- canonical split SHA-256:
  `da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67`

## Document format and visual language

- Page size: A4 landscape throughout so four-column evaluation panels remain
  legible.
- Length: 15 pages, including the evidence and references page.
- Language: Korean body text with original English model and metric names.
- Typography: a Korean sans-serif font embedded in the PDF, with a restrained
  navy, cyan, amber, and gray palette.
- Footer: report title, `Round 1 / 36-image validation`, and page number.
- Every score page must state that the 100-point value is an operational
  harness score, not an accuracy percentage.
- Missing or inapplicable metrics must display as `N/A`, never zero.

## Page structure

1. Cover - project, report scope, date, and one-line conclusion.
2. Executive conclusion - provisional top three, product interpretation, and
   immediate recommendation.
3. Problem definition - fixed-CCTV crowd density, zone heatmap, and multi-camera
   calibration/merge context.
4. Evaluation design - input, outputs, 36-image split, technical score weights,
   and rights-independent decision logic.
5. Overall comparison - technical score chart, count/spatial/runtime summary,
   and explicit non-ranking limitation.
6. STEERER - architecture family, input/output/count method, metrics,
   demonstrated functions, strengths, limitations, rights, and examples.
7. DM-Count - the same model-analysis template.
8. PET - the same model-analysis template, clearly marked research-only.
9. MPCount - the same model-analysis template.
10. APGCC - the same model-analysis template.
11. CSRNet - the same model-analysis template and baseline interpretation.
12. Density-family visual comparison - DM-Count, STEERER, MPCount, and CSRNet
    on the common selected sample `fd9373c130ac1077`.
13. Point/hybrid visual comparison - STEERER, PET, and APGCC on the common
    selected sample `8e0a4dbe072725a1`.
14. Commercialization and decision matrix - code, dataset, pretrained weight,
    derived weight, and deployment rights shown independently from technical
    score.
15. Conclusion and roadmap - recommended candidates, CCTV calibration/merge
    validation, field data collection, drone-domain follow-up, and the final
    fine-tuning gate; include evidence hashes and compact references.

## Model analysis contract

Each model page must include:

1. Architecture family and practical structure at a level useful to an AI
   researcher.
2. Input contract: RGB image/frame and relevant preprocessing assumptions.
3. Native output: density map, point set, or hybrid output.
4. Count method: density integral or predicted point count.
5. Raw metrics: MAE, RMSE, MAPE reference, signed bias, zone MAE, applicable
   GAME/density or localization metrics, median latency, FPS, and peak VRAM.
6. Demonstrated operational capabilities: zone aggregation, density map,
   point localization, and confidence.
7. Fixed-CCTV strengths, known weaknesses, and calibration implications.
8. Overall rights scope plus component-level rights status.
9. Two real curated panels: one representative lower-error case and one
   limitation case selected deterministically from the accepted panel set.

The report must avoid declaring a panel "best" unless its selection rule and
error value support that wording. Use "lower-error example" and "limitation
example" instead.

## Visual example selection

Only existing accepted panels may be used; no model inference is rerun. Pull
only the panels needed for the report into `tmp/pdfs/` and keep them out of Git.

- Per-model pages: choose two panels among each run's 12 hash-verified curated
  panels using recorded per-sample error and density band. Prefer one median or
  lower-error example and one high-error/high-density limitation example.
- Density comparison: use the already selected common sample
  `fd9373c130ac1077` for DM-Count, STEERER, MPCount, and CSRNet.
- Point/hybrid comparison: use the already selected common sample
  `8e0a4dbe072725a1` for STEERER, PET, and APGCC.
- Every image caption must name the model, sample, density band when available,
  ground-truth count, predicted count, and relevant spatial metric.

## Interpretation and conclusion

The report conclusion must distinguish three layers:

1. Technical screen: STEERER, DM-Count, and PET are the provisional top three.
2. Product screen: PET remains research-only; STEERER and DM-Count are the
   leading product candidates but still require component-level due diligence.
3. Next validation: validate fixed-CCTV ROI, perspective correction, zone
   calibration, camera-overlap ownership, and merge behavior before any
   fine-tuning decision.

The report must not recommend training every model. Fine-tuning remains a final
gate for at most one rights-eligible winner after static and field/domain
validation.

## Generation and quality assurance

- Generate the PDF programmatically from frozen JSON/CSV evidence and accepted
  image paths.
- Save the final file under `output/pdf/` with a stable descriptive filename.
- Save intermediate assets only under `tmp/pdfs/`.
- Render every PDF page to PNG using Poppler.
- Visually inspect all rendered pages for Korean glyph coverage, clipping,
  overlap, panel legibility, table fit, consistent margins, headers, footers,
  and page numbering.
- Extract text with `pdftotext` or `pypdf` and verify all six model names,
  the shortlist, the non-ranking disclaimer, dataset/split identity, and rights
  labels.
- Verify the final page count is 15 and record the PDF SHA-256.

## Out of scope

- New inference, training, or fine-tuning
- Test-split access
- Queue advancement
- New rights approval or legal conclusion
- Unverified performance claims
- Full raw dataset, checkpoint, or density-array inclusion in Git
