# Round 1 Share Report Model Pages Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the separated model tables and output galleries with six model-detail pages that each contain the model's actual accepted output image, while removing `RTX 5090` from the cover and preserving exactly 12 pages.

**Architecture:** Add an explicit model-page and asset-selection contract, then render one reusable detail page for each model using accepted comparison metrics, report content, rights scope, and packaged assets. Keep the existing cover, summary, evaluation, performance, rights, and conclusion renderers, with only the cover execution-environment wording changed.

**Tech Stack:** Python 3.12, ReportLab, pypdf, Pillow, pytest, Poppler, existing DroneAI integrity and report-content helpers.

## Global Constraints

- Keep exactly 12 A4 portrait pages.
- Pages 4-9 must be STEERER, DM-Count, PET, MPCount, APGCC, and CSRNet detail pages in that order.
- Every model page must contain at least one actual packaged output image from the accepted asset manifest.
- Remove `RTX 5090` and `5090` from the cover.
- Reuse accepted metrics, split identities, comparison JSON, and 19 packaged assets without inference or regeneration.
- Do not run inference, training, fine-tuning, test-split access, or queue advancement.
- Keep PET research-only and preserve all component-rights and non-ranking boundaries.
- Keep the PDF and rendered PNGs untracked and do not merge to `main` before user approval.

---

### Task 1: Enforce the model-page contract

**Files:**
- Modify: `tests/test_round1_share_report.py`
- Modify: `src/droneai/round1_share_report.py`

**Interfaces:**
- Consumes: accepted `comparison["models"]`, `assets["comparisons"]`, and `ModelContent` entries.
- Produces: `model_page_asset_contract() -> dict[str, tuple[str, str]]` and a new 12-page renderer order.

- [ ] **Step 1: Write failing contract and PDF tests**

Update the page contract assertion:

```python
def test_report_page_contract_has_six_model_detail_pages() -> None:
    pages = report_page_contract()
    assert pages == (
        ("cover",),
        ("summary",),
        ("evaluation",),
        ("steerer",),
        ("dm-count",),
        ("pet",),
        ("mpcount",),
        ("apgcc",),
        ("csrnet",),
        ("performance",),
        ("rights",),
        ("conclusion",),
    )
```

Add the asset contract test:

```python
def test_model_page_asset_contract_covers_every_model() -> None:
    assert model_page_asset_contract() == {
        "steerer": ("points", "steerer"),
        "dm-count": ("density", "dm-count"),
        "pet": ("points", "pet"),
        "mpcount": ("density", "mpcount"),
        "apgcc": ("points", "apgcc"),
        "csrnet": ("density", "csrnet"),
    }
```

Add a generated-PDF test that checks the cover and model pages:

```python
def test_each_model_page_has_its_output_image_and_no_5090_cover(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-19",
    )
    reader = PdfReader(output)
    assert "5090" not in (reader.pages[0].extract_text() or "")
    for page, model_id in zip(reader.pages[3:9], MODEL_PAGE_ORDER, strict=True):
        text = page.extract_text() or ""
        assert DISPLAY_NAMES[model_id] in text
        assert model_output_note(model_id) in " ".join(text.split())
        assert "MAE" in text and "RMSE" in text and "FPS" in text and "VRAM" in text
        assert len(page.images) >= 1
```

- [ ] **Step 2: Run RED**

```powershell
& $python -m pytest tests/test_round1_share_report.py -q
```

Expected: failures because the old page contract groups models and output images on separate pages, and the cover contains `RTX 5090`.

- [ ] **Step 3: Implement the contract and model-page renderer**

Define:

```python
MODEL_PAGE_ORDER = ("steerer", "dm-count", "pet", "mpcount", "apgcc", "csrnet")
MODEL_PAGE_ASSETS = {
    "steerer": ("points", "steerer"),
    "dm-count": ("density", "dm-count"),
    "pet": ("points", "pet"),
    "mpcount": ("density", "mpcount"),
    "apgcc": ("points", "apgcc"),
    "csrnet": ("density", "csrnet"),
}


def model_page_asset_contract() -> dict[str, tuple[str, str]]:
    return dict(MODEL_PAGE_ASSETS)
```

Change the cover execution line from `RTX 5090 / batch 1 / fine-tuning 없음` to
`batch 1 / fine-tuning 없음`.

Implement `_page_model_detail(c, ctx, *, model_id: str, number: str)` with:

- a section title using the display name and family;
- six metric cells for score, MAE, RMSE, spatial metric, FPS, and VRAM;
- readable wrapped input, native output, count method, and CCTV interpretation;
- one full-width `draw_figure` panel selected by `MODEL_PAGE_ASSETS`;
- the existing `model_output_note(model_id)`;
- strengths/limitations and `rights_scope` without changing claim boundaries.

Add `comparison_by_id` to the rendering context and replace the former model and
output renderers with six lambdas calling `_page_model_detail` in
`MODEL_PAGE_ORDER`. Renumber performance to 09, rights to 10, and conclusion to
11 while keeping the cover unnumbered.

- [ ] **Step 4: Run GREEN and focused regression tests**

```powershell
& $python -m pytest tests/test_round1_share_report.py tests/test_round1_pdf_report.py -q
git diff --check
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit code and tests**

```powershell
git add src/droneai/round1_share_report.py tests/test_round1_share_report.py
git commit -m "Show output image on every model report page"
```

---

### Task 2: Generate and verify the model-page review PDF

**Files:**
- Create locally: `output/pdf/DroneAI_Round1_CCTV_모델별_출력사진_공유용_검수본.pdf`
- Create locally: `tmp/pdfs/round1-share-report-model-pages/rendered-v1/`

**Interfaces:**
- Consumes: committed renderer, accepted comparison JSON, content JSON, and assets manifest.
- Produces: one untracked 12-page PDF and 12 rendered PNGs.

- [ ] **Step 1: Run the full suite**

```powershell
& $python -m pytest -q
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Generate the new PDF without overwriting prior copies**

```powershell
& $python scripts/build_round1_share_report.py `
  --comparison results/round1-cctv-comparison-34ad450/comparison.json `
  --assets-manifest tmp/pdfs/round1-model-analysis-1dc5974/assets/assets-manifest.json `
  --content configs/reporting/round1_model_analysis_ko.json `
  --output output/pdf/DroneAI_Round1_CCTV_모델별_출력사진_공유용_검수본.pdf `
  --report-date 2026-07-19
```

Expected: `PASS_RESEARCH_ONLY`, `pages=12`.

- [ ] **Step 3: Render and inspect every model page**

Render at 150 DPI with Poppler. Inspect pages 4-9 individually at original detail
and confirm each page shows the correct model name, actual input/output composite,
metrics, interpretation, and rights scope. Inspect contact sheets for all 12 pages
for clipping, overlap, Korean glyph errors, footer errors, and excess empty space.

- [ ] **Step 4: Verify evidence and Git scope**

Require 12 A4 portrait pages, no `5090` on page 1, six model pages with at least
one embedded image each, all model output notes and claim-boundary tokens, 12
rendered PNGs, 19 matching packaged asset hashes, and an untracked final PDF.
Record byte size and SHA-256.

- [ ] **Step 5: Push and synchronize home5090**

```powershell
git push origin agent/steerer-ucf-qnrf-adapter
ssh home5090-pop "git -C /home/lucita/crowd-counting-lab pull --ff-only"
```

Verify Windows, host, and container commits match. Do not add the PDF to Git and
do not merge to `main` before user review.
