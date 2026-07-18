# Readable Round 1 Share Report Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Reflow the existing Round 1 share report into an exact 12-page A4 portrait PDF with readable body, table, caption, header, and footer typography while preserving all accepted evidence and claim boundaries.

**Architecture:** Keep the existing comparison, content, asset validation, CLI, and atomic-output interfaces. Add an explicit typography/layout contract, split dense model and output sections into multiple pages, and regenerate under a new filename. Automated tests enforce page count, page geometry, minimum font sizes, model grouping, output grouping, and required text; Poppler rendering is the final visual gate.

**Tech Stack:** Python 3.12, ReportLab, Pillow, pypdf, Poppler, pytest, existing DroneAI PDF and integrity helpers.

## Global Constraints

- Generate exactly 12 A4 portrait pages.
- Use body text at 10.5pt or larger with 15pt or larger leading.
- Use table headers and table body at 8.5pt or larger.
- Use figure captions at 8.5pt or larger with 11pt or larger leading.
- Use report headers and footers at 8pt or larger.
- Use full-content-width output panels for the seven common comparison figures.
- Preserve UCF-QNRF validation 36-image cross-domain compatibility smoke scope.
- Preserve canonical split SHA-256 `da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67`.
- Preserve PET as research-only and never claim production approval or an official model ranking.
- Do not run inference, training, fine-tuning, test-split access, or queue advancement.
- Keep generated PDFs and rendered images untracked.
- Do not merge to `main` until the user reviews the new PDF.

---

### Task 1: Enforce readable typography and 12-page layout

**Files:**
- Modify: `src/droneai/round1_share_report.py`
- Modify: `tests/test_round1_share_report.py`

**Interfaces:**
- Consumes: the existing `build_round1_share_pdf(...)` inputs and accepted comparison/assets/content files.
- Produces: the unchanged public builder and a new testable `readability_contract() -> dict[str, float]` plus `report_page_contract() -> tuple[tuple[str, ...], ...]`.

- [ ] **Step 1: Change tests first to require 12 pages and minimum typography**

```python
from droneai.round1_share_report import readability_contract, report_page_contract

def test_share_report_is_exact_twelve_page_a4_portrait(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH, manifest, CONTENT_PATH, tmp_path / "share.pdf",
        report_date="2026-07-19",
    )
    reader = PdfReader(output)
    assert len(reader.pages) == 12
    assert all(float(page.mediabox.height) > float(page.mediabox.width) for page in reader.pages)

def test_readability_contract_has_no_small_report_text() -> None:
    contract = readability_contract()
    assert contract["body_size"] >= 10.5
    assert contract["body_leading"] >= 15
    assert contract["table_header_size"] >= 8.5
    assert contract["table_body_size"] >= 8.5
    assert contract["caption_size"] >= 8.5
    assert contract["caption_leading"] >= 11
    assert contract["header_footer_size"] >= 8

def test_report_page_contract_splits_dense_content() -> None:
    pages = report_page_contract()
    assert len(pages) == 12
    assert pages[3] == ("steerer", "dm-count", "pet")
    assert pages[4] == ("mpcount", "apgcc", "csrnet")
    assert pages[6] == ("dm-count", "steerer")
    assert pages[7] == ("mpcount", "csrnet")
    assert pages[8] == ("steerer", "pet")
    assert pages[9] == ("apgcc",)
```

- [ ] **Step 2: Run RED**

```powershell
$env:PYTHONPATH='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages'
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round1_share_report.py -q
```

Expected: failures because the current report has nine pages and the new contracts do not exist.

- [ ] **Step 3: Add explicit typography and page contracts**

```python
PAGE_COUNT = 12
BODY_SIZE = 10.5
BODY_LEADING = 15.0
TABLE_HEADER_SIZE = 8.5
TABLE_BODY_SIZE = 8.5
CAPTION_SIZE = 8.5
CAPTION_LEADING = 11.0
HEADER_FOOTER_SIZE = 8.0

def readability_contract() -> dict[str, float]:
    return {
        "body_size": BODY_SIZE,
        "body_leading": BODY_LEADING,
        "table_header_size": TABLE_HEADER_SIZE,
        "table_body_size": TABLE_BODY_SIZE,
        "caption_size": CAPTION_SIZE,
        "caption_leading": CAPTION_LEADING,
        "header_footer_size": HEADER_FOOTER_SIZE,
    }

def report_page_contract() -> tuple[tuple[str, ...], ...]:
    return (
        ("cover",), ("summary",), ("evaluation",),
        ("steerer", "dm-count", "pet"),
        ("mpcount", "apgcc", "csrnet"),
        ("performance",),
        ("dm-count", "steerer"),
        ("mpcount", "csrnet"),
        ("steerer", "pet"),
        ("apgcc",), ("rights",), ("conclusion",),
    )
```

- [ ] **Step 4: Reflow tables and prose around the minimum typography**

Update `draw_report_header`, `draw_report_footer`, `draw_report_table`,
`draw_figure`, `_bullets`, and evidence text to consume the constants. Split the
model table into two renderers with three 115-125pt rows each. Keep the
performance table on one page with 8.5pt text and at least 52pt rows. Retain all
six models and the same numeric values.

- [ ] **Step 5: Split output figures into four full-width pages**

Implement one reusable renderer that accepts a title, subtitle, group name, and
ordered model IDs. Density I renders DM-Count and STEERER; Density II renders
MPCount and CSRNet; Point/Hybrid I renders STEERER and PET; Point/Hybrid II
renders APGCC. Each panel uses the full 511pt content width and a caption at
8.5pt or larger. The page narrative uses 10.5pt text.

- [ ] **Step 6: Assemble the exact 12-page renderer list**

Use the page order from `report_page_contract()`. Number sections 01 through 11
after the cover, update table and figure captions consistently, and make every
non-cover footer display `page / 12`.

- [ ] **Step 7: Run GREEN and commit**

```powershell
& $python -m pytest tests/test_round1_share_report.py tests/test_round1_pdf_report.py -q
git diff --check
git add src/droneai/round1_share_report.py tests/test_round1_share_report.py
git commit -m "Enlarge Round 1 share report typography"
```

Expected: all focused tests pass.

---

### Task 2: Generate and visually verify the enlarged review copy

**Files:**
- Create locally: `output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_확대검수본.pdf`
- Create locally: `tmp/pdfs/round1-share-report-readable/rendered/`

**Interfaces:**
- Consumes: committed Task 1 renderer, frozen comparison JSON, verified assets manifest, Korean content JSON.
- Produces: one untracked 12-page PDF, 12 rendered PNGs, contact sheets, and final SHA-256.

- [ ] **Step 1: Run focused and full tests**

```powershell
& $python -m pytest tests/test_round1_share_report.py tests/test_round1_pdf_report.py -q
& $python -m pytest -q
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Generate the enlarged PDF**

```powershell
& $python scripts/build_round1_share_report.py `
  --comparison results/round1-cctv-comparison-34ad450/comparison.json `
  --assets-manifest tmp/pdfs/round1-model-analysis-1dc5974/assets/assets-manifest.json `
  --content configs/reporting/round1_model_analysis_ko.json `
  --output output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_확대검수본.pdf `
  --report-date 2026-07-19
```

Expected: `PASS_RESEARCH_ONLY`, `pages=12`, non-empty output.

- [ ] **Step 3: Render and inspect every page**

Render at 150 DPI with bundled Poppler. Create contact sheets for pages 1-4,
5-8, and 9-12. Inspect all contact sheets and pages 4-11 at original detail.
Reject the report for text below the declared contract, clipped table rows,
caption wrapping outside figures, unreadable panel internals, Korean tofu,
overlap, missing footer, or excessive empty space. Regenerate under a new
filename and repeat until the latest render has no visible defects.

- [ ] **Step 4: Verify geometry, text, hashes, and Git scope**

Use pypdf to require 12 A4 portrait pages and all six models, UCF-QNRF,
validation 36 images, no fine-tuning, cross-domain compatibility smoke, PET
research-only, non-ranking disclaimer, section/table/figure captions, and final
roadmap. Recompute all 19 packaged asset hashes. Require 12 rendered PNGs and
confirm the final PDF is untracked. Record byte size and SHA-256.

- [ ] **Step 5: Push only code and synchronize home5090**

Push the feature branch and run `git pull --ff-only` on the Pop!_OS host repo.
Verify host and container commits match. Do not add the PDF.

- [ ] **Step 6: Deliver before merge**

Return the enlarged PDF link and verification evidence. Preserve the feature
worktree and wait for user confirmation before merging to `main`.
