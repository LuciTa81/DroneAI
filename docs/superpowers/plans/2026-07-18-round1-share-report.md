# Round 1 공유용 기술 보고서 Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a Korean, A4-portrait, nine-page technical report that shares the frozen Round 1 fixed-CCTV model comparison through narrative sections, tables, and hash-verified output figures.

**Architecture:** A dedicated ReportLab renderer consumes the already accepted comparison JSON, Korean model content, and packaged panel manifest. It validates the frozen evaluation scope and every image hash before atomically writing an exact nine-page portrait PDF. Poppler rendering and pypdf extraction form the delivery gate; no inference or training is required.

**Tech Stack:** Python 3.12, ReportLab, Pillow, pypdf, Poppler, pytest, existing DroneAI integrity and report-content modules.

## Global Constraints

- Generate exactly nine A4 portrait pages in Korean.
- Use only `ucf-qnrf-kaggle-apache` split `dm-count-upstream-qnrf-val-smoke-36-v1`, role `validation`, 36 samples.
- Require canonical split SHA-256 `da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67`.
- Use the existing official-checkpoint cross-domain compatibility smoke results; do not describe them as a formal model ranking.
- Do not run inference, training, fine-tuning, test-split access, or queue advancement.
- Treat PET as a research-only comparison model.
- Do not equate `PASS_COMMERCIAL_CANDIDATE` with production or deployment approval.
- Use only SHA-256-verified common panels: density `img_0097` and point/hybrid `img_0062`.
- Use report-style continuous text, numbered sections, tables, figure numbers, captions, small headers, and page footers; do not use presentation-style large cards.
- Keep the PDF and rendered intermediates outside Git until the user explicitly approves publication.

---

### Task 1: Implement and test the portrait report renderer

**Files:**
- Create: `src/droneai/round1_share_report.py`
- Create: `scripts/build_round1_share_report.py`
- Create: `tests/test_round1_share_report.py`

**Interfaces:**
- Consumes: comparison JSON, assets manifest and image package, Korean content JSON, output path, explicit report date.
- Produces: `build_round1_share_pdf(comparison_path: str | Path, assets_manifest_path: str | Path, content_path: str | Path, output_path: str | Path, *, report_date: str) -> Path`.

- [ ] **Step 1: Write the failing page-contract and asset-integrity tests**

```python
def test_share_report_is_exact_nine_page_a4_portrait(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-18",
    )
    reader = PdfReader(output)
    assert len(reader.pages) == 9
    for page in reader.pages:
        assert float(page.mediabox.height) > float(page.mediabox.width)
        assert abs(float(page.mediabox.width) - A4[0]) < 1
        assert abs(float(page.mediabox.height) - A4[1]) < 1

def test_share_report_contains_required_claim_boundaries(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH, manifest, CONTENT_PATH, tmp_path / "share.pdf",
        report_date="2026-07-18",
    )
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    for token in (
        "STEERER", "DM-Count", "PET", "MPCount", "APGCC", "CSRNet",
        "UCF-QNRF", "validation 36장", "fine-tuning 없음",
        "cross-domain compatibility smoke", "PET는 연구 비교군",
        "공식 모델 순위가 아님",
    ):
        assert token in text

def test_share_report_rejects_changed_asset_hash(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    payload = json.loads(manifest.read_text(encoding="utf-8"))
    target = manifest.parent / payload["assets"][0]["packaged_path"]
    target.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        build_round1_share_pdf(
            COMPARISON_PATH, manifest, CONTENT_PATH, tmp_path / "share.pdf",
            report_date="2026-07-18",
        )
```

- [ ] **Step 2: Run RED**

```powershell
$env:PYTHONPATH='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages'
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round1_share_report.py -q
```

Expected: collection fails because `droneai.round1_share_report` does not exist.

- [ ] **Step 3: Implement strict input validation and report primitives**

Implement these constants and public function:

```python
PAGE_SIZE = A4
PAGE_COUNT = 9

def build_round1_share_pdf(
    comparison_path: str | Path,
    assets_manifest_path: str | Path,
    content_path: str | Path,
    output_path: str | Path,
    *,
    report_date: str,
) -> Path:
    ...
```

Validation must require the frozen dataset ID, split ID, validation role, 36
samples, canonical split hash, `ranking_eligible=False`, shortlist label
`PROVISIONAL_TECHNICAL_SHORTLIST`, six exact model IDs, and common comparison
samples `img_0097` and `img_0062`. Recompute every referenced packaged image
SHA-256 before opening the output file.

Add report primitives for:

```python
def draw_report_header(canvas, section: str, *, regular_font: str) -> None: ...
def draw_report_footer(canvas, page_number: int, *, regular_font: str) -> None: ...
def draw_section_title(canvas, number: str, title: str, subtitle: str, *, regular_font: str, bold_font: str) -> None: ...
def draw_report_table(canvas, rows: list[list[str]], column_widths: tuple[float, ...], x: float, y: float, *, row_height: float, fonts: tuple[str, str]) -> float: ...
def draw_figure(canvas, image_path: Path, caption: str, x: float, y: float, width: float, height: float, *, regular_font: str) -> None: ...
```

Use Malgun Gothic via the existing `register_korean_fonts` helper. Use the
existing `draw_wrapped` helper for paragraphs. The body uses 8.5-9.5 pt type,
11-14 pt leading, restrained navy/cyan rules, no metric-card grid, and no
landscape pages.

- [ ] **Step 4: Implement the nine fixed report pages**

Render pages in this exact order:

1. Cover with title, scope, date, and one-line recommendation.
2. Executive summary with numbered findings and interpretation limits.
3. Evaluation design with dataset/split table and common/family-specific metrics.
4. Six-model architecture/input/output/count/CCTV comparison table.
5. Performance table with technical score, MAE, RMSE, spatial metric, FPS, VRAM and narrative interpretation.
6. Density output comparison with DM-Count, STEERER, MPCount, CSRNet `img_0097` figures in a 2x2 report grid.
7. Point/hybrid output comparison with STEERER, PET, APGCC `img_0062` figures and localization interpretation.
8. Rights-component table plus fixed-CCTV calibration and overlap ownership guidance.
9. Final recommendation, ordered roadmap, evidence identities, and non-ranking disclaimer.

Every non-cover page must contain a small report header and a `page / 9` footer.
Section-based figure captions must begin with `그림 5-` or `그림 6-`; model,
performance, and rights tables must begin with `표 3-1`, `표 4-1`, and
`표 7-1` respectively.

- [ ] **Step 5: Add an atomic CLI**

Create `scripts/build_round1_share_report.py` with required flags
`--comparison`, `--assets-manifest`, `--content`, `--output`, and
`--report-date`. Write to `.{output.name}.tmp.pdf`, validate nine pages with
pypdf, rename atomically, print structured `PASS_RESEARCH_ONLY` JSON on success,
and return structured JSON with exit code 2 on failure.

- [ ] **Step 6: Run GREEN, focused verification, and commit**

```powershell
$env:PYTHONPATH='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages'
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round1_share_report.py tests/test_round1_pdf_report.py -q
git diff --check
git add src/droneai/round1_share_report.py scripts/build_round1_share_report.py tests/test_round1_share_report.py
git commit -m "Generate Round 1 shareable technical report"
```

Expected: all focused tests pass and only private `output/` and `tmp/` remain untracked.

---

### Task 2: Generate and visually verify the share report

**Files:**
- Create locally: `output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_검토본.pdf`
- Create locally: `tmp/pdfs/round1-share-report/rendered/`

**Interfaces:**
- Consumes: committed Task 1 renderer, frozen comparison JSON, existing verified panel package.
- Produces: one untracked nine-page PDF, rendered PNG evidence, and final SHA-256.

- [ ] **Step 1: Run the focused and full test suites**

```powershell
$env:PYTHONPATH='C:\Users\user\.cache\codex-runtimes\codex-primary-runtime\dependencies\python\Lib\site-packages'
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest tests/test_round1_share_report.py tests/test_round1_pdf_report.py -q
& 'C:\Users\user\Documents\Codex\2026-07-13\new-chat-4\work\DroneAI\.venv\Scripts\python.exe' -m pytest -q
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Generate the PDF from accepted evidence**

```powershell
& $python scripts/build_round1_share_report.py `
  --comparison results/round1-cctv-comparison-34ad450/comparison.json `
  --assets-manifest tmp/pdfs/round1-model-analysis-1dc5974/assets/assets-manifest.json `
  --content configs/reporting/round1_model_analysis_ko.json `
  --output output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_검토본.pdf `
  --report-date 2026-07-18
```

Expected: `PASS_RESEARCH_ONLY`, `pages=9`, and a non-empty PDF.

- [ ] **Step 3: Render and inspect all pages**

```powershell
& $pdfinfo output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_검토본.pdf
& $pdftoppm -png -r 150 output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_검토본.pdf tmp/pdfs/round1-share-report/rendered/page
```

Create contact sheets for pages 1-3, 4-6, and 7-9, inspect all three with
`view_image`, and inspect pages 4-8 at original detail. Reject the report for
Korean tofu glyphs, clipped rows, overlap, illegible figures, slide-like empty
space, missing captions, inconsistent margins, or missing header/footer. Revise,
regenerate under a new filename, and repeat until the latest render has zero
defects.

- [ ] **Step 4: Verify page geometry, text, assets, and final hash**

Use pypdf to require nine portrait A4 pages and all required tokens from Task 1.
Recompute every packaged panel SHA-256 from the assets manifest. Record the PDF
byte size and SHA-256. Require `git ls-files -- output/pdf/...` to return no path.

- [ ] **Step 5: Push only renderer code and synchronize home5090**

Push the implementation commit, then run `git pull --ff-only` on the Pop!_OS
host repository `/home/lucita/crowd-counting-lab`. Verify host and container see
the same commit. Do not add the PDF or intermediate images.

- [ ] **Step 6: Deliver the local review copy**

Return a clickable local link, nine-page count, byte size, SHA-256, test results,
and the claim-boundary note. Stop before external publication, model advancement,
or fine-tuning.
