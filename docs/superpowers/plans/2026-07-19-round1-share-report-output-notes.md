# Round 1 Share Report Output Notes Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Correct the oversized model-table header and add concise, model-specific output interpretation to all accepted output panels while preserving the exact 12-page report.

**Architecture:** Extend the existing table renderer with an optional header height while leaving its default behavior unchanged. Store six model-specific output notes in the report module and append them to the existing verified panel captions, so no inference or evidence regeneration is required.

**Tech Stack:** Python 3.12, ReportLab, pypdf, pytest, Poppler, existing DroneAI report integrity helpers.

## Global Constraints

- Keep exactly 12 A4 portrait pages.
- Reuse only accepted comparison JSON and the 19 packaged assets.
- Do not run inference, training, fine-tuning, test-split access, or queue advancement.
- Keep PET research-only and preserve all existing rights and non-ranking disclaimers.
- Keep captions at least 8.5pt with at least 11pt leading.
- Keep the generated PDF and rendered PNGs untracked.
- Do not merge to `main` before user approval of the revised PDF.

---

### Task 1: Correct the model-table header height

**Files:**
- Modify: `src/droneai/round1_share_report.py`
- Modify: `tests/test_round1_share_report.py`

**Interfaces:**
- Consumes: `draw_report_table(..., row_height: float, fonts: tuple[str, str])`.
- Produces: `draw_report_table(..., header_row_height: float | None = None) -> float` and `model_table_height_contract() -> dict[str, float]`.

- [ ] **Step 1: Write the failing tests**

Add a contract test and a real ReportLab drawing test:

```python
def test_model_table_uses_compact_header_and_readable_body() -> None:
    contract = model_table_height_contract()
    assert contract == {"header": 42.0, "body": 126.0}


def test_report_table_can_size_header_independently() -> None:
    buffer = BytesIO()
    pdf = canvas.Canvas(buffer, pagesize=A4)
    end_y = draw_report_table(
        pdf,
        [["Header"], ["Body 1"], ["Body 2"]],
        (120,),
        42,
        700,
        row_height=60,
        header_row_height=30,
        fonts=register_korean_fonts(),
    )
    assert end_y == 550
```

- [ ] **Step 2: Run RED**

```powershell
& $python -m pytest tests/test_round1_share_report.py -q
```

Expected: import/signature failures because the new contract and header option do not exist.

- [ ] **Step 3: Implement the minimal table change**

```python
MODEL_TABLE_HEADER_HEIGHT = 42.0
MODEL_TABLE_BODY_HEIGHT = 126.0


def model_table_height_contract() -> dict[str, float]:
    return {"header": MODEL_TABLE_HEADER_HEIGHT, "body": MODEL_TABLE_BODY_HEIGHT}
```

Inside `draw_report_table`, calculate each row independently:

```python
current_height = (
    header_row_height
    if row_index == 0 and header_row_height is not None
    else row_height
)
row_y = y - current_height
```

Use `current_height` for the row rectangle, text origin, and maximum line count.
Call the model table with `row_height=126` and `header_row_height=42`.

- [ ] **Step 4: Run GREEN**

```powershell
& $python -m pytest tests/test_round1_share_report.py -q
```

Expected: all share-report tests pass.

---

### Task 2: Add output interpretation for every model

**Files:**
- Modify: `src/droneai/round1_share_report.py`
- Modify: `tests/test_round1_share_report.py`

**Interfaces:**
- Consumes: the existing `_panel_caption(panel)` and accepted density/point panels.
- Produces: `model_output_note(model_id: str) -> str` and captions containing one note for each of the six model IDs.

- [ ] **Step 1: Write the failing tests**

```python
def test_every_model_has_a_concise_output_note() -> None:
    for model_id in MODELS:
        note = model_output_note(model_id)
        assert note.startswith(f"출력 해석({DISPLAY_NAMES[model_id]}):")
        assert len(note) <= 115


def test_share_report_contains_every_model_output_note(tmp_path: Path) -> None:
    manifest = make_assets(tmp_path)
    output = build_round1_share_pdf(
        COMPARISON_PATH,
        manifest,
        CONTENT_PATH,
        tmp_path / "share.pdf",
        report_date="2026-07-19",
    )
    text = "\n".join(page.extract_text() or "" for page in PdfReader(output).pages)
    normalized = " ".join(text.split())
    for model_id in MODELS:
        assert " ".join(model_output_note(model_id).split()) in normalized
```

- [ ] **Step 2: Run RED**

```powershell
& $python -m pytest tests/test_round1_share_report.py -q
```

Expected: failure because `model_output_note` does not exist and the captions do not contain the notes.

- [ ] **Step 3: Implement six concise notes and append them to captions**

Create notes describing native output, count calculation, and CCTV interpretation:

```python
MODEL_OUTPUT_NOTES = {
    "dm-count": "출력 해석(DM-Count): density map 합이 count이며 zone별 합으로 구역 밀집도를 읽는다.",
    "steerer": "출력 해석(STEERER): density 합으로 count하고 point를 zone 위치 후보로 함께 활용한다.",
    "pet": "출력 해석(PET): confidence를 통과한 point 개수가 count이며 좌표를 zone에 직접 배정한다.",
    "mpcount": "출력 해석(MPCount): full-resolution density 합으로 count하고 zone별 질량을 비교한다.",
    "apgcc": "출력 해석(APGCC): confidence point 개수가 count이며 위치 오차와 누락을 함께 확인한다.",
    "csrnet": "출력 해석(CSRNet): 비음수 density 합이 count이며 흐린 분포와 zone 오차를 확인한다.",
}


def model_output_note(model_id: str) -> str:
    try:
        return MODEL_OUTPUT_NOTES[model_id]
    except KeyError as exc:
        raise ValueError(f"unknown model output note: {model_id}") from exc
```

Append `"\n" + model_output_note(model_id)` to every density and point panel caption.

- [ ] **Step 4: Run GREEN and focused regression tests**

```powershell
& $python -m pytest tests/test_round1_share_report.py tests/test_round1_pdf_report.py -q
git diff --check
```

Expected: all focused tests pass.

- [ ] **Step 5: Commit code and tests**

```powershell
git add src/droneai/round1_share_report.py tests/test_round1_share_report.py
git commit -m "Explain model outputs in Round 1 report"
```

---

### Task 3: Generate and verify the revised review PDF

**Files:**
- Create locally: `output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_출력설명검수본.pdf`
- Create locally: `tmp/pdfs/round1-share-report-output-notes/rendered-v1/`

**Interfaces:**
- Consumes: committed renderer, accepted comparison JSON, content JSON, and assets manifest.
- Produces: one untracked 12-page review PDF plus 12 rendered PNGs.

- [ ] **Step 1: Run the full suite**

```powershell
& $python -m pytest -q
git diff --check
```

Expected: zero failures.

- [ ] **Step 2: Generate a new PDF without overwriting prior review copies**

```powershell
& $python scripts/build_round1_share_report.py `
  --comparison results/round1-cctv-comparison-34ad450/comparison.json `
  --assets-manifest tmp/pdfs/round1-model-analysis-1dc5974/assets/assets-manifest.json `
  --content configs/reporting/round1_model_analysis_ko.json `
  --output output/pdf/DroneAI_Round1_CCTV_모델비교_공유용_출력설명검수본.pdf `
  --report-date 2026-07-19
```

Expected: `PASS_RESEARCH_ONLY`, `pages=12`.

- [ ] **Step 3: Render and inspect**

Render all pages at 150 DPI with Poppler. Inspect pages 4-5 to confirm the dark
header is compact and body rows remain unclipped. Inspect pages 7-10 at original
detail to confirm all six model notes are readable, remain inside their panels,
and do not obscure the output images.

- [ ] **Step 4: Recompute evidence and Git-scope checks**

Require 12 A4 portrait pages, all six output notes, the existing claim-boundary
tokens, 12 rendered PNGs, 19 matching packaged asset SHA-256 values, and an
untracked final PDF. Record the PDF size and SHA-256.

- [ ] **Step 5: Push and synchronize home5090**

```powershell
git push origin agent/steerer-ucf-qnrf-adapter
ssh home5090-pop "git -C /home/lucita/crowd-counting-lab pull --ff-only"
```

Verify Windows, host, and container commits match. Do not add the PDF to Git and
do not merge to `main` before the user reviews the revised PDF.
