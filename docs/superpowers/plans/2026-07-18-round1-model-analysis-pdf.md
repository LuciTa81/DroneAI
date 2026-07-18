# Round 1 Model Analysis PDF Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a private, Korean, 15-page A4-landscape PDF that explains all six Round 1 crowd-counting models, shows hash-verified real examples, separates technical performance from commercial rights, and ends with a fixed-CCTV recommendation and pre-fine-tuning roadmap.

**Architecture:** A tested evidence-packager runs inside home5090 against the accepted SSD runs and copies only selected panels plus a SHA-256 manifest. A deterministic ReportLab renderer on Windows consumes the frozen comparison JSON, a reviewed Korean editorial config, and the verified panel package to create exactly 15 pages. Poppler rendering and pypdf text checks form the final delivery gate.

**Tech Stack:** Python 3.12, ReportLab, Pillow, pypdf, Poppler (`pdfinfo`, `pdftoppm`, `pdftotext`), pytest, SSH/SCP, existing DroneAI integrity helpers.

## Global Constraints

- Use only `ucf-qnrf-kaggle-apache` split `dm-count-upstream-qnrf-val-smoke-36-v1`, role `validation`, 36 samples.
- Canonical split SHA-256 must equal `da1d947aad73d45be573d52a6462fa8022948fba2b1026a3bfff76b11a4b5c67`.
- Do not run inference, training, fine-tuning, test access, or queue advancement.
- Technical score is an operational 100-point harness score, never an accuracy percentage.
- Preserve component-level code, dataset, pretrained-weight, derived-weight, and deployment rights independently.
- Use only accepted, hash-verified panel files; keep intermediate panels under `tmp/pdfs/` and out of Git.
- Generate exactly 15 A4-landscape pages in Korean with embedded Korean fonts.
- Keep the draft PDF private and unpushed until the user approves it.
- Missing metrics render as `N/A`, never zero.

---

### Task 1: Build the hash-verified PDF input package

**Files:**
- Create: `src/droneai/pdf_input_package.py`
- Create: `scripts/build_round1_pdf_inputs.py`
- Create: `tests/test_pdf_input_package.py`

**Interfaces:**
- Consumes: `comparison.json`, home5090 `results_root`, each run's `selection_manifest.json`, `predictions.csv`, and `evidence-manifest.json`.
- Produces: `build_pdf_input_package(comparison_path: str | Path, results_root: str | Path, output_dir: str | Path) -> Path`; output contains `assets-manifest.json` and content-addressed PNG files.

- [ ] **Step 1: Write fixture tests for deterministic selection and hashes**

```python
def test_package_selects_high_density_best_and_largest_failure(tmp_path):
    comparison, results = make_pdf_fixture(tmp_path)
    output = build_pdf_input_package(comparison, results, tmp_path / "package")
    manifest = json.loads((output / "assets-manifest.json").read_text())
    steerer = manifest["models"]["steerer"]
    assert steerer["lower_error"]["category"] == "best"
    assert steerer["lower_error"]["density_band"] == "high"
    assert steerer["limitation"]["category"] == "failure"
    assert steerer["limitation"]["normalized_error"] == max(
        item["normalized_error"] for item in steerer["failure_candidates"]
    )

def test_package_requires_frozen_common_samples(tmp_path):
    comparison, results = make_pdf_fixture(tmp_path)
    output = build_pdf_input_package(comparison, results, tmp_path / "package")
    manifest = json.loads((output / "assets-manifest.json").read_text())
    assert set(manifest["comparisons"]["density"]) == {
        "dm-count", "steerer", "mpcount", "csrnet"
    }
    assert {item["sample_id"] for item in manifest["comparisons"]["density"].values()} == {
        manifest["comparisons"]["density_sample_id"]
    }
    assert set(manifest["comparisons"]["points"]) == {"steerer", "pet", "apgcc"}

def test_package_rejects_panel_hash_change(tmp_path):
    comparison, results = make_pdf_fixture(tmp_path)
    (results / "dm-count/run/figures/panel.png").write_bytes(b"changed")
    with pytest.raises(ValueError, match="artifact hash mismatch"):
        build_pdf_input_package(comparison, results, tmp_path / "package")
```

- [ ] **Step 2: Run RED**

Run:

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_pdf_input_package.py -q
```

Expected: collection fails with `ModuleNotFoundError: droneai.pdf_input_package`.

- [ ] **Step 3: Implement package validation and selection**

Implement these exact public types and function:

```python
@dataclass(frozen=True)
class SelectedPanel:
    model_id: str
    role: str
    sample_id: str
    category: str
    density_band: str
    ground_truth_count: float
    predicted_count: float
    normalized_error: float
    spatial_metric_name: str
    spatial_metric_value: float | None
    source_sha256: str
    packaged_path: str

PUBLIC_SIGNATURE = (
    "build_pdf_input_package(comparison_path: str | Path, "
    "results_root: str | Path, output_dir: str | Path) -> Path"
)
```

Implementation rules:

- Verify `comparison.json` contains the six frozen model IDs and canonical split hash.
- Resolve each `representative_panels` path beneath `results_root`; reject absolute paths, `..` escape after normalization, missing files, and evidence-manifest hash mismatches.
- Zip `selection_manifest.panel_paths` with `selection_manifest.selections` by index and join each selection to its exact `predictions.csv` row by `sample_id`.
- Select `lower_error` from category `best`, preferring density band `high`, then the smallest normalized error, then lexical sample ID.
- Select `limitation` from category `failure`, preferring the largest normalized error, then lexical sample ID.
- Resolve the density comparison panel basename suffix `fd9373c130ac1077` and verify one common sample ID across DM-Count, STEERER, MPCount, and CSRNet.
- Resolve the point/hybrid suffix `8e0a4dbe072725a1` and verify one common sample ID across STEERER, PET, and APGCC.
- Copy each unique source panel once using the concrete filename template
  `{model_id}--{sample_id}--{sha12}.png`.
- Write `assets-manifest.json` atomically with all captions, source/result-relative paths, source SHA-256, packaged SHA-256, roles, and comparison groups.

- [ ] **Step 4: Add the thin CLI and structured errors**

```python
def main(argv: Sequence[str] | None = None) -> int:
    args = parser.parse_args(argv)
    try:
        output = build_pdf_input_package(
            args.comparison, args.results_root, args.output_dir
        )
        print(json.dumps({"status": "PASS_RESEARCH_ONLY", "output_dir": str(output)}))
        return 0
    except Exception as exc:
        print(json.dumps({
            "status": "FAIL",
            "error_type": type(exc).__name__,
            "error": str(exc),
        }), file=sys.stderr)
        return 2
```

CLI flags must be exactly `--comparison`, `--results-root`, and `--output-dir`.

- [ ] **Step 5: Run GREEN and commit locally**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_pdf_input_package.py -q
git diff --check
git add src/droneai/pdf_input_package.py scripts/build_round1_pdf_inputs.py tests/test_pdf_input_package.py
git commit -m "Package verified Round 1 PDF examples"
```

Expected: focused tests pass and the worktree is clean.

---

### Task 2: Freeze reviewed Korean model-analysis copy

**Files:**
- Create: `configs/reporting/round1_model_analysis_ko.json`
- Create: `src/droneai/pdf_report_content.py`
- Create: `tests/test_pdf_report_content.py`

**Interfaces:**
- Consumes: editorial JSON and frozen comparison model IDs.
- Produces: `load_report_content(path: str | Path) -> ReportContent` with six ordered `ModelContent` records.

- [ ] **Step 1: Write schema and claim-boundary tests**

```python
def test_content_covers_exact_six_models():
    content = load_report_content("configs/reporting/round1_model_analysis_ko.json")
    assert tuple(model.model_id for model in content.models) == (
        "dm-count", "steerer", "pet", "mpcount", "apgcc", "csrnet"
    )

def test_content_keeps_pet_research_only_and_no_production_claims():
    payload = Path("configs/reporting/round1_model_analysis_ko.json").read_text()
    assert "PET은 연구 비교군" in payload
    assert "배포 승인 완료" not in payload
    assert "공식 모델 순위" not in payload
```

- [ ] **Step 2: Run RED**

Run `pytest tests/test_pdf_report_content.py -q`.

Expected: collection fails because `droneai.pdf_report_content` is absent.

- [ ] **Step 3: Write the six-model Korean content configuration**

For every model, provide these non-empty fields:

```json
{
  "model_id": "dm-count",
  "paper": "Distribution Matching for Crowd Counting (NeurIPS 2020)",
  "family": "density",
  "backbone": "VGG-19 convolutional features",
  "structure": ["VGG-19 E feature extractor", "bilinear 2x upsampling", "density regression head"],
  "input": "ImageNet 정규화를 적용한 RGB 이미지",
  "native_output": "비음수 density map",
  "count_method": "density map 질량의 합",
  "strengths": ["질량 보존형 heatmap", "구역별 density 집계"],
  "limitations": ["도메인 변화", "고밀도 undercount", "큰 activation"],
  "cctv_interpretation": "고정형 CCTV heatmap baseline과 zone calibration 후보",
  "reference_source": "accepted model-brief.md"
}
```

Use the accepted briefs for all architecture claims:

- DM-Count: VGG-19 density, distribution matching, mass sum.
- STEERER: HRNet-W48, multi-resolution density plus merged points, selective inheritance.
- PET: VGG16-BN point-query quadtree, thresholded points and confidences, research-only.
- MPCount: VGG16-BN plus 1024-slot density memory bank, domain-generalized density.
- APGCC: VGG16-BN, ASPP, IFI decoder, confidence and point-offset heads.
- CSRNet: VGG-16 frontend, dilation-2 backend, stride-8 density baseline.

- [ ] **Step 4: Implement strict dataclass loading**

```python
@dataclass(frozen=True)
class ModelContent:
    model_id: str
    paper: str
    family: str
    backbone: str
    structure: tuple[str, ...]
    input: str
    native_output: str
    count_method: str
    strengths: tuple[str, ...]
    limitations: tuple[str, ...]
    cctv_interpretation: str
    reference_source: str

@dataclass(frozen=True)
class ReportContent:
    title: str
    subtitle: str
    models: tuple[ModelContent, ...]
    conclusion: tuple[str, ...]
    roadmap: tuple[str, ...]
```

Reject missing keys, empty Korean copy, duplicate/unknown model IDs, fewer than
two strengths or limitations, and any `PRODUCTION_APPROVED` claim.

- [ ] **Step 5: Run GREEN and commit locally**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_pdf_report_content.py -q
git diff --check
git add configs/reporting/round1_model_analysis_ko.json src/droneai/pdf_report_content.py tests/test_pdf_report_content.py
git commit -m "Freeze Korean Round 1 report content"
```

---

### Task 3: Generate the exact 15-page ReportLab PDF

**Files:**
- Create: `src/droneai/round1_pdf_report.py`
- Create: `scripts/build_round1_model_analysis_pdf.py`
- Create: `tests/test_round1_pdf_report.py`

**Interfaces:**
- Consumes: comparison JSON, assets manifest/package directory, Korean content JSON, font files, and an explicit report date.
- Produces: `build_round1_pdf(comparison_path: str | Path, assets_manifest_path: str | Path, content_path: str | Path, output_path: str | Path, *, report_date: str) -> Path`.

- [ ] **Step 1: Write page-count, text, and missing-asset tests**

```python
def test_report_has_exact_pages_and_required_text(pdf_fixture, tmp_path):
    output = build_round1_pdf(*pdf_fixture, tmp_path / "report.pdf", report_date="2026-07-18")
    reader = PdfReader(output)
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    assert len(reader.pages) == 15
    for token in ("STEERER", "DM-Count", "PET", "MPCount", "APGCC", "CSRNet"):
        assert token in text
    assert "공식 모델 순위가 아님" in text
    assert "PROVISIONAL_TECHNICAL_SHORTLIST" in text

def test_report_rejects_changed_asset_hash(pdf_fixture, tmp_path):
    comparison, manifest, content = pdf_fixture
    image = next((Path(manifest).parent / "images").iterdir())
    image.write_bytes(b"changed")
    with pytest.raises(ValueError, match="asset hash mismatch"):
        build_round1_pdf(comparison, manifest, content, tmp_path / "report.pdf", report_date="2026-07-18")
```

- [ ] **Step 2: Run RED**

Run `pytest tests/test_round1_pdf_report.py -q`.

Expected: collection fails because `droneai.round1_pdf_report` is absent.

- [ ] **Step 3: Implement font, page, and primitive helpers**

Implement:

```python
PAGE_SIZE = landscape(A4)
PAGE_COUNT = 15

def register_korean_fonts() -> tuple[str, str]:
    candidates = (
        (Path("C:/Windows/Fonts/malgun.ttf"), Path("C:/Windows/Fonts/malgunbd.ttf")),
        (Path("C:/Windows/Fonts/NotoSansKR-Regular.ttf"), Path("C:/Windows/Fonts/NotoSansKR-Bold.ttf")),
    )
    for regular, bold in candidates:
        if regular.is_file() and bold.is_file():
            pdfmetrics.registerFont(TTFont("DroneAI-Regular", regular))
            pdfmetrics.registerFont(TTFont("DroneAI-Bold", bold))
            return "DroneAI-Regular", "DroneAI-Bold"
    raise FileNotFoundError("Korean regular/bold font pair is unavailable")
```

Add helpers with these exact signatures: `draw_footer(canvas, page_number: int)
-> None`, `draw_wrapped(canvas, text: str, x: float, y: float, width: float, *,
font: str, size: float, leading: float) -> float`, `draw_metric_card(canvas,
title: str, value: str, x: float, y: float, width: float, height: float) ->
None`, and `draw_panel(canvas, image_path: Path, caption: str, x: float, y:
float, width: float, height: float) -> None`.

Use `ImageReader`, preserve aspect ratio, add a light border, and never crop
evaluation columns. Use only ASCII hyphens in generated copy.

- [ ] **Step 4: Implement the 15 fixed pages**

Create one renderer per page number and call `canvas.showPage()` exactly once
after each renderer. Pages must follow the approved design verbatim. Technical
score charts are drawn directly from `comparison.models`; model pages use
`comparison`, `ModelContent`, and two selected assets; comparison pages use the
fixed common-sample groups; rights page uses all five component states.

The renderer must validate before writing:

```python
assert comparison["ranking_eligible"] is False
assert comparison["dataset"]["split_role"] == "validation"
assert comparison["dataset"]["expected_samples"] == 36
assert comparison["shortlist"]["model_ids"] == ["steerer", "dm-count", "pet"]
assert comparison["shortlist"]["label"] == "PROVISIONAL_TECHNICAL_SHORTLIST"
```

- [ ] **Step 5: Add CLI and atomic output**

The CLI requires `--comparison`, `--assets-manifest`, `--content`, `--output`,
and `--report-date`. Write to `.{output.name}.tmp.pdf`, parse it with pypdf, require 15
pages, then rename to the final path. On failure, delete only the temporary file
and emit structured JSON to stderr with exit code 2.

- [ ] **Step 6: Run GREEN, full focused tests, and commit locally**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_pdf_input_package.py tests/test_pdf_report_content.py tests/test_round1_pdf_report.py -q
git diff --check
git add src/droneai/round1_pdf_report.py scripts/build_round1_model_analysis_pdf.py tests/test_round1_pdf_report.py
git commit -m "Generate private Round 1 model analysis PDF"
```

---

### Task 4: Produce, render, inspect, and deliver the private report

**Files:**
- Create locally: `tmp/pdfs/round1-model-analysis/assets/`
- Create locally: `tmp/pdfs/round1-model-analysis/rendered/`
- Create: `output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf`

**Interfaces:**
- Consumes: committed Tasks 1-3, clean home5090 accepted runs, bundled PDF runtime.
- Produces: one private PDF and a recorded SHA-256; no GitHub push.

- [ ] **Step 1: Load bundled PDF dependencies and verify tools**

Use `codex_app__load_workspace_dependencies`, then run:

```powershell
& $python -c "import reportlab, PIL, pypdf; print('PDF_DEPS_OK')"
& $pdftoppm -v
& $pdfinfo -v
```

Expected: imports succeed and Poppler version commands exit 0.

- [ ] **Step 2: Run focused and full repository verification**

```powershell
& '.venv\Scripts\python.exe' -m pytest tests/test_pdf_input_package.py tests/test_pdf_report_content.py tests/test_round1_pdf_report.py -q
& '.venv\Scripts\python.exe' -m pytest -q
git diff --check
```

Expected: zero test failures.

- [ ] **Step 3: Sync code to home5090 without publishing the private PDF**

Push only implementation commits if required to make the container execute the
asset packager; do not add or push `output/pdf/`. Then run `git pull --ff-only`
on home5090 and require a clean matching commit.

- [ ] **Step 4: Package selected panels on home5090**

```bash
cd /workspace
/workspace/.venvs/harness/bin/python scripts/build_round1_pdf_inputs.py \
  --comparison results/round1-cctv-comparison-34ad450/comparison.json \
  --results-root /workspace/data/results \
  --output-dir /workspace/data/results/private-round1-pdf-assets-$(git rev-parse --short HEAD)
```

Expected: `PASS_RESEARCH_ONLY`, six lower-error assets, six limitation assets,
four density comparison references, three point/hybrid comparison references,
and no inference output.

- [ ] **Step 5: Pull only the package and verify hashes on Windows**

```powershell
$commit=(git rev-parse --short HEAD)
scp -r "home5090-pop:/mnt/crowd-data/CrowdCounting/results/private-round1-pdf-assets-$commit" tmp/pdfs/round1-model-analysis/assets
```

Recompute every packaged image SHA-256 from `assets-manifest.json`; abort on any
missing file, mismatch, duplicate model role, or common-sample disagreement.

- [ ] **Step 6: Generate the final PDF**

```powershell
& $python scripts/build_round1_model_analysis_pdf.py \
  --comparison results/round1-cctv-comparison-34ad450/comparison.json \
  --assets-manifest tmp/pdfs/round1-model-analysis/assets/assets-manifest.json \
  --content configs/reporting/round1_model_analysis_ko.json \
  --output output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf \
  --report-date 2026-07-18
```

Expected: structured success JSON and a non-empty PDF.

- [ ] **Step 7: Render and visually inspect all pages**

```powershell
& $pdfinfo output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf
& $pdftoppm -png -r 130 output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf tmp/pdfs/round1-model-analysis/rendered/page
```

Create contact sheets for pages 1-5, 6-10, and 11-15, inspect them with
`view_image`, then inspect any dense page individually at original detail. Reject
the PDF for Korean tofu glyphs, clipping, overlap, illegible captions, stretched
panels, table overflow, inconsistent margins, missing footer, or wrong page
number. Regenerate and repeat until all 15 pages pass.

- [ ] **Step 8: Verify text, page count, and final hash**

```powershell
& $pdftotext output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf tmp/pdfs/round1-model-analysis/report.txt
& $python -c "from pypdf import PdfReader; p=PdfReader(r'output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf'); assert len(p.pages)==15"
Get-FileHash -Algorithm SHA256 output/pdf/DroneAI_Round1_CCTV_모델분석_개인검토용.pdf
```

Require extracted text to contain all six model names, `UCF-QNRF`, `36`,
`PROVISIONAL_TECHNICAL_SHORTLIST`, `연구 전용`, and the non-ranking disclaimer.

- [ ] **Step 9: Deliver and stop before publication or fine-tuning**

Return a clickable local PDF link, page count, file size, SHA-256, validation
summary, and the private-draft caveat. Do not commit or push the PDF until the
user explicitly approves it for leadership sharing.
