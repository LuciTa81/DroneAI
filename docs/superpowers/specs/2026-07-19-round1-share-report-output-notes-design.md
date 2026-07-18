# Round 1 Share Report Output Notes Design

## Goal

Correct the oversized header spacing in the two model-structure tables and make
every model's accepted Round 1 output example self-explanatory without reducing
the readability of the 12-page A4 portrait report.

## Scope

- Keep the report at exactly 12 A4 portrait pages.
- Keep the existing six-model comparison, metrics, rights boundaries, and
  conclusions unchanged.
- Reuse only the already verified packaged figures and accepted comparison JSON.
- Do not run inference, training, fine-tuning, test-split access, or queue
  advancement.
- Keep the generated review PDF and render artifacts untracked.

## Table Spacing Correction

The model-structure tables currently use one row height for both the header and
the three body rows. This makes the dark header unusually tall. Extend the table
renderer with an optional header-row height so these two tables can use a compact
header while retaining the existing body height required for readable wrapped
model descriptions. Other report tables retain their current layout unless they
explicitly opt in.

## Model Output Explanations

The accepted report already contains output panels for all six models:

- DM-Count: density map
- STEERER: density and point/hybrid views
- PET: point predictions
- MPCount: density map
- APGCC: point predictions
- CSRNet: operational density map

Keep these full-width panels and add a concise model-specific output note to each
caption. Each note states:

1. the native output form;
2. how the count is produced;
3. how an operator should interpret or aggregate it for CCTV zones.

The existing GT, predicted count, sample ID, density band, and spatial metric
remain visible. STEERER appears in both density and point/hybrid sections because
both outputs are operationally relevant.

## Readability and Layout

- Model-table header height: compact and visually consistent with other tables.
- Model-table body height: unchanged unless visual verification proves clipping.
- Figure captions: at least 8.5pt with at least 11pt leading.
- Each output explanation must fit inside the panel caption area without
  truncation, overlap, or reducing the output image to an unreadable size.
- If a caption needs more room, increase the caption area within the existing
  panel rather than adding thumbnail summary pages.

## Verification

Automated tests must fail first and then require:

- an independently configurable header-row height;
- model-specific output explanations for all six models;
- the existing exact 12-page contract and required claim boundaries;
- no change to the accepted model metrics or evidence identities.

Generate a new review PDF filename, render all 12 pages with Poppler, inspect the
two model-table pages and four output pages at original detail, and recompute all
19 packaged asset hashes. The PDF remains `PASS_RESEARCH_ONLY` and is not merged
to `main` until the user approves the revised copy.
