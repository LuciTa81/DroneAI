from pathlib import Path


def test_steerer_review_bundle_preserves_hashed_bytes_across_platforms() -> None:
    lines = Path(".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "results/steerer/**/*.json text eol=lf" in lines
    assert "results/steerer/**/*.csv text eol=lf" in lines
    assert "results/steerer/**/*.md text eol=lf" in lines
    assert "results/steerer/**/*.png binary" in lines


def test_pet_review_bundle_preserves_hashed_bytes_across_platforms() -> None:
    lines = Path(".gitattributes").read_text(encoding="utf-8").splitlines()

    assert "results/pet/**/*.json text eol=lf" in lines
    assert "results/pet/**/*.csv text eol=lf" in lines
    assert "results/pet/**/*.md text eol=lf" in lines
    assert "results/pet/**/*.png binary" in lines
