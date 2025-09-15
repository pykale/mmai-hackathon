"""Tests for clinical notes (text) loading utilities.

This suite validates the public APIs in ``mmai25_hackathon.load_data.text``:

- ``load_mimic_iv_notes(note_path, ...)``: loads a selected notes CSV (e.g., radiology), verifies required ID columns,
  optionally merges detail CSV, strips/filters empty ``text``, and returns a ``pd.DataFrame``.
- ``extract_text_from_note(note, ...)``: extracts the ``text`` field and optionally returns metadata.

Prerequisite
------------
Optional real-data integration uses:
``${PWD}/MMAI25Hackathon/mimic-iv/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note``.
If unavailable, integration tests are skipped; unit tests still validate core behavior with synthetic CSVs.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from mmai25_hackathon.load_data.text import extract_text_from_note, load_mimic_iv_notes

# Optional real dataset path for integration-style checks
TEXT_ROOT = (
    Path.cwd() / "MMAI25Hackathon" / "mimic-iv" / "mimic-iv-note-deidentified-free-text-clinical-notes-2.2" / "note"
)


@pytest.fixture(scope="module")
def real_notes_root() -> Path:
    if not TEXT_ROOT.exists():
        pytest.skip(f"Notes root not found: {TEXT_ROOT}")
    return TEXT_ROOT


def _w(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_csv(index=False))


@pytest.mark.parametrize("as_str", [True, False])
def test_invalid_notes_path_raises(tmp_path: Path, as_str: bool) -> None:
    arg = str(tmp_path / "missing") if as_str else (tmp_path / "missing")
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_notes(arg, subset="radiology")


def test_missing_main_csv_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_notes(tmp_path, subset="radiology")


def test_required_id_columns_missing_raises(tmp_path: Path) -> None:
    # Missing note_id
    _w(tmp_path / "radiology.csv", pd.DataFrame({"subject_id": [1], "text": ["t"]}))
    with pytest.raises(KeyError):
        load_mimic_iv_notes(tmp_path, subset="radiology")


def test_missing_detail_when_requested_raises(tmp_path: Path) -> None:
    _w(
        tmp_path / "radiology.csv",
        pd.DataFrame({"note_id": [1], "subject_id": [101], "text": ["hello"]}),
    )
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_notes(tmp_path, subset="radiology", include_detail=True)


def test_load_notes_filters_empty_text_and_merges_detail(tmp_path: Path) -> None:
    # Main notes: include one empty/whitespace-only text to exercise filtering
    _w(
        tmp_path / "radiology.csv",
        pd.DataFrame(
            {
                "note_id": [1, 2],
                "subject_id": [101, 102],
                "text": ["  hello  ", "   "],
                "note_type": ["r", "r"],
            }
        ),
    )
    # Detail includes an extra column to verify merge and absence of suffix when no collision
    _w(
        tmp_path / "radiology_detail.csv",
        pd.DataFrame({"note_id": [1, 2], "subject_id": [101, 102], "detail_field": ["A", "B"]}),
    )

    df = load_mimic_iv_notes(tmp_path, subset="radiology", include_detail=True)
    assert not df.empty, "Notes DataFrame is unexpectedly empty"
    assert set(["note_id", "subject_id"]).issubset(df.columns), f"Missing ID columns: {list(df.columns)}"
    assert "detail_field" in df.columns, f"Merged detail column missing: columns={list(df.columns)}"

    # After filtering empty text, only note_id==1 should remain
    assert set(df["note_id"]) == {1}, f"Expected only note_id=1 after filtering, got {set(df['note_id'])}"
    assert (
        df.loc[df["note_id"] == 1, "text"].iloc[0] == "hello"
    ), f"Text should be stripped; got {df.loc[df['note_id'] == 1, 'text'].iloc[0]!r}"


def test_detail_missing_required_id_columns_raises(tmp_path: Path) -> None:
    _w(
        tmp_path / "radiology.csv",
        pd.DataFrame({"note_id": [1], "subject_id": [101], "text": ["x"]}),
    )
    # Detail exists but missing 'subject_id'
    _w(
        tmp_path / "radiology_detail.csv",
        pd.DataFrame({"note_id": [1], "extra": ["A"]}),
    )
    with pytest.raises(KeyError):
        load_mimic_iv_notes(tmp_path, subset="radiology", include_detail=True)


def test_load_notes_without_text_column_warns(tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
    _w(
        tmp_path / "radiology.csv",
        pd.DataFrame({"note_id": [1], "subject_id": [101], "note_type": ["r"]}),
    )
    caplog.set_level(logging.WARNING)
    df = load_mimic_iv_notes(tmp_path, subset="radiology", include_detail=False, subset_cols=["note_type"])
    assert "text" not in df.columns, f"'text' unexpectedly present: {list(df.columns)}"
    assert any(
        "do not include a 'text'" in r.getMessage() for r in caplog.records
    ), "Expected warning about missing 'text' column"


def test_extract_text_from_note_success_and_metadata() -> None:
    note = pd.Series({"note_id": 1, "subject_id": 101, "text": "Patient stable.", "note_type": "Discharge"})
    t = extract_text_from_note(note)
    assert t == "Patient stable.", f"Unexpected text extracted: {t!r}"

    t2, meta = extract_text_from_note(note, include_metadata=True)
    assert t2 == "Patient stable.", f"Unexpected text extracted: {t2!r}"
    assert meta.get("note_id") == 1 and meta.get("subject_id") == 101, f"Unexpected metadata returned: {meta}"


def test_extract_text_from_note_missing_text_raises() -> None:
    with pytest.raises(KeyError):
        extract_text_from_note(pd.Series({"note_id": 1, "subject_id": 101}))


@pytest.mark.parametrize("as_str", [True, False])
def test_integration_load_notes_if_available(real_notes_root: Path, as_str: bool) -> None:
    arg = str(real_notes_root) if as_str else real_notes_root
    df = load_mimic_iv_notes(arg, subset="radiology", include_detail=False)
    # It's acceptable for the subset to be empty if the sample doesn't include radiology,
    # but the loader should return a DataFrame with columns present.
    assert isinstance(df, pd.DataFrame), "Expected DataFrame from load_mimic_iv_notes"
    if not df.empty:
        assert {"note_id", "subject_id"}.issubset(df.columns), f"Missing required ID columns: {df.columns}"
        if "text" in df.columns:
            assert (df["text"].astype(str).str.len() > 0).any(), "All texts are empty after loading"
