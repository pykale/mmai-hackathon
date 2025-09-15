"""Tests for MIMIC-IV Electrocardiogram (ECG) loading utilities.

This suite validates the public APIs in ``mmai25_hackathon.load_data.ecg``:

- ``load_mimic_iv_ecg_record_list(ecg_path, ...)``: parses ``record_list.csv``, verifies
  the expected dataset layout (``files/`` subfolder), resolves absolute ``.hea``/``.dat`` paths,
  applies optional row filtering, and returns a ``pd.DataFrame`` containing only rows with both files.
- ``load_ecg_record(hea_path)``: reads an ECG record via ``wfdb.rdsamp`` given a ``.hea`` path.

Prerequisite
------------
The tests assume the real dataset may be available under the fixed path:
``${PWD}/MMAI25Hackathon/mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0``.
If that directory, its ``files/`` subfolder, or ``record_list.csv`` is missing, the integration
tests are skipped. Unit-level behavior and error handling are still validated via temporary data.
"""

import logging
from pathlib import Path
from typing import Dict, Tuple

import numpy as np
import pandas as pd
import pytest

# Ensure wfdb is available; otherwise, skip this module's tests at collection time
pytest.importorskip("wfdb")

from mmai25_hackathon.load_data.ecg import load_ecg_record, load_mimic_iv_ecg_record_list  # noqa: E402

# Fixed dataset path (if available locally or fetched during CI)
ECG_ROOT = Path.cwd() / "MMAI25Hackathon" / "mimic-iv" / "mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0"


@pytest.fixture(scope="module")
def ecg_root() -> Path:
    """Return the dataset root or skip the module-level integration tests if missing."""
    if not ECG_ROOT.exists():
        pytest.skip(f"Dataset root not found: {ECG_ROOT}")
    files_dir = ECG_ROOT / "files"
    if not files_dir.exists():
        pytest.skip(f"Dataset 'files' subdir not found under: {ECG_ROOT}")
    records_csv = ECG_ROOT / "record_list.csv"
    if not records_csv.exists():
        pytest.skip(f"'record_list.csv' not found under: {ECG_ROOT}")
    return ECG_ROOT


@pytest.fixture(scope="module")
def record_df(ecg_root: Path) -> pd.DataFrame:
    """Load the record list once for the module to speed up tests."""
    return load_mimic_iv_ecg_record_list(ecg_root)


@pytest.mark.parametrize("use_str_path", [True, False])
def test_record_list_and_signal_loading(caplog: pytest.LogCaptureFixture, ecg_root: Path, use_str_path: bool) -> None:
    # Capture loader INFO logs if emitted
    caplog.set_level(logging.INFO)

    path_arg = str(ecg_root) if use_str_path else ecg_root
    df = load_mimic_iv_ecg_record_list(path_arg)

    # Basic metadata checks
    assert isinstance(df, pd.DataFrame), f"Expected a DataFrame from load_mimic_iv_ecg_record_list, got {type(df)!r}"
    assert not df.empty, (
        "Record list DataFrame is unexpectedly empty; check that 'record_list.csv' contains rows and that"
        " corresponding '.hea' and '.dat' files exist."
    )
    for col in ("ecg_path", "hea_path", "dat_path"):
        assert col in df.columns, f"Expected column '{col}' to be present; available columns: {list(df.columns)}"

    # Optional assertion on log messages if available
    if caplog.records:
        assert any(
            "Mapping ECG file paths" in rec.getMessage() or "Found" in rec.getMessage() for rec in caplog.records
        ), "Expected mapping or discovery log message"

    # Sample one record, ensure paths are absolute/exist, and load signals
    sample_hea = Path(str(df.iloc[0]["hea_path"]))  # type: ignore[index]
    sample_dat = Path(str(df.iloc[0]["dat_path"]))  # type: ignore[index]
    assert sample_hea.is_absolute(), f"hea_path should be absolute, got: {sample_hea}"
    assert sample_dat.is_absolute(), f"dat_path should be absolute, got: {sample_dat}"
    assert sample_hea.exists(), f"Expected .hea file to exist, missing: {sample_hea}"
    assert sample_dat.exists(), f"Expected .dat file to exist, missing: {sample_dat}"
    assert (
        sample_hea.suffix.lower() == ".hea"
    ), f"hea_path must end with .hea, got suffix: {sample_hea.suffix} (path={sample_hea})"
    assert (
        sample_dat.suffix.lower() == ".dat"
    ), f"dat_path must end with .dat, got suffix: {sample_dat.suffix} (path={sample_dat})"

    signals, fields = load_ecg_record(sample_hea)
    assert isinstance(signals, np.ndarray), f"signals should be np.ndarray, got {type(signals)!r}"
    assert signals.ndim == 2, f"signals should be 2D (T, L), got shape {signals.shape}"
    assert signals.shape[1] > 0, f"signals should have >0 leads, got shape {signals.shape}"
    assert isinstance(fields, dict), f"fields should be dict, got {type(fields)!r}"
    # WFDB commonly provides sampling frequency under 'fs'
    if "fs" in fields:
        assert float(fields["fs"]) > 0, f"Sampling frequency 'fs' must be > 0, got {fields['fs']!r}"


def test_paths_are_absolute_and_exist_on_head(record_df: pd.DataFrame) -> None:
    head_hea = record_df["hea_path"].astype(str).head(10).tolist()
    head_dat = record_df["dat_path"].astype(str).head(10).tolist()

    for hp, dp in zip(head_hea, head_dat):
        hp = Path(hp)
        dp = Path(dp)
        assert hp.is_absolute(), f"hea_path is not absolute: {hp}"
        assert dp.is_absolute(), f"dat_path is not absolute: {dp}"
        assert hp.exists(), f"Resolved .hea does not exist: {hp}"
        assert dp.exists(), f"Resolved .dat does not exist: {dp}"
        assert hp.suffix.lower() == ".hea", f"hea_path must end with .hea, got {hp.suffix} (path={hp})"
        assert dp.suffix.lower() == ".dat", f"dat_path must end with .dat, got {dp.suffix} (path={dp})"


def test_filter_rows_subject_subset_is_consistent(ecg_root: Path, record_df: pd.DataFrame) -> None:
    # Choose a subject_id present in the loaded record list to make the test robust
    some_subject = int(record_df["subject_id"].iloc[0])
    filtered = load_mimic_iv_ecg_record_list(ecg_root, filter_rows={"subject_id": [some_subject]})

    assert (
        not filtered.empty
    ), f"Filtered subject_id is unexpectedly empty; subject={some_subject} not found in record_list.csv"
    assert set(filtered["subject_id"].unique()) == {
        some_subject
    }, f"Filtered subject_id has unexpected values; expected only {some_subject}, got {set(filtered['subject_id'].unique())}"

    # The filtered set must be a subset of the unfiltered rows for that subject
    all_rows = record_df[record_df["subject_id"] == some_subject]
    assert set(filtered["hea_path"]).issubset(
        set(all_rows["hea_path"])
    ), "Filtered rows mismatch for 'hea_path'; filtered set must be a subset of unfiltered rows for the same subject"
    assert set(filtered["dat_path"]).issubset(
        set(all_rows["dat_path"])
    ), "Filtered rows mismatch for 'dat_path'; filtered set must be a subset of unfiltered rows for the same subject"


def test_loading_nonexistent_hea_raises() -> None:
    with pytest.raises(FileNotFoundError):
        load_ecg_record(Path("/definitely/not/here.hea"))


def test_invalid_ecg_base_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_ecg_record_list(tmp_path / "missing_root")


def test_files_folder_missing_raises(tmp_path: Path) -> None:
    (tmp_path / "record_list.csv").write_text(pd.DataFrame({"path": ["p100/p101/s133/133"]}).to_csv(index=False))
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_ecg_record_list(tmp_path)


def test_record_list_not_found_raises(tmp_path: Path) -> None:
    (tmp_path / "files").mkdir()
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_ecg_record_list(tmp_path)


def test_mapping_and_filtering_with_temp_csv(tmp_path: Path) -> None:
    # Create folder layout: <tmp>/files and a synthetic record_list.csv
    files_dir = tmp_path / "files"
    files_dir.mkdir()

    # Synthetic CSV with whitespace in 'path' to exercise .str.strip()
    df = pd.DataFrame(
        {
            "subject_id": [101, 101, 102],
            "study_id": [133, 999, 200],
            # one valid pair (both .hea/.dat), one missing .dat, one missing .hea
            "path": [
                " files/p100/p101/s133/133 ",
                " files/p100/p101/s999/999 ",
                " files/p200/p201/s200/200 ",
            ],
        }
    )
    (tmp_path / "record_list.csv").write_text(df.to_csv(index=False))

    # Create files for the first entry only
    rec1 = files_dir / "p100" / "p101" / "s133"
    rec1.mkdir(parents=True)
    (rec1 / "133.hea").write_text("dummy header")
    (rec1 / "133.dat").write_bytes(b"\x00\x01\x02")

    # Second: create only .hea
    rec2 = files_dir / "p100" / "p101" / "s999"
    rec2.mkdir(parents=True)
    (rec2 / "999.hea").write_text("dummy header")

    # Third: create only .dat
    rec3 = files_dir / "p200" / "p201" / "s200"
    rec3.mkdir(parents=True)
    (rec3 / "200.dat").write_bytes(b"\x00\x01\x02")

    out = load_mimic_iv_ecg_record_list(tmp_path, filter_rows={"subject_id": [101, 102]})

    # Only the first row should remain (both files present)
    assert len(out) == 1, f"Expected only one mapped row with both .hea/.dat present; got {len(out)} rows."
    row = out.iloc[0]
    assert int(row["subject_id"]) == 101, f"Unexpected subject_id in mapped row: {row['subject_id']!r}"
    assert Path(row["hea_path"]).is_absolute(), f"hea_path should be absolute, got: {row['hea_path']}"
    assert Path(row["dat_path"]).is_absolute(), f"dat_path should be absolute, got: {row['dat_path']}"
    assert Path(row["hea_path"]).exists(), f"Mapped .hea missing on disk: {row['hea_path']}"
    assert Path(row["dat_path"]).exists(), f"Mapped .dat missing on disk: {row['dat_path']}"
    assert Path(row["hea_path"]).suffix == ".hea", f"hea_path must end with .hea: {row['hea_path']}"
    assert Path(row["dat_path"]).suffix == ".dat", f"dat_path must end with .dat: {row['dat_path']}"


@pytest.mark.parametrize("as_str", [True, False])
def test_load_ecg_record_invokes_wfdb_with_stem(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, as_str: bool) -> None:
    # Prepare a dummy .hea file to pass the existence check
    hea_file = tmp_path / "example" / "rec.hea"
    hea_file.parent.mkdir(parents=True)
    hea_file.write_text("header")

    captured_arg: Dict[str, str] = {}

    def fake_rdsamp(arg: str) -> Tuple[np.ndarray, Dict[str, float]]:  # type: ignore[override]
        captured_arg["arg"] = arg
        return np.zeros((10, 3), dtype=float), {"fs": 500.0}

    # Monkeypatch wfdb.rdsamp to avoid requiring a valid WFDB record on disk
    import wfdb as _wfdb

    monkeypatch.setattr(_wfdb, "rdsamp", fake_rdsamp, raising=True)

    arg_in = str(hea_file) if as_str else hea_file
    sig, meta = load_ecg_record(arg_in)

    # Expect the stem path (without suffix) to be passed to rdsamp
    assert captured_arg.get("arg") and (
        captured_arg["arg"].endswith("/example/rec") or captured_arg["arg"].endswith("\\example\\rec")
    ), f"wfdb.rdsamp should receive the stem path; got arg={captured_arg!r}"
    assert sig.shape == (10, 3), f"Unexpected dummy signal shape, expected (10,3), got {sig.shape}"
    assert meta["fs"] == 500.0, f"Unexpected dummy fs, expected 500.0, got {meta.get('fs')!r}"
