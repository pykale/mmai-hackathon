"""Tests for MIMIC-IV Echocardiogram (ECHO) loading utilities.

This suite validates the public APIs in ``mmai25_hackathon.load_data.echo``:

- ``load_mimic_iv_echo_record_list(echo_path, ...)``: parses ``echo-record-list.csv``, verifies
  the expected dataset layout (``files/`` subfolder), resolves absolute DICOM paths from
  ``dicom_filepath``, applies optional row filtering, and returns a ``pd.DataFrame`` containing only
  rows with existing files.
- ``load_echo_dicom(path)``: reads an ECHO DICOM via ``pydicom.dcmread`` and returns frames (T,H,W) and metadata.

Prerequisite
------------
The tests assume the real dataset may be available under the fixed path:
``${PWD}/MMAI25Hackathon/mimic-iv/mimic-iv-echo-0.1.physionet.org``.
If that directory, its ``files/`` subfolder, or ``echo-record-list.csv`` is missing, the integration
tests are skipped. Unit-level behavior and error handling are still validated via temporary data.
"""

import logging
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# Ensure pydicom is available; otherwise, skip this module's tests at collection time
pytest.importorskip("pydicom")

from mmai25_hackathon.load_data.echo import load_echo_dicom, load_mimic_iv_echo_record_list  # noqa: E402

ECHO_ROOT = Path.cwd() / "MMAI25Hackathon" / "mimic-iv" / "mimic-iv-echo-0.1.physionet.org"


@pytest.fixture(scope="module")
def echo_root() -> Path:
    if not ECHO_ROOT.exists():
        pytest.skip(f"Dataset root not found: {ECHO_ROOT}")
    files_dir = ECHO_ROOT / "files"
    if not files_dir.exists():
        pytest.skip(f"Dataset 'files' subdir not found: {ECHO_ROOT}")
    records_csv = ECHO_ROOT / "echo-record-list.csv"
    if not records_csv.exists():
        pytest.skip(f"'echo-record-list.csv' not found under: {ECHO_ROOT}")
    return ECHO_ROOT


@pytest.fixture(scope="module")
def echo_df(echo_root: Path) -> pd.DataFrame:
    return load_mimic_iv_echo_record_list(echo_root)


@pytest.mark.parametrize("use_str_path", [True, False])
def test_record_list_mapping_and_paths(caplog: pytest.LogCaptureFixture, echo_root: Path, use_str_path: bool) -> None:
    caplog.set_level(logging.INFO)

    arg = str(echo_root) if use_str_path else echo_root
    df = load_mimic_iv_echo_record_list(arg)

    assert isinstance(df, pd.DataFrame), f"Expected DataFrame, got {type(df)!r}"
    assert not df.empty, "ECHO record list is empty; ensure 'echo-record-list.csv' has rows and DICOM files exist."
    assert "echo_path" in df.columns, f"Missing 'echo_path' column; columns: {list(df.columns)}"

    # Inspect one record
    p = Path(str(df.iloc[0]["echo_path"]))  # type: ignore[index]
    assert p.is_absolute(), f"echo_path should be absolute, got: {p}"
    assert p.exists(), f"Resolved DICOM does not exist: {p}"
    assert p.suffix.lower() == ".dcm", f"Expected .dcm suffix, got {p.suffix} (path={p})"

    # Optional logging checks
    if caplog.records:
        assert any(
            "Mapping ECHO DICOM" in r.getMessage() or "Found" in r.getMessage() for r in caplog.records
        ), "Expected mapping/discovery log messages in INFO logs"


def test_invalid_echo_base_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_echo_record_list(tmp_path / "missing")


def test_files_folder_missing_raises(tmp_path: Path) -> None:
    (tmp_path / "echo-record-list.csv").write_text(
        pd.DataFrame({"dicom_filepath": ["files/p100/p101/s133/133.dcm"]}).to_csv(index=False)
    )
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_echo_record_list(tmp_path)


def test_record_list_not_found_raises(tmp_path: Path) -> None:
    (tmp_path / "files").mkdir()
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_echo_record_list(tmp_path)


def test_mapping_and_filtering_with_temp_csv(tmp_path: Path) -> None:
    files_dir = tmp_path / "files"
    files_dir.mkdir()

    df = pd.DataFrame(
        {
            "subject_id": [101, 102, 102],
            "study_id": [133, 200, 201],
            "dicom_filepath": [
                " files/p100/p101/s133/133.dcm ",  # valid
                " files/p200/p201/s200/200.dcm ",  # missing
                " files/p200/p201/s201/201.dcm ",  # valid
            ],
        }
    )
    (tmp_path / "echo-record-list.csv").write_text(df.to_csv(index=False))

    # Create only 133.dcm and 201.dcm
    (files_dir / "p100" / "p101" / "s133").mkdir(parents=True)
    (files_dir / "p100" / "p101" / "s133" / "133.dcm").write_bytes(b"DICOM")
    (files_dir / "p200" / "p201" / "s201").mkdir(parents=True)
    (files_dir / "p200" / "p201" / "s201" / "201.dcm").write_bytes(b"DICOM")

    out = load_mimic_iv_echo_record_list(tmp_path, filter_rows={"subject_id": [101, 102]})
    assert len(out) == 2, f"Expected two existing DICOMs, got {len(out)}"
    assert set(out["study_id"]) == {133, 201}, f"Unexpected study_ids mapped: {set(out['study_id'])}"
    for p in out["echo_path"].astype(str):
        pth = Path(p)
        assert pth.is_absolute(), f"echo_path should be absolute, got {pth}"
        assert pth.exists(), f"Mapped DICOM missing on disk: {pth}"


def test_loading_nonexistent_dicom_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_echo_dicom(tmp_path / "not_here.dcm")


@pytest.mark.parametrize("as_str", [True, False])
def test_load_echo_dicom_with_monkeypatch(monkeypatch: pytest.MonkeyPatch, tmp_path: Path, as_str: bool) -> None:
    # Create a dummy file to pass existence check
    dcm = tmp_path / "a" / "b.dcm"
    dcm.parent.mkdir(parents=True)
    dcm.write_bytes(b"DICOM")

    class FakeDicom:
        # Start with single-frame (H,W) to exercise branch
        def __init__(self) -> None:
            self.pixel_array = np.ones((8, 6), dtype=np.float32)
            self.RescaleSlope = 2.0
            self.RescaleIntercept = -1.0

        def __iter__(self):
            # Simulate iteration over DICOM elements with keyword/value
            class E:
                def __init__(self, k: str, v: Any) -> None:
                    self.keyword, self.value = k, v

            yield E("Rows", 8)
            yield E("Columns", 6)
            yield E("NumberOfFrames", 1)

    # Monkeypatch the symbol actually used by the loader (echo.dcmread)
    from mmai25_hackathon.load_data import echo as echo_mod

    monkeypatch.setattr(echo_mod, "dcmread", lambda p: FakeDicom(), raising=True)

    path_arg = str(dcm) if as_str else dcm
    frames, meta = load_echo_dicom(path_arg)
    assert frames.shape == (1, 8, 6), f"Expected (1,H,W) after expand, got {frames.shape}"
    # Check rescale applied: 1*2-1 = 1
    assert float(frames.mean()) == pytest.approx(1.0), f"Unexpected rescaled mean: {frames.mean()}"
    assert {"Rows", "Columns"}.issubset(meta.keys()), f"Missing expected metadata keys: {meta.keys()}"
