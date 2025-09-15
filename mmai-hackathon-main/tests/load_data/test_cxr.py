"""Tests for MIMIC-IV Chest X-ray (CXR) loading utilities.

This suite validates two public APIs:

- ``load_mimic_cxr_metadata(path, ...)``: parses the CXR metadata CSV, verifies
  the expected dataset layout (``files/`` subfolder), resolves absolute image
  paths, applies optional row filtering, and returns a ``pd.DataFrame`` with a
  ``cxr_path`` column pointing to existing ``.jpg`` files.
- ``load_chest_xray_image(path, to_gray=True)``: opens a chest X-ray image at
  the given path, optionally converting to grayscale.

Prerequisite
------------
The tests assume the real dataset is available during CI under the fixed path:
``${PWD}/MMAI25Hackathon/mimic-iv/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0``.
If that directory or its ``files/`` subfolder is missing, the tests are skipped
rather than failing.
"""

import logging
from pathlib import Path

import pandas as pd
import pytest

from mmai25_hackathon.load_data.cxr import (
    load_chest_xray_image,
    load_mimic_cxr_metadata,
)

# Fixed dataset path (fetched during CI)
CXR_ROOT = Path.cwd() / "MMAI25Hackathon" / "mimic-iv" / "mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0"


@pytest.fixture(scope="module")
def cxr_root() -> Path:
    """Return the dataset root or skip the module if missing."""
    if not CXR_ROOT.exists():
        pytest.skip(f"Dataset root not found: {CXR_ROOT}")
    files_dir = CXR_ROOT / "files"
    if not files_dir.exists():
        pytest.skip(f"Dataset 'files' subdir not found under: {CXR_ROOT}")
    return CXR_ROOT


@pytest.fixture(scope="module")
def metadata_df(cxr_root: Path) -> pd.DataFrame:
    """Load metadata once for the module to speed up tests."""
    return load_mimic_cxr_metadata(cxr_root)


@pytest.mark.parametrize("use_str_path", [True, False])
def test_metadata_and_image_loading(caplog: pytest.LogCaptureFixture, cxr_root: Path, use_str_path: bool):
    # Ensure INFO logs from the loader are captured if emitted
    caplog.set_level(logging.INFO)

    path_arg = str(cxr_root) if use_str_path else cxr_root
    df = load_mimic_cxr_metadata(path_arg)

    # Basic metadata checks
    assert isinstance(df, pd.DataFrame), "Expected a DataFrame from load_mimic_cxr_metadata"
    assert not df.empty, "Metadata DataFrame is unexpectedly empty"
    assert "cxr_path" in df.columns, "Expected column 'cxr_path' to be present"

    # If the implementation logs mapping info, it should be visible here.
    # We don't require it, but if present we assert the message contains 'Mapped'.
    if caplog.records:
        assert any("Mapped" in rec.getMessage() for rec in caplog.records), "Expected mapping log message"

    # Sample one image and check both grayscale and RGB branches
    sample_path = Path(str(df.iloc[0]["cxr_path"]))  # type: ignore[index]
    assert sample_path.is_absolute(), "cxr_path should be absolute"
    assert sample_path.exists(), f"Image path does not exist: {sample_path}"

    img_gray = load_chest_xray_image(sample_path)
    assert img_gray.mode == "L"

    img_rgb = load_chest_xray_image(sample_path, to_gray=False)
    assert img_rgb.mode == "RGB"


def test_paths_are_absolute_and_exist_on_head(metadata_df: pd.DataFrame):
    head_paths = metadata_df["cxr_path"].astype(str).head(10).tolist()

    for p in head_paths:
        pth = Path(p)
        assert pth.is_absolute(), f"Path is not absolute: {pth}"
        assert pth.exists(), f"Resolved path does not exist: {pth}"
        assert pth.suffix.lower() == ".jpg"


def test_filter_rows_train_subset_is_consistent(cxr_root: Path, metadata_df: pd.DataFrame):
    # Use an ID that exists in the dataset. 101 is commonly present in the public splits.
    train_df = load_mimic_cxr_metadata(cxr_root, filter_rows={"subject_id": [101]})

    assert not train_df.empty, "Filtered subject_id is unexpectedly empty"
    assert set(train_df["subject_id"].unique()) == {101}, "Filtered subject_id has unexpected values"

    # The filtered set must be a subset of the unfiltered rows with subject_id==101
    all_train = metadata_df[metadata_df["subject_id"] == 101]
    assert set(train_df["cxr_path"]).issubset(set(all_train["cxr_path"])), "Filtered rows mismatch"


def test_loading_nonexistent_image_raises(cxr_root: Path):
    missing = cxr_root / "files" / "__definitely_not_here__.jpg"
    with pytest.raises(FileNotFoundError):
        load_chest_xray_image(missing)


def test_invalid_metadata_path_raises(cxr_root: Path):
    with pytest.raises(FileNotFoundError):
        load_mimic_cxr_metadata(cxr_root / "nonexistent_dir")


def test_missing_dicom_id_column_raises(tmp_path: Path):
    """If required DICOM ID columns are absent, the loader must raise ``KeyError``."""
    df = pd.DataFrame({"subject_id": [1, 2], "study_id": [10, 20], "other_column": ["A", "B"]})
    (tmp_path / "files").mkdir()
    (tmp_path / "metadata.csv").write_text(df.to_csv(index=False))

    with pytest.raises(KeyError):
        load_mimic_cxr_metadata(tmp_path, filter_rows={"subject_id": [1]})


def test_files_folder_missing_raises(tmp_path: Path):
    """If ``files/`` is missing, the loader must raise ``FileNotFoundError``."""
    df = pd.DataFrame({"subject_id": [1, 2], "study_id": [10, 20], "dicom_id": ["img1", "img2"]})
    (tmp_path / "metadata.csv").write_text(df.to_csv(index=False))

    with pytest.raises(FileNotFoundError):
        load_mimic_cxr_metadata(tmp_path, filter_rows={"subject_id": [1]})


def test_metadata_not_found_raises(tmp_path: Path):
    """If no metadata CSV is present, the loader must raise ``FileNotFoundError``."""
    (tmp_path / "files").mkdir()

    with pytest.raises(FileNotFoundError):
        load_mimic_cxr_metadata(tmp_path, filter_rows={"subject_id": [1]})
