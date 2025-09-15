"""
MIMIC-IV Echocardiogram (ECHO) loading utilities.

Functions:
load_mimic_iv_echo_record_list(echo_path, filter_rows=None)
    Loads `echo-record-list.csv`, verifies dataset layout (expects `files/`), constructs absolute
    `echo_path` for each DICOM from `dicom_filepath`, filters rows if provided, and drops paths that
    do not exist. Returns a `pd.DataFrame`.

load_echo_dicom(path)
    Reads an ECHO DICOM (cine or single-frame) with pydicom. Returns `(frames, metadata)` where
    `frames` is `np.ndarray` shaped `(T, H, W)` (or `(1, H, W)`), rescaled via `RescaleSlope` and
    `RescaleIntercept`, and `metadata` is a dict of DICOM keywords to values.

Preview CLI:
`python -m mmai25_hackathon.load_data.echo --data-path /path/to/mimic-iv-echo-...`
Prints a preview of the record list and loads one example DICOM to report shape and selected metadata.
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
from pydicom import dcmread
from sklearn.utils._param_validation import validate_params

from .tabular import read_tabular

__all__ = ["load_mimic_iv_echo_record_list", "load_echo_dicom"]


@validate_params({"echo_path": [str, Path], "filter_rows": [None, dict]}, prefer_skip_nested_validation=True)
def load_mimic_iv_echo_record_list(
    echo_path: Union[str, Path], filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None
) -> pd.DataFrame:
    """
    Load the MIMIC-IV-ECHO `echo-record-list.csv` file as a DataFrame and maps DICOM file paths.
    The path must contain a `files` subdirectory with the DICOM files.

    High-level steps:
    - Validate dataset root, ensure `files/` and `echo-record-list.csv` exist.
    - Load CSV via `read_tabular`, optionally applying `filter_rows`.
    - Strip `dicom_filepath`, resolve absolute `echo_path` under the root.
    - Keep only rows whose `echo_path` exists on disk.
    - Return the filtered DataFrame.

    Args:
        echo_path (str): The root directory of the MIMIC-IV-ECHO dataset.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of `echo-record-list.csv`.

    Raises:
        FileNotFoundError: If the specified `echo_path` does not exist or if the CSV file cannot be found.

    Examples:
        >>> df = load_mimic_iv_echo_record_list("path/to/mimic-iv-echo")
        >>> print(df.head())
           subject_id  study_id acquisition_datetime                dicom_filepath                                          echo_path
        0         101       133     03/10/2204 13:14  files/p100/p101/s133/133.dcm  mimic-iv/mimic-iv-echo-0.1.physionet.org/files...
        1         101       231     03/10/2204 13:17  files/p100/p101/s231/231.dcm  mimic-iv/mimic-iv-echo-0.1.physionet.org/files...
        2         101       378     03/10/2204 13:18  files/p100/p101/s378/378.dcm  mimic-iv/mimic-iv-echo-0.1.physionet.org/files...
        3         102       484     03/10/2204 13:18  files/p100/p102/s484/484.dcm  mimic-iv/mimic-iv-echo-0.1.physionet.org/files...
        4         102       548     03/10/2204 13:18  files/p100/p102/s548/548.dcm  mimic-iv/mimic-iv-echo-0.1.physionet.org/files...
    """
    if isinstance(echo_path, str):
        echo_path = Path(echo_path)

    if not echo_path.exists():
        raise FileNotFoundError(f"MIMIC-IV-ECHO path not found: {echo_path}")

    if not (echo_path / "files").exists():
        raise FileNotFoundError(f"Expected 'files' subdirectory not found under: {echo_path}")

    records_path = echo_path / "echo-record-list.csv"
    if not records_path.exists():
        raise FileNotFoundError(f"'echo-record-list.csv' not found in: {echo_path}")

    logger = logging.getLogger(f"{__name__}.load_mimic_iv_echo_record_list")
    logger.info("Loading ECHO record list from: %s", records_path)
    df_records = read_tabular(records_path, filter_rows=filter_rows)

    logger.info("Loaded %d records from %s", len(df_records), records_path)
    logger.info("Mapping ECHO DICOM file paths under: %s", echo_path / "files")

    df_records["dicom_filepath"] = df_records["dicom_filepath"].astype(str).str.strip()
    df_records["echo_path"] = df_records["dicom_filepath"].map(lambda rel_path: str(echo_path / rel_path))

    existing_files = df_records["echo_path"].map(os.path.exists)
    logger.info("Found %d records with existing DICOM files.", existing_files.sum())

    return df_records[existing_files].copy()


@validate_params({"path": [str, Path]}, prefer_skip_nested_validation=True)
def load_echo_dicom(path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an ECHO DICOM (supports multi-frame cine) using pydicom.

    High-level steps:
    - Coerce `path` to `Path` and validate it exists.
    - Read DICOM via `pydicom.dcmread` and extract `pixel_array`.
    - If 2D, expand dims to shape `(1, H, W)`.
    - Apply rescale using `RescaleSlope` and `RescaleIntercept` if present.
    - Collect metadata from DICOM elements into a dictionary.
    - Return `(frames, metadata)`.

    Args:
        path (Union[str, Path]): The path to the ECHO DICOM file.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
            - frames: np.ndarray with shape (T, H, W) for multi-frame, or (1, H, W) for single image
            - metadata: metadata from the DICOM file as a dictionary (e.g., Rows, Columns, NumberOfFrames, FrameTime, CineRate, etc.)

    Examples:
        >>> frames, meta = load_echo_dicom("path/to/echo.dcm")
        >>> print("Frames shape:", frames.shape)
        Frames shape: (58, 708, 1016, 3)
        >>> print("Meta:", {k: meta[k] for k in ("NumberOfFrames", "Rows", "Columns", "FrameTime", "CineRate")})
        Meta: {'NumberOfFrames': '58', 'Rows': 708, 'Columns': 1016, 'FrameTime': '33.6842', 'CineRate': '30'}
    """
    if isinstance(path, str):
        path = Path(path)

    if not path.exists():
        raise FileNotFoundError(f"ECHO DICOM not found: {path}")

    logger = logging.getLogger(f"{__name__}.load_echo_dicom")
    logger.info("Loading ECHO DICOM: %s", path)

    echo = dcmread(path)
    frames = echo.pixel_array  # shape (num_frames, height, width) or (height, width)
    if frames.ndim == 2:
        frames = np.expand_dims(frames, 0)  # single image to (1, H, W)

    logger.info("Loaded frames with shape: %s", frames.shape)
    logger.info("Adjusting pixel values using RescaleSlope and RescaleIntercept if present.")
    intercept = float(getattr(echo, "RescaleIntercept", 0.0))
    slope = float(getattr(echo, "RescaleSlope", 1.0))
    frames = frames * slope + intercept

    metadata = {elem.keyword: elem.value for elem in echo if elem.keyword}
    logger.info("Extracted %d metadata fields from DICOM.", len(metadata))

    return frames, metadata


if __name__ == "__main__":
    import argparse

    # Example script:
    # python -m mmai25_hackathon.load_data.echo --data-path mimic-iv/mimic-iv-echo-0.1.physionet.org
    parser = argparse.ArgumentParser(description="Load MIMIC-IV-ECHO metadata and DICOM files.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the MIMIC-IV-ECHO dataset directory (containing 'files' subdirectory).",
        default="MMAI25Hackathon/mimic-iv/mimic-iv-echo-0.1.physionet.org",
    )
    args = parser.parse_args()

    print("Loading MIMIC-IV-ECHO record list...")
    records = load_mimic_iv_echo_record_list(args.data_path)
    print(records.head()[["subject_id", "study_id", "echo_path"]])

    # Example of loading a DICOM file
    if not records.empty:
        print()
        print(f"Loading first ECHO DICOM from: {records.iloc[0]['echo_path']}")
        example_path = records.iloc[0]["echo_path"]
        frames, meta = load_echo_dicom(example_path)
        meta_filtered = {
            k: meta[k] for k in ("NumberOfFrames", "Rows", "Columns", "FrameTime", "CineRate") if k in meta
        }
        print(f"Loaded frames shape: {frames.shape}")
        print(f"Metadata sample: {meta_filtered}")
