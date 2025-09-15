"""
MIMIC-IV Electrocardiogram (ECG) loading utilities.

Functions:
load_mimic_iv_ecg_record_list(ecg_path, filter_rows=None)
    Loads `record_list.csv`, verifies dataset layout (expects a `files/` subdirectory), constructs absolute
    `ecg_path` for each record from the CSV `path` column, derives the corresponding `.hea` and `.dat` paths,
    filters rows if provided, and keeps only rows where both files exist. Returns a `pd.DataFrame` with the
    added columns: `ecg_path`, `hea_path`, and `dat_path`.

load_ecg_record(hea_path)
    Reads an ECG record with `wfdb.rdsamp` using the provided `.hea` file path (the stem is passed to WFDB).
    Returns `(signals, metadata)` where `signals` is a `np.ndarray` shaped `(T, L)` (time samples x leads),
    and `metadata` is a dict of WFDB header fields (e.g., `fs`, `n_sig`).

Preview CLI:
`python -m mmai25_hackathon.load_data.ecg --data-path /path/to/mimic-iv-ecg-...`
Prints a preview of the record list (including `hea_path` and `dat_path`), then loads one example record
to report the array shape and selected metadata (sampling frequency and number of leads).
"""

import logging
import os
from pathlib import Path
from typing import Any, Dict, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import wfdb
from sklearn.utils._param_validation import validate_params

from .tabular import read_tabular

__all__ = ["load_mimic_iv_ecg_record_list", "load_ecg_record"]


@validate_params({"ecg_path": [str, Path], "filter_rows": [None, dict]}, prefer_skip_nested_validation=True)
def load_mimic_iv_ecg_record_list(
    ecg_path: Union[str, Path], filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None
) -> pd.DataFrame:
    """
    Load the MIMIC-IV-ECG `record_list.csv` file as a DataFrame and maps available `.dat` and `.hea` files
    to their respective paths. The path must contain a `files` subdirectory with the ECG files.

    High-level steps:
    - Validate dataset root, ensure `files/` and `record_list.csv` exist.
    - Load CSV via `read_tabular`, optionally applying `filter_rows`.
    - Strip `path`, resolve absolute `ecg_path`, and derive `hea_path`/`dat_path`.
    - Keep only rows for which both `.hea` and `.dat` exist.
    - Return the filtered DataFrame.

    Args:
        ecg_path (Union[str, Path]): The root directory of the MIMIC-IV-ECG dataset.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: A DataFrame containing the contents of `record_list.csv` with additional columns:
            - `ecg_path`: Full path to the original ECG file.
            - `hea_path`: Full path to the corresponding `.hea` file.
            - `dat_path`: Full path to the corresponding `.dat` file.
        Only rows with both `.hea` and `.dat` files present are included.

    Raises:
        FileNotFoundError: If the specified `ecg_path` does not exist, `files` subdirectory is missing,
            or if the `record_list.csv` file cannot be found.

    Examples:
        >>> df = load_mimic_iv_ecg_record_list("path/to/mimic-iv-ecg")
        >>> print(df.head())
           subject_id                                           hea_path                                           dat_path
        0         101  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...
        1         101  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...
        2         101  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...
        3         102  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...
        4         102  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...  mimic-iv/mimic-iv-ecg-diagnostic-electrocardio...
    """
    if isinstance(ecg_path, str):
        ecg_path = Path(ecg_path)

    if not ecg_path.exists():
        raise FileNotFoundError(f"MIMIC-IV-ECG base path not found: {ecg_path}")

    if not (ecg_path / "files").exists():
        raise FileNotFoundError(f"Expected 'files' subdirectory not found under: {ecg_path}")

    records_path = ecg_path / "record_list.csv"
    if not records_path.exists():
        raise FileNotFoundError(f"'record_list.csv' not found in: {ecg_path}")

    df_records = read_tabular(records_path, filter_rows=filter_rows)

    logger = logging.getLogger(f"{__name__}.load_mimic_iv_ecg_record_list")
    logger.info("Loaded %d records from %s", len(df_records), records_path)
    logger.info("Mapping ECG file paths under: %s", ecg_path / "files")

    df_records["path"] = df_records["path"].astype(str).str.strip()
    df_records["ecg_path"] = df_records["path"].map(lambda rel_path: str(ecg_path / rel_path))
    df_records["hea_path"] = df_records["ecg_path"].map(lambda x: str(Path(x).with_suffix(".hea")))
    df_records["dat_path"] = df_records["ecg_path"].map(lambda x: str(Path(x).with_suffix(".dat")))

    existing_hea = df_records["hea_path"].map(os.path.exists)
    existing_dat = df_records["dat_path"].map(os.path.exists)

    # Only collect records with both .hea and .dat present
    available_files = existing_hea & existing_dat
    logger.info("Found %d records with both .hea and .dat files present.", available_files.sum())
    return df_records[available_files].copy()


@validate_params({"hea_path": [str, Path]}, prefer_skip_nested_validation=True)
def load_ecg_record(hea_path: Union[str, Path]) -> Tuple[np.ndarray, Dict[str, Any]]:
    """
    Load an ECG record given a .hea file path using wfdb.

    High-level steps:
    - Coerce `hea_path` to `Path` and validate the file exists.
    - Call `wfdb.rdsamp` with the stem (path without suffix).
    - Return the sampled `signals` and `fields` metadata.

    Args:
        hea_path (Union[str, Path]): The path to the .hea file.

    Returns:
        Tuple[np.ndarray, Dict[str, Any]]: A tuple containing:
            - signals (np.ndarray): The ECG signal data with shape (signal_length, num_leads).
            - fields (Dict[str, Any]): A dictionary of metadata fields from the .hea file.

    Raises:
        FileNotFoundError: If the specified .hea file does not exist.

    Examples:
        >>> signals, fields = load_ecg_wfdb("path/to/record.hea")
        >>> print(signals.shape)  # (signal_length, num_leads) e.g.
        (5000, 12)
        >>> print(fields["fs"])  # Sampling frequency
        500
    """
    if isinstance(hea_path, str):
        hea_path = Path(hea_path)

    if not hea_path.exists():
        raise FileNotFoundError(f"ECG .hea path not found: {hea_path}")

    logger = logging.getLogger(f"{__name__}.load_ecg_record")
    logger.info("Loading ECG record from: %s", hea_path)
    signals, fields = wfdb.rdsamp(hea_path.with_suffix("").as_posix())

    logger.info("Loaded ECG signals with shape: %s", signals.shape)
    logger.info("Metadata fields: %s", list(fields.keys()))

    return signals, fields


if __name__ == "__main__":
    import argparse

    # Example script:
    # python -m mmai25_hackathon.load_data.ecg --data-path mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0
    parser = argparse.ArgumentParser(description="Load MIMIC-IV-ECG metadata and records.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the MIMIC-IV-ECG dataset root (should contain 'files' subdirectory).",
        default="MMAI25Hackathon/mimic-iv/mimic-iv-ecg-diagnostic-electrocardiogram-matched-subset-1.0",
    )
    args = parser.parse_args()

    print("Loading MIMIC-IV-ECG record list...")
    records = load_mimic_iv_ecg_record_list(args.data_path)
    print(records.head()[["subject_id", "hea_path", "dat_path"]])

    # Example of loading a record
    if not records.empty:
        print()
        print(f"Loading first ECG record from: {records.iloc[0]['hea_path']}")
        signals, fields = load_ecg_record(records.iloc[0]["hea_path"])
        print(f"Loaded ECG signals with shape: {signals.shape}")
        print(f"Metadata fields: {list(fields.keys())}")
        print(f"Sampling frequency: {fields.get('fs', 'N/A')} Hz")
        print(f"Number of leads: {fields.get('n_sig', 'N/A')}")
