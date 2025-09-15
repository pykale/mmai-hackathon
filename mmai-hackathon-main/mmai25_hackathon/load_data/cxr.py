"""
Chest x-ray (CXR) loading utilities for MIMIC-CXR.

Functions:
load_mimic_cxr_metadata(cxr_path, filter_rows=None)
    Scans the dataset directory for a metadata CSV, loads it, optionally filters rows,
    and adds a `cxr_path` column pointing to each JPEG file (by DICOM ID). Returns a
    `pd.DataFrame`. Raises `FileNotFoundError` if the dataset/CSV is missing and
    `KeyError` if no suitable DICOM ID column is found.

load_chest_xray_image(path, to_gray=True)
    Opens a chest x-ray image with PIL. Converts to grayscale ("L") when `to_gray=True`,
    otherwise returns RGB. Returns a `PIL.Image.Image`. Raises `FileNotFoundError` if the
    image path does not exist.

Preview CLI:
`python -m mmai25_hackathon.load_data.cxr --data-path /path/to/mimic-cxr-jpg-...`
Loads metadata, prints a preview, and opens a sample image.
"""

import logging
import os
from pathlib import Path
from typing import Dict, Optional, Sequence, Union

import pandas as pd
from PIL import Image
from sklearn.utils._param_validation import validate_params

from .tabular import read_tabular

__all__ = ["load_mimic_cxr_metadata", "load_chest_xray_image"]

METADATA_PATTERNS = ("*metadata*.csv", "*mimic-cxr*-metadata*.csv", "mimic-cxr-2.0.0-metadata.csv")
DICOM_ID_COLUMN_CANDIDATES = ("dicom_id", "dicom", "image_id")


@validate_params({"cxr_path": [Path, str], "filter_rows": [None, dict]}, prefer_skip_nested_validation=True)
def load_mimic_cxr_metadata(
    cxr_path: Union[Path, str], filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None
) -> pd.DataFrame:
    """
    Loads the MIMIC CXR metadata and maps available DICOM IDs to their corresponding image file paths.

    High-level steps:
    - Validate dataset root, ensure `files/` exists and find a metadata CSV by pattern.
    - Load metadata via `read_tabular`, optionally applying `filter_rows`; normalise columns to lower case.
    - Identify a DICOM ID column, compute absolute `cxr_path` by rglob under `files/`.
    - Keep only rows that successfully map to an existing `.jpg`.
    - Return the filtered DataFrame.

    Args:
        cxr_path (Union[Path, str]): The root directory of the MIMIC CXR dataset.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: A DataFrame containing the metadata with an additional column `cxr_path`
            that provides the full path to each image file. Rows without a corresponding image file are excluded.

    Raises:
        FileNotFoundError: If the specified `cxr_path` does not exist or if the metadata CSV cannot be found.
        KeyError: If no suitable DICOM ID column is found in the metadata CSV.

    Examples:
        >>> df_metadata = load_mimic_cxr_metadata("path/to/mimic-cxr")
        >>> print(df_metadata.head()[["subject_id", "cxr_path"]])
            subject_id                                           cxr_path
        0           101  mimic-iv/mimic-cxr-jpg-chest-radiographs-with-...
        1           101  mimic-iv/mimic-cxr-jpg-chest-radiographs-with-...
        2           101  mimic-iv/mimic-cxr-jpg-chest-radiographs-with-...
        3           101  mimic-iv/mimic-cxr-jpg-chest-radiographs-with-...
        4           101  mimic-iv/mimic-cxr-jpg-chest-radiographs-with-...
    """
    if isinstance(cxr_path, str):
        cxr_path = Path(cxr_path)

    if not cxr_path.exists():
        raise FileNotFoundError(f"MIMIC CXR path not found: {cxr_path}")

    if not (cxr_path / "files").exists():
        raise FileNotFoundError(f"Expected 'files' subdirectory not found under: {cxr_path}")

    # find metadata csv given patterns
    metadata_path = None
    for pat in METADATA_PATTERNS:
        for subpath in cxr_path.rglob(pat):
            # Stop if multiple found, take the first
            if metadata_path is None:
                metadata_path = subpath
                break

    if metadata_path is None:
        raise FileNotFoundError(f"Metadata could not be found in {cxr_path} given patterns: {METADATA_PATTERNS}")

    logger = logging.getLogger(f"{__name__}.load_mimic_cxr_metadata")
    logger.info("Loading CXR metadata from: %s", metadata_path)
    df_metadata = read_tabular(metadata_path, filter_rows=filter_rows)
    df_metadata.columns = df_metadata.columns.str.lower()
    dicom_id_col = df_metadata.columns.intersection(DICOM_ID_COLUMN_CANDIDATES)

    if len(dicom_id_col) == 0:
        raise KeyError(f"No suitable DICOM ID column found. Expected one of: {DICOM_ID_COLUMN_CANDIDATES}")
    dicom_id_col = dicom_id_col[0]  # take the first match

    logger.info("Using DICOM ID column: %s", dicom_id_col)
    logger.info("Found %d metadata entries in: %s", len(df_metadata), metadata_path)
    logger.info("Mapping DICOM IDs to image files under: %s", cxr_path / "files")

    # image path column
    df_metadata["cxr_path"] = (
        df_metadata[dicom_id_col]
        .astype(str)
        .str.strip()
        .map(lambda x: str(next((cxr_path / "files").rglob(f"{x}.jpg"), "")))
    )

    df_metadata = df_metadata[df_metadata["cxr_path"] != ""].copy()
    logger.info("Mapped %d metadata entries to existing image files.", len(df_metadata))

    return df_metadata


@validate_params({"path": [Path, str], "to_gray": ["boolean"]}, prefer_skip_nested_validation=True)
def load_chest_xray_image(path: Union[str, Path], to_gray: bool = True) -> Image.Image:
    """
    Loads a chest X-ray image from the specified path.

    High-level steps:
    - Validate the image path exists.
    - Open image with PIL and convert to `L` (grayscale) or `RGB` based on `to_gray`.
    - Return the converted `Image`.

    Args:
        path (Union[str, Path]): The file path to the chest X-ray image.
        to_gray (bool): If True, convert the image to grayscale. Default is True.

    Returns:
        Image.Image: The loaded chest X-ray image.

    Raises:
        FileNotFoundError: If the specified image file does not exist.

    Examples:
        >>> image = load_chest_xray_image("path/to/image.jpg", to_gray=True)
        >>> image.show()
    """
    if isinstance(path, Path):
        path = str(path)

    if not os.path.exists(path):
        raise FileNotFoundError(f"Image not found: {path}")

    logger = logging.getLogger(f"{__name__}.load_chest_xray_image")
    logger.info("Loading image: %s", path)

    img = Image.open(path)
    img = img.convert("L") if to_gray else img.convert("RGB")
    logger.info("Loaded image size: %s, mode: %s", img.size, img.mode)

    return img


if __name__ == "__main__":
    import argparse

    # Example script:
    # python -m mmai25_hackathon.load_data.cxr --data-path mimic-iv/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0
    parser = argparse.ArgumentParser(description="Load MIMIC CXR metadata and images.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the MIMIC CXR dataset directory.",
        default="MMAI25Hackathon/mimic-iv/mimic-cxr-jpg-chest-radiographs-with-structured-labels-2.1.0",
    )
    args = parser.parse_args()

    print("Loading MIMIC CXR metadata...")
    metadata = load_mimic_cxr_metadata(args.data_path)
    print(metadata.head()[["subject_id", "cxr_path"]])

    # Example of loading an image
    if not metadata.empty:
        print()
        print(f"Loading first chest x-ray image from: {metadata.iloc[0]['cxr_path']}")
        example_path = metadata.iloc[0]["cxr_path"]
        image = load_chest_xray_image(example_path)
        image.show()
