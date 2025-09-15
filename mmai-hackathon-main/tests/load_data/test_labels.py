"""Tests for labels utilities.

This suite validates the public APIs in ``mmai25_hackathon.load_data.labels``:

- ``load_labels_from_dataframe(df_or_path, ...)``: extracts label column(s) from a DataFrame or CSV path,
  optionally setting an index, and returns a DataFrame named ``label`` (single column) or original names (multi).
- ``one_hot_encode_labels(labels, ...)``: one-hot encodes categorical label columns to ``float32`` dtypes.

Prerequisite
------------
Optional real-data integration uses:
``${PWD}/MMAI25Hackathon/molecule-protein-interaction/dataset.csv``.
If unavailable, integration tests are skipped; unit tests still validate core behavior.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmai25_hackathon.load_data.labels import (
    load_labels_from_dataframe,
    one_hot_encode_labels,
)

# Optional real dataset path for integration-style checks
LABELS_DATASET_CSV = Path.cwd() / "MMAI25Hackathon" / "molecule-protein-interaction" / "dataset.csv"


@pytest.fixture(scope="module")
def labels_csv() -> Path:
    if not LABELS_DATASET_CSV.exists():
        pytest.skip(f"Labels dataset CSV not found: {LABELS_DATASET_CSV}")
    return LABELS_DATASET_CSV


def test_load_labels_single_column(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "Y": [0, 1], "extra": ["a", "b"]})
    csv = tmp_path / "labels.csv"
    csv.write_text(df.to_csv(index=False))

    out_df = load_labels_from_dataframe(df, label_col="Y", index_col="id")
    assert list(out_df.columns) == ["id", "label"], f"Expected columns ['id', 'label'], got {list(out_df.columns)}"
    assert out_df.shape[0] == 2, f"Unexpected number of rows: {out_df.shape}"

    out_csv = load_labels_from_dataframe(str(csv), label_col="Y")
    assert list(out_csv.columns) == ["label"], f"Expected single column 'label', got {list(out_csv.columns)}"

    with pytest.raises(ValueError):
        load_labels_from_dataframe(df, label_col="missing")


def test_one_hot_encode_labels_single_and_multi_columns() -> None:
    df = pd.DataFrame({"label": ["cat", "dog", "cat", "mouse"]})
    oh = one_hot_encode_labels(df)
    assert {
        "label_cat",
        "label_dog",
        "label_mouse",
    }.issubset(oh.columns), f"Missing expected one-hot columns: {list(oh.columns)}"
    assert all(oh.dtypes == np.float32), f"One-hot dtypes must be float32, got {oh.dtypes}"

    df2 = pd.DataFrame({"a": ["x", "y"], "b": ["u", "v"]})
    oh2 = one_hot_encode_labels(df2, columns=["a", "b"])
    assert any(c.startswith("a_") for c in oh2.columns), f"Expected one-hot columns for 'a': {list(oh2.columns)}"
    assert any(c.startswith("b_") for c in oh2.columns), f"Expected one-hot columns for 'b': {list(oh2.columns)}"


def test_load_labels_dataframe_filter_rows() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "Y": [0, 1, 0]})
    out = load_labels_from_dataframe(
        df,
        label_col="Y",
        index_col="id",
        filter_rows={"id": [1, 3]},
    )
    assert out.shape[0] == 2, f"Expected 2 rows after filtering, got shape {out.shape}"
    assert set(out["id"].values) == {1, 3}, f"Unexpected remaining ids: {out['id'].tolist()}"


def test_load_labels_with_filter_rows(tmp_path: Path) -> None:
    # Create a small CSV with an ID column to filter on
    df = pd.DataFrame({"id": [1, 2, 3], "Y": [0, 1, 0]})
    csv = tmp_path / "labels.csv"
    csv.write_text(df.to_csv(index=False))

    # Apply a filter to keep only id==2; include index_col so filter can apply
    out = load_labels_from_dataframe(str(csv), label_col="Y", index_col="id", filter_rows={"id": [2]})
    assert out.shape[0] == 1, f"Expected 1 row after filtering, got shape {out.shape}"
    assert 2 in out["id"].values, f"Expected remaining id to be 2, got {out['id'].tolist()}"


def test_load_labels_from_real_dataset(labels_csv: Path) -> None:
    out = load_labels_from_dataframe(str(labels_csv), label_col="Y")
    assert not out.empty, f"No label rows loaded from {labels_csv}"
    assert list(out.columns) == ["label"], f"Expected single column 'label', got {list(out.columns)}"
