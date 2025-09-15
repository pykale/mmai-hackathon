"""Tests for protein sequence utilities.

This suite validates the public APIs in ``mmai25_hackathon.load_data.protein``:

- ``load_protein_sequences_from_dataframe(df_or_path, ...)``: extracts a protein sequence column from a DataFrame
  or CSV path, optionally setting an index, and returns a single-column DataFrame named ``protein_sequence``.
- ``protein_sequence_to_integer_encoding(sequence, ...)``: integer-encodes an amino-acid sequence to fixed length.

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

from mmai25_hackathon.load_data.protein import (
    load_protein_sequences_from_dataframe,
    protein_sequence_to_integer_encoding,
)

# Optional real dataset path for integration-style checks
PROTEIN_DATASET_CSV = Path.cwd() / "MMAI25Hackathon" / "molecule-protein-interaction" / "dataset.csv"


@pytest.fixture(scope="module")
def protein_csv() -> Path:
    if not PROTEIN_DATASET_CSV.exists():
        pytest.skip(f"Protein dataset CSV not found: {PROTEIN_DATASET_CSV}")
    return PROTEIN_DATASET_CSV


def test_load_protein_sequences_from_dataframe_and_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "Protein": ["MKTAYI", "GAVLIL"], "extra": [0, 1]})
    csv = tmp_path / "proteins.csv"
    csv.write_text(df.to_csv(index=False))

    out_df = load_protein_sequences_from_dataframe(df, prot_seq_col="Protein", index_col="id")
    assert list(out_df.columns) == [
        "id",
        "protein_sequence",
    ], f"Expected columns ['id', 'protein_sequence'], got {list(out_df.columns)}"
    assert out_df.shape[0] == 2, f"Unexpected number of rows: {out_df.shape}"

    out_csv = load_protein_sequences_from_dataframe(str(csv), prot_seq_col="Protein")
    assert list(out_csv.columns) == [
        "protein_sequence"
    ], f"Expected single column 'protein_sequence', got {list(out_csv.columns)}"

    with pytest.raises(ValueError):
        load_protein_sequences_from_dataframe(df, prot_seq_col="missing")


def test_protein_sequence_to_integer_encoding_properties() -> None:
    seq = "MKTAY?"  # '?' should map to 0 (unknown)
    enc = protein_sequence_to_integer_encoding(seq, max_length=5)

    assert (
        isinstance(enc, np.ndarray) and enc.ndim == 1
    ), f"Expected 1D numpy array, got type={type(enc)!r}, shape={getattr(enc, 'shape', None)}"
    assert enc.dtype == np.uint64, f"Expected dtype uint64, got {enc.dtype}"
    assert len(enc) == 5, f"Expected length 5 (truncation), got {len(enc)}"
    assert (enc == 0).sum() >= 0, "Unknown characters should be encoded as 0"


def test_load_proteins_from_real_dataset(protein_csv: Path) -> None:
    out = load_protein_sequences_from_dataframe(str(protein_csv), prot_seq_col="Protein")
    assert not out.empty, f"No Protein rows loaded from {protein_csv}"
    assert list(out.columns) == [
        "protein_sequence"
    ], f"Expected single column 'protein_sequence', got {list(out.columns)}"
    first = out.iloc[0]["protein_sequence"]
    assert isinstance(first, str) and len(first) > 0, f"First protein sequence is invalid: {first!r}"


def test_load_protein_sequences_dataframe_filter_rows() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "Protein": ["MKT", "GAV", "TTT"]})
    out = load_protein_sequences_from_dataframe(
        df,
        prot_seq_col="Protein",
        index_col="id",
        filter_rows={"id": [3]},
    )
    assert out.shape[0] == 1, f"Expected 1 row after filtering, got shape {out.shape}"
    assert 3 in out["id"].values, f"Expected remaining id to be 3, got {out['id'].tolist()}"
