"""Tests for molecular SMILES utilities.

This suite validates the public APIs in ``mmai25_hackathon.load_data.molecule``:

- ``load_smiles_from_dataframe(df_or_path, ...)``: extracts a SMILES column from a DataFrame or CSV path,
  optionally setting an index, and returns a single-column DataFrame named ``smiles``.
- ``smiles_to_graph(smiles, ...)``: converts a SMILES string to a PyG ``Data`` graph; flags forwarded.

Prerequisite
------------
Optional real-data integration uses:
``${PWD}/MMAI25Hackathon/molecule-protein-interaction/dataset.csv``.
If unavailable, integration tests are skipped; unit tests still validate core behavior and conversion via monkeypatching.
"""

from pathlib import Path

import pandas as pd
import pytest

# Ensure torch_geometric is available; otherwise, skip this module's tests
pytest.importorskip("torch_geometric")
from torch_geometric.data import Data  # noqa: E402

from mmai25_hackathon.load_data.molecule import load_smiles_from_dataframe, smiles_to_graph  # noqa: E402

# Optional real dataset path for integration-style checks
MOLECULE_DATASET_CSV = Path.cwd() / "MMAI25Hackathon" / "molecule-protein-interaction" / "dataset.csv"


@pytest.fixture(scope="module")
def molecule_csv() -> Path:
    if not MOLECULE_DATASET_CSV.exists():
        pytest.skip(f"Molecule dataset CSV not found: {MOLECULE_DATASET_CSV}")
    return MOLECULE_DATASET_CSV


def test_load_smiles_from_dataframe_and_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2], "SMILES": ["CCO", "C1=CC=CC=C1"], "extra": [0, 1]})
    csv = tmp_path / "molecules.csv"
    csv.write_text(df.to_csv(index=False))

    # From DataFrame with index
    out_df = load_smiles_from_dataframe(df, smiles_col="SMILES", index_col="id")
    assert list(out_df.columns) == ["id", "smiles"], f"Expected columns ['id', 'smiles'], got {list(out_df.columns)}"
    assert out_df.shape[0] == 2, f"Unexpected number of rows: {out_df.shape}"

    # From CSV
    out_csv = load_smiles_from_dataframe(str(csv), smiles_col="SMILES")
    assert list(out_csv.columns) == ["smiles"], f"Expected columns ['smiles'], got {list(out_csv.columns)}"

    # Missing column error
    with pytest.raises(ValueError):
        load_smiles_from_dataframe(df, smiles_col="missing")


def test_smiles_to_graph_monkeypatched(monkeypatch: pytest.MonkeyPatch) -> None:
    # Monkeypatch torch_geometric.utils.smiles.from_smiles to return a tiny Data object
    from mmai25_hackathon.load_data import molecule as mol_mod

    def fake_from_smiles(s: str, with_h: bool, kek: bool) -> Data:  # type: ignore[override]
        return Data(x=None, edge_index=None, edge_attr=None, smiles=s, with_h=with_h, kek=kek)

    monkeypatch.setattr(mol_mod, "from_smiles", fake_from_smiles, raising=True)

    g = smiles_to_graph("CCO", with_hydrogen=True, kekulize=False)
    assert isinstance(g, Data), f"Expected Data, got {type(g)!r}"
    assert getattr(g, "smiles", None) == "CCO", f"SMILES not propagated: {getattr(g, 'smiles', None)!r}"
    assert (
        getattr(g, "with_h", None) is True and getattr(g, "kek", None) is False
    ), f"Flags not forwarded correctly: with_h={getattr(g, 'with_h', None)}, kek={getattr(g, 'kek', None)}"


def test_load_smiles_from_real_dataset(molecule_csv: Path) -> None:
    out = load_smiles_from_dataframe(str(molecule_csv), smiles_col="SMILES")
    assert not out.empty, f"No SMILES rows loaded from {molecule_csv}"
    assert list(out.columns) == ["smiles"], f"Expected single column 'smiles', got {list(out.columns)}"
    first = out.iloc[0]["smiles"]
    assert isinstance(first, str) and len(first) > 0, f"First SMILES is invalid: {first!r}"


def test_load_smiles_dataframe_filter_rows() -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "SMILES": ["CCO", "CCC", "CCN"]})
    out = load_smiles_from_dataframe(
        df,
        smiles_col="SMILES",
        index_col="id",
        filter_rows={"id": [2]},
    )
    assert out.shape[0] == 1, f"Expected 1 row after filtering, got shape {out.shape}"
    assert 2 in out["id"].values, f"Expected remaining id to be 2, got {out['id'].tolist()}"
