"""Tests for MIMIC-IV Electronic Health Record (EHR) utilities.

This suite validates the public API in ``mmai25_hackathon.load_data.ehr``:

- ``load_mimic_iv_ehr(ehr_path, ...)``: discovers available tables for selected module(s),
  loads CSVs with optional column selection and row filtering, and merges tables by overlapping keys.

Prerequisite
------------
No external dataset required. The tests use synthetic CSVs under temporary directories to validate
behavior, including error handling for missing modules/tables and merge semantics.
"""

from pathlib import Path

import pandas as pd
import pytest

from mmai25_hackathon.load_data.ehr import load_mimic_iv_ehr


def _write_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(df.to_csv(index=False))


def test_invalid_ehr_base_path_raises(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_ehr(tmp_path / "missing")


def test_missing_selected_module_subfolder_raises(tmp_path: Path) -> None:
    # Create only 'hosp'
    (tmp_path / "hosp").mkdir()
    with pytest.raises(FileNotFoundError):
        load_mimic_iv_ehr(tmp_path, module="icu")


def test_no_available_tables_raises(tmp_path: Path) -> None:
    (tmp_path / "hosp").mkdir()
    with pytest.raises(ValueError):
        load_mimic_iv_ehr(tmp_path, module="hosp")


def test_requested_missing_tables_raises(tmp_path: Path) -> None:
    # Create one table only
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    _write_csv(hosp / "admissions.csv", pd.DataFrame({"subject_id": [1], "hadm_id": [10], "admittime": ["x"]}))

    with pytest.raises(ValueError):
        load_mimic_iv_ehr(
            tmp_path, module="hosp", tables=["admissions", "transfers"], index_cols=["subject_id", "hadm_id"]
        )


@pytest.mark.parametrize("as_str", [True, False])
def test_merge_success_on_shared_keys(tmp_path: Path, as_str: bool) -> None:
    hosp = tmp_path / "hosp"
    icu = tmp_path / "icu"
    hosp.mkdir()
    icu.mkdir()

    _write_csv(
        hosp / "admissions.csv",
        pd.DataFrame(
            {
                "subject_id": [101, 102],
                "hadm_id": [1, 2],
                "admittime": ["t1", "t2"],
            }
        ),
    )
    _write_csv(
        icu / "icustays.csv",
        pd.DataFrame(
            {
                "subject_id": [101, 102],
                "hadm_id": [1, 2],
                "first_careunit": ["A", "B"],
            }
        ),
    )

    root_arg = str(tmp_path) if as_str else tmp_path
    df = load_mimic_iv_ehr(
        root_arg,
        module="both",
        tables=["admissions", "icustays"],
        index_cols=["subject_id", "hadm_id"],
        subset_cols={"admissions": ["admittime"], "icustays": ["first_careunit"]},
        filter_rows={"subject_id": [101]},
        merge=True,
        join="inner",
    )

    assert not df.empty, "Merged EHR DataFrame is unexpectedly empty"
    assert set(["subject_id", "hadm_id", "admittime", "first_careunit"]).issubset(
        df.columns
    ), f"Missing expected columns after merge; got: {list(df.columns)}"
    assert set(df["subject_id"]) == {101}, f"Filter_rows not applied as expected: {set(df['subject_id'])}"


def test_merge_multiple_components_raises(tmp_path: Path) -> None:
    hosp = tmp_path / "hosp"
    icu = tmp_path / "icu"
    hosp.mkdir()
    icu.mkdir()

    _write_csv(hosp / "patients.csv", pd.DataFrame({"subject_id": [1, 2], "gender": ["M", "F"]}))
    _write_csv(icu / "caregiver.csv", pd.DataFrame({"icustay_id": [10, 20], "role": ["x", "y"]}))

    with pytest.raises(ValueError):
        load_mimic_iv_ehr(
            tmp_path,
            module="both",
            tables=["patients", "caregiver"],
            index_cols=["subject_id", "icustay_id"],
            merge=True,
        )


def test_merge_false_returns_dict(tmp_path: Path) -> None:
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    _write_csv(hosp / "admissions.csv", pd.DataFrame({"subject_id": [1], "hadm_id": [10], "admittime": ["t"]}))

    dfs = load_mimic_iv_ehr(tmp_path, module="hosp", tables=["admissions"], merge=False)
    assert isinstance(dfs, dict), f"Expected dict of DataFrames, got {type(dfs)!r}"
    assert set(dfs.keys()) == {"admissions"}, f"Unexpected keys in result: {set(dfs.keys())}"
    assert not dfs["admissions"].empty, "admissions DataFrame unexpectedly empty"


def test_autodiscover_tables_when_none(tmp_path: Path) -> None:
    hosp = tmp_path / "hosp"
    hosp.mkdir()
    _write_csv(hosp / "admissions.csv", pd.DataFrame({"subject_id": [7], "hadm_id": [70], "admittime": ["t"]}))

    # With tables=None and module='hosp', admissions should be discovered (covers available_tables[table]=path branch)
    df = load_mimic_iv_ehr(tmp_path, module="hosp", tables=None, index_cols=["subject_id", "hadm_id"], merge=True)
    assert isinstance(df, pd.DataFrame) and not df.empty, "Autodiscovered admissions table should load successfully"


# Optional real dataset integration
EHR_ROOT = Path.cwd() / "MMAI25Hackathon" / "mimic-iv" / "mimic-iv-3.1"


@pytest.fixture(scope="module")
def ehr_root() -> Path:
    if not EHR_ROOT.exists():
        pytest.skip(f"EHR root not found: {EHR_ROOT}")
    return EHR_ROOT


def test_integration_load_and_merge_real_ehr_if_available(ehr_root: Path) -> None:
    # Try a small, common pair of tables
    hosp_adm = ehr_root / "hosp" / "admissions.csv"
    icu_stays = ehr_root / "icu" / "icustays.csv"
    if not hosp_adm.exists() or not icu_stays.exists():
        pytest.skip("Required EHR tables (admissions or icustays) not found; skipping integration test")

    df = load_mimic_iv_ehr(
        ehr_root,
        module="both",
        tables=["admissions", "icustays"],
        index_cols=["subject_id", "hadm_id"],
        subset_cols={"admissions": ["admittime"], "icustays": ["first_careunit"]},
        merge=True,
        join="inner",
    )
    # We can't guarantee non-emptiness in all subsets, but can assert type and columns when present
    assert isinstance(df, pd.DataFrame), "Expected merged DataFrame"
    if not df.empty:
        assert {"subject_id", "hadm_id"}.issubset(df.columns), f"Merged keys missing in columns: {df.columns}"
