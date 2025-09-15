"""Tests for tabular utilities: ``read_tabular`` and ``merge_multiple_dataframes``.

This suite validates the public APIs in ``mmai25_hackathon.load_data.tabular``:

- ``read_tabular(path, ...)``: loads a CSV, optionally selects/indexes columns, and applies row filtering.
- ``merge_multiple_dataframes(dfs, ...)``: merges frames by overlapping key columns, or concatenates when no keys.

Prerequisite
------------
Optional real-data integration uses:
``${PWD}/MMAI25Hackathon/mimic-iv/mimic-iv-3.1``.
If unavailable, the integration test is skipped; unit tests still validate selection, filtering,
merge grouping, suffix behavior, and error handling using synthetic CSVs and DataFrames.
"""

from pathlib import Path

import pandas as pd
import pytest

from mmai25_hackathon.load_data.tabular import merge_multiple_dataframes, read_tabular


def test_read_tabular_selects_and_filters(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1, 2, 3], "a": [10, 20, 30], "b": [0.1, 0.2, 0.3]})
    p = tmp_path / "data.csv"
    p.write_text(df.to_csv(index=False))

    # Select subset/index cols in order and filter rows
    out = read_tabular(p, subset_cols=["b", "missing"], index_cols="id", filter_rows={"id": [1, 3]})
    assert list(out.columns) == ["id", "b"], f"Unexpected columns: {list(out.columns)}"
    assert out["id"].tolist() == [1, 3], f"Row filter not applied as expected: {out['id'].tolist()}"


def test_read_tabular_raises_on_invalid_selection_when_requested(tmp_path: Path) -> None:
    df = pd.DataFrame({"id": [1], "x": [5]})
    p = tmp_path / "d.csv"
    p.write_text(df.to_csv(index=False))

    with pytest.raises(ValueError):
        read_tabular(p, subset_cols=["missing"], raise_errors=True)

    with pytest.raises(ValueError):
        read_tabular(p, index_cols=["missing"], raise_errors=True)


def test_merge_multiple_dataframes_concat_when_no_keys() -> None:
    df1 = pd.DataFrame({"a": [1, 2]})
    df2 = pd.DataFrame({"b": [10, 20]})
    comps = merge_multiple_dataframes([df1, df2], index_cols=None)
    assert (
        len(comps) == 1 and comps[0][0] == ()
    ), f"Expected single concat component, got {[(k, d.shape) for k, d in comps]}"
    assert list(comps[0][1].columns) == ["a", "b"], f"Unexpected columns after concat: {list(comps[0][1].columns)}"


def test_merge_multiple_dataframes_by_overlap_keys() -> None:
    df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
    df2 = pd.DataFrame({"id": [1, 2], "b": [0.1, 0.2]})
    df3 = pd.DataFrame({"site": ["A", "B"], "c": [5, 6]})
    comps = merge_multiple_dataframes(
        [df1, df2, df3], dfs_name=["X", "Y", "Z"], index_cols=["id", "site"], join="inner"
    )
    # Should produce two components keyed by 'id' and 'site'
    keys_list = [keys for keys, _ in comps]
    assert ("id",) in keys_list and ("site",) in keys_list, f"Unexpected key components: {keys_list}"


def test_merge_multiple_dataframes_invalid_join() -> None:
    with pytest.raises(ValueError):
        merge_multiple_dataframes([pd.DataFrame()], index_cols=["id"], join="bad")


def test_merge_multiple_dataframes_empty_input() -> None:
    assert merge_multiple_dataframes([]) == [], "Expected empty list for empty input"


def test_merge_multiple_dataframes_labels_length_mismatch() -> None:
    with pytest.raises(ValueError):
        merge_multiple_dataframes([pd.DataFrame(), pd.DataFrame()], dfs_name=["a"], index_cols=["id"])


def test_merge_no_subsets_returns_empty() -> None:
    # Provide index_cols that none of the DataFrames contain to cover df_by_subset empty branch
    df1 = pd.DataFrame({"x": [1]})
    df2 = pd.DataFrame({"y": [2]})
    comps = merge_multiple_dataframes([df1, df2], index_cols=["id"])  # no 'id' in frames
    assert comps == [], f"Expected empty components when no frames share the provided keys; got {comps}"


def test_merge_greedy_overlap_path() -> None:
    # Three groups: ('id',), ('id','site'), and ('site') to exercise greedy overlap selection and merge
    df_id = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
    df_id_site = pd.DataFrame({"id": [1, 2], "site": ["A", "B"], "b": [0.1, 0.2]})
    df_site = pd.DataFrame({"site": ["A", "B"], "c": [5, 6]})

    comps = merge_multiple_dataframes(
        [df_id, df_id_site, df_site],
        dfs_name=["X", "Y", "Z"],
        index_cols=["id", "site"],
        join="inner",
    )
    # All three can merge into a single component via greedy merging
    assert len(comps) == 1, f"Expected a single merged component, got {[(k, d.shape) for k, d in comps]}"


# Optional real EHR root for integration-style checks
EHR_ROOT = Path.cwd() / "MMAI25Hackathon" / "mimic-iv" / "mimic-iv-3.1"


@pytest.fixture(scope="module")
def ehr_root() -> Path:
    if not EHR_ROOT.exists():
        pytest.skip(f"EHR root not found: {EHR_ROOT}")
    return EHR_ROOT


def test_read_and_merge_real_ehr_if_available(ehr_root: Path) -> None:
    hosp_adm = ehr_root / "hosp" / "admissions.csv"
    icu_stays = ehr_root / "icu" / "icustays.csv"
    if not hosp_adm.exists() or not icu_stays.exists():
        pytest.skip("Required EHR tables (admissions or icustays) not found; skipping integration test")

    # Load minimal subsets with expected keys
    adm_df = read_tabular(
        hosp_adm,
        subset_cols=["admittime"],
        index_cols=["subject_id", "hadm_id"],
        raise_errors=False,
    )
    stays_df = read_tabular(
        icu_stays,
        subset_cols=["first_careunit"],
        index_cols=["subject_id", "hadm_id"],
        raise_errors=False,
    )

    comps = merge_multiple_dataframes(
        [adm_df, stays_df],
        dfs_name=["admissions", "icustays"],
        index_cols=["subject_id", "hadm_id"],
        join="inner",
    )
    key_sets = [keys for keys, _ in comps]
    assert ("hadm_id", "subject_id") in key_sets or (
        "subject_id",
        "hadm_id",
    ) in key_sets, f"Expected a merged component on subject_id/hadm_id, got keys: {key_sets}"
