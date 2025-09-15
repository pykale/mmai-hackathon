"""
MIMIC-IV Electronic Health Record (EHR) loading and merging utilities.

Functions:
load_mimic_iv_ehr(
    ehr_path, module='hosp'|'icu'|'both', tables=None, index_cols=None,
    subset_cols=None, filter_rows=None, merge=True, join='inner', raise_errors=True
)
    Discovers and loads CSV tables from the selected MIMIC-IV module(s). Optionally selects
    per-table columns, filters rows, and merges tables by overlapping key columns (via
    `merge_multiple_dataframes`). Returns a dict of DataFrames when `merge=False` or a single
    merged DataFrame when `merge=True`. Raises `FileNotFoundError` if the dataset/subfolders are
    missing and `ValueError` for invalid table names or disjoint merge components.

Notes:
- Column selection and row filtering are delegated to `read_tabular`.
- `index_cols` are used as merge keys only (the DataFrame index is not set by this helper).

Preview CLI:
`python -m mmai25_hackathon.load_data.ehr --data-path /path/to/mimic-iv-3.1`
Loads a small example (e.g., ICU stays + admissions), merges on `subject_id, hadm_id`, and prints a preview.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Union

import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from .tabular import merge_multiple_dataframes, read_tabular

__all__ = ["load_mimic_iv_ehr"]

MIMIC_IV_EHR_AVAILABLE_TABLES = {
    "hosp": (
        "admissions",
        "diagnoses_icd",
        "drgcodes",
        "emar",
        "emar_detail",
        "hcpcsevents",
        "labevents",
        "microbiologyevents",
        "omr",
        "patients",
        "pharmacy",
        "poe",
        "poe_detail",
        "prescriptions",
        "procedures_icd",
        "provider",
        "services",
        "transfers",
        "d_hcpcs",
        "d_icd_diagnoses",
        "d_icd_procedures",
        "d_labitems",
    ),
    "icu": (
        "caregiver",
        "chartevents",
        "d_items",
        "datetimeevents",
        "icustays",
        "ingredientevents",
        "inputevents",
        "outputevents",
        "procedureevents",
    ),
}


@validate_params(
    {
        "ehr_path": [str, Path],
        "module": [StrOptions({"hosp", "icu", "both"})],
        "tables": [None, "array-like"],
        "index_cols": [None, list, str],
        "subset_cols": [None, dict],
        "filter_rows": [None, dict],
        "merge": ["boolean"],
        "join": [StrOptions({"inner", "outer", "left", "right"})],
        "raise_errors": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def load_mimic_iv_ehr(
    ehr_path: Union[str, Path],
    module: Literal["hosp", "icu", "both"] = "hosp",
    tables: Optional[Sequence[str]] = None,
    index_cols: Optional[Union[List[str], str]] = None,
    subset_cols: Optional[Dict[str, Sequence[str]]] = None,
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
    merge: bool = True,
    join: Literal["inner", "outer", "left", "right"] = "inner",
    raise_errors: bool = True,
) -> Union[Dict[str, pd.DataFrame], pd.DataFrame]:
    """
    Query, load, and aggregate MIMIC-IV EHR data from specified module(s) and tables.

    High-level steps:
    - Coerce `ehr_path` to `Path` and validate root; select modules (`hosp`, `icu`, or both) and check subfolders (optional).
    - Discover available `tables` under selected modules (or validate requested names).
    - Load each table via `read_tabular` with optional `subset_cols`, `index_cols`, and `filter_rows`.
    - When `merge=False`, return a dict of DataFrames keyed by table name.
    - When `merge=True`, call `merge_multiple_dataframes` on loaded tables; if multiple disjoint components remain, raise.
    - Return the merged DataFrame.

    Args:
        ehr_path (Union[str, Path]): Path to the root folder containing `hosp` and/or `icu` subfolders.
        module (Literal['hosp', 'icu', 'both']): Module(s) to load data from. The 'hosp' module contains hospital-wide
            data, while the 'icu' module contains intensive care unit-specific data. Default: 'hosp'.
        tables (Optional[Sequence[str]]): Specific sequences of tables to load. If None, all available tables in the selected
            module(s) will be loaded. Default: None.
        index_cols (Optional[List[str]]): Columns to use as keys for merging tables. If None and merge=True, will try to do
            naive concatenation. Default: None.
        subset_cols (Optional[Dict[str, List[str]]]): Per-table column selection. If provided, only these columns
            will be loaded from each table. Default: None.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): A dictionary to filter rows in the DataFrame.
            Keys are column names and values are the values to filter by. Default: None.
        merge (bool): Whether to merge the loaded tables into components based on shared keys. Default: True.
        join (str): Merge strategy to use when merging tables given merge=True. Options include 'inner', 'outer', 'left', and 'right'.
            Default: 'inner'.
        raise_errors (bool): If True, will raise an error for the following criterias:
            1. `modules` not found in `ehr_path`. Will fetch existing ones if False.
            2. `subset_cols` or `index_cols` provided but none of the specified columns are found in the DataFrame.
            3. `filter_rows` provided but none of the specified values are found in the DataFrame.

    Returns:
        Union[Dict[str, pd.DataFrame], pd.DataFrame]: If merge is False, returns a dictionary of DataFrames
            keyed by table names. If merge is True, returns a single merged DataFrame.

    Raises:
        FileNotFoundError: If `ehr_path` does not exist or if the specified `modules` subfolder is not found
            and `raise_errors` is True.
        ValueError: If no available tables are found for the specified `modules`, if any of the requested `tables`
            are not available, or if merging results in multiple components with exclusive keys.

    Examples:
        >>> # Load specific tables from both modules and merge them on 'subject_id' and 'hadm_id'
        >>> df = load_mimic_iv_ehr(
        ...     ehr_path="path/to/mimic-iv-3.1",
        ...     module="both",
        ...     tables=["icustays", "admissions"],
        ...     index_cols=["subject_id", "hadm_id"],
        ...     subset_cols={"icustays": ["first_careunit"], "admissions": ["admittime"]},
        ...     merge=True,
        ...     join="inner",
        ... )
        >>> print(df.head())
            subject_id  hadm_id         admittime                                   first_careunit
        0          101        1  24/02/2196 14:38                                   Neuro Stepdown
        1          101        2  17/09/2153 17:08  Neuro Surgical Intensive Care Unit (Neuro SICU)
        2          101        3  18/08/2134 02:02                               Neuro Intermediate
        3          102        4  13/11/2111 23:39                              Trauma SICU (TSICU)
        4          102        5  04/08/2113 18:46                              Trauma SICU (TSICU)
        5          103        6  12/12/2132 01:43                              Trauma SICU (TSICU)
    """
    if isinstance(ehr_path, str):
        ehr_path = Path(ehr_path)

    if not ehr_path.exists():
        raise FileNotFoundError(f"MIMIC-IV EHR path not found: '{ehr_path}'")

    # Check if hosp and/or icu directories exist, expect to be validated
    # later will add sklearn params validation
    selected_modules = ["hosp", "icu"] if module == "both" else [module]

    # need to check availability of selected modules
    # sklearn param validation doesn't support this
    for mod in selected_modules:
        if not (ehr_path / mod).exists() and raise_errors:
            raise FileNotFoundError(f"Expected subfolder '{mod}' not found in {ehr_path}")

    # generate dictionary of available tables to load given selected modules, tables, and paths
    available_tables = {}
    for mod in selected_modules:
        if tables is None:
            for table in MIMIC_IV_EHR_AVAILABLE_TABLES[mod]:
                path = ehr_path / mod / f"{table}.csv"
                if path.exists():
                    available_tables[table] = path
            continue

        for table in tables:
            path = ehr_path / mod / f"{table}.csv"
            if table in MIMIC_IV_EHR_AVAILABLE_TABLES[mod] and path.exists():
                available_tables[table] = path

    logger = logging.getLogger(f"{__name__}.load_mimic_iv_ehr")
    logger.info("Selected modules: %s", selected_modules)
    logger.info("Available tables to load: %s", list(available_tables.keys()))

    # Validate we have at least one table to load
    if len(available_tables) == 0:
        raise ValueError(f"No available tables found for modules: {selected_modules}")

    # Check available tables if any missing
    if tables is not None:
        missing_tables = set(tables) - set(available_tables.keys())
        if missing_tables:
            raise ValueError(f"The following requested tables are not available: {missing_tables}")

    # Load tables
    logger.info("Loading tables from: %s", ehr_path)
    dfs = {
        table: read_tabular(
            path,
            (subset_cols or {}).get(table, None),
            index_cols,
            filter_rows,
            raise_errors=raise_errors,
        )
        for table, path in available_tables.items()
    }

    if not merge:
        return dfs

    logging.info("Merging tables on keys: %s using '%s' join", index_cols, join)
    aggregated_dfs = merge_multiple_dataframes(
        list(dfs.values()), dfs_name=list(dfs.keys()), index_cols=index_cols, join=join
    )

    if len(aggregated_dfs) != 1:
        # Find exclusive keys between aggregated dataframes
        all_keys = [set(keys) for keys, _ in aggregated_dfs]
        exclusive_keys = set().union(*all_keys) - set().intersection(*all_keys)
        raise ValueError(
            f"Merging resulted in multiple components with exclusive keys: {exclusive_keys}. "
            "Consider using a different set of index_cols or setting merge=False when "
            "loading tables with exclusive/disjoint keys."
        )

    _, merged_df = aggregated_dfs[0]
    logger.info("Merged DataFrame shape: %s", merged_df.shape)

    return merged_df


if __name__ == "__main__":
    import argparse

    # Example script given the relative path to folder mimic-iv
    # containing mimic-iv-3.1 that has hosp and icu subfolders:
    # python -m mmai25_hackathon.load_data.ehr --data-path mimic-iv/mimic-iv-3.1
    parser = argparse.ArgumentParser(description="Fetch MIMIC-IV EHR data example.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the MIMIC-IV EHR root directory (mimic-iv-3.1).",
        default="MMAI25Hackathon/mimic-iv/mimic-iv-3.1",
    )
    args = parser.parse_args()

    print("Loading MIMIC-IV EHR data example...")
    dfs_new = load_mimic_iv_ehr(
        ehr_path=args.data_path,
        module="both",
        tables=["icustays", "admissions"],
        index_cols=["subject_id", "hadm_id"],
        subset_cols={
            "icustays": ["first_careunit"],
            "admissions": ["admittime"],
        },
        filter_rows={"subject_id": [101]},
        merge=True,
        join="inner",
    )

    print(dfs_new.head())
