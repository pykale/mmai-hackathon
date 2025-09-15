"""
Tabular utilities for loading a CSV and merging multiple DataFrames by overlapping key columns.

Functions:
read_tabular(path, subset_cols=None, index_cols=None, filter_rows=None, sep=",", raise_errors=True)
    Thin wrapper around `pandas.read_csv`. Optionally selects key columns first (order-preserving),
    keeps only requested columns, and filters rows via a `{column: allowed_values}` mapping.
    Note: does **not** set the DataFrame index; `index_cols` are treated as key columns only.

merge_multiple_dataframes(dfs, dfs_name=None, index_cols=None, join="outer")
    Greedily merges a sequence of DataFrames into connected components based on overlapping key columns.
    First merges frames that share the same subset of keys, then merges groups whose key sets overlap.
    Returns a list of `(keys_tuple, merged_df)`. Name collisions get suffixes from `dfs_name` or `_df{i}`.

Preview CLI:
`python -m mmai25_hackathon.load_data.tabular --data-path BASE_PATH --index-cols ... --subset-cols ... --join outer`
Recursively loads `*.csv`, then groups/merges and prints a preview for each component.
"""

import logging
from pathlib import Path
from typing import Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

__all__ = ["read_tabular", "merge_multiple_dataframes"]


@validate_params(
    {
        "path": [Path, str],
        "subset_cols": [None, list, str],
        "index_cols": [None, list, str],
        "filter_rows": [None, dict],
        "sep": [str],
        "raise_errors": ["boolean"],
    },
    prefer_skip_nested_validation=True,
)
def read_tabular(
    path: Union[str, Path],
    subset_cols: Optional[Union[List[str], str]] = None,
    index_cols: Optional[Union[List[str], str]] = None,
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
    sep: str = ",",
    raise_errors: bool = True,
) -> pd.DataFrame:
    """
    Reads a single tabular text file into a DataFrame.

    If `subset_cols` and/or `index_cols` are provided, will select only those columns
    that exist in the DataFrame. The order will be `index_cols` followed by `subset_cols`.

    High-level steps:
    - Read CSV via `pandas.read_csv` with separator `sep`.
    - Normalise `subset_cols`/`index_cols` to lists; compute intersections with available columns.
    - When `raise_errors` and none of the requested columns exist, raise `ValueError`.
    - Order columns with `index_cols` first then `subset_cols`; filter rows per `filter_rows` where possible.
    - Return the resulting DataFrame.

    Args:
        path (Union[str, Path]): Path to the tabular text file.
        subset_cols (Optional[Union[List[str], str]]): If provided, will select only these columns if they exist in the DataFrame.
            Default: None.
        index_cols (Optional[Union[List[str], str]]): If provided, will select these columns as the index of the DataFrame.
            Default: None.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.
        sep (str): Value separator in the tabular text file. Default: ",".
        raise_errors (bool): If True, will raise an error if none of the specified `subset_cols`
            or `index_cols` are found in the DataFrame. Default: True.

    Returns:
        pd.DataFrame: The loaded DataFrame with complete or partial columns selected.

    Raises:
        ValueError: If `raise_errors` is True and none of the specified `subset_cols`
            or `index_cols` are found in the DataFrame.

    Examples:
        >>> df = read_tabular("data.csv")
        >>> print(df.head())
    """
    # Just provide thin wrapper around pd.read_csv for function availability.
    logger = logging.getLogger(f"{__name__}.read_tabular")
    logger.info("Reading tabular data from: %s", path)

    df = pd.read_csv(path, sep=sep)

    if isinstance(subset_cols, str):
        subset_cols = [subset_cols]

    if isinstance(index_cols, str):
        index_cols = [index_cols]

    selected_index_cols = pd.Index([])
    if index_cols is not None:
        # preserves the order of index_cols as provided when sort=False
        logger.info("Selecting index columns: %s", index_cols)
        selected_index_cols = df.columns.intersection(index_cols, sort=False)
        logger.info("Found index columns in DataFrame: %s", selected_index_cols.to_list())

    selected_subset_cols = df.columns.difference(selected_index_cols, sort=False)
    if subset_cols is not None:
        logger.info("Selecting subset columns: %s", subset_cols)
        selected_subset_cols = df.columns.intersection(subset_cols, sort=False)
        logger.info("Found subset columns in DataFrame: %s", selected_subset_cols.to_list())

    if raise_errors and subset_cols is not None and len(selected_subset_cols) == 0:
        raise ValueError(f"No valid subset_cols found in DataFrame for: {subset_cols}")

    if raise_errors and index_cols is not None and len(selected_index_cols) == 0:
        raise ValueError(f"No valid index_cols found in DataFrame for: {index_cols}")

    # Reorder the dataframe to have index_cols first then subset_cols
    logger.info(
        "Final selected columns: %s",
        selected_index_cols.to_list() + selected_subset_cols.to_list(),
    )
    selected_cols = selected_index_cols.union(selected_subset_cols, sort=False)

    df = df if len(selected_cols) == 0 else df[df.columns.intersection(selected_cols, sort=False)]
    logger.info("Loaded DataFrame shape: %s", df.shape)

    logger.info("Applying row filters: %s", filter_rows)
    for col, valid_vals in (filter_rows or {}).items():
        if col in df.columns:
            logger.info(
                "Filtering rows on column '%s' with %d valid values.",
                col,
                len(valid_vals),
            )
            df = df[df[col].isin(valid_vals)]

    return df


@validate_params(
    {
        "dfs": ["array-like"],
        "dfs_name": [None, "array-like"],
        "index_cols": [None, list, str],
        "join": [StrOptions({"outer", "inner", "left", "right"})],
    },
    prefer_skip_nested_validation=True,
)
def merge_multiple_dataframes(
    dfs: Sequence[pd.DataFrame],
    dfs_name: Optional[Sequence[str]] = None,
    index_cols: Optional[List[str]] = None,
    join: Literal["outer", "inner", "left", "right"] = "outer",
) -> List[Tuple[Tuple[str, ...], pd.DataFrame]]:
    """
    Merge a sequence of DataFrames by shared keys until disjoint components remain.

    High-level steps:
    - Validate `join` option and `dfs_name` length.
    - If `dfs` is empty, return []. If `index_cols` is falsy, concatenate columns and return single component.
    - Group frames by the exact subset of provided keys they contain; if none, return [].
    - Merge within each group using suffixes determined by `dfs_name`.
    - Greedily merge groups whose key sets overlap until no overlaps remain.
    - Return components as `(sorted_keys_tuple, DataFrame)` pairs.

    Args:
        dfs (Sequence[pd.DataFrame]): Sequences of dataframes to merge.
        dfs_name (Optional[Sequence[str]]): Optional names (same length as `dfs`) used to derive merge suffixes.
        index_cols (Optional[List[str]]): Key columns to use for merging. If None/empty,
            will concatenate all frames. Default: None.
        subset_cols (Optional[List[str]]): If provided, will pre select these columns from each dataframe
            if any of them exist in the dataframe. Default: None.
        join (Literal["outer", "inner", "left", "right"]): Dataframe merging strategy. Default: "outer".

    Returns:
        List[Tuple[Tuple[str, ...], pd.DataFrame]]: A list of tuples where each tuple contains:
            - A tuple of key column names used for merging that component.
            - The merged DataFrame for that component.
            - Accounts for column collisions by adding suffixes and disjoining index columns.

    Examples:
        >>> df1 = pd.DataFrame({"id": [1, 2], "a": [10, 20]})
        >>> df2 = pd.DataFrame({"id": [1, 2], "b": [0.1, 0.2]})
        >>> df3 = pd.DataFrame({"site": ["A", "B"], "c": [5, 6]})
        >>> comps = merge_multiple_dataframes(
        ...     [df1, df2, df3],
        ...     dfs_name=["X", "Y", "Z"],
        ...     index_cols=["id", "site"],
        ... )
        >>> [keys for keys, _ in comps]
        [('id',), ('site',)]
        >>> # The first component merges df1 & df2 on 'id'; non-key collisions would get
        >>> # suffixes '_X' and '_Y'. The second component is just df3 keyed by 'site'.
    """
    if dfs_name is not None and len(dfs_name) != len(dfs):
        raise ValueError(
            f"Length of `dfs_name` must match length of `dfs`. Found {len(dfs_name)} and {len(dfs)} respectively."
        )

    if not dfs:
        return []

    logger = logging.getLogger(f"{__name__}.merge_multiple_dataframes")

    # Concatenate-only mode
    if not index_cols:
        logger.info("No index_cols provided; concatenating all DataFrames column-wise.")
        return [((), pd.concat(list(dfs), axis="columns", join=join))]

    # Prepare suffix labels
    labels = [f"_{name}" for name in (dfs_name or [f"df{i}" for i in range(len(dfs))])]

    # Bucket frames by the exact subset of keys they actually contain
    logger.info("Merging DataFrames by overlapping keys: %s", index_cols)
    df_by_subset = {}  # type: ignore[var-annotated]
    for df, label in zip(dfs, labels):
        subset = tuple(col for col in index_cols if col in df.columns)
        if subset:
            df_by_subset.setdefault(subset, []).append((df, label))

    if not df_by_subset:
        return []

    # Merge within each exact key-subset first
    logger.info("Found %d groups by exact key subsets.", len(df_by_subset))
    groups = []  # (keys, df, last_suffix)
    for subset, items in df_by_subset.items():
        (merged_df, left_suffix), *rest = items
        for df, right_suffix in rest:
            merged_df = merged_df.merge(
                df,
                on=list(subset),
                how=join,
                suffixes=(left_suffix, right_suffix),
            )
            left_suffix = right_suffix
        groups.append((frozenset(subset), merged_df, left_suffix))

    # Greedy pairwise merging across groups until no overlaps remain
    logger.info("Merging %d groups by overlapping keys.", len(groups))
    while True:
        best = None
        best_score = None  # (-overlap_size, combined_cells)
        n = len(groups)

        for i in range(n):
            keys_i, df_i, sfx_i = groups[i]
            for j in range(i + 1, n):
                keys_j, df_j, sfx_j = groups[j]
                overlap = keys_i & keys_j
                if not overlap:
                    continue
                score = (-len(overlap), df_i.size + df_j.size)
                if best_score is None or score < best_score:
                    best_score = score
                    best = (i, j, sorted(overlap))

        if best is None:
            break

        i, j, on_cols = best
        keys_i, df_i, sfx_i = groups[i]
        keys_j, df_j, sfx_j = groups[j]
        merged_df = df_i.merge(df_j, on=on_cols, how=join, suffixes=(sfx_i, sfx_j))
        groups[i] = (keys_i | keys_j, merged_df, sfx_j)
        del groups[j]

    # Materialize as (sorted_keys_tuple, DataFrame), sorted by keys for consistency
    merged_key_df_pairs = [
        (tuple(sorted(keys)), df) for (keys, df, _) in sorted(groups, key=lambda g: tuple(sorted(g[0])))
    ]

    logging.info("Final merged components: %d", len(merged_key_df_pairs))
    for keys, df in merged_key_df_pairs:
        logging.info("Merged keys: %s, shape: %s", keys, df.shape)

    return merged_key_df_pairs


if __name__ == "__main__":
    import argparse

    # Example script (assuming folder mimic-iv/mimic-iv-3.1 is in the current directory)
    # python -m mmai25_hackathon.load_data.tabular --data-path mimic-iv/mimic-iv-3.1 --index-cols subject_id hadm_id --subset-cols language --join outer
    # NOTE: Expect increase in row count given we are doing outer join and each dataframes may or will have
    #       different relational structures (i.e., admissions to icustays in MIMIC-IV has one-to-many
    #       relationship w.r.t. subject_id and hadm_id)

    parser = argparse.ArgumentParser(description="Read and aggregate tabular CSV files.")
    parser.add_argument(
        "--data-path",
        help="Data path for the CSV files.",
        default="MMAI25Hackathon/mimic-iv/mimic-iv-3.1",
    )
    parser.add_argument(
        "--index-cols",
        nargs="+",
        default=["subject_id", "hadm_id"],
        help="Columns to use as index.",
    )
    parser.add_argument("--subset-cols", nargs="+", default=["language"], help="Columns to subset.")
    parser.add_argument("--join", default="outer", help="Join type for merging DataFrames.")
    args = parser.parse_args()

    # Recursive glob for CSV files
    csv_files = list(Path(args.data_path).rglob("*.csv"))

    # Load multiple dataframes
    dfs = [
        read_tabular(
            f,
            index_cols=args.index_cols,
            subset_cols=args.subset_cols,
            raise_errors=False,
        )
        for f in csv_files
    ]
    dfs_name = [f.stem for f in csv_files]

    # Merge dataframes
    components = merge_multiple_dataframes(dfs, dfs_name=dfs_name, index_cols=args.index_cols, join=args.join)

    for keys, comp_df in components:
        print(f"Component keys: {keys}")
        print(comp_df.head())
        print()
