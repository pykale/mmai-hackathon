"""
Labels loading and encoding utilities.

Functions:
load_labels_from_dataframe(df, label_col, index_col=None)
    Loads labels from a DataFrame or CSV (uses `read_tabular` when a path is provided). Supports single- or multi-column
    labels for classification/regression. Optionally sets an index. Returns a DataFrame named "label" for a single column
    or the original column names for multiple columns.

one_hot_encode_labels(labels, columns="label")
    One-hot encodes categorical label columns using `pandas.get_dummies`. Supports single or multiple columns and returns
    a `pd.DataFrame` with `float32` dtypes.

Preview CLI:
`python -m mmai25_hackathon.load_data.labels --data-path /path/to/labels.csv`
Reads the CSV (expects a label column named `Y` in this demo), prints the first five labels, then prints a preview of
one-hot–encoded labels.
"""

from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import validate_params

from .tabular import read_tabular

__all__ = ["load_labels_from_dataframe", "one_hot_encode_labels"]


@validate_params(
    {
        "df": [pd.DataFrame, str],
        "label_col": [str, "array-like"],
        "index_col": [None, str],
        "filter_rows": [None, dict],
    },
    prefer_skip_nested_validation=True,
)
def load_labels_from_dataframe(
    df: Union[pd.DataFrame, str],
    label_col: Union[str, Sequence[str]],
    index_col: str = None,
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
) -> pd.DataFrame:
    """
    Loads supervision labels from a DataFrame or CSV file. Will read the CSV if a path is provided.

    High-level steps:
    - If `df` is a path, load via `read_tabular` selecting `label_col` and optional `index_col`; apply `filter_rows`.
    - If `df` is a DataFrame and `filter_rows` is provided, apply row filters where columns exist.
    - Validate that the requested label column(s) are present.
    - If a single column, optionally set index and return DataFrame named `"label"`.
    - If multiple columns, return the DataFrame as-is.

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        label_col (Union[str, Sequence[str]]): Column name or sequence of column names for labels.
        index_col (str, optional): Column to set as index. Default: None.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: A DataFrame containing the labels with name `"label"` if a single column is provided,
                      or the original column names if multiple columns are provided.

    Examples:
        >>> df = pd.DataFrame(
        ...     {"id": [1, 2, 3], "feature1": [0.5, 0.6, 0.7], "feature2": [1.5, 1.6, 1.7], "label": [0, 1, 0]}
        ... )
        >>> labels = load_labels_from_dataframe(df, label_col="label", index_col="id")
        >>> print(labels)
            label
        id
        1      0
        2      1
        3      0
    """
    if isinstance(df, str):
        df = read_tabular(df, subset_cols=label_col, index_cols=index_col, filter_rows=filter_rows)
    else:
        # Apply filter_rows to provided DataFrame for consistency
        for col, valid_vals in (filter_rows or {}).items():
            if col in df.columns:
                df = df[df[col].isin(valid_vals)]

    if label_col not in df.columns:
        raise ValueError(f"Column '{label_col}' not found in DataFrame.")

    if isinstance(label_col, (list, tuple)) and len(label_col) > 1:
        return df

    if index_col is not None:
        df = df.set_index(index_col)

    return df[label_col].to_frame("label").reset_index(drop=index_col is None)


@validate_params({"labels": [pd.DataFrame], "columns": [str, "array-like"]}, prefer_skip_nested_validation=True)
def one_hot_encode_labels(labels: pd.DataFrame, columns: Union[Sequence[str], str] = "label") -> pd.DataFrame:
    """
    One-hot encodes categorical labels in a DataFrame.

    High-level steps:
    - Coerce `columns` to a list of column names.
    - Call `pandas.get_dummies` with `dtype=np.float32` on the specified columns.
    - Return the one-hot encoded DataFrame.

    Args:
        labels (pd.DataFrame): DataFrame containing the labels to be one-hot encoded.
        columns (Union[Sequence[str], str]): Column name or sequence of column names to be one-hot encoded. Default: "label".

    Returns:
        pd.DataFrame: A DataFrame with one-hot encoded labels.

    Examples:
        >>> df = pd.DataFrame({"label": ["cat", "dog", "cat", "mouse"]})
        >>> one_hot_labels = one_hot_encode_labels(df)
        >>> print(one_hot_labels)
           label_cat  label_dog  label_mouse
        0          1          0            0
        1          0          1            0
        2          1          0            0
        3          0          0            1
    """
    if isinstance(columns, str):
        columns = [columns]
    return pd.get_dummies(labels, columns=columns, dtype=np.float32)


if __name__ == "__main__":
    import argparse

    # Example script: python -m mmai25_hackathon.load_data.labels --data-path MMAI25Hackathon/molecule-protein-interaction/dataset.csv
    parser = argparse.ArgumentParser(description="Process supervision labels for regression/classification.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV file containing supervision labels.",
        default="MMAI25Hackathon/molecule-protein-interaction/dataset.csv",
    )
    args = parser.parse_args()

    # Take from Peizhen's csv file for DrugBAN training
    df = load_labels_from_dataframe(args.data_path, label_col="Y")
    for i, label in enumerate(df["label"].head(5), 1):
        print(i, label)

    one_hot_df = one_hot_encode_labels(df, columns="label")
    print(one_hot_df.head(5))
