"""
Protein sequence loading and integer-encoding utilities.

Functions:
load_protein_sequences_from_dataframe(df, prot_seq_col, index_col=None)
    Loads protein sequences from a DataFrame or CSV (uses `read_tabular` if a path is given). Optionally sets an index
    and returns a one-column DataFrame named "protein_sequence" (index reset if `index_col` is None).

protein_sequence_to_integer_encoding(sequence, max_length=1200)
    Converts an amino-acid sequence to a fixed-length integer array using an A–Z alphabet (excluding 'J'); 0 is reserved
    for padding/unknown. Returns a NumPy array of shape `(max_length,)`.

Preview CLI:
`python -m mmai25_hackathon.load_data.protein --data-path /path/to/proteins.csv`
Reads the CSV, prints a small preview of the protein sequences, and encodes the first few entries, printing each array’s
shape and the count of unknown (0) tokens.
"""

import logging
from numbers import Integral
from typing import Dict, Optional, Sequence, Union

import numpy as np
import pandas as pd
from sklearn.utils._param_validation import Interval, validate_params

from .tabular import read_tabular

__all__ = ["load_protein_sequences_from_dataframe", "protein_sequence_to_integer_encoding"]

# Generate character set for protein sequences between A-Z (except J)
CHARPROTSET = [chr(i) for i in range(ord("A"), ord("Z") + 1) if chr(i) != "J"]
# Zero is used for padding or unknown characters hence increment from one.
CHARPROTSET = {letter: idx for idx, letter in enumerate(CHARPROTSET, 1)}


@validate_params(
    {"df": [pd.DataFrame, str], "prot_seq_col": [str], "index_col": [None, str], "filter_rows": [None, dict]},
    prefer_skip_nested_validation=True,
)
def load_protein_sequences_from_dataframe(
    df: Union[pd.DataFrame, str],
    prot_seq_col: str,
    index_col: str = None,
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
) -> pd.DataFrame:
    """
    Loads protein sequences from a DataFrame or CSV file. Will read the CSV if a path is provided.

    High-level steps:
    - If `df` is a path, load via `read_tabular` selecting `prot_seq_col` and optional `index_col`; apply `filter_rows`.
    - If `df` is a DataFrame and `filter_rows` is provided, apply row filters where columns exist.
    - Validate `prot_seq_col` exists; optionally set DataFrame index.
    - Return a one-column DataFrame named `"protein_sequence"` (index preserved if set).

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        prot_seq_col (str): Column name for protein sequences.
        index_col (str, optional): Column to set as index. Default: None.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: A single column DataFrame containing the protein sequences with name `"protein_sequence"`.

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": [1, 2, 3],
        ...         "protein_sequence": ["MKTAYIAKQRQISFVKSH", "GAVLILLLV", "TTPSYVAFTDTER"],
        ...     }
        ... )
        >>> sequences = load_protein_sequences_from_dataframe(df, prot_seq_col="protein_sequence", index_col="id")
        >>> print(sequences)
            protein_sequence
        id
        1  MKTAYIAKQRQISFVKSH
        2         GAVLILLLV
        3      TTPSYVAFTDTER
    """
    if isinstance(df, str):
        df = read_tabular(df, subset_cols=prot_seq_col, index_cols=index_col, filter_rows=filter_rows)
    else:
        for col, valid_vals in (filter_rows or {}).items():
            if col in df.columns:
                df = df[df[col].isin(valid_vals)]

    if prot_seq_col not in df.columns:
        raise ValueError(f"Column '{prot_seq_col}' not found in DataFrame.")

    if index_col is not None:
        df = df.set_index(index_col)

    logger = logging.getLogger(f"{__name__}.load_protein_sequences_from_dataframe")
    logger.info("Fetched %d protein sequences from column '%s'.", len(df), prot_seq_col)
    return df[prot_seq_col].to_frame("protein_sequence").reset_index(drop=index_col is None)


@validate_params(
    {"sequence": [str], "max_length": [Interval(Integral, 1, None, closed="left")]}, prefer_skip_nested_validation=True
)
def protein_sequence_to_integer_encoding(sequence: str, max_length: int = 1200) -> np.ndarray:
    """
    Converts a protein sequence into an integer-encoded representation.

    High-level steps:
    - Allocate a zero-initialised array of length `max_length` (dtype `uint64`).
    - For each character up to `max_length`, map A–Z (excluding 'J') using a lookup (unknown→0).
    - Return the encoded array.

    Args:
        sequence (str): The protein sequence to encode.
        max_length (int): The maximum length of the output array.

    Returns:
        np.ndarray: An array of shape (max_length,) containing the integer-encoded representation.

    Examples:
        >>> seq = "MKTAYIAKQRQISFVKSH"
        >>> encoded = protein_sequence_to_integer_encoding(seq, max_length=5)
        >>> print(encoded)
        [13 11 20  1 25]
        >>> encoded = protein_sequence_to_integer_encoding(seq, max_length=25)
        >>> print(encoded)
        [13 11 20  1 25  9  1 11 17 18 17  9 19  6 22 11 19  8  0  0  0  0  0  0  0]
    """
    # Initialize an array of zeros
    encoded_sequence = np.zeros(max_length, dtype=np.uint64)
    for i, char in enumerate(sequence[:max_length]):
        # If character is not in CHARPROTSET, it will be skipped and assumed to be unknown
        encoded_sequence[i] = CHARPROTSET.get(char, 0)

    logger = logging.getLogger(f"{__name__}.protein_sequence_to_integer_encoding")
    logger.info("Encoded sequence: %s", encoded_sequence)
    logger.info("Original sequence length: %d, Encoded length: %d", len(sequence), len(encoded_sequence))
    logger.info("Unknown characters (encoded as 0) count: %d", np.sum(encoded_sequence == 0))
    return encoded_sequence


if __name__ == "__main__":
    import argparse

    # Example script: python -m mmai25_hackathon.load_data.protein --data-path MMAI25Hackathon/molecule-protein-interaction/dataset.csv
    parser = argparse.ArgumentParser(description="Process protein sequences.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV file containing protein sequences.",
        default="MMAI25Hackathon/molecule-protein-interaction/dataset.csv",
    )
    args = parser.parse_args()

    # Take from Peizhen's csv file for DrugBAN training
    df = load_protein_sequences_from_dataframe(args.data_path, prot_seq_col="Protein")
    for i, prot_seq in enumerate(df["protein_sequence"].head(5), 1):
        integer_encoding = protein_sequence_to_integer_encoding(prot_seq)
        print(i, integer_encoding)
