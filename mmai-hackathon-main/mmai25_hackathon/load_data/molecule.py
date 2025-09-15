"""
Molecular (SMILES) loading and graph conversion utilities.

Functions:
load_smiles_from_dataframe(df, smiles_col, index_col=None)
    Fetches SMILES strings from a DataFrame or CSV. Uses `read_tabular` when a path is provided. Optionally sets an index,
    and returns a one-column DataFrame named `"smiles"` (index reset if `index_col` is None).

smiles_to_graph(smiles, with_hydrogen=False, kekulize=False)
    Converts a SMILES string into a PyTorch Geometric `Data` object via `torch_geometric.utils.smiles.from_smiles`.
    Returns a graph `Data` with typical keys: `x` (node features), `edge_index` (COO connectivity), and `edge_attr`.
    Flags `with_hydrogen` and `kekulize` are forwarded to the underlying conversion.

Preview CLI:
`python -m mmai25_hackathon.load_data.molecule --data-path /path/to/dataset.csv`
Reads the CSV, prints a small preview of the SMILES column, and converts the first few entries to graphs, printing each
graph’s summary (e.g., number of nodes/edges and feature sizes).
"""

import logging
from typing import Dict, Optional, Sequence, Union

import pandas as pd
from sklearn.utils._param_validation import validate_params
from torch_geometric.data import Data
from torch_geometric.utils.smiles import from_smiles

from .tabular import read_tabular

__all__ = ["load_smiles_from_dataframe", "smiles_to_graph"]


@validate_params(
    {"df": [pd.DataFrame, str], "smiles_col": [str], "index_col": [None, str], "filter_rows": [None, dict]},
    prefer_skip_nested_validation=True,
)
def load_smiles_from_dataframe(
    df: Union[pd.DataFrame, str],
    smiles_col: str,
    index_col: str = None,
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
) -> pd.DataFrame:
    """
    Fetches SMILES strings from a DataFrame or CSV file. Will read the CSV if a path is provided.

    High-level steps:
    - If `df` is a path, load via `read_tabular` selecting `smiles_col` and optional `index_col`; apply `filter_rows`.
    - If `df` is a DataFrame and `filter_rows` is provided, apply row filters where columns exist.
    - Validate `smiles_col` exists; optionally set DataFrame index.
    - Return a one-column DataFrame named `"smiles"` (index preserved if set).

    Args:
        df (Union[pd.DataFrame, str]): DataFrame or path to CSV file.
        smiles_col (str): Column name for SMILES representations.
        index_col (str, optional): Column to set as index. Default: None.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: A single column DataFrame containing the SMILES strings with name `"smiles"`.

    Examples:
        >>> df = pd.DataFrame(
        ...     {
        ...         "id": [1, 2, 3],
        ...         "smiles": ["CCO", "C1=CC=CC=C1", "CC(=O)O"],
        ...     }
        ... )
        >>> smiles = load_smiles_from_dataframe(df, smiles_col="smiles", index_col="id")
        >>> print(smiles)
            smiles
        id
        1      CCO
        2  C1=CC=CC=C1
        3   CC(=O)O
    """
    if isinstance(df, str):
        df = read_tabular(df, subset_cols=smiles_col, index_cols=index_col, filter_rows=filter_rows)
    else:
        for col, valid_vals in (filter_rows or {}).items():
            if col in df.columns:
                df = df[df[col].isin(valid_vals)]

    if smiles_col not in df.columns:
        raise ValueError(f"Column '{smiles_col}' not found in DataFrame.")

    logger = logging.getLogger(f"{__name__}.load_smiles_from_dataframe")

    if index_col is not None:
        df = df.set_index(index_col)
        logger.info("Setting index column to '%s'.", index_col)

    logger.info("Fetched %d SMILES strings from column '%s'.", len(df), smiles_col)
    return df[smiles_col].to_frame("smiles").reset_index(drop=index_col is None)


@validate_params({"smiles": [str], "with_hydrogen": [bool], "kekulize": [bool]}, prefer_skip_nested_validation=True)
def smiles_to_graph(smiles: str, with_hydrogen: bool = False, kekulize: bool = False) -> Data:
    """
    Converts a SMILES string to a molecular graph representation.

    High-level steps:
    - Forward to `torch_geometric.utils.smiles.from_smiles` with `with_hydrogen` and `kekulize` flags.
    - Return the resulting `torch_geometric.data.Data` graph.

    Args:
        smiles (str): The SMILES string to convert.
        with_hydrogen (bool): Store hydrogens in the graph if True. Default: False
        kekulize (bool): Converts aromatic bonds to single/double bonds if True. Default: False

    Returns:
        torch_geometric.data.Data: The molecular graph representation containing:
            - `x`: Node feature matrix with shape [num_nodes, num_node_features].
            - `edge_index`: Graph connectivity in COO format with shape [2, num_edges].
            - `edge_attr`: Edge feature matrix with shape [num_edges, num_edge_features].
            - `num_nodes`: Number of nodes in the graph.
            - `num_edges`: Number of edges in the graph.
            - `smiles`: The original SMILES string as a graph attribute.

    Examples:
        >>> smiles = "CCO"
        >>> graph = smiles_to_graph(smiles)
        >>> print(graph)
        Data(x=[3, 9], edge_index=[2, 4], edge_attr=[4, 3], smiles='CCO')
    """
    logger = logging.getLogger(f"{__name__}.smiles_to_graph")
    logger.info("Converting SMILES to graph: %s", smiles)
    return from_smiles(smiles, with_hydrogen, kekulize)


if __name__ == "__main__":
    import argparse

    # Example script: python -m mmai25_hackathon.load_data.molecule --data-path MMAI25Hackathon/molecule-protein-interaction/dataset.csv
    parser = argparse.ArgumentParser(description="Process SMILES strings.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the CSV file containing SMILES strings.",
        default="MMAI25Hackathon/molecule-protein-interaction/dataset.csv",
    )
    args = parser.parse_args()

    # Take from Peizhen's csv file for DrugBAN training
    df = load_smiles_from_dataframe(args.data_path, smiles_col="SMILES")
    for i, smiles in enumerate(df["smiles"].head(5), 1):
        graph = smiles_to_graph(smiles)
        print(i, graph)
