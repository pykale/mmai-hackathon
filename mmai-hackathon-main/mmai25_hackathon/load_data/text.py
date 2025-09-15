"""
MIMIC-IV clinical notes (free-text) loading utilities.

Functions:
load_mimic_iv_notes(note_path, subset='radiology', include_detail=False, subset_cols=None)
    Loads the selected notes CSV (`radiology.csv` or `discharge.csv`), verifies required ID columns
    (`note_id`, `subject_id`), optionally merges `<subset>_detail.csv` when `include_detail=True`,
    applies optional `subset_cols`, strips/filters empty `text`, and returns a `pd.DataFrame` indexed by
    [`note_id`, `subject_id`].

extract_text_from_note(note, include_metadata=False)
    Extracts the `text` field from a single note `pd.Series`. When `include_metadata=True`, returns
    `(text, metadata_dict)` where `metadata_dict` is the note’s fields excluding `text`.

Preview CLI:
`python -m mmai25_hackathon.load_data.text --data-path /path/to/mimic-iv-note-.../note --subset radiology --note-id 12345678`
Prints a preview of the loaded notes (columns like `note_id`, `subject_id`, `hadm_id`, `note_type`, `text`) and then
retrieves the note matching the provided `note_id`, printing its full text and selected metadata (e.g., `subject_id`,
`hadm_id`, `note_type`).
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Literal, Optional, Sequence, Tuple, Union

import pandas as pd
from sklearn.utils._param_validation import StrOptions, validate_params

from .tabular import merge_multiple_dataframes, read_tabular

__all__ = ["load_mimic_iv_notes", "extract_text_from_note"]

REQUIRED_ID_COLS = ["note_id", "subject_id"]


@validate_params(
    {
        "note_path": [str, Path],
        "subset": [StrOptions({"radiology", "discharge"})],
        "include_detail": ["boolean"],
        "subset_cols": [None, list],
        "filter_rows": [None, dict],
    },
    prefer_skip_nested_validation=True,
)
def load_mimic_iv_notes(
    note_path: Union[str, Path],
    subset: Literal["radiology", "discharge"] = "radiology",
    include_detail: bool = False,
    subset_cols: Optional[List[str]] = ["hadm_id", "note_type", "note_seq", "charttime", "storetime", "text"],
    filter_rows: Optional[Dict[str, Union[Sequence, pd.Index]]] = None,
) -> pd.DataFrame:
    """
    Load de-identified free-text clinical notes for a selected MIMIC-IV subset and
    optionally merge the corresponding detail CSV.

    High-level steps:
    - Validate the input directory exists; resolve ``note_path`` to ``Path``.
    - Load ``<subset>.csv`` via ``read_tabular`` with ``subset_cols`` plus required IDs; apply ``filter_rows`` when provided.
    - Ensure required ID columns (``note_id``, ``subject_id``) are present.
    - When ``include_detail`` is True, load ``<subset>_detail.csv`` (also applying ``filter_rows``), validate IDs, and left‑merge on IDs.
    - If ``text`` exists, ``str.strip`` and drop empty rows; otherwise log a warning.
    - Return the resulting ``pd.DataFrame``.

    Args:
        note_path (Union[str, Path]): Directory containing the notes CSV files
            (for example: ``.../mimic-iv-note-.../note``).
        subset (Literal['radiology', 'discharge']): Which note subset to load. Default: ``'radiology'``.
        include_detail (bool): If True, left-join ``<subset>_detail.csv`` on ``['note_id', 'subject_id']``.
            Default: False.
        subset_cols (Optional[List[str]]): Columns to load from the main notes CSV in addition to the
            required ID columns. Defaults to a small set including ``'text'``.
        filter_rows (Optional[Dict[str, Union[Sequence, pd.Index]]]): Row filters as
            {column: allowed_values}. Applied where columns exist. Default: None.

    Returns:
        pd.DataFrame: Notes for the requested subset. When ``text`` exists, values are trimmed and
        empty rows removed; when ``include_detail=True``, columns from the detail CSV may be present.

    Raises:
        FileNotFoundError: If ``note_path`` or the main CSV (``<subset>.csv``) is missing, or if
            ``include_detail=True`` and ``<subset>_detail.csv`` is missing.
        KeyError: If the required ID columns ``['note_id', 'subject_id']`` are absent from the main
            or (when requested) detail CSV.

    Examples:
        >>> from mmai25_hackathon.load_data.text import load_mimic_iv_notes
        >>> base = "MMAI25Hackathon/mimic-iv/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note"
        >>> df = load_mimic_iv_notes(base, subset="radiology", include_detail=True)
        >>> df.head()[["note_id", "subject_id", "note_type", "text"]]
           note_id  subject_id  hadm_id note_type                                               text
        0        1         101        1        DS  EXAMINATION: CHEST (PA AND LAT)INDICATION: ___...
        1        2         101        2        DS  EXAMINATION: LIVER OR GALLBLADDER US (SINGLE O...
        2        3         101        3        DS  INDICATION: ___ HCV cirrhosis c/b ascites, hiv...
        3        4         102        4        DS  EXAMINATION: Ultrasound-guided paracentesis.IN...
        4        5         102        5        DS  EXAMINATION: Ultrasound-guided paracentesis.IN...
    """
    if isinstance(note_path, str):
        note_path = Path(note_path)

    if not note_path.exists():
        raise FileNotFoundError(f"Notes folder not found: {note_path}")

    subset_path = note_path / f"{subset}.csv"

    if not subset_path.exists():
        raise FileNotFoundError(f"Missing main notes CSV: {subset_path}")

    logger = logging.getLogger(f"{__name__}.load_mimic_iv_notes")
    logger.info("Loading notes from: %s", subset_path)
    df_notes = read_tabular(subset_path, subset_cols=subset_cols, index_cols=REQUIRED_ID_COLS, filter_rows=filter_rows)
    logger.info("Loaded %d notes from: %s", len(df_notes), subset_path)

    id_cols_available = df_notes.columns.intersection(REQUIRED_ID_COLS).to_list()
    if len(id_cols_available) < len(REQUIRED_ID_COLS):
        raise KeyError(f"{subset_path} must contain columns: {REQUIRED_ID_COLS}. Found: {id_cols_available}")

    detail_path = note_path / f"{subset}_detail.csv"
    if include_detail and not detail_path.exists():
        raise FileNotFoundError(f"Missing detail notes CSV: {detail_path}")

    if include_detail:
        logger.info("Including detail from: %s", detail_path)
        df_detail = read_tabular(detail_path, index_cols=REQUIRED_ID_COLS, filter_rows=filter_rows)
        logger.info("Loaded %d detail rows from: %s", len(df_detail), detail_path)
        id_cols_available = df_detail.columns.intersection(REQUIRED_ID_COLS).to_list()
        if len(id_cols_available) < len(REQUIRED_ID_COLS):
            raise KeyError(f"{detail_path} must contain columns: {REQUIRED_ID_COLS}. Found: {id_cols_available}")
        logger.info("Merging detail into main notes on: %s", REQUIRED_ID_COLS)
        df_notes = merge_multiple_dataframes((df_notes, df_detail), ("notes", "detail"), REQUIRED_ID_COLS, "left")
        # Unpack df_notes from list of (paired_keys, df_notes)
        df_notes = df_notes[0]
        _, df_notes = df_notes

    text_included = "text" in df_notes.columns
    if text_included:
        df_notes["text"] = df_notes["text"].astype(str).str.strip()
        available_texts = df_notes["text"] != ""
        df_notes = df_notes[available_texts].copy()
        logger.info("After filtering, %d notes have non-empty text.", len(df_notes))
    else:
        logger.warning("The loaded notes do not include a 'text' column.")

    return df_notes


@validate_params({"note": [pd.Series], "include_metadata": ["boolean"]}, prefer_skip_nested_validation=True)
def extract_text_from_note(note: pd.Series, include_metadata: bool = False) -> Union[str, Tuple[str, Dict[str, Any]]]:
    """
    Extracts the text from a note Series, optionally returning metadata.

    High-level steps:
    - Validate that the input Series contains a ``'text'`` field; otherwise, raise ``KeyError``.
    - When ``include_metadata`` is False, return only the note text.
    - When ``include_metadata`` is True, return a tuple of ``(text, metadata_dict)`` where
      ``metadata_dict`` is the note’s fields excluding ``'text'``.

    Args:
        note (pd.Series): A pandas Series representing a note, expected to contain a 'text' column.
        include_metadata (bool): If True, return a tuple of (text, metadata_dict). Default is False.

    Returns:
        Union[str, Tuple[str, Dict[str, Any]]]: The note text; optionally with the note's metadata as a dictionary.

    Raises:
        KeyError: If the 'text' column is not present in the note Series.

    Examples:
        >>> note = pd.Series(
        ...     {"note_id": 1, "subject_id": 101, "text": "Patient is stable.", "note_type": "Discharge summary"}
        ... )
        >>> extract_text_from_note(note)
        'Patient is stable.'
        >>> extract_text_from_note(note, include_metadata=True)
        ('Patient is stable.', {'note_id': 1, 'subject_id': 101, 'note_type': 'Discharge summary'})
    """
    if "text" not in note:
        raise KeyError("The note does not include a 'text' column.")

    logger = logging.getLogger(f"{__name__}.extract_text_from_note")
    logger.info("Extracting text from note with ID: %s", note.get("note_id", "unknown"))

    text = note["text"]
    if not include_metadata:
        return text

    logger.info("Including metadata in the output.")

    metadata = note.drop("text").to_dict()
    return text, metadata


if __name__ == "__main__":
    import argparse

    # Example script:
    # python -m mmai25_hackathon.load_data.text --data-path mimic-iv/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note --subset radiology --note-id 1
    parser = argparse.ArgumentParser(description="Load MIMIC-IV free-text clinical notes.")
    parser.add_argument(
        "--data-path",
        type=str,
        help="Path to the MIMIC-IV notes directory (containing CSV files).",
        default="MMAI25Hackathon/mimic-iv/mimic-iv-note-deidentified-free-text-clinical-notes-2.2/note",
    )
    parser.add_argument(
        "--subset",
        type=str,
        choices=["radiology", "discharge"],
        help="Which note subset to load (radiology or discharge).",
        default="radiology",
    )
    parser.add_argument("--note-id", type=int, help="The note_id of the note to retrieve.", default=1)
    args = parser.parse_args()

    print(f"Loading {args.subset} notes from: {args.data_path}")
    data = load_mimic_iv_notes(args.data_path, subset=args.subset, include_detail=True)
    print(data.head()[["note_id", "subject_id", "hadm_id", "note_type", "text"]])
    print()

    print(f"Retrieving text for note_id={args.note_id}")
    try:
        text, metadata = extract_text_from_note(
            data.loc[data["note_id"] == args.note_id].squeeze(), include_metadata=True
        )
        print("Note text:")
        print(text)
        print("Metadata:")
        print(metadata)
    except KeyError as e:
        print(e)
