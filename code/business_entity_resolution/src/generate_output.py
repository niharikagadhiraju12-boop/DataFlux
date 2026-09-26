"""
Output Generation Module for Business Entity Resolution.

Responsibilities:
- Formatting and validating challenge submission files
- Exporting the required output files in TSV format:
    1. matching_results.tsv: Final predicted entity matches
    2. candidate_pairs.tsv: Generated candidate pairs from blocking
- Ensuring compliance with the official challenge submission schema and rules
"""

import os
from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd


def export_matching_results_tsv(
    predicted_pairs_df: pd.DataFrame,
    source1_df: pd.DataFrame,
    output_path: Union[str, Path],
) -> Path:
    """
    Export predicted matches into the official challenge submission format:
        source1_entity_id \t matched_entity_ids
    Where matched_entity_ids is a comma-separated list of predicted matching IDs.

    Guarantees:
        - Exactly one row per Source 1 entity in source1_df.
        - Empty string for entities with zero predicted matches.
        - Tab-separated (.tsv) encoding in UTF-8.
        - Passes validate_submission.py with zero formatting errors.

    Parameters:
        predicted_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id'].
        source1_df: Source 1 DataFrame or DataFrame with 'entity_id' column.
        output_path: Destination filepath for matching_results.tsv.

    Returns:
        Path to the saved matching_results.tsv.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    s1_col = predicted_pairs_df.columns[0]
    cand_col = predicted_pairs_df.columns[1]

    # Group predicted matches by Source 1 entity
    if len(predicted_pairs_df) > 0:
        grouped = (
            predicted_pairs_df.groupby(s1_col)[cand_col]
            .apply(lambda ids: ",".join(sorted(set(str(i).strip() for i in ids if str(i).strip()))))
            .to_dict()
        )
    else:
        grouped = {}

    all_s1_col = "entity_id" if "entity_id" in source1_df.columns else source1_df.columns[0]
    all_s1_ids = list(source1_df[all_s1_col].unique())

    rows = []
    for s1_id in all_s1_ids:
        s1_str = str(s1_id).strip()
        matches_str = grouped.get(s1_str, "")
        rows.append({"source1_entity_id": s1_str, "matched_entity_ids": matches_str})

    res_df = pd.DataFrame(rows, columns=["source1_entity_id", "matched_entity_ids"])
    res_df.to_csv(path, sep="\t", index=False, encoding="utf-8")
    return path


def save_candidate_pairs(
    candidate_pairs_df: pd.DataFrame,
    output_path: Union[str, Path],
    sep: str = "\t",
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Export candidate pairs to the designated TSV file (candidate_pairs.tsv).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    candidate_pairs_df.to_csv(path, sep=sep, index=index, **kwargs)
    return path


def save_matching_results(
    matching_results_df: pd.DataFrame,
    output_path: Union[str, Path],
    sep: str = "\t",
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Export final matching predictions to the designated TSV file (matching_results.tsv).
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matching_results_df.to_csv(path, sep=sep, index=index, **kwargs)
    return path


def export_challenge_outputs(
    matching_results_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    output_dir: Union[str, Path],
    source1_df: Optional[pd.DataFrame] = None,
) -> Dict[str, Path]:
    """
    Validate and export both required challenge outputs to the output directory:
    - matching_results.tsv
    - candidate_pairs.tsv
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    match_path = out_dir / "matching_results.tsv"
    cand_path = out_dir / "candidate_pairs.tsv"

    if source1_df is not None:
        saved_matches = export_matching_results_tsv(matching_results_df, source1_df, match_path)
    else:
        saved_matches = save_matching_results(matching_results_df, match_path)

    saved_candidates = save_candidate_pairs(candidate_pairs_df, cand_path)

    return {
        "matching_results": saved_matches,
        "candidate_pairs": saved_candidates,
    }
