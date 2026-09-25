"""
Output Generation Module for Business Entity Resolution.

Responsibilities:
- Formatting and validating challenge submission files
- Exporting the required output files in TSV format:
    1. matching_results.tsv: Final predicted entity matches
    2. candidate_pairs.tsv: Generated candidate pairs from blocking
- Ensuring compliance with the official challenge submission schema and rules

NOTE: Do not create fake output data or mock files now. Output generation
will only be triggered when the real pipeline executes on official challenge data.
"""

from pathlib import Path
from typing import Any, Dict, Optional, Union
import pandas as pd


def save_candidate_pairs(
    candidate_pairs_df: pd.DataFrame,
    output_path: Union[str, Path],
    sep: str = "\t",
    index: bool = False,
    **kwargs: Any,
) -> Path:
    """
    Export candidate pairs to the designated TSV file (candidate_pairs.tsv).

    Parameters:
        candidate_pairs_df: DataFrame of candidate pairs.
        output_path: Target file path for candidate_pairs.tsv.
        sep: Separator character, defaults to tab ('\\t').
        index: Whether to write row index (defaults to False).
        **kwargs: Additional parameters passed to to_csv.

    Returns:
        Path to the saved file.
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

    Parameters:
        matching_results_df: DataFrame of predicted matches.
        output_path: Target file path for matching_results.tsv.
        sep: Separator character, defaults to tab ('\\t').
        index: Whether to write row index (defaults to False).
        **kwargs: Additional parameters passed to to_csv.

    Returns:
        Path to the saved file.
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    matching_results_df.to_csv(path, sep=sep, index=index, **kwargs)
    return path


def export_challenge_outputs(
    matching_results_df: pd.DataFrame,
    candidate_pairs_df: pd.DataFrame,
    output_dir: Union[str, Path],
) -> Dict[str, Path]:
    """
    Validate and export both required challenge outputs to the output directory:
    - matching_results.tsv
    - candidate_pairs.tsv

    Parameters:
        matching_results_df: DataFrame containing final predicted matching pairs.
        candidate_pairs_df: DataFrame containing candidate pairs from blocking.
        output_dir: Output directory path.

    Returns:
        Dictionary mapping output file names to their saved paths.
    """
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    match_path = out_dir / "matching_results.tsv"
    cand_path = out_dir / "candidate_pairs.tsv"

    saved_matches = save_matching_results(matching_results_df, match_path)
    saved_candidates = save_candidate_pairs(candidate_pairs_df, cand_path)

    return {
        "matching_results": saved_matches,
        "candidate_pairs": saved_candidates,
    }
