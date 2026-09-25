"""
Preprocessing Module for Business Entity Resolution.

Responsibilities:
- Loading challenge TSV files
- Validating loaded data structures
- Text normalization
- Business name normalization
- Address normalization
- Providing reusable preprocessing functions across the pipeline

NOTE: Dataset column names and specific schema assumptions are deferred
until the official challenge files are provided and inspected.
"""

from pathlib import Path
from typing import Any, Dict, List, Optional, Union
import re
import pandas as pd


def load_tsv_file(
    file_path: Union[str, Path],
    sep: str = "\t",
    encoding: str = "utf-8",
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Load a challenge-provided TSV file into a pandas DataFrame.

    Parameters:
        file_path: Path to the TSV file.
        sep: Separator character, defaults to tab ('\\t').
        encoding: File encoding, defaults to 'utf-8'.
        **kwargs: Additional parameters passed to pd.read_csv.

    Returns:
        pd.DataFrame containing the loaded data.
    """
    path = Path(file_path)
    if not path.is_file():
        raise FileNotFoundError(f"Challenge data file not found: {path}")

    df = pd.read_csv(path, sep=sep, encoding=encoding, **kwargs)
    return df


def validate_data(
    df: pd.DataFrame,
    required_columns: Optional[List[str]] = None,
) -> bool:
    """
    Validate basic integrity of the loaded DataFrame.

    Parameters:
        df: Loaded pandas DataFrame.
        required_columns: Optional list of column names that must exist.

    Returns:
        True if validation passes, raises ValueError otherwise.
    """
    if df.empty:
        raise ValueError("Loaded DataFrame is empty.")

    if required_columns is not None:
        missing = [col for col in required_columns if col not in df.columns]
        if missing:
            raise ValueError(f"Missing required columns in dataset: {missing}")

    return True


def normalize_text(text: Optional[str]) -> str:
    """
    Generic text normalization: lowercasing, whitespace stripping,
    and removing redundant whitespace.

    Parameters:
        text: Input string or None.

    Returns:
        Normalized text string.
    """
    if text is None or pd.isna(text):
        return ""
    text = str(text).lower().strip()
    text = re.sub(r"\s+", " ", text)
    return text


def normalize_business_name(name: Optional[str]) -> str:
    """
    Normalize business legal/trade name for matching.
    Applies lowercasing, stripping, punctuation handling, and basic cleaning.

    Parameters:
        name: Business name string.

    Returns:
        Normalized business name string.
    """
    if name is None or pd.isna(name):
        return ""
    cleaned = normalize_text(name)
    # Further business-name specific transformations (e.g., standardizing suffixes)
    # will be configured once the official dataset formats are inspected.
    return cleaned


def normalize_address(address: Optional[str]) -> str:
    """
    Normalize address fields for matching.
    Applies lowercasing, whitespace cleaning, and character normalization.

    Parameters:
        address: Address string.

    Returns:
        Normalized address string.
    """
    if address is None or pd.isna(address):
        return ""
    cleaned = normalize_text(address)
    # Address normalization details to be refined after inspecting challenge data.
    return cleaned
