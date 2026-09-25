"""
Feature Engineering Module for Business Entity Resolution.

Responsibilities:
- Extract pairwise comparison features for candidate entity pairs
- Produce numerical features for ML-based entity matching
"""

from typing import Any

import pandas as pd
from difflib import SequenceMatcher


def text_normalize(value: Any) -> str:
    """Convert a value to a safe normalized string."""
    if pd.isna(value):
        return ""
    return str(value).strip().lower()


def token_jaccard(a: Any, b: Any) -> float:
    """Calculate Jaccard similarity between whitespace-separated tokens."""
    a_tokens = set(text_normalize(a).split())
    b_tokens = set(text_normalize(b).split())

    if not a_tokens and not b_tokens:
        return 1.0

    if not a_tokens or not b_tokens:
        return 0.0

    return len(a_tokens & b_tokens) / len(a_tokens | b_tokens)


def edit_similarity(a: Any, b: Any) -> float:
    """Calculate normalized character-level similarity."""
    a = text_normalize(a)
    b = text_normalize(b)

    if not a and not b:
        return 1.0

    if not a or not b:
        return 0.0

    return SequenceMatcher(None, a, b).ratio()


def length_difference(a: Any, b: Any) -> int:
    """Calculate absolute difference in character lengths."""
    a = text_normalize(a)
    b = text_normalize(b)

    return abs(len(a) - len(b))


def extract_pair_features(
    candidate_pairs: pd.DataFrame,
    source_a: pd.DataFrame,
    source_b: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Extract numerical pairwise comparison features.

    Parameters
    ----------
    candidate_pairs:
        DataFrame containing:
        - source1_entity_id
        - candidate_entity_id

    source_a:
        Source 1 entity records.

    source_b:
        Target entity records from Source 2 or Source 3.

    Returns
    -------
    pd.DataFrame
        Candidate pairs with entity attributes and numerical
        similarity features.
    """

    # Select only the columns required for feature extraction.
    a = source_a[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    b = source_b[
        [
            "entity_id",
            "business_name",
            "business_address",
            "country",
        ]
    ].copy()

    # Rename columns to distinguish Source 1 and target attributes.
    a = a.rename(
        columns={
            "entity_id": "source1_entity_id",
            "business_name": "name_a",
            "business_address": "address_a",
            "country": "country_a",
        }
    )

    b = b.rename(
        columns={
            "entity_id": "candidate_entity_id",
            "business_name": "name_b",
            "business_address": "address_b",
            "country": "country_b",
        }
    )

    # Attach entity attributes to candidate pairs.
    df = candidate_pairs.merge(
        a,
        on="source1_entity_id",
        how="left",
    ).merge(
        b,
        on="candidate_entity_id",
        how="left",
    )

    # ------------------------------------------------------------------
    # Business-name features
    # ------------------------------------------------------------------

    name_a_norm = df["name_a"].fillna("").astype(str).str.strip().str.lower()
    name_b_norm = df["name_b"].fillna("").astype(str).str.strip().str.lower()

    df["name_exact"] = (
        name_a_norm == name_b_norm
    ).astype(int)

    df["name_jaccard"] = [
        token_jaccard(a, b)
        for a, b in zip(df["name_a"], df["name_b"])
    ]

    df["name_edit_similarity"] = [
        edit_similarity(a, b)
        for a, b in zip(df["name_a"], df["name_b"])
    ]

    df["name_length_diff"] = [
        length_difference(a, b)
        for a, b in zip(df["name_a"], df["name_b"])
    ]

    # ------------------------------------------------------------------
    # Address features
    # ------------------------------------------------------------------

    address_a_norm = (
        df["address_a"].fillna("").astype(str).str.strip().str.lower()
    )

    address_b_norm = (
        df["address_b"].fillna("").astype(str).str.strip().str.lower()
    )

    df["address_exact"] = (
        address_a_norm == address_b_norm
    ).astype(int)

    df["address_jaccard"] = [
        token_jaccard(a, b)
        for a, b in zip(df["address_a"], df["address_b"])
    ]

    df["address_edit_similarity"] = [
        edit_similarity(a, b)
        for a, b in zip(df["address_a"], df["address_b"])
    ]

    df["address_length_diff"] = [
        length_difference(a, b)
        for a, b in zip(df["address_a"], df["address_b"])
    ]

    # ------------------------------------------------------------------
    # Country feature
    # ------------------------------------------------------------------

    country_a = (
        df["country_a"].fillna("").astype(str).str.strip().str.lower()
    )

    country_b = (
        df["country_b"].fillna("").astype(str).str.strip().str.lower()
    )

    df["country_match"] = (
        (country_a != "")
        & (country_b != "")
        & (country_a == country_b)
    ).astype(int)

    # ------------------------------------------------------------------
    # Token-count features
    # ------------------------------------------------------------------

    df["name_token_count_diff"] = [
        abs(
            len(text_normalize(a).split())
            - len(text_normalize(b).split())
        )
        for a, b in zip(df["name_a"], df["name_b"])
    ]

    df["address_token_count_diff"] = [
        abs(
            len(text_normalize(a).split())
            - len(text_normalize(b).split())
        )
        for a, b in zip(df["address_a"], df["address_b"])
    ]

    return df