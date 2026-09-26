"""
Feature Engineering Module for Business Entity Resolution.

Responsibilities:
- Extract pairwise comparison features for candidate entity pairs
- Produce numerical features for ML-based entity matching
- Integrate shared Unicode-safe preprocessing logic from Member 1
"""

import re
import unicodedata
from typing import Any
from difflib import SequenceMatcher

import pandas as pd

# Import Member 1's shared Unicode-safe preprocessing logic
try:
    from .preprocessing import normalize_country, normalize_text
except ImportError:
    try:
        from src.preprocessing import normalize_country, normalize_text
    except ImportError:
        try:
            from preprocessing import normalize_country, normalize_text
        except ImportError:
            def normalize_text(value: Any) -> str:
                if pd.isna(value):
                    return ""
                text = unicodedata.normalize("NFKC", str(value)).casefold()
                text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
                return re.sub(r"\s+", " ", text).strip()

            def normalize_country(value: Any) -> str:
                if pd.isna(value):
                    return ""
                text = unicodedata.normalize("NFKC", str(value)).casefold()
                return re.sub(r"\s+", " ", text).strip()


def text_normalize(value: Any) -> str:
    """
    Safely convert any value to normalized text using shared preprocessing logic.
    Applies NFKC normalization, Unicode-aware lowercasing (casefold),
    and removes punctuation while preserving multilingual alphanumeric tokens.
    """
    return normalize_text(value)


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

    name_a_norm = [text_normalize(x) for x in df["name_a"]]
    name_b_norm = [text_normalize(x) for x in df["name_b"]]

    df["name_exact"] = [
        int(x != "" and x == y)
        for x, y in zip(name_a_norm, name_b_norm)
    ]

    df["name_jaccard"] = [
        token_jaccard(x, y)
        for x, y in zip(name_a_norm, name_b_norm)
    ]

    df["name_edit_similarity"] = [
        edit_similarity(x, y)
        for x, y in zip(name_a_norm, name_b_norm)
    ]

    df["name_length_diff"] = [
        length_difference(x, y)
        for x, y in zip(name_a_norm, name_b_norm)
    ]

    # ------------------------------------------------------------------
    # Address features
    # ------------------------------------------------------------------

    address_a_norm = [text_normalize(x) for x in df["address_a"]]
    address_b_norm = [text_normalize(x) for x in df["address_b"]]

    df["address_exact"] = [
        int(x != "" and x == y)
        for x, y in zip(address_a_norm, address_b_norm)
    ]

    df["address_jaccard"] = [
        token_jaccard(x, y)
        for x, y in zip(address_a_norm, address_b_norm)
    ]

    df["address_edit_similarity"] = [
        edit_similarity(x, y)
        for x, y in zip(address_a_norm, address_b_norm)
    ]

    df["address_length_diff"] = [
        length_difference(x, y)
        for x, y in zip(address_a_norm, address_b_norm)
    ]

    # ------------------------------------------------------------------
    # Country feature
    # ------------------------------------------------------------------

    country_a = [normalize_country(x) for x in df["country_a"]]
    country_b = [normalize_country(x) for x in df["country_b"]]

    df["country_match"] = [
        int(a != "" and b != "" and a == b)
        for a, b in zip(country_a, country_b)
    ]

    # ------------------------------------------------------------------
    # Token-count features
    # ------------------------------------------------------------------

    df["name_token_count_diff"] = [
        abs(len(x.split()) - len(y.split()))
        for x, y in zip(name_a_norm, name_b_norm)
    ]

    df["address_token_count_diff"] = [
        abs(len(x.split()) - len(y.split()))
        for x, y in zip(address_a_norm, address_b_norm)
    ]

    return df
