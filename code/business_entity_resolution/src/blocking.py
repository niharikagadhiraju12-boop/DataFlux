"""
Blocking and Candidate Generation Module for Business Entity Resolution.

Responsibilities:
- Candidate pair generation to reduce the O(N x M) comparison space while achieving high recall
- Multi-pass blocking system:
    1. Country blocking (partition candidates strictly by country label, supporting open sets like US, India, France)
    2. Rare/informative name-token blocking (inverted indexing on informative name tokens, avoiding generic corporate suffixes)
    3. Character n-gram TF-IDF name retrieval (character-level n-gram representations robust to typos, accents, and concatenations)
    4. Address-assisted token blocking (capturing records with non-Latin transliterated or heavily corrupted names)
- Union and deduplication of candidate pairs
- Downstream candidate generation between Source 1 & Source 2 and Source 1 & Source 3
- Candidate recall evaluation and missed-pair diagnostic analysis
- Deterministic, reproducible output conforming to competition requirements
"""

import os
import re
import unicodedata
from collections import defaultdict
from typing import Any, Dict, List, Optional, Set, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer

# Import Member 1's preprocessing functions
try:
    from .preprocessing import (
        normalize_address,
        normalize_country,
        normalize_name,
        normalize_text,
    )
except ImportError:
    try:
        from src.preprocessing import (
            normalize_address,
            normalize_country,
            normalize_name,
            normalize_text,
        )
    except ImportError:
        try:
            from preprocessing import (
                normalize_address,
                normalize_country,
                normalize_name,
                normalize_text,
            )
        except ImportError:
            # Fallback implementation if imported from non-standard path
            def normalize_text(value: Any) -> str:
                if pd.isna(value):
                    return ""
                text = unicodedata.normalize("NFKC", str(value)).casefold()
                text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
                return re.sub(r"\s+", " ", text).strip()

            def normalize_name(value: Any) -> str:
                return normalize_text(value)

            def normalize_address(value: Any) -> str:
                return normalize_text(value)

            def normalize_country(value: Any) -> str:
                if pd.isna(value):
                    return ""
                text = unicodedata.normalize("NFKC", str(value)).casefold()
                return re.sub(r"\s+", " ", text).strip()


# Common corporate suffixes and high-frequency terms to avoid as sole blocking keys.
# These terms appear in hundreds of thousands of business names and would explode candidate sizes.
BUSINESS_STOPWORDS = {
    "inc", "inc.", "llc", "corp", "corporation", "ltd", "limited", "private", "pvt",
    "company", "co", "enterprises", "enterprise", "services", "service", "center",
    "group", "agency", "industries", "industry", "associates", "holdings", "the",
    "and", "of", "in", "for", "at", "to", "llp", "pc", "sarl", "sasu", "gmbh",
    "solutions", "systems", "management", "consulting", "international", "tech",
    "technologies", "technology", "store", "shop", "hotel", "restaurant", "market",
    "marketing", "trade", "trading", "hospital", "foods", "food", "bank", "school",
    "college", "institute", "foundation", "trust", "care", "health", "pharma",
    "pharmaceuticals", "club", "cafe", "bar", "spa", "sons", "brothers", "de", "la", "le"
}

# Standard address terms that do not provide distinctive discriminating value
ADDRESS_STOPWORDS = {
    "road", "rd", "street", "st", "avenue", "ave", "lane", "ln", "drive", "dr",
    "court", "ct", "boulevard", "blvd", "way", "highway", "hwy", "floor", "flr",
    "fl", "suite", "ste", "unit", "apt", "apartment", "building", "bldg", "near",
    "opp", "opposite", "behind", "beside", "at", "po", "post", "block", "sector",
    "nagar", "colony", "plot", "flat", "door", "no", "room", "city", "town", "dist",
    "district", "state", "west", "east", "north", "south", "null", "none"
}


def load_sources(
    source_path: str,
    sep: str = "\t",
    nrows: Optional[int] = None,
    usecols: Optional[List[str]] = None,
) -> pd.DataFrame:
    """
    Load a business entity TSV file with appropriate typing and missing value handling.

    Parameters:
        source_path: Filepath to the TSV file.
        sep: Delimiter character (default '\\t').
        nrows: Optional row limit for sampling or testing.
        usecols: Optional subset of columns to load.

    Returns:
        pd.DataFrame containing the entity records.
    """
    if not os.path.isfile(source_path):
        raise FileNotFoundError(f"File not found: {source_path}")

    df = pd.read_csv(
        source_path,
        sep=sep,
        nrows=nrows,
        usecols=usecols,
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
    )
    # Standardize column names
    df.columns = [c.strip().lower() for c in df.columns]
    return df


def normalize_for_blocking(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure normalized text columns exist on the DataFrame using Member 1's preprocessing logic.
    If pre-normalized columns already exist, they are preserved.

    Parameters:
        df: Input DataFrame with at least entity_id, business_name, business_address, country.

    Returns:
        DataFrame copy with 'norm_name', 'norm_address', and 'norm_country' columns.
    """
    df = df.copy()

    if "norm_name" not in df.columns:
        if "business_name" in df.columns:
            df["norm_name"] = df["business_name"].apply(normalize_name)
        else:
            df["norm_name"] = ""

    if "norm_address" not in df.columns:
        if "business_address" in df.columns:
            df["norm_address"] = df["business_address"].apply(normalize_address)
        else:
            df["norm_address"] = ""

    if "norm_country" not in df.columns:
        if "country" in df.columns:
            df["norm_country"] = df["country"].apply(normalize_country)
        else:
            df["norm_country"] = ""

    return df


def extract_informative_name_tokens(
    normalized_name: str,
    min_len: int = 3,
) -> List[str]:
    """
    Extract meaningful, discriminating tokens from a normalized business name.
    Filters out common corporate legal suffixes and stopwords.

    Parameters:
        normalized_name: Preprocessed business name string.
        min_len: Minimum token character length.

    Returns:
        List of informative token strings.
    """
    tokens = normalized_name.split()
    return [
        t for t in tokens
        if len(t) >= min_len and t not in BUSINESS_STOPWORDS and not t.isdigit()
    ]


def extract_informative_address_tokens(
    normalized_address: str,
) -> List[str]:
    """
    Extract distinctive tokens from a normalized address (e.g., street numbers,
    pin codes, unique locality words).

    Parameters:
        normalized_address: Preprocessed address string.

    Returns:
        List of distinctive address token strings.
    """
    tokens = normalized_address.split()
    res = []
    for t in tokens:
        if t in ADDRESS_STOPWORDS:
            continue
        # House/street numbers and alphanumeric identifiers
        if any(c.isdigit() for c in t) and len(t) >= 2:
            res.append(t)
        # Distinctive alphabetic locality names
        elif len(t) >= 4 and not t.isdigit():
            res.append(t)
    return res


def country_block(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
) -> Dict[str, Tuple[pd.DataFrame, pd.DataFrame]]:
    """
    Partition Source 1 and Target DataFrames by normalized country.
    
    Why: True business entities in this challenge are strictly within the same country.
    Partitioning by country eliminates cross-country comparisons (reducing ~60% of search space)
    and dynamically handles open country sets (US, India, France, etc.).

    Parameters:
        source1_df: DataFrame of Source 1 records (must have 'norm_country').
        target_df: DataFrame of Target (S2/S3) records (must have 'norm_country').

    Returns:
        Dict mapping country_label -> (s1_country_subframe, target_country_subframe).
    """
    partitions = {}
    s1_countries = set(source1_df["norm_country"].unique())
    tgt_countries = set(target_df["norm_country"].unique())
    all_countries = s1_countries.intersection(tgt_countries)

    for country in all_countries:
        if not country:
            continue
        s1_sub = source1_df[source1_df["norm_country"] == country].reset_index(drop=True)
        tgt_sub = target_df[target_df["norm_country"] == country].reset_index(drop=True)
        if len(s1_sub) > 0 and len(tgt_sub) > 0:
            partitions[country] = (s1_sub, tgt_sub)

    return partitions


def rare_token_block(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    max_token_freq: int = 150,
    min_token_len: int = 3,
) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs using an inverted index of rare/informative name tokens.

    Why: Businesses sharing an uncommon word (e.g. 'Novinoviaria', 'Dahlia', 'Orelee')
    are highly likely candidate matches. Capping token frequency prevents candidate
    explosions from common dictionary terms.

    Parameters:
        source1_df: Preprocessed Source 1 records.
        target_df: Preprocessed Target records.
        max_token_freq: Maximum document frequency in target set for a token to be indexed.
        min_token_len: Minimum token character length.

    Returns:
        Set of candidate pairs: (source1_entity_id, candidate_entity_id).
    """
    candidates = set()

    # Build inverted index on target names: token -> list of target entity IDs
    token_to_targets = defaultdict(list)
    for _, row in target_df.iterrows():
        tokens = set(extract_informative_name_tokens(row["norm_name"], min_len=min_token_len))
        for t in tokens:
            token_to_targets[t].append(row["entity_id"])

    # Filter out tokens exceeding maximum frequency threshold
    rare_index = {
        tok: eids for tok, eids in token_to_targets.items() if len(eids) <= max_token_freq
    }

    # Match S1 records against the inverted index
    for _, row in source1_df.iterrows():
        s1_id = row["entity_id"]
        tokens = set(extract_informative_name_tokens(row["norm_name"], min_len=min_token_len))
        for t in tokens:
            for meid in rare_index.get(t, []):
                candidates.add((s1_id, meid))

    return candidates


def tfidf_name_block(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    k_top: int = 25,
    min_similarity: float = 0.18,
    chunk_size: int = 500,
    ngram_range: Tuple[int, int] = (3, 4),
) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs via character n-gram TF-IDF cosine similarity.

    Why: Character n-grams are robust against misspellings ('Wilblims' vs 'Williams'),
    diacritics/accents ('Nónet' vs 'Nonet'), concatenations ('maurewilliamscolombier.com'),
    and word transpositions.

    Uses batched matrix multiplication to ensure O(chunk_size) memory usage.

    Parameters:
        source1_df: Preprocessed Source 1 records.
        target_df: Preprocessed Target records.
        k_top: Number of top candidate matches to retrieve per S1 record.
        min_similarity: Minimum cosine similarity threshold to qualify as a candidate.
        chunk_size: Batch size of S1 records for vector multiplication.
        ngram_range: Character n-gram bounds (default (3, 4)).

    Returns:
        Set of candidate pairs: (source1_entity_id, candidate_entity_id).
    """
    candidates = set()
    s1_names = source1_df["norm_name"].tolist()
    tgt_names = target_df["norm_name"].tolist()

    if not s1_names or not tgt_names:
        return candidates

    # Fit TF-IDF vectorizer on combined corpus
    tfidf = TfidfVectorizer(
        analyzer="char_wb",
        ngram_range=ngram_range,
        min_df=1,
        sublinear_tf=True,
    )
    tfidf.fit(tgt_names + s1_names)

    tgt_vecs = tfidf.transform(tgt_names)
    s1_vecs = tfidf.transform(s1_names)

    # Process S1 in chunks to conserve memory
    n_s1 = len(source1_df)
    for start in range(0, n_s1, chunk_size):
        end = min(start + chunk_size, n_s1)
        s1_chunk = s1_vecs[start:end]
        sims = (s1_chunk * tgt_vecs.T).toarray()

        for i in range(len(sims)):
            s1_id = source1_df.iloc[start + i]["entity_id"]
            row_sims = sims[i]

            if len(row_sims) <= k_top:
                top_indices = np.argsort(-row_sims)
            else:
                top_indices = np.argpartition(-row_sims, k_top)[:k_top]
                top_indices = top_indices[np.argsort(-row_sims[top_indices])]

            for tidx in top_indices:
                if row_sims[tidx] >= min_similarity:
                    tgt_id = target_df.iloc[tidx]["entity_id"]
                    candidates.add((s1_id, tgt_id))

    return candidates


def rare_address_token_block(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    max_addr_freq: int = 30,
) -> Set[Tuple[str, str]]:
    """
    Generate candidate pairs by indexing distinctive address identifiers (street numbers,
    PIN codes, unique locality words).

    Why: In multi-source entity resolution, business names often suffer extreme corruption
    or non-Latin transliteration (e.g. Hindi/Tamil scripts for Indian businesses) while
    addresses retain common alphanumeric door/street numbers and postal codes.

    Parameters:
        source1_df: Preprocessed Source 1 records.
        target_df: Preprocessed Target records.
        max_addr_freq: Maximum document frequency for an address token.

    Returns:
        Set of candidate pairs: (source1_entity_id, candidate_entity_id).
    """
    candidates = set()

    addr_to_tgt = defaultdict(list)
    for _, row in target_df.iterrows():
        tokens = set(extract_informative_address_tokens(row["norm_address"]))
        for t in tokens:
            addr_to_tgt[t].append(row["entity_id"])

    addr_index = {
        tok: eids for tok, eids in addr_to_tgt.items() if len(eids) <= max_addr_freq
    }

    for _, row in source1_df.iterrows():
        s1_id = row["entity_id"]
        tokens = set(extract_informative_address_tokens(row["norm_address"]))
        token_hits = defaultdict(int)

        for t in tokens:
            weight = 2 if any(c.isdigit() for c in t) else 1
            for meid in addr_index.get(t, []):
                token_hits[meid] += weight

        # Require match on at least 2 distinct address tokens or 1 numeric token
        for meid, score in token_hits.items():
            if score >= 2:
                candidates.add((s1_id, meid))

    return candidates


def generate_candidates(
    source1_df: pd.DataFrame,
    target_df: pd.DataFrame,
    use_rare_tokens: bool = True,
    use_tfidf: bool = True,
    use_address_tokens: bool = True,
    k_top: int = 25,
    min_similarity: float = 0.18,
    max_token_freq: int = 150,
    max_addr_freq: int = 30,
) -> pd.DataFrame:
    """
    Execute the multi-pass blocking pipeline:
        1. Country partitioning
        2. Rare name-token blocking
        3. Character n-gram TF-IDF retrieval
        4. Distinctive address-token blocking (optional)
        5. Union and deduplication
    
    Guarantees:
        - Candidates are strictly S1-S2 or S1-S3 (no S1-S1 self-matches).
        - Multiple candidates per S1 are supported.
        - Output is deterministically sorted for reproducibility.

    Parameters:
        source1_df: DataFrame containing Source 1 records.
        target_df: DataFrame containing Target (Source 2 and/or Source 3) records.
        use_rare_tokens: Whether to enable rare token blocking.
        use_tfidf: Whether to enable TF-IDF retrieval.
        use_address_tokens: Whether to enable address token blocking.
        k_top: Top-K TF-IDF candidates per S1 record.
        min_similarity: Minimum TF-IDF cosine similarity.
        max_token_freq: Document frequency cap for name tokens.
        max_addr_freq: Document frequency cap for address tokens.

    Returns:
        pd.DataFrame with columns ['source1_entity_id', 'candidate_entity_id'].
    """
    s1_norm = normalize_for_blocking(source1_df)
    tgt_norm = normalize_for_blocking(target_df)

    # Filter targets to valid S2 and S3 IDs only
    tgt_norm = tgt_norm[tgt_norm["entity_id"].str.startswith(("S2-", "S3-"))].copy()

    partitions = country_block(s1_norm, tgt_norm)
    all_pairs: Set[Tuple[str, str]] = set()

    for country, (s1_sub, tgt_sub) in partitions.items():
        if use_rare_tokens:
            rare_pairs = rare_token_block(
                s1_sub, tgt_sub, max_token_freq=max_token_freq
            )
            all_pairs.update(rare_pairs)

        if use_tfidf:
            tfidf_pairs = tfidf_name_block(
                s1_sub, tgt_sub, k_top=k_top, min_similarity=min_similarity
            )
            all_pairs.update(tfidf_pairs)

        if use_address_tokens:
            addr_pairs = rare_address_token_block(
                s1_sub, tgt_sub, max_addr_freq=max_addr_freq
            )
            all_pairs.update(addr_pairs)

    # Format into sorted, deterministic DataFrame
    pairs_list = sorted(list(all_pairs))
    if pairs_list:
        candidates_df = pd.DataFrame(
            pairs_list, columns=["source1_entity_id", "candidate_entity_id"]
        )
    else:
        candidates_df = pd.DataFrame(
            columns=["source1_entity_id", "candidate_entity_id"]
        )

    return candidates_df


def generate_candidate_pairs(
    source_a: pd.DataFrame,
    source_b: pd.DataFrame,
    blocking_keys: Optional[List[str]] = None,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Generate candidate matching pairs between two entity sources.
    Implements the standard project scaffolding interface.

    Parameters:
        source_a: Source 1 DataFrame.
        source_b: Target DataFrame (Source 2 or Source 3).
        blocking_keys: Optional list of specific blocking keys (kept for interface compatibility).
        **kwargs: Strategy configurations passed to generate_candidates.

    Returns:
        pd.DataFrame of candidate pairs ['source1_entity_id', 'candidate_entity_id'].
    """
    return generate_candidates(source_a, source_b, **kwargs)


def block_source1_source2(
    source_1: pd.DataFrame,
    source_2: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Perform blocking and generate candidate pairs between Source 1 and Source 2.
    """
    return generate_candidates(source_1, source_2, **kwargs)


def block_source1_source3(
    source_1: pd.DataFrame,
    source_3: pd.DataFrame,
    **kwargs: Any,
) -> pd.DataFrame:
    """
    Perform blocking and generate candidate pairs between Source 1 and Source 3.
    """
    return generate_candidates(source_1, source_3, **kwargs)


def export_candidate_pairs_tsv(
    candidate_pairs_df: pd.DataFrame,
    source1_df: pd.DataFrame,
    output_path: str,
) -> None:
    """
    Export candidate pairs into the official challenge submission format:
        source1_entity_id \\t candidate_entity_ids
    Where candidate_entity_ids is a comma-separated list of candidate IDs.
    
    Guarantees:
        - Exactly one row per Source 1 entity in source1_df.
        - Empty string for entities with zero candidates.
        - Tab-separated (.tsv) encoding in UTF-8.
        - Passable directly to utils/validate_submission.py.

    Parameters:
        candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id'].
        source1_df: Source 1 DataFrame containing all required entity IDs.
        output_path: Destination filepath for candidate_pairs.tsv.
    """
    os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)

    # Group candidate IDs by source1_entity_id
    grouped = (
        candidate_pairs_df.groupby("source1_entity_id")["candidate_entity_id"]
        .apply(lambda ids: ",".join(sorted(set(ids))))
        .to_dict()
    )

    all_s1_ids = list(source1_df["entity_id"].unique())
    rows = []
    for s1_id in all_s1_ids:
        c_str = grouped.get(s1_id, "")
        rows.append({"source1_entity_id": s1_id, "candidate_entity_ids": c_str})

    res_df = pd.DataFrame(rows, columns=["source1_entity_id", "candidate_entity_ids"])
    res_df.to_csv(output_path, sep="\t", index=False, encoding="utf-8")


def evaluate_candidate_recall(
    candidate_pairs_df: pd.DataFrame,
    ground_truth: Union[pd.DataFrame, str],
    source1_df: Optional[pd.DataFrame] = None,
    target_df: Optional[pd.DataFrame] = None,
    max_missed_examples: int = 10,
) -> Dict[str, Any]:
    """
    Calculate blocking candidate recall against ground truth labels.

    Computes:
        - Candidate Recall = Captured True Pairs / Total True Pairs
        - Total ground truth pairs
        - Captured pairs count
        - Missed pairs count
        - Total candidate pairs generated
        - Candidate reduction ratio (if target_df and source1_df provided)
        - Sample of missed true pairs for diagnostics

    Parameters:
        candidate_pairs_df: DataFrame with ['source1_entity_id', 'candidate_entity_id'].
        ground_truth: Ground truth DataFrame or path to train_ground_truth.tsv.
        source1_df: Optional Source 1 DataFrame for reduction ratio and missed-pair details.
        target_df: Optional Target DataFrame for reduction ratio and missed-pair details.
        max_missed_examples: Number of missed pair examples to extract.

    Returns:
        Dict of evaluation metrics and diagnostic details.
    """
    if isinstance(ground_truth, str):
        gt_df = pd.read_csv(ground_truth, sep="\t", dtype=str, keep_default_na=False)
    else:
        gt_df = ground_truth.copy()

    gt_df.columns = [c.strip().lower() for c in gt_df.columns]
    s1_col = "source1_entity_id"
    match_col = "matched_entity_ids"

    # Restrict ground truth to S1 entities present in source1_df if provided
    if source1_df is not None:
        eval_s1_ids = set(source1_df["entity_id"].unique())
        gt_df = gt_df[gt_df[s1_col].isin(eval_s1_ids)]

    # Parse ground truth pairs
    gt_pairs: Set[Tuple[str, str]] = set()
    for _, row in gt_df.iterrows():
        s1_id = str(row[s1_col]).strip()
        m_str = str(row[match_col]).strip()
        if m_str and m_str != "nan":
            for mid in m_str.split(","):
                mid = mid.strip()
                if mid:
                    gt_pairs.add((s1_id, mid))

    # Candidate pairs set
    c_s1_col = candidate_pairs_df.columns[0]
    c_tgt_col = candidate_pairs_df.columns[1]
    candidate_pairs: Set[Tuple[str, str]] = set(
        zip(candidate_pairs_df[c_s1_col], candidate_pairs_df[c_tgt_col])
    )

    total_gt = len(gt_pairs)
    captured = len(gt_pairs.intersection(candidate_pairs))
    missed = total_gt - captured
    recall = captured / max(1, total_gt)
    total_candidates = len(candidate_pairs)

    reduction_ratio = None
    if source1_df is not None and target_df is not None:
        total_possible = len(source1_df) * len(target_df)
        reduction_ratio = 1.0 - (total_candidates / max(1, total_possible))

    # Sample missed pairs for investigation
    missed_set = gt_pairs.difference(candidate_pairs)
    missed_examples = []

    s1_lookup = {}
    if source1_df is not None:
        s1_lookup = {r["entity_id"]: r.to_dict() for _, r in source1_df.iterrows()}

    tgt_lookup = {}
    if target_df is not None:
        tgt_lookup = {r["entity_id"]: r.to_dict() for _, r in target_df.iterrows()}

    for s1_id, tgt_id in list(missed_set)[:max_missed_examples]:
        missed_examples.append({
            "source1_id": s1_id,
            "target_id": tgt_id,
            "source1_record": s1_lookup.get(s1_id, None),
            "target_record": tgt_lookup.get(tgt_id, None),
        })

    report = {
        "candidate_recall": recall,
        "total_ground_truth_pairs": total_gt,
        "captured_pairs": captured,
        "missed_pairs": missed,
        "total_candidate_pairs": total_candidates,
        "reduction_ratio": reduction_ratio,
        "missed_examples": missed_examples,
    }

    return report
