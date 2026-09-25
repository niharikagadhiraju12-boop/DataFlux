import re
import unicodedata

import pandas as pd


def normalize_text(value):
    """
    Basic Unicode-safe text normalization.
    """
    if pd.isna(value):
        return ""

    text = str(value)
    text = unicodedata.normalize("NFKC", text)
    text = text.casefold()
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(value):
    """
    Normalize business names.
    """
    return normalize_text(value)


def normalize_address(value):
    """
    Normalize business addresses.
    """
    return normalize_text(value)


def normalize_country(value):
    """
    Normalize country values.
    """
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.casefold()
    text = re.sub(r"\s+", " ", text).strip()

    return text