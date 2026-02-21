from __future__ import annotations

from functools import lru_cache


@lru_cache(maxsize=1)
def _get_nemo_normalizer():
    try:
        from nemo_text_processing.text_normalization.normalize import Normalizer
    except Exception:
        return None

    return Normalizer(input_case="cased", lang="en")


def normalize_text(text: str) -> str:
    normalizer = _get_nemo_normalizer()
    if normalizer is None:
        return text

    try:
        normalized = normalizer.normalize(text, verbose=False)
    except TypeError:
        normalized = normalizer.normalize(text)

    return normalized.strip() if normalized else text
