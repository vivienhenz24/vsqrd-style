from functools import lru_cache

from misaki.en import G2P

from mifi.fe.txt_norm import normalize_text


@lru_cache(maxsize=1)
def _load_g2p() -> G2P:
    return G2P()


def text_to_phonemes(text: str) -> list[str]:
    normalized = normalize_text(text)
    _, tokens = _load_g2p()(normalized)
    return [tk.phonemes for tk in tokens if tk.phonemes and tk.phonemes.strip()]
