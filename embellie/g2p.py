from g2p_en import G2p

from embellie.txt_norm import normalize_text


_G2P = G2p()


def text_to_phonemes(text: str) -> list[str]:
    normalized_text = normalize_text(text)
    phonemes = _G2P(normalized_text)
    return [token for token in phonemes if token and token.strip()]
