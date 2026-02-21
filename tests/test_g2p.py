from types import SimpleNamespace

import mifi.fe.g2p as g2p_module


def _fake_g2p(*phonemes):
    """Return a callable that mimics misaki G2P returning given phoneme strings per token."""
    tokens = [SimpleNamespace(phonemes=p, whitespace=' ') for p in phonemes]
    return lambda text: ("", tokens)


def test_text_to_phonemes_filters_empty_tokens(monkeypatch) -> None:
    monkeypatch.setattr(g2p_module, "normalize_text", lambda text: text)
    monkeypatch.setattr(g2p_module, "_load_g2p", lambda: _fake_g2p("hɛloʊ", " ", "", "wɜːld"))

    tokens = g2p_module.text_to_phonemes("hello world")

    assert tokens == ["hɛloʊ", "wɜːld"]
    assert all(tok.strip() for tok in tokens)


def test_text_to_phonemes_runs_normalization_first(monkeypatch) -> None:
    seen = []
    monkeypatch.setattr(g2p_module, "normalize_text", lambda text: seen.append(text) or "normalized")
    monkeypatch.setattr(g2p_module, "_load_g2p", lambda: _fake_g2p("nɔːrməlˌaɪzd"))

    g2p_module.text_to_phonemes("RAW INPUT")

    assert seen == ["RAW INPUT"]


def test_text_to_phonemes_returns_nonempty_list(monkeypatch) -> None:
    monkeypatch.setattr(g2p_module, "normalize_text", lambda text: text)
    monkeypatch.setattr(g2p_module, "_load_g2p", lambda: _fake_g2p("hɛloʊ"))

    tokens = g2p_module.text_to_phonemes("hello")

    assert tokens
    assert all(isinstance(t, str) for t in tokens)
