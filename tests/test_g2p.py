import embellie.g2p as g2p_module


def test_text_to_phonemes_filters_empty_tokens(monkeypatch) -> None:
    monkeypatch.setattr(g2p_module, "normalize_text", lambda text: text)
    monkeypatch.setattr(
        g2p_module,
        "_G2P",
        lambda _: ["HH", "AH0", " ", "", "L", "OW1"],
    )

    tokens = g2p_module.text_to_phonemes("hello world")

    assert tokens
    assert tokens == ["HH", "AH0", "L", "OW1"]
    assert all(token.strip() for token in tokens)


def test_text_to_phonemes_two_sentences(monkeypatch) -> None:
    monkeypatch.setattr(g2p_module, "normalize_text", lambda text: text)
    monkeypatch.setattr(
        g2p_module,
        "_G2P",
        lambda _: ["DH", "IH1", "S", ".", " ", "", "IH1", "Z", ".", " "],
    )

    tokens = g2p_module.text_to_phonemes("This is. It is.")

    assert tokens == ["DH", "IH1", "S", ".", "IH1", "Z", "."]
    assert all(token.strip() for token in tokens)


def test_text_to_phonemes_runs_normalization_first(monkeypatch) -> None:
    monkeypatch.setattr(g2p_module, "normalize_text", lambda _: "normalized text")
    monkeypatch.setattr(g2p_module, "_G2P", lambda text: [text])

    tokens = g2p_module.text_to_phonemes("RAW INPUT")

    assert tokens == ["normalized text"]
