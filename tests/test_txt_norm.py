import mifi.txt_norm as txt_norm


def test_normalize_text_falls_back_when_nemo_unavailable(monkeypatch) -> None:
    monkeypatch.setattr(txt_norm, "_get_nemo_normalizer", lambda: None)

    assert txt_norm.normalize_text("Raw Text") == "Raw Text"


def test_normalize_text_uses_nemo_normalizer_and_strips(monkeypatch) -> None:
    class FakeNormalizer:
        def normalize(self, text: str, verbose: bool = False) -> str:
            return f"  normalized: {text}  "

    monkeypatch.setattr(txt_norm, "_get_nemo_normalizer", lambda: FakeNormalizer())

    assert txt_norm.normalize_text("hello") == "normalized: hello"


def test_normalize_text_handles_normalizer_without_verbose(monkeypatch) -> None:
    class LegacyNormalizer:
        def normalize(self, text: str) -> str:
            return "legacy"

    monkeypatch.setattr(txt_norm, "_get_nemo_normalizer", lambda: LegacyNormalizer())

    assert txt_norm.normalize_text("anything") == "legacy"
