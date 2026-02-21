from mifi.fe.txt_norm import normalize_text


def test_plain_text_unchanged() -> None:
    assert normalize_text("hello world") == "hello world"


def test_strips_whitespace() -> None:
    assert normalize_text("  hello  ") == "hello"


def test_integer_expanded() -> None:
    result = normalize_text("I have 3 cats")
    assert "3" not in result
    assert "three" in result


def test_float_expanded() -> None:
    result = normalize_text("score is 3.5")
    assert "3.5" not in result


def test_currency_dollar() -> None:
    result = normalize_text("costs $5")
    assert "$" not in result
    assert "five" in result
    assert "dollar" in result


def test_currency_singular() -> None:
    result = normalize_text("$1 only")
    assert "dollar" in result
    assert "dollars" not in result


def test_ordinal_expanded() -> None:
    result = normalize_text("finished 1st")
    assert "1st" not in result
    assert "first" in result
