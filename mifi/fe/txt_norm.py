from __future__ import annotations

import re

from num2words import num2words

_CURRENCY = {'$': 'dollar', '£': 'pound', '€': 'euro', '¥': 'yen'}


def _expand_ordinal(m: re.Match) -> str:
    return num2words(int(m.group(1)), to='ordinal')


def _expand_currency(m: re.Match) -> str:
    symbol, amount_str = m.group(1), m.group(2).replace(',', '')
    amount = float(amount_str)
    n = int(amount) if amount.is_integer() else amount
    word = _CURRENCY.get(symbol, symbol)
    plural = 's' if amount != 1 else ''
    return f'{num2words(n)} {word}{plural}'


def _expand_number(m: re.Match) -> str:
    s = m.group(0).replace(',', '')
    try:
        return num2words(float(s) if '.' in s else int(s))
    except Exception:
        return m.group(0)


def normalize_text(text: str) -> str:
    text = re.sub(r'\b(\d+)(st|nd|rd|th)\b', _expand_ordinal, text)
    text = re.sub(r'([$£€¥])([0-9,]+(?:\.[0-9]+)?)', _expand_currency, text)
    text = re.sub(r'\b[0-9,]+(?:\.[0-9]+)?\b', _expand_number, text)
    return text.strip()
