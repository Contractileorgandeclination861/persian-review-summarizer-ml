"""
پیش‌پردازش و جمله‌بندی متن فارسی.

اگر `hazm` نصب باشد از نرمال‌سازی و جمله‌بندی حرفه‌ای آن استفاده می‌شود؛ در غیر
این صورت به نسخه‌ی سبک مبتنی بر regex برمی‌گردیم تا پروژه همیشه آفلاین کار کند.
"""

from __future__ import annotations

import re
import unicodedata

try:
    from hazm import Normalizer as _HazmNormalizer
    from hazm import sent_tokenize as _hazm_sent_tokenize

    _HAS_HAZM = True
except Exception:  # pragma: no cover
    _HAS_HAZM = False

_DIGIT_MAP = {
    "٠": "۰", "١": "۱", "٢": "۲", "٣": "۳", "٤": "۴",
    "٥": "۵", "٦": "۶", "٧": "۷", "٨": "۸", "٩": "۹",
    "0": "۰", "1": "۱", "2": "۲", "3": "۳", "4": "۴",
    "5": "۵", "6": "۶", "7": "۷", "8": "۸", "9": "۹",
}
_CHAR_MAP = {"ي": "ی", "ك": "ک", "ة": "ه", "أ": "ا", "إ": "ا", "ؤ": "و", "ئ": "ی"}
_DIACRITICS = re.compile(r"[ً-ْٰـ]")
_EMOJI = re.compile(
    "[\U0001F300-\U0001FAFF\U00002600-\U000027BF\U0001F1E6-\U0001F1FF"
    "\U00002190-\U000021FF\U00002B00-\U00002BFF️]+",
    flags=re.UNICODE,
)
_URL = re.compile(r"https?://\S+|www\.\S+")
_MULTISPACE = re.compile(r"[ \t]{2,}")
_SENT_SPLIT = re.compile(r"(?<=[.!?؟…])\s+|\n+")


class PersianTextPreprocessor:
    """پاک‌سازی و نرمال‌سازی متن فارسی."""

    def __init__(self) -> None:
        self._hazm = _HazmNormalizer() if _HAS_HAZM else None

    def clean(self, text: str) -> str:
        if not text or not text.strip():
            return ""
        text = unicodedata.normalize("NFC", text)
        text = _URL.sub(" ", text)
        text = _EMOJI.sub(" ", text)
        text = _DIACRITICS.sub("", text)
        text = "".join(_CHAR_MAP.get(c, c) for c in text)
        text = "".join(_DIGIT_MAP.get(c, c) for c in text)
        if self._hazm is not None:
            text = self._hazm.normalize(text)
        text = _MULTISPACE.sub(" ", text)
        return text.strip()

    def clean_many(self, texts) -> list[str]:
        return [self.clean(t) for t in texts]


def split_sentences(text: str) -> list[str]:
    """متن را به جملات تقسیم کن (hazm در صورت وجود، وگرنه regex)."""
    text = (text or "").strip()
    if not text:
        return []
    if _HAS_HAZM:
        sents = _hazm_sent_tokenize(text)
    else:
        sents = _SENT_SPLIT.split(text)
    return [s.strip() for s in sents if s and s.strip()]


__all__ = ["PersianTextPreprocessor", "split_sentences"]
