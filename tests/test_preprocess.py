"""تست‌های پیش‌پردازش و جمله‌بندی فارسی."""

from __future__ import annotations

from summarizer.preprocess import PersianTextPreprocessor, split_sentences


def test_empty():
    assert PersianTextPreprocessor().clean("  ") == ""


def test_arabic_normalized():
    out = PersianTextPreprocessor().clean("علي كتاب")
    assert "ي" not in out and "ك" not in out


def test_digits_persianized():
    out = PersianTextPreprocessor().clean("قیمت 1500")
    assert "1500" not in out and "۱۵۰۰" in out


def test_emoji_and_url_removed():
    out = PersianTextPreprocessor().clean("عالی 😍 https://x.ir")
    assert "😍" not in out and "http" not in out


def test_split_sentences():
    sents = split_sentences("جمله اول. جمله دوم! جمله سوم؟")
    assert len(sents) == 3


def test_split_empty():
    assert split_sentences("") == []
