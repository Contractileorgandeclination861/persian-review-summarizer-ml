"""تست‌های موتور استخراجی (TextRank) — بدون نیاز به دانلود مدل."""

from __future__ import annotations

from summarizer.config import Settings
from summarizer.extractive import ExtractiveSummarizer


def _engine(n: int = 2) -> ExtractiveSummarizer:
    return ExtractiveSummarizer(Settings(summary_sentences=n))


def test_short_text_returned_as_is():
    eng = _engine(3)
    out = eng.summarize("یک جمله کوتاه است")
    assert out  # چون جملات کمتر از حد است، خودش برمی‌گردد


def test_summarize_selects_subset():
    eng = _engine(2)
    text = (
        "دوربین گوشی عالی است و عکس‌ها واضح‌اند. "
        "باتری ضعیف است و زود خالی می‌شود. "
        "صفحه‌نمایش روشن و باکیفیت است. "
        "قیمت گوشی کمی بالا است. "
        "پردازنده سریع است و هنگ نمی‌کند."
    )
    out = eng.summarize(text)
    assert isinstance(out, str) and out
    # خلاصه باید کوتاه‌تر از متن اصلی باشد
    assert len(out) < len(text)


def test_summarize_reviews():
    eng = _engine(2)
    reviews = [
        "کیفیت ساخت عالیه و بدنه محکمه.",
        "باتری ضعیفه و زود خالی میشه.",
        "دوربین فوق‌العاده‌ست و عکس‌ها واضحن.",
        "قیمت کمی بالاست ولی ارزش داره.",
    ]
    out = eng.summarize_reviews(reviews)
    assert isinstance(out, str) and out


def test_deterministic():
    """خروجی باید تکرارپذیر باشد (الگوریتم قطعی است)."""
    eng = _engine(2)
    text = "جمله یک درباره کیفیت. جمله دو درباره قیمت. جمله سه درباره باتری. جمله چهار درباره دوربین."
    assert eng.summarize(text) == eng.summarize(text)
