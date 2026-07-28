"""تست‌های تحلیلگر نظرات (خلاصه + احساسات) با موتور استخراجی."""

from __future__ import annotations

import pytest

from summarizer.aggregator import ReviewAnalyzer
from summarizer.config import Settings
from summarizer.extractive import ExtractiveSummarizer


def _analyzer(max_reviews: int = 500) -> ReviewAnalyzer:
    settings = Settings(method="extractive", summary_sentences=2, max_reviews=max_reviews)
    return ReviewAnalyzer(
        summarizer=ExtractiveSummarizer(settings), settings=settings
    )


def test_analyze_returns_summary_and_sentiment():
    analyzer = _analyzer()
    reviews = [
        "کیفیت ساخت عالیه ولی قیمتش گرونه.",
        "باتری خوبی داره و یه روز کامل دووم میاره.",
        "پشتیبانی ضعیف بود و دیر جواب دادن.",
        "دوربین فوق‌العاده‌ست و عکس‌ها واضحن.",
    ]
    result = analyzer.analyze(reviews)
    assert result.summary
    assert result.review_count == 4
    assert result.sentiment.total == 4
    assert result.method == "extractive"


def test_empty_reviews():
    analyzer = _analyzer()
    result = analyzer.analyze(["", "   "])
    assert result.summary == ""
    assert result.review_count == 0


def test_max_reviews_limit():
    analyzer = _analyzer(max_reviews=2)
    with pytest.raises(ValueError):
        analyzer.analyze(["a", "b", "c"])


def test_summarize_one():
    analyzer = _analyzer()
    out = analyzer.summarize_one(
        "دوربین عالیه و عکس‌ها واضحن. باتری ضعیفه. صفحه‌نمایش خوبه. قیمت بالاست."
    )
    assert isinstance(out, str)
