"""تست‌های استخراج بینش (کلیدواژه، Pros/Cons، امتیاز)."""

from __future__ import annotations

from summarizer import insights

_REVIEWS = [
    "دوربین گوشی عالیه و عکس‌ها واضحن.",
    "باتری ضعیفه و زود خالی میشه.",
    "کیفیت ساخت خوبه ولی قیمت گرونه.",
    "پشتیبانی خوب نبود و دیر جواب دادن.",
]


def test_keywords_returns_list():
    kw = insights.extract_keywords(_REVIEWS, top_n=5)
    assert isinstance(kw, list)
    assert len(kw) <= 5
    assert all(isinstance(k, str) for k in kw)


def test_keywords_empty():
    assert insights.extract_keywords([]) == []


def test_pros_cons_split():
    pros, cons = insights.extract_pros_cons(_REVIEWS)
    assert isinstance(pros, list) and isinstance(cons, list)
    # حداقل یک نکته‌ی مثبت و یک نکته‌ی منفی باید تشخیص داده شود
    assert len(pros) >= 1
    assert len(cons) >= 1


def test_rating_range():
    assert insights.estimate_rating(0, 0, 0) == 0.0
    assert insights.estimate_rating(10, 0, 0) == 5.0
    assert insights.estimate_rating(0, 10, 0) == 1.0
    mid = insights.estimate_rating(5, 5, 0)
    assert 1.0 <= mid <= 5.0


def test_rating_positive_bias():
    """امتیاز نظرات مثبت باید بیشتر از منفی باشد."""
    high = insights.estimate_rating(8, 1, 1)
    low = insights.estimate_rating(1, 8, 1)
    assert high > low
