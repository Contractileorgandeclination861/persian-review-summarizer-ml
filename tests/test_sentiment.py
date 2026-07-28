"""تست‌های تحلیل احساسات."""

from __future__ import annotations

from summarizer import sentiment


def test_positive_score():
    assert sentiment.score_text("کیفیت عالی و راضی هستم") > 0


def test_negative_score():
    assert sentiment.score_text("افتضاح بود و خیلی بد") < 0


def test_neutral_score():
    assert sentiment.score_text("محصول را دیروز خریدم") == 0


def test_analyze_counts():
    res = sentiment.analyze(["عالی بود", "بد بود", "خریدم"])
    assert res.positive == 1
    assert res.negative == 1
    assert res.neutral == 1
    assert res.total == 3


def test_label():
    assert sentiment.analyze(["عالی", "خوب", "بد"]).label == "عمدتاً مثبت"
    assert sentiment.analyze([]).label == "نامشخص"
