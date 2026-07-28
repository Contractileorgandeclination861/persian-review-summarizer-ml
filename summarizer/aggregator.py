"""
تحلیلگر نظرات (لایه‌ی سطح‌بالا).

مجموعه‌ای از نظرات را می‌گیرد و یک گزارش تحلیلی کامل برمی‌گرداند:
  • خلاصه‌ی کوتاه (با موتور انتخاب‌شده: استخراجی یا تولیدی)
  • آمار احساسات (مثبت/منفی/خنثی) + امتیاز ستاره‌ای تخمینی
  • کلیدواژه‌های پرتکرار
  • نکات مثبت و منفی (Pros & Cons)

هم رابط کاربری (Streamlit) و هم خط فرمان از همین کلاس استفاده می‌کنند.
"""

from __future__ import annotations

from dataclasses import dataclass, field

from summarizer import insights, sentiment
from summarizer.base import BaseSummarizer
from summarizer.config import Settings, get_settings
from summarizer.engine import get_summarizer
from summarizer.preprocess import PersianTextPreprocessor
from summarizer.sentiment import SentimentResult


@dataclass(frozen=True)
class ReviewAnalysis:
    """خروجی کامل تحلیل مجموعه‌ای از نظرات."""

    summary: str
    sentiment: SentimentResult
    review_count: int
    method: str
    rating: float = 0.0
    keywords: list[str] = field(default_factory=list)
    pros: list[str] = field(default_factory=list)
    cons: list[str] = field(default_factory=list)


class ReviewAnalyzer:
    """موتور اصلی تحلیل نظرات فارسی (خلاصه + احساسات + بینش)."""

    def __init__(
        self,
        summarizer: BaseSummarizer | None = None,
        preprocessor: PersianTextPreprocessor | None = None,
        settings: Settings | None = None,
    ) -> None:
        self._settings = settings or get_settings()
        self._summarizer = summarizer or get_summarizer(self._settings)
        self._pre = preprocessor or PersianTextPreprocessor()

    @property
    def method(self) -> str:
        return self._summarizer.name

    def analyze(self, reviews: list[str]) -> ReviewAnalysis:
        """مجموعه‌ای از نظرات را تحلیل کن و گزارش کامل برگردان."""
        if len(reviews) > self._settings.max_reviews:
            raise ValueError(
                f"تعداد نظرات ({len(reviews)}) از حد مجاز "
                f"({self._settings.max_reviews}) بیشتر است."
            )

        cleaned = [c for c in self._pre.clean_many(reviews) if c]
        if not cleaned:
            return ReviewAnalysis(
                summary="",
                sentiment=SentimentResult(0, 0, 0, 0),
                review_count=0,
                method=self.method,
            )

        summary = self._summarizer.summarize_reviews(cleaned)
        senti = sentiment.analyze(cleaned)
        pros, cons = insights.extract_pros_cons(cleaned)

        return ReviewAnalysis(
            summary=summary,
            sentiment=senti,
            review_count=len(cleaned),
            method=self.method,
            rating=insights.estimate_rating(
                senti.positive, senti.negative, senti.neutral
            ),
            keywords=insights.extract_keywords(cleaned),
            pros=pros,
            cons=cons,
        )

    def summarize_one(self, text: str) -> str:
        """خلاصه‌ی یک متن تکی."""
        cleaned = self._pre.clean(text)
        return self._summarizer.summarize(cleaned) if cleaned else ""


__all__ = ["ReviewAnalyzer", "ReviewAnalysis"]
