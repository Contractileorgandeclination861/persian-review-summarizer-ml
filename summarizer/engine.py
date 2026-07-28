"""
کارخانه‌ی انتخاب موتور خلاصه‌سازی بر اساس تنظیمات."""

from __future__ import annotations

from summarizer.base import BaseSummarizer
from summarizer.config import Settings, get_settings


def get_summarizer(settings: Settings | None = None) -> BaseSummarizer:
    """موتور مناسب را بر اساس `METHOD` بساز."""
    settings = settings or get_settings()
    method = settings.method.lower()

    if method == "extractive":
        from summarizer.extractive import ExtractiveSummarizer

        return ExtractiveSummarizer(settings)
    if method == "abstractive":
        from summarizer.abstractive import AbstractiveSummarizer

        return AbstractiveSummarizer(settings)

    raise ValueError(
        f"METHOD نامعتبر است: {method!r}. مقادیر مجاز: extractive، abstractive"
    )


__all__ = ["get_summarizer"]
