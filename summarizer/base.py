"""
رابط انتزاعی موتور خلاصه‌سازی.

هر دو موتور (استخراجی و تولیدی) این رابط را پیاده‌سازی می‌کنند تا لایه‌ی بالاتر
(تحلیلگر نظرات) مستقل از نوع موتور بماند.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from summarizer.preprocess import split_sentences


class BaseSummarizer(ABC):
    """رابط پایه‌ی موتورهای خلاصه‌سازی."""

    name: str = "base"

    @abstractmethod
    def summarize(self, text: str) -> str:
        """یک متن را خلاصه کن."""
        raise NotImplementedError

    def summarize_reviews(self, reviews: list[str]) -> str:
        """
        خلاصه‌ی تجمیعی از چند نظر.

        پیاده‌سازی پیش‌فرض: همه‌ی نظرات را به هم می‌چسباند و خلاصه می‌کند. موتورهایی
        که راهبرد بهتری دارند (مثل map-reduce در موتور تولیدی) این متد را override
        می‌کنند.
        """
        joined = " ".join(r for r in reviews if r and r.strip())
        return self.summarize(joined)

    # ابزار مشترک
    @staticmethod
    def _sentences(text: str) -> list[str]:
        return split_sentences(text)


__all__ = ["BaseSummarizer"]
