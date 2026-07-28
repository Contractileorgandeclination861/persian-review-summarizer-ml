"""
خلاصه‌ساز محلی نظرات فارسی (کاملاً آفلاین، مبتنی بر ML/DL).

این پکیج دو موتور خلاصه‌سازی دارد:
  • ExtractiveSummarizer  — یادگیری ماشین (TextRank با TF-IDF + PageRank)، بدون دانلود مدل
  • AbstractiveSummarizer — یادگیری عمیق (مدل mT5 روی سیستم محلی)

و یک لایه‌ی تحلیل (ReviewAnalyzer) که مجموعه‌ای از نظرات را می‌گیرد و یک
«پیام کوتاه» جمع‌بندی + آمار احساسات برمی‌گرداند.
"""

from summarizer.aggregator import ReviewAnalyzer, ReviewAnalysis
from summarizer.base import BaseSummarizer
from summarizer.engine import get_summarizer

__version__ = "1.0.0"

__all__ = [
    "BaseSummarizer",
    "get_summarizer",
    "ReviewAnalyzer",
    "ReviewAnalysis",
]
