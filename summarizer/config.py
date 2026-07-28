"""
تنظیمات مرکزی خلاصه‌ساز (config.py).

همه‌ی پارامترها از `.env` یا متغیرهای محیطی خوانده می‌شوند تا هیچ مقداری در کد
هاردکد نشود.
"""

from __future__ import annotations

from dataclasses import dataclass
import os


def _get(name: str, default: str) -> str:
    return os.environ.get(name, default)


def _get_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


@dataclass(frozen=True)
class Settings:
    """تنظیمات برنامه (بدون وابستگی خارجی، سبک و آفلاین)."""

    # روش خلاصه‌سازی: "extractive" (ML، بدون دانلود) یا "abstractive" (DL، مدل mT5)
    method: str = _get("METHOD", "extractive")

    # --- موتور استخراجی (TextRank) ---
    #: تعداد جملات نهایی در خلاصه‌ی استخراجی
    summary_sentences: int = _get_int("SUMMARY_SENTENCES", 3)

    # --- موتور تولیدی (mT5) ---
    hf_model_name: str = _get("HF_MODEL_NAME", "csebuetnlp/mT5_multilingual_XLSum")
    device: str = _get("DEVICE", "cpu")
    max_input_length: int = _get_int("MAX_INPUT_LENGTH", 1024)
    max_output_length: int = _get_int("MAX_OUTPUT_LENGTH", 90)
    min_output_length: int = _get_int("MIN_OUTPUT_LENGTH", 15)
    num_beams: int = _get_int("NUM_BEAMS", 4)

    # --- عمومی ---
    max_reviews: int = _get_int("MAX_REVIEWS", 500)


def get_settings() -> Settings:
    """نمونه‌ی تنظیمات را بساز (هر بار از محیط می‌خواند تا در تست قابل‌کنترل باشد)."""
    return Settings()
