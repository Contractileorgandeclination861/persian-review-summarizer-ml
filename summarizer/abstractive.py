"""
موتور خلاصه‌سازی تولیدی (Abstractive) — یادگیری عمیق با مدل mT5.

از مدل `csebuetnlp/mT5_multilingual_XLSum` استفاده می‌کند که یک مدل ترنسفورمر
seq2seq چندزبانه است و روی فارسی هم آموزش دیده. برخلاف روش استخراجی، این موتور
جملات جدید تولید می‌کند (نه صرفاً انتخاب جمله).

  • کاملاً محلی اجرا می‌شود؛ فقط بار اول مدل (~۲.۲ گیگابایت) از HuggingFace
    دانلود می‌شود و پس از آن آفلاین کار می‌کند.
  • بارگذاری مدل «تنبل» است تا import برنامه سریع بماند.
  • برای چند نظر از راهبرد map-reduce استفاده می‌کند: ابتدا هر نظر خلاصه می‌شود،
    سپس خلاصه‌ها با هم خلاصه می‌شوند تا یک «پیام کوتاه» نهایی به دست آید.
"""

from __future__ import annotations

import re

from summarizer.base import BaseSummarizer
from summarizer.config import Settings, get_settings

_WHITESPACE = re.compile(r"\s+")


class AbstractiveSummarizer(BaseSummarizer):
    """خلاصه‌ساز تولیدی مبتنی بر ترنسفورمر mT5."""

    name = "abstractive"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()
        self._tokenizer = None
        self._model = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import AutoModelForSeq2SeqLM, AutoTokenizer

        name = self._settings.hf_model_name
        self._tokenizer = AutoTokenizer.from_pretrained(name)
        self._model = AutoModelForSeq2SeqLM.from_pretrained(name)
        self._model.to(self._settings.device)
        self._model.eval()

    def summarize(self, text: str) -> str:
        text = _WHITESPACE.sub(" ", (text or "")).strip()
        if not text:
            return ""

        self._ensure_loaded()
        import torch

        s = self._settings
        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            max_length=s.max_input_length,
            truncation=True,
        ).to(s.device)

        with torch.no_grad():
            output_ids = self._model.generate(
                **inputs,
                max_length=s.max_output_length,
                min_length=s.min_output_length,
                num_beams=s.num_beams,
                no_repeat_ngram_size=2,
                length_penalty=1.0,
                early_stopping=True,
            )

        return self._tokenizer.decode(
            output_ids[0], skip_special_tokens=True
        ).strip()

    def summarize_reviews(self, reviews: list[str]) -> str:
        """راهبرد map-reduce برای چند نظر."""
        clean = [r for r in reviews if r and r.strip()]
        if not clean:
            return ""
        if len(clean) == 1:
            return self.summarize(clean[0])

        # مرحله‌ی map: خلاصه‌ی هر نظر (نظرات خیلی کوتاه را دست‌نخورده نگه می‌داریم)
        partial = [
            r if len(r) < 120 else self.summarize(r) for r in clean
        ]
        # مرحله‌ی reduce: خلاصه‌ی خلاصه‌ها → پیام کوتاه نهایی
        combined = " ".join(partial)
        return self.summarize(combined)


__all__ = ["AbstractiveSummarizer"]
