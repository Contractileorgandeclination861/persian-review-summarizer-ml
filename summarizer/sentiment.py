"""
تحلیل احساسات سبک و آفلاین (مبتنی بر واژه‌نامه).

این یک تحلیل heuristic و کاملاً محلی است: تعداد واژه‌های مثبت و منفی هر نظر را
می‌شمارد و برچسب مثبت/منفی/خنثی می‌دهد. هدف افزودن یک «آمار سریع» درباره‌ی
محصول در کنار خلاصه است، نه یک مدل دقیق طبقه‌بندی.

می‌توان بعداً این ماژول را با یک مدل یادگیری‌عمیق فارسی (مثلاً ParsBERT) جایگزین
کرد بدون آنکه بقیه‌ی کد تغییر کند.
"""

from __future__ import annotations

from dataclasses import dataclass

_POSITIVE = {
    # پایه
    "عالی", "خوب", "بی‌نظیر", "بینظیر", "فوق‌العاده", "فوقالعاده",
    "راضی", "دوست", "پیشنهاد", "کیفیت", "سریع", "زیبا", "قشنگ",
    "ارزش", "مناسب", "بهترین", "لذت", "ممنون", "تشکر", "شیک", "محکم",
    "خوشحال", "دقیق", "منصفانه", "ماندگار", "قوی", "پایدار", "راحت", "سالم",
    "واضح", "شفاف", "روشن", "زنده",
    # فرم‌های محاوره‌ای/صرفی رایج
    "عالیه", "خوبه", "خوبی", "راضی‌ام", "راضیام", "محکمی", "سریعه",
    "باکیفیته", "واضحن", "زنده‌ان", "زنده‌ن",
}
_NEGATIVE = {
    # پایه
    "بد", "ضعیف", "افتضاح", "خراب", "مشکل", "ایراد", "گران", "گرون",
    "دیر", "کند", "نارضایتی", "ناراضی", "زشت", "بی‌کیفیت", "بیکیفیت", "شکسته",
    "معیوب", "بدقول", "بی‌ادب", "بیادب", "سرد", "متاسف", "متأسف",
    "پشیمان", "بدترین", "کلاهبرداری", "دروغ", "سنگین",
    # فرم‌های محاوره‌ای/صرفی رایج
    "بده", "ضعیفه", "گرونه", "گرونه‌", "دیره", "کنده", "خرابه",
    "نمی‌ارزه", "نمیارزه", "هنگ",
}

# واژه‌های نفی: اگر کنار یک واژه‌ی مثبت بیایند، آن را منفی می‌کنند («خوب نبود»)
_NEGATIONS = {"نبود", "نیست", "نه", "نداره", "نداشت", "ندارد", "نبوده", "نمی‌کنه", "نمیکنه"}


@dataclass(frozen=True)
class SentimentResult:
    """نتیجه‌ی تحلیل احساسات یک مجموعه نظر."""

    positive: int
    negative: int
    neutral: int
    total: int

    @property
    def positive_ratio(self) -> float:
        return self.positive / self.total if self.total else 0.0

    @property
    def label(self) -> str:
        """برچسب کلی احساسات مجموعه."""
        if self.total == 0:
            return "نامشخص"
        if self.positive > self.negative:
            return "عمدتاً مثبت"
        if self.negative > self.positive:
            return "عمدتاً منفی"
        return "متعادل"


def score_text(text: str) -> int:
    """
    امتیاز یک متن: مثبت (>0)، منفی (<0)، خنثی (=0).

    آگاه به نفی: اگر یک واژه‌ی مثبت در همسایگی یک واژه‌ی نفی باشد (مثل «خوب نبود»)
    به‌عنوان منفی شمرده می‌شود.
    """
    # علائم نگارشی را از دو سرِ هر توکن جدا می‌کنیم تا «نبود،» = «نبود» شناخته شود
    _punct = "،.!?؟:;()[]«»\"'…-ـ"
    tokens = [t.strip(_punct) for t in (text or "").split()]
    tokens = [t for t in tokens if t]
    n = len(tokens)
    pos = neg = 0
    for i, tok in enumerate(tokens):
        negated = (i + 1 < n and tokens[i + 1] in _NEGATIONS) or (
            i > 0 and tokens[i - 1] in _NEGATIONS
        )
        if tok in _POSITIVE:
            neg += 1 if negated else 0
            pos += 0 if negated else 1
        elif tok in _NEGATIVE:
            neg += 1
    return pos - neg


def analyze(reviews: list[str]) -> SentimentResult:
    """تحلیل احساسات یک فهرست نظر."""
    pos = neg = neu = 0
    for r in reviews:
        s = score_text(r)
        if s > 0:
            pos += 1
        elif s < 0:
            neg += 1
        else:
            neu += 1
    return SentimentResult(positive=pos, negative=neg, neutral=neu, total=len(reviews))


__all__ = ["SentimentResult", "analyze", "score_text"]
