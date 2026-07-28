"""
استخراج بینش (Insights) از مجموعه‌ی نظرات.

قابلیت‌های اینجا، خروجی را از یک «خلاصه‌ی ساده» به یک «گزارش تحلیلی» ارتقا می‌دهند:

  • extract_keywords  — مهم‌ترین کلیدواژه‌ها با TF-IDF
  • extract_pros_cons — تفکیک جملات به نکات مثبت (Pros) و منفی (Cons)
  • estimate_rating   — تخمین امتیاز ستاره‌ای (۱ تا ۵) از روی احساسات

همه کاملاً محلی و بدون دانلود مدل کار می‌کنند.
"""

from __future__ import annotations

import numpy as np

from summarizer.extractive import _tokenize
from summarizer.preprocess import split_sentences
from summarizer.sentiment import score_text


def extract_keywords(reviews: list[str], top_n: int = 8) -> list[str]:
    """مهم‌ترین کلیدواژه‌های مجموعه‌ی نظرات را با TF-IDF برگردان."""
    from sklearn.feature_extraction.text import TfidfVectorizer

    docs = [r for r in reviews if r and r.strip()]
    if not docs:
        return []

    vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None)
    try:
        matrix = vectorizer.fit_transform(docs)
    except ValueError:
        return []

    scores = np.asarray(matrix.sum(axis=0)).ravel()
    terms = vectorizer.get_feature_names_out()
    top_idx = scores.argsort()[::-1][:top_n]
    return [str(terms[i]) for i in top_idx if scores[i] > 0]


def extract_pros_cons(
    reviews: list[str], max_each: int = 5
) -> tuple[list[str], list[str]]:
    """
    جملات را بر اساس احساسات به نکات مثبت و منفی تفکیک کن.

    از تحلیل احساسات آگاه به نفی در سطح «جمله» استفاده می‌کند.
    """
    pros: list[str] = []
    cons: list[str] = []
    seen_pos: set[str] = set()
    seen_neg: set[str] = set()

    for review in reviews:
        for sentence in split_sentences(review):
            key = sentence.strip()
            if not key:
                continue
            score = score_text(sentence)
            if score > 0 and key not in seen_pos:
                pros.append(key)
                seen_pos.add(key)
            elif score < 0 and key not in seen_neg:
                cons.append(key)
                seen_neg.add(key)

    return pros[:max_each], cons[:max_each]


def estimate_rating(positive: int, negative: int, neutral: int) -> float:
    """
    امتیاز ستاره‌ای تخمینی (۱.۰ تا ۵.۰) از روی نسبت احساسات.

    وزن‌دهی: هر نظر مثبت=۱، خنثی=۰.۵، منفی=۰.
    """
    total = positive + negative + neutral
    if total == 0:
        return 0.0
    ratio = (positive * 1.0 + neutral * 0.5) / total
    return round(1.0 + 4.0 * ratio, 1)


__all__ = ["extract_keywords", "extract_pros_cons", "estimate_rating"]
