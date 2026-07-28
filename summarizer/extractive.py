"""
موتور خلاصه‌سازی استخراجی — الگوریتم TextRank (یادگیری ماشین بدون‌ناظر).

روش کار:
  ۱. متن به جملات شکسته می‌شود.
  ۲. هر جمله با TF-IDF به بردار تبدیل می‌شود (نماینده‌ی اهمیت واژه‌ها).
  ۳. شباهت کسینوسی بین همه‌ی جفت‌جملات، یک گراف می‌سازد.
  ۴. الگوریتم PageRank (همان ایده‌ی گوگل) مهم‌ترین جملات را رتبه‌بندی می‌کند.
  ۵. چند جمله‌ی برتر، به ترتیب اصلی، به‌عنوان خلاصه برگردانده می‌شوند.

این روش کاملاً آفلاین است، هیچ مدلی دانلود نمی‌کند و بسیار سریع است.
"""

from __future__ import annotations

import numpy as np

from summarizer.base import BaseSummarizer
from summarizer.config import Settings, get_settings

# فهرست کوتاه واژه‌های ایست فارسی (برای بهبود TF-IDF)
_PERSIAN_STOPWORDS = {
    "و", "در", "به", "از", "که", "این", "را", "با", "است", "برای", "آن",
    "یک", "خود", "تا", "کرد", "بر", "هم", "نیز", "می", "شد", "ما", "یا",
    "اما", "ولی", "هر", "همه", "بود", "شده", "دارد", "کنم", "کنید", "خیلی",
    "هست", "من", "تو", "او", "ها", "های", "یه", "رو",
}


def _tokenize(text: str) -> list[str]:
    """توکن‌سازی ساده‌ی سازگار با فارسی برای TF-IDF."""
    return [w for w in text.split() if w not in _PERSIAN_STOPWORDS and len(w) > 1]


class ExtractiveSummarizer(BaseSummarizer):
    """خلاصه‌ساز استخراجی مبتنی بر TextRank."""

    name = "extractive"

    def __init__(self, settings: Settings | None = None) -> None:
        self._settings = settings or get_settings()

    def summarize(self, text: str, num_sentences: int | None = None) -> str:
        sentences = self._sentences(text)
        n = num_sentences or self._settings.summary_sentences

        # اگر متن به‌اندازه‌ی کافی جمله ندارد، خودش را برگردان
        if len(sentences) <= n:
            return " ".join(sentences)

        ranked_idx = self._rank_sentences(sentences)
        top_idx = sorted(ranked_idx[:n])  # حفظ ترتیب اصلی برای خوانایی
        return " ".join(sentences[i] for i in top_idx)

    def summarize_reviews(self, reviews: list[str]) -> str:
        """
        خلاصه‌ی تجمیعی نظرات: همه‌ی جملات همه‌ی نظرات را کنار هم می‌گذارد و
        مهم‌ترین‌ها را با TextRank انتخاب می‌کند.
        """
        all_sentences: list[str] = []
        for r in reviews:
            all_sentences.extend(self._sentences(r))

        n = self._settings.summary_sentences
        if len(all_sentences) <= n:
            return " ".join(all_sentences)

        ranked_idx = self._rank_sentences(all_sentences)
        top_idx = sorted(ranked_idx[:n])
        return " ".join(all_sentences[i] for i in top_idx)

    # ------------------------------------------------------------- internal
    def _rank_sentences(self, sentences: list[str]) -> list[int]:
        """اندیس جملات را از مهم به کم‌اهمیت با TextRank برگردان."""
        from sklearn.feature_extraction.text import TfidfVectorizer
        from sklearn.metrics.pairwise import cosine_similarity
        import networkx as nx

        vectorizer = TfidfVectorizer(tokenizer=_tokenize, token_pattern=None)
        try:
            tfidf = vectorizer.fit_transform(sentences)
        except ValueError:
            # اگر هیچ واژه‌ی معناداری نماند، به ترتیب طول جمله برگرد
            return sorted(
                range(len(sentences)),
                key=lambda i: len(sentences[i]),
                reverse=True,
            )

        sim = cosine_similarity(tfidf)
        np.fill_diagonal(sim, 0.0)

        graph = nx.from_numpy_array(sim)
        try:
            scores = nx.pagerank(graph, max_iter=200)
        except nx.PowerIterationFailedConvergence:  # pragma: no cover
            scores = {i: float(sim[i].sum()) for i in range(len(sentences))}

        return sorted(scores, key=scores.get, reverse=True)


__all__ = ["ExtractiveSummarizer"]
