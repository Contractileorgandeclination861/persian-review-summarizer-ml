"""
رابط کاربری محلی (Streamlit) برای خلاصه‌ساز نظرات فارسی.

اجرا:
    streamlit run app.py

کاملاً آفلاین اجرا می‌شود. با روش «استخراجی» هیچ دانلودی لازم نیست؛ با روش
«تولیدی» بار اول مدل mT5 دانلود می‌شود.
"""

from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import streamlit as st

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from summarizer.aggregator import ReviewAnalyzer  # noqa: E402
from summarizer.config import Settings  # noqa: E402
from summarizer.engine import get_summarizer  # noqa: E402

st.set_page_config(page_title="خلاصه‌ساز نظرات فارسی", page_icon="🧠", layout="centered")

st.markdown(
    """
    <style>
        .stApp { direction: rtl; text-align: right; }
        textarea, input { direction: rtl; text-align: right; }
        h1, h2, h3, p, label, .stMarkdown { text-align: right; }
    </style>
    """,
    unsafe_allow_html=True,
)


@st.cache_resource(show_spinner="در حال آماده‌سازی موتور…")
def _build_analyzer(method: str) -> ReviewAnalyzer:
    settings = Settings(method=method)
    return ReviewAnalyzer(summarizer=get_summarizer(settings), settings=settings)


def main() -> None:
    st.title("🧠 خلاصه‌ساز محلی نظرات فارسی")
    st.caption("یادگیری ماشین و عمیق • کاملاً آفلاین • بدون هیچ API")

    method = st.sidebar.radio(
        "روش خلاصه‌سازی:",
        options=["extractive", "abstractive"],
        format_func=lambda m: {
            "extractive": "🔹 استخراجی (ML · سریع · بدون دانلود)",
            "abstractive": "🧠 تولیدی (Deep Learning · مدل mT5)",
        }[m],
    )
    if method == "abstractive":
        st.sidebar.warning("بار اول مدل mT5 (~۲.۲گ) دانلود می‌شود.")

    st.markdown("نظرات را وارد کن (هر نظر در یک خط) یا فایل CSV آپلود کن:")

    raw = st.text_area(
        "نظرات:",
        height=220,
        placeholder="کیفیت ساخت عالیه ولی قیمتش گرونه.\nباتری خوبی داره.\nپشتیبانی ضعیف بود.",
    )
    uploaded = st.file_uploader("یا فایل CSV (ستون text):", type=["csv"])

    if st.button("📝 خلاصه کن", type="primary"):
        reviews: list[str] = []
        if uploaded is not None:
            df = pd.read_csv(uploaded)
            col = "text" if "text" in df.columns else df.columns[0]
            reviews = [str(t) for t in df[col].dropna().tolist()]
        else:
            reviews = [ln for ln in raw.splitlines() if ln.strip()]

        if not reviews:
            st.warning("حداقل یک نظر وارد کن.")
            return

        analyzer = _build_analyzer(method)
        with st.spinner(f"در حال تحلیل {len(reviews)} نظر…"):
            result = analyzer.analyze(reviews)

        st.subheader("📌 خلاصه‌ی نظرات")
        st.success(result.summary or "_خلاصه‌ای تولید نشد._")

        st.subheader("📊 تحلیل احساسات")
        s = result.sentiment
        stars = "★" * round(result.rating) + "☆" * (5 - round(result.rating))
        st.markdown(f"### {stars}  &nbsp; {result.rating} از ۵")
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("تعداد نظرات", s.total)
        c2.metric("مثبت 😊", s.positive)
        c3.metric("منفی 😞", s.negative)
        c4.metric("خنثی 😐", s.neutral)
        st.info(f"جمع‌بندی احساسات: **{s.label}**")

        if result.keywords:
            st.subheader("🏷️ کلیدواژه‌ها")
            st.markdown(
                " ".join(
                    f"<span style='background:#eef;padding:4px 10px;border-radius:12px;"
                    f"margin:2px;display:inline-block'>{k}</span>"
                    for k in result.keywords
                ),
                unsafe_allow_html=True,
            )

        col_pro, col_con = st.columns(2)
        with col_pro:
            st.subheader("✅ نکات مثبت")
            if result.pros:
                for p in result.pros:
                    st.markdown(f"- {p}")
            else:
                st.caption("—")
        with col_con:
            st.subheader("⚠️ نکات منفی")
            if result.cons:
                for c in result.cons:
                    st.markdown(f"- {c}")
            else:
                st.caption("—")


if __name__ == "__main__":
    main()
