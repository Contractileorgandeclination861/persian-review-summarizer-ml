"""
خط فرمان (CLI) خلاصه‌ساز نظرات فارسی.

نمونه‌ها:
    # از فایل CSV (ستون text)
    python cli.py --file data/sample_reviews.csv

    # چند نظر مستقیم
    python cli.py --reviews "کیفیت عالیه" "قیمت گرونه" "پشتیبانی ضعیف بود"

    # با موتور یادگیری عمیق
    python cli.py --file data/sample_reviews.csv --method abstractive
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

from summarizer.aggregator import ReviewAnalyzer  # noqa: E402
from summarizer.config import Settings  # noqa: E402
from summarizer.engine import get_summarizer  # noqa: E402


def _load_reviews(args: argparse.Namespace) -> list[str]:
    if args.reviews:
        return args.reviews
    if args.file:
        import pandas as pd

        df = pd.read_csv(args.file)
        col = args.column if args.column in df.columns else df.columns[0]
        return [str(t) for t in df[col].dropna().tolist()]
    raise SystemExit("یکی از --file یا --reviews را بده.")


def main() -> None:
    parser = argparse.ArgumentParser(description="خلاصه‌ساز محلی نظرات فارسی")
    parser.add_argument("--file", "-f", help="مسیر فایل CSV")
    parser.add_argument("--column", "-c", default="text", help="نام ستون نظرات")
    parser.add_argument("--reviews", "-r", nargs="+", help="نظرات به‌صورت مستقیم")
    parser.add_argument(
        "--method",
        "-m",
        default="extractive",
        choices=["extractive", "abstractive"],
        help="روش خلاصه‌سازی (پیش‌فرض: extractive)",
    )
    args = parser.parse_args()

    reviews = _load_reviews(args)
    settings = Settings(method=args.method)
    analyzer = ReviewAnalyzer(summarizer=get_summarizer(settings), settings=settings)

    print(f"\n⏳ در حال تحلیل {len(reviews)} نظر با موتور «{args.method}»…\n")
    result = analyzer.analyze(reviews)

    s = result.sentiment
    stars = "★" * round(result.rating) + "☆" * (5 - round(result.rating))

    print("═" * 62)
    print("📌 خلاصه‌ی نظرات:")
    print(f"   {result.summary}")
    print("─" * 62)
    print(f"⭐ امتیاز تخمینی: {stars}  ({result.rating} از ۵)")
    print(
        f"📊 احساسات: مثبت={s.positive} | منفی={s.negative} | خنثی={s.neutral}"
        f"  →  {s.label}"
    )
    if result.keywords:
        print(f"🏷️  کلیدواژه‌ها: {'، '.join(result.keywords)}")
    if result.pros:
        print("✅ نکات مثبت:")
        for p in result.pros:
            print(f"   + {p}")
    if result.cons:
        print("⚠️  نکات منفی:")
        for c in result.cons:
            print(f"   - {c}")
    print("═" * 62)


if __name__ == "__main__":
    main()
