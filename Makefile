.PHONY: help install test lint ui cli-demo

help:  ## نمایش راهنما
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
		awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-12s\033[0m %s\n", $$1, $$2}'

install:  ## نصب وابستگی‌ها
	pip install -r requirements.txt

test:  ## اجرای تست‌ها
	PYTEST_DISABLE_PLUGIN_AUTOLOAD=1 pytest tests/

lint:  ## بررسی کیفیت کد
	ruff check .

ui:  ## اجرای رابط کاربری (Streamlit)
	streamlit run app.py

cli-demo:  ## نمونه‌ی خط فرمان روی داده‌ی نمونه
	python cli.py --file data/sample_reviews.csv --method extractive
