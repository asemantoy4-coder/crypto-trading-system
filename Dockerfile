# استفاده از نسخه سبک پایتون
FROM python:3.10-slim

# تنظیم پوشه کاری
WORKDIR /app

# نصب ابزارهای مورد نیاز سیستم برای پانداز
RUN apt-get update && apt-get install -y \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# کپی کردن فایل نیازمندی‌ها
COPY requirements.txt .

# نصب کتابخانه‌ها
RUN pip install --no-cache-dir -r requirements.txt

# کپی کردن کل کدها به سرور
COPY . .

# اجرای برنامه
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
