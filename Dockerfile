FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

# ابتدا کتابخانه‌های اصلی را نصب می‌کنیم
RUN pip install --no-cache-dir -r requirements.txt

# حالا pandas-ta را با دستور مستقیم نصب می‌کنیم
RUN pip install --no-cache-dir pandas_ta

COPY . .

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]
