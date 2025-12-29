# تغییر از 3.10 به 3.12
FROM python:3.12-slim

# بقیه دستورات ثابت بماند...
WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# نصب مستقیم نسخه پایدار
RUN pip install --no-cache-dir pandas_ta

COPY . .

CMD ["python", "main.py"]
