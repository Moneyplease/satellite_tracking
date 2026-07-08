FROM python:3.10-slim

WORKDIR /app

# ติดตั้งคอมไพเลอร์ที่จำเป็นสำหรับพวกไลบรารีดาราศาสตร์ (ถ้ามี)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /lib/apt/lists/*

# ก๊อปปี้ไฟล์ requirements และติดตั้ง
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# ก๊อปปี้โค้ดทั้งหมดของ backend เข้าไป
COPY . .

EXPOSE 8000

CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8000"]