FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app/backend \
    MPLCONFIGDIR=/tmp/foresight_mplconfig \
    PORT=8000

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    libglib2.0-0 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir \
      "fastapi>=0.115" \
      "uvicorn[standard]>=0.32" \
      "python-multipart>=0.0.12" \
      "google-genai>=3.6.1" \
      "pymunk>=7.0" \
      "scipy>=1.14" \
      "opencv-contrib-python-headless>=4.10" \
      "numpy>=2.0" \
      "pillow>=11.0" \
      "pydantic>=2.10" \
      "python-dotenv>=1.0" \
      "matplotlib>=3.9" \
      "pyyaml>=6.0"

COPY backend ./backend
COPY phase0/vidio.py ./phase0/vidio.py
COPY eval/results.json eval/chart.png ./eval/

RUN mkdir -p /app/outputs /app/logs /tmp/foresight_mplconfig

EXPOSE 8000

CMD ["sh", "-c", "uvicorn foresight.api:app --app-dir backend --host 0.0.0.0 --port ${PORT:-8000}"]
