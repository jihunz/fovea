FROM python:3.11-slim

WORKDIR /app

# System libs for Pillow/OpenCV (ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./
RUN pip install --no-cache-dir -r requirements.txt

COPY . ./

ENV PYTHONUNBUFFERED=1 \
    FOVEA_DATA_DIR=/app/data \
    FOVEA_MODEL_DIR=/app/model

EXPOSE 8000
CMD ["uvicorn", "fovea.main:app", "--host", "0.0.0.0", "--port", "8000"]
