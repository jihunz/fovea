FROM python:3.11-slim

ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# System libs for Pillow / OpenCV (ultralytics)
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 libglib2.0-0 libsm6 libxext6 libxrender1 \
    && rm -rf /var/lib/apt/lists/*

# INSTALL_AI=0 builds a small image without numpy/torch/ultralytics/cloud SDKs (labeling, review and export
# still work). torch comes from the CPU wheel index: the default CUDA wheels add several GB for no use here.
ARG INSTALL_AI=1
ARG TORCH_INDEX_URL=https://download.pytorch.org/whl/cpu
COPY requirements.txt ./
RUN if [ "$INSTALL_AI" = "0" ]; then \
        sed -n '1,/^# Optional — AI/p' requirements.txt > /tmp/req.txt; \
    else \
        cp requirements.txt /tmp/req.txt; \
    fi \
    && pip install --extra-index-url "$TORCH_INDEX_URL" -r /tmp/req.txt \
    && rm /tmp/req.txt

# Run as an unprivileged user. docker-compose maps it to your own uid/gid so label files and exports
# written to bind-mounted folders stay owned by you (not root).
RUN useradd --create-home --uid 1000 fovea \
    && mkdir -p /app/data /app/model && chown -R fovea:fovea /app
COPY --chown=fovea:fovea . ./
USER fovea

ENV FOVEA_DATA_DIR=/app/data \
    FOVEA_MODEL_DIR=/app/model \
    YOLO_CONFIG_DIR=/app/data/ultralytics \
    XDG_CACHE_HOME=/app/data/.cache \
    HOME=/home/fovea

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=30s --retries=3 \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://127.0.0.1:8000/api/plugins', timeout=4)" || exit 1
# python -m fovea: cancels running jobs on stop and shuts down within a few seconds (docker stop waits 10 s).
CMD ["python", "-m", "fovea", "--host", "0.0.0.0", "--port", "8000"]
