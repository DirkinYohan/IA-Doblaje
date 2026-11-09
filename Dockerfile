# syntax=docker/dockerfile:1

FROM python:3.14-slim AS models
WORKDIR /build
RUN pip install --no-cache-dir --index-url https://download.pytorch.org/whl/cpu torch \
 && pip install --no-cache-dir transformers==5.18.0 sentencepiece==0.2.2 ctranslate2==4.8.2 huggingface-hub==1.33.0
COPY scripts/fetch_models.py scripts/fetch_models.py
RUN python scripts/fetch_models.py --models-dir /build/models

FROM python:3.14-slim
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HOST=0.0.0.0 \
    PORT=10000 \
    FORWARDED_ALLOW_IPS=* \
    IA_LIVE_PRELOAD=1 \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1
RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg libmagic1 \
 && rm -rf /var/lib/apt/lists/*
WORKDIR /srv
COPY requirements-deploy.txt .
RUN pip install --no-cache-dir -r requirements-deploy.txt \
 && pip install --no-cache-dir --no-deps faster-whisper==1.0.3
COPY --from=models /build/models ./models
COPY app ./app
COPY configs ./configs
RUN mkdir -p data/input data/output data/temporary data/mail
EXPOSE 10000
CMD ["python", "-m", "app.presentation.api"]
