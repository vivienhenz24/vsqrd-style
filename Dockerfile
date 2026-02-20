FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    NLTK_DATA=/usr/local/share/nltk_data

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY embellie ./embellie

RUN pip install --upgrade pip && pip install -e ".[tn]" \
    && python -c "import nltk; nltk.download('averaged_perceptron_tagger_eng', download_dir='/usr/local/share/nltk_data')"

CMD ["python", "-m", "embellie.main"]
