FROM pytorch/pytorch:2.5.1-cuda12.4-cudnn9-runtime

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1

WORKDIR /workspace

RUN apt-get update && apt-get install -y --no-install-recommends \
    ffmpeg \
    git \
    gcc \
    && rm -rf /var/lib/apt/lists/*

COPY pyproject.toml README.md ./
COPY mifi ./mifi
COPY g2p ./g2p

RUN pip install --upgrade pip && pip install -e "." \
    && python -c "from misaki.en import G2P; G2P()"

CMD ["python", "-m", "mifi.main"]
