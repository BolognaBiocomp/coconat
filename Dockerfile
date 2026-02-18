# Base Image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /app/coconat

ENV DEBIAN_FRONTEND=noninteractive
ENV PIP_BREAK_SYSTEM_PACKAGES=1 PIP_DISABLE_PIP_VERSION_CHECK=1
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 python3-pip python3-venv python3-dev \
        ca-certificates curl git build-essential && \
    rm -rf /var/lib/apt/lists/* && \
    pip install torch torchvision transformers && \
    pip install --no-cache-dir numpy biopython fair-esm sentencepiece && \
    apt-get -y update && \
    apt-get -y install vim

COPY . .

ENTRYPOINT ["/app/coconat/coconat.py"]
