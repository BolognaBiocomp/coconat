# Base Image
FROM nvidia/cuda:12.8.0-base-ubuntu24.04

WORKDIR /app/coconat

RUN python -m pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy biopython fair-esm transformers[torch]==4.31.0 sentencepiece && \
    apt-get -y update && \
    apt-get -y install vim

COPY . .

ENTRYPOINT ["/app/coconat/coconat.py"]
