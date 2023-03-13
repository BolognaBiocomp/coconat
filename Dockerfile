# Base Image
FROM python:3.8-slim-buster

WORKDIR /app/coconat

RUN python -m pip install --upgrade pip && \
    pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir numpy biopython fair-esm tensorflow

COPY . .

ENTRYPOINT ["/bin/bash"]
