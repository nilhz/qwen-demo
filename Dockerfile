# docker build -t qwen .
# docker run -it --gpus all --name qwen qwen

FROM nvidia/cuda:12.2.0-devel-ubuntu22.04

ENV DEBIAN_FRONTEND=noninteractive
# !!! Importante para habilitar o suporte a CUDA no llama-cpp-python
ENV CMAKE_ARGS="-DGGML_CUDA=on"

WORKDIR /app
COPY ./app.py requirements.txt ./
COPY ./LLM ./LLM

RUN apt-get update && apt-get install -y \
    python3 \
    python3-pip \
    python3-dev \
    build-essential \
    cmake \
    git \
    ninja-build \
    && rm -rf /var/lib/apt/lists/*

RUN python3 -m pip install --upgrade pip && \
    python3 -m pip install --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cu122 --no-cache-dir -r requirements.txt

CMD ["python3", "app.py"]