# syntax=docker/dockerfile:1.6

FROM mambaorg/micromamba:1.5.7-focal

ARG MAMBA_DOCKERFILE_ACTIVATE=1
WORKDIR /workspace

USER root
ENV DEBIAN_FRONTEND=noninteractive
RUN apt-get update && apt-get install -y --no-install-recommends \
    bash \
    build-essential \
    ffmpeg \
    git \
    libgl1 \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
        wget \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

COPY environment.yml /tmp/environment.yml
RUN micromamba create -y -n depthedit -f /tmp/environment.yml && \
    micromamba clean -a -y

ENV PYTHONPATH=/workspace

COPY . .

RUN chmod +x docker/run_gradio.sh

ENTRYPOINT ["micromamba", "run", "-n", "depthedit"]
CMD ["bash", "docker/run_gradio.sh"]
