FROM pytorch/pytorch:2.1.0-cuda11.8-cudnn8-runtime
ENV DEBIAN_FRONTEND noninteractive
RUN apt-get update -y && \
    apt-get install -y --no-install-recommends \
    build-essential \
    curl \
    unzip \
    wget \
    git \
    && \
    apt-get clean && \
    rm -rf /var/lib/apt/lists/*
COPY requirements.txt requirements.txt
RUN python3 -m pip install --no-cache-dir --upgrade pip &&  pip install -r requirements.txt
WORKDIR /workspace
CMD ["python3"]
