# Use a lightweight Python base image
FROM python:3.10-slim

# Install system dependencies and build tools for compiling C extensions
RUN apt-get update && apt-get install -y \
    ffmpeg \
    libsndfile1 \
    git \
    wget \
    build-essential \
    python3-dev \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN python3 -m pip install --upgrade pip

# Install Python packages for audio preprocessing, VAD, and Triton client
RUN pip install --no-cache-dir \
    requests \
    numpy \
    torchaudio \
    webrtcvad \
    tritonclient[all] \
    soundfile \
    resampy

# Set working directory
WORKDIR /client

COPY harvard.wav .

# Copy client script into the container
COPY client.py .

# Ensure Python buffers output immediately (important for logs)
ENV PYTHONUNBUFFERED=1

# Use ENTRYPOINT so args are appended automatically
ENTRYPOINT ["python", "client.py"]
