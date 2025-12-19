FROM nvcr.io/nvidia/tritonserver:24.05-py3

RUN apt-get update && apt-get install -y python3 python3-pip ffmpeg

RUN pip install --no-cache-dir --upgrade pip \
    && pip install --no-cache-dir \
        transformers==4.38.2 \
        torch==2.2.1 \
        accelerate==0.22.0 \
        torchaudio \
        faster-whisper==1.0.3 \
        librosa==0.10.1 \
        numpy==1.26.4 \
        soundfile==0.12.1 \
        openai-whisper==20231117 \
        resampy 