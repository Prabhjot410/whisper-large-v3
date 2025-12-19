#!/bin/bash
docker run --gpus device=1 -it --rm \
  --shm-size=1G \
  -v $(pwd)/model_repository:/opt/tritonserver/model_repository \
  -p 5000:8000 \
  triton_whisper \
  tritonserver --model-repository=/opt/tritonserver/model_repository