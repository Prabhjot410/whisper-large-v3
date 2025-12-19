#!/usr/bin/env bash
set -e

IMAGE_NAME="whisper-triton-client"
TRITON_URL="localhost:5000"
MODEL_NAME="whisper_large_v3"

AUDIO_FILE="hindi.wav"

if [ ! -f "$AUDIO_FILE" ]; then
  echo "❌ Audio file not found: $AUDIO_FILE"
  exit 1
fi

AUDIO_ABS=$(realpath "$AUDIO_FILE")
AUDIO_DIR=$(dirname "$AUDIO_ABS")
AUDIO_NAME=$(basename "$AUDIO_ABS")

echo "🎧 Audio  : $AUDIO_ABS"
echo "🚀 Triton : $TRITON_URL"
echo "🧠 Model  : $MODEL_NAME"
echo "-----------------------------------"

docker run --rm \
  --network host \
  -v "$AUDIO_DIR:/audio:ro" \
  "$IMAGE_NAME" \
  "/audio/$AUDIO_NAME" \
  --triton-url "$TRITON_URL" \
  --model-name "$MODEL_NAME" \
  --mode align \
  --format srt \
  --beam 3 \
  --temp 0.1 \
  --vad 0

