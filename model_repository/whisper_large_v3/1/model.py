import os
import json
import base64
import tempfile
import time
import numpy as np
import triton_python_backend_utils as pb_utils
from faster_whisper import WhisperModel

FW = None


# ---------------- helpers ----------------

def _get_first(req, name):
    t = pb_utils.get_input_tensor_by_name(req, name)
    return None if t is None else t.as_numpy()[0]


def _b(x):
    return x.decode("utf-8") if x is not None else ""


def _i(x, d=0):
    try:
        return int(x)
    except:
        return d


def _f(x, d=0.0):
    try:
        return float(x)
    except:
        return d


def _write_audio_temp(audio_bytes):
    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=".wav", dir="/dev/shm")
    tmp.write(audio_bytes)
    tmp.close()
    return tmp.name


def _to_srt(segments):
    from datetime import timedelta
    out = []
    for i, s in enumerate(segments, 1):
        st = str(timedelta(seconds=s["start"]))[:-3]
        et = str(timedelta(seconds=s["end"]))[:-3]
        out.append(f"{i}\n{st} --> {et}\n{s['text']}\n")
    return "\n".join(out)


def _to_vtt(segments):
    from datetime import timedelta
    out = ["WEBVTT\n"]
    for s in segments:
        st = str(timedelta(seconds=s["start"]))[:-3]
        et = str(timedelta(seconds=s["end"]))[:-3]
        out.append(f"{st} --> {et}\n{s['text']}\n")
    return "\n".join(out)


def _to_tsv(segments):
    lines = ["start\tend\ttext"]
    for s in segments:
        lines.append(
            f"{int(s['start']*1000)}\t{int(s['end']*1000)}\t{s['text'].strip()}"
        )
    return "\n".join(lines)


# ---------------- triton model ----------------

class TritonPythonModel:

    def initialize(self, args):
        global FW
        import torch

        model = os.environ.get("WHISPER_MODEL", "large-v3")
        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute = "float16" if device == "cuda" else "int8"

        FW = WhisperModel(model, device=device, compute_type=compute)
        print("Loaded Faster-Whisper:", model)

    def execute(self, requests):
        responses = []

        for req in requests:
            audio = base64.b64decode(_get_first(req, "AUDIO_B64"))
            mode = _b(_get_first(req, "MODE")) or "transcribe"
            fmt = _b(_get_first(req, "OUTPUT_FORMAT")) or "text"
            lang = _b(_get_first(req, "LANGUAGE")) or None

            beam = _i(_get_first(req, "BEAM_SIZE"), 5)
            temp = _f(_get_first(req, "TEMPERATURE"), 0.0)
            vad = _i(_get_first(req, "VAD_FILTER"), 0) == 1

            path = _write_audio_temp(audio)

            start = time.perf_counter()
            seg_iter, info = FW.transcribe(
                path,
                task="translate" if mode == "translate" else "transcribe",
                language=lang,
                beam_size=beam,
                temperature=temp,
                vad_filter=vad,
            )
            segs = list(seg_iter)
            elapsed = time.perf_counter() - start

            segments = [
                {"start": s.start, "end": s.end, "text": s.text}
                for s in segs
            ]

            # ---------- ALIGN MODE (TSV ONLY) ----------
            if mode == "align":
                output = _to_tsv(segments)
                responses.append(
                    pb_utils.InferenceResponse(
                        output_tensors=[
                            pb_utils.Tensor("OUTPUT", np.array([output.encode()], object)),
                            pb_utils.Tensor("INFERENCE_TIME_SEC", np.array([elapsed], np.float32)),
                        ]
                    )
                )
                continue

            # ---------- TRANSCRIBE / TRANSLATE ----------
            if fmt == "json":
                payload = {
                    "text": "".join(s["text"] for s in segments),
                    "segments": segments,
                    "language": info.language,
                }
                output = json.dumps(payload)

            elif fmt == "srt":
                output = _to_srt(segments)

            elif fmt == "vtt":
                output = _to_vtt(segments)

            else:
                output = "".join(s["text"] for s in segments)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[
                        pb_utils.Tensor("OUTPUT", np.array([output.encode()], object)),
                        pb_utils.Tensor("INFERENCE_TIME_SEC", np.array([elapsed], np.float32)),
                    ]
                )
            )

        return responses
