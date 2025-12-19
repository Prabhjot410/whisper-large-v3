import os
import json
import base64
import tempfile
import time

import numpy as np
import triton_python_backend_utils as pb_utils
from faster_whisper import WhisperModel

FW = None
USE_FASTER = True


# ---------------- helpers ----------------

def _get_first(request, name):
    t = pb_utils.get_input_tensor_by_name(request, name)
    return None if t is None else t.as_numpy()[0]


def _i(x, d=0):
    try:
        return int(x) if x is not None else d
    except Exception:
        return d


def _f(x, d=0.0):
    try:
        return float(x) if x is not None else d
    except Exception:
        return d


def _b(x):
    try:
        return x.decode("utf-8") if x is not None else None
    except Exception:
        return None


def _write_audio_temp(audio_bytes: bytes) -> str:
    h = audio_bytes[:12]
    if h.startswith(b"RIFF"):
        ext = ".wav"
    elif h.startswith(b"fLaC"):
        ext = ".flac"
    elif h[:3] == b"ID3" or h[:2] == b"\xff\xfb":
        ext = ".mp3"
    elif h[4:8] == b"ftyp":
        ext = ".m4a"
    else:
        ext = ".audio"

    tmp = tempfile.NamedTemporaryFile(delete=False, suffix=ext, dir="/dev/shm")
    tmp.write(audio_bytes)
    tmp.flush()
    tmp.close()
    return tmp.name


def _to_srt(segments):
    from datetime import timedelta
    lines = []
    for i, s in enumerate(segments, 1):
        start = str(timedelta(seconds=float(s["start"])))[:-3]
        end = str(timedelta(seconds=float(s["end"])))[:-3]
        lines.append(f"{i}\n{start} --> {end}\n{s['text']}\n")
    return "\n".join(lines)


def _to_vtt(segments):
    from datetime import timedelta
    lines = ["WEBVTT\n"]
    for s in segments:
        start = str(timedelta(seconds=float(s["start"])))[:-3]
        end = str(timedelta(seconds=float(s["end"])))[:-3]
        lines.append(f"{start} --> {end}\n{s['text']}\n")
    return "\n".join(lines)


# ---------------- whisper runner ----------------

def _run_faster_whisper(FW, path, task, lang, beam, temp, vad, word_ts):
    seg_iter, _ = FW.transcribe(
        path,
        task=task,
        language=lang,
        beam_size=beam,
        temperature=temp,
        vad_filter=vad,
        word_timestamps=word_ts,
    )
    segs = list(seg_iter)
    text = "".join(s.text for s in segs)
    segments = [{"start": s.start, "end": s.end, "text": s.text} for s in segs]
    return text, segments


# ---------------- triton model ----------------

class TritonPythonModel:

    def initialize(self, args):
        global FW, USE_FASTER
        import torch

        cfg = json.loads(args["model_config"])
        params = cfg.get("parameters", {})

        self.model_name = os.environ.get(
            "WHISPER_MODEL",
            params.get("WHISPER_MODEL", {}).get("string_value", "large-v3"),
        )

        device = "cuda" if torch.cuda.is_available() else "cpu"
        compute_type = "float16" if device == "cuda" else "int8"

        print("=== Whisper Init ===")
        print("Model:", self.model_name)
        print("Device:", device)

        try:
            FW = WhisperModel(
                self.model_name,
                device=device,
                compute_type=compute_type,
            )
            print("Loaded Faster-Whisper")
        except Exception as e:
            USE_FASTER = False
            raise RuntimeError(f"Failed to load Faster-Whisper: {e}")

    # ---------------- main execution ----------------

    def execute(self, requests):
        responses = []

        for req in requests:
            a64 = _get_first(req, "AUDIO_B64")
            if a64 is None:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError("AUDIO_B64 is required")
                    )
                )
                continue

            try:
                audio_bytes = base64.b64decode(a64, validate=True)
            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(f"Invalid AUDIO_B64: {e}")
                    )
                )
                continue

            mode = _b(_get_first(req, "MODE")) or "transcribe"

            # ---------- ALIGN MODE ----------
            if mode == "align":
                out = [
                    pb_utils.Tensor("TEXT", np.array([b""], dtype=object)),
                    pb_utils.Tensor(
                        "SEGMENTS",
                        np.array(
                            [json.dumps({"message": "Alignment not implemented"}).encode()],
                            dtype=object,
                        ),
                    ),
                    pb_utils.Tensor("SRT", np.array([b""], dtype=object)),
                    pb_utils.Tensor("VTT", np.array([b""], dtype=object)),
                    pb_utils.Tensor(
                        "INFERENCE_TIME_MS",
                        np.array([0.0], dtype=np.float32),
                    ),
                ]
                responses.append(pb_utils.InferenceResponse(output_tensors=out))
                continue

            # ---------- WHISPER MODE ----------
            task = mode  # transcribe | translate
            lang = _b(_get_first(req, "LANGUAGE")) if task == "transcribe" else None

            beam = _i(_get_first(req, "BEAM_SIZE"), 5)
            temp = _f(_get_first(req, "TEMPERATURE"), 0.0)
            vad = _i(_get_first(req, "VAD_FILTER"), 0) == 1
            word_ts = _i(_get_first(req, "WORD_TIMESTAMPS"), 0) == 1
            ret_srt = _i(_get_first(req, "RETURN_SRT"), 0) == 1
            ret_vtt = _i(_get_first(req, "RETURN_VTT"), 0) == 1

            path = _write_audio_temp(audio_bytes)

            try:
                # 🔥 Measure inference time
                start = time.perf_counter()

                text, segments = _run_faster_whisper(
                    FW, path, task, lang, beam, temp, vad, word_ts
                )
                ## for miliseconds
                # end = time.perf_counter()
                # inference_time_ms = (end - start) * 1000.0

                # for seconds
                end = time.perf_counter()
                inference_time_sec = end - start

                print(f"[DEBUG] Inference time (ms): {inference_time_sec}")

                srt = _to_srt(segments) if ret_srt else ""
                vtt = _to_vtt(segments) if ret_vtt else ""

                out = [
                    pb_utils.Tensor("TEXT", np.array([text.encode()], dtype=object)),
                    pb_utils.Tensor(
                        "SEGMENTS",
                        np.array([json.dumps(segments).encode()], dtype=object),
                    ),
                    pb_utils.Tensor("SRT", np.array([srt.encode()], dtype=object)),
                    pb_utils.Tensor("VTT", np.array([vtt.encode()], dtype=object)),
                    pb_utils.Tensor(
                        "INFERENCE_TIME_SEC",
                        np.array([[inference_time_sec]], dtype=np.float32),
                    )

                    ## for milisconds
                    # pb_utils.Tensor(
                    #     "INFERENCE_TIME_MS",
                    #     np.array([[inference_time_ms]], dtype=np.float32),
                    # ),
                ]

                responses.append(pb_utils.InferenceResponse(output_tensors=out))

            except Exception as e:
                responses.append(
                    pb_utils.InferenceResponse(
                        error=pb_utils.TritonError(str(e))
                    )
                )

        return responses

    def finalize(self):
        print("Finalizing Whisper backend")
