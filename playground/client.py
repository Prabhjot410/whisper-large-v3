import sys
import json
import numpy as np
import tritonclient.http as httpclient
import argparse
import base64


def main(args):
    # ---------------- Load audio ----------------
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes)

    # ---------------- Triton client ----------------
    triton_client = httpclient.InferenceServerClient(
        url=args.triton_url, verbose=False
    )

    inputs = []

    # AUDIO_B64
    inp = httpclient.InferInput("AUDIO_B64", [1], "BYTES")
    inp.set_data_from_numpy(np.array([audio_b64], dtype=object))
    inputs.append(inp)

    # MODE
    inp = httpclient.InferInput("MODE", [1], "BYTES")
    inp.set_data_from_numpy(
        np.array([args.mode.encode("utf-8")], dtype=object)
    )
    inputs.append(inp)

    # LANGUAGE
    inp = httpclient.InferInput("LANGUAGE", [1], "BYTES")
    inp.set_data_from_numpy(
        np.array(
            [args.language.encode("utf-8") if args.language else b""],
            dtype=object,
        )
    )
    inputs.append(inp)

    # BEAM_SIZE
    inp = httpclient.InferInput("BEAM_SIZE", [1], "INT32")
    inp.set_data_from_numpy(np.array([args.beam_size], dtype=np.int32))
    inputs.append(inp)

    # TEMPERATURE
    inp = httpclient.InferInput("TEMPERATURE", [1], "FP32")
    inp.set_data_from_numpy(np.array([args.temperature], dtype=np.float32))
    inputs.append(inp)

    # VAD_FILTER
    inp = httpclient.InferInput("VAD_FILTER", [1], "INT32")
    inp.set_data_from_numpy(np.array([args.vad_filter], dtype=np.int32))
    inputs.append(inp)

    # WORD_TIMESTAMPS
    inp = httpclient.InferInput("WORD_TIMESTAMPS", [1], "INT32")
    inp.set_data_from_numpy(np.array([args.word_timestamps], dtype=np.int32))
    inputs.append(inp)

    # RETURN_SRT
    inp = httpclient.InferInput("RETURN_SRT", [1], "INT32")
    inp.set_data_from_numpy(np.array([args.return_srt], dtype=np.int32))
    inputs.append(inp)

    # RETURN_VTT
    inp = httpclient.InferInput("RETURN_VTT", [1], "INT32")
    inp.set_data_from_numpy(np.array([args.return_vtt], dtype=np.int32))
    inputs.append(inp)

    # ---------------- Requested outputs ----------------
    outputs = [
        httpclient.InferRequestedOutput("TEXT"),
        httpclient.InferRequestedOutput("SEGMENTS"),
        httpclient.InferRequestedOutput("SRT"),
        httpclient.InferRequestedOutput("VTT"),
        httpclient.InferRequestedOutput("INFERENCE_TIME_SEC"),
    ]

    # ---------------- Inference ----------------
    results = triton_client.infer(
        model_name=args.model_name,
        inputs=inputs,
        outputs=outputs,
    )

    # ---------------- Decode outputs ----------------
    text = results.as_numpy("TEXT")[0].decode("utf-8")
    segments = json.loads(results.as_numpy("SEGMENTS")[0].decode("utf-8"))
    srt = results.as_numpy("SRT")[0].decode("utf-8")
    vtt = results.as_numpy("VTT")[0].decode("utf-8")
    time_ms = results.as_numpy("INFERENCE_TIME_SEC")[0]
    time_arr = results.as_numpy("INFERENCE_TIME_SEC")

    print("\n=== MODE:", args.mode, "===")

    if time_arr is None:
        print("❌ INFERENCE_TIME_Sec not returned by Triton")
    else:
        time_ms = float(time_arr[0][0])
        print(f"Inference time: {time_ms:.2f} sec")



    print("\n=== TEXT ===\n", text)

    if segments:
        print("\n=== SEGMENTS ===")
        for seg in segments:
            print(f"[{seg['start']:.2f} - {seg['end']:.2f}] {seg['text']}")

    if args.return_srt and srt.strip():
        print("\n=== SRT ===\n", srt)

    if args.return_vtt and vtt.strip():
        print("\n=== VTT ===\n", vtt)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Triton Whisper HTTP Client")

    parser.add_argument("audio_file")
    parser.add_argument("--triton-url", default="localhost:8000")
    parser.add_argument("--model-name", default="whisper_large_v3")

    parser.add_argument(
        "--mode",
        choices=["transcribe", "translate", "align"],
        default="transcribe",
    )

    parser.add_argument("--language", default="")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vad-filter", type=int, choices=[0, 1], default=0)
    parser.add_argument("--word-timestamps", type=int, choices=[0, 1], default=1)
    parser.add_argument("--return-srt", type=int, choices=[0, 1], default=1)
    parser.add_argument("--return-vtt", type=int, choices=[0, 1], default=1)

    args = parser.parse_args()
    main(args)
