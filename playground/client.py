import argparse
import base64
import json
import numpy as np
import tritonclient.http as httpclient


def main(args):
    # -------- Read audio --------
    with open(args.audio_file, "rb") as f:
        audio_bytes = f.read()

    audio_b64 = base64.b64encode(audio_bytes)

    # -------- Triton client --------
    client = httpclient.InferenceServerClient(
        url=args.triton_url, verbose=False
    )

    inputs = []

    def add_str(name, val):
        inp = httpclient.InferInput(name, [1], "BYTES")
        inp.set_data_from_numpy(
            np.array([val.encode("utf-8")], dtype=object)
        )
        inputs.append(inp)

    def add_int(name, val):
        inp = httpclient.InferInput(name, [1], "INT32")
        inp.set_data_from_numpy(np.array([val], dtype=np.int32))
        inputs.append(inp)

    def add_float(name, val):
        inp = httpclient.InferInput(name, [1], "FP32")
        inp.set_data_from_numpy(np.array([val], dtype=np.float32))
        inputs.append(inp)

    # -------- Inputs --------
    inp = httpclient.InferInput("AUDIO_B64", [1], "BYTES")
    inp.set_data_from_numpy(np.array([audio_b64], dtype=object))
    inputs.append(inp)

    add_str("MODE", args.mode)
    add_str("OUTPUT_FORMAT", args.format)

    # Language: send empty only if user provides
    add_str("LANGUAGE", args.language if args.language else "")

    add_int("BEAM_SIZE", args.beam_size)
    add_float("TEMPERATURE", args.temperature)
    add_int("VAD_FILTER", args.vad_filter)

    # -------- Outputs --------
    outputs = [
        httpclient.InferRequestedOutput("OUTPUT"),
        httpclient.InferRequestedOutput("INFERENCE_TIME_SEC"),
    ]

    # -------- Infer --------
    result = client.infer(
        model_name=args.model_name,
        inputs=inputs,
        outputs=outputs,
    )

    output = result.as_numpy("OUTPUT")[0].decode("utf-8")
    t = float(result.as_numpy("INFERENCE_TIME_SEC")[0])

    print(f"\n⏱ Inference Time: {t:.2f} sec\n")

    # -------- Display --------
    if args.mode == "align":
        print(output)
        return

    if args.format == "json":
        print(json.dumps(json.loads(output), indent=2, ensure_ascii=False))
    else:
        print(output)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Triton Whisper Client")

    parser.add_argument("audio_file")

    parser.add_argument("--triton-url", default="localhost:5000")
    parser.add_argument("--model-name", default="whisper_large_v3")

    parser.add_argument(
        "--mode",
        choices=["transcribe", "translate", "align"],
        default="transcribe",
    )

    parser.add_argument(
        "--format",
        choices=["text", "json", "srt", "vtt", "tsv"],
        default="text",
    )

    parser.add_argument("--language", default="")
    parser.add_argument("--beam-size", type=int, default=5)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--vad-filter", type=int, choices=[0, 1], default=0)

    args = parser.parse_args()
    main(args)
