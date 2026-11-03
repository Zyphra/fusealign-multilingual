"""Compare test-model-forward output with the Python package (or with saved predictions).

    uv run --with soundfile python ggml/tests/model/model_parity.py --cpp CPP.jsonl --inputs INPUTS.jsonl \
        [--checkpoint SAFETENSORS] [--device cpu|cuda] [--save-python PY.jsonl]
    uv run python ggml/tests/model/model_parity.py --cpp CPP.jsonl --reference REF.jsonl

Only the clips in the C++ output are compared (it skips long-form and under-one-patch clips). With --inputs, the
reference is fusealign_hub.load_aligner(checkpoint, device, batch_size=1).intervals_batch on those rows; with
--reference, a JSONL of {"clip_id", "intervals"[, "presence"]} rows (e.g. a file written by --save-python).
Reports % identical word intervals, placed/unplaced agreement, % of edges within 1/5/20 ms among words placed on
both sides, and the max |presence difference| when both sides carry presence.
"""

import argparse
import json
from pathlib import Path
import sys


def read_jsonl(path: Path) -> list[dict]:
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def python_reference(cpp_ids: set[str], inputs: Path, checkpoint: str | None, device: str | None) -> dict[str, dict]:
    import soundfile
    import torch

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from fusealign_hub import load_aligner

    rows = [row for row in read_jsonl(inputs) if row["clip_id"] in cpp_ids]
    aligner = load_aligner(checkpoint=checkpoint, device=device, batch_size=1)
    wavs = []
    for row in rows:
        audio, rate = soundfile.read(row["audio"], dtype="float32")
        if rate != 16000 or audio.ndim != 1:
            sys.exit(f"{row['audio']}: expected 16 kHz mono")
        wavs.append(torch.from_numpy(audio))
    results = aligner.intervals_batch(wavs, [row["words"] for row in rows])
    return {
        row["clip_id"]: {"clip_id": row["clip_id"], "intervals": [list(iv) if iv else None for iv in intervals],
                         "presence": presence}
        for row, (intervals, presence) in zip(rows, results)
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--cpp", type=Path, required=True, help="test-model-forward output")
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--inputs", type=Path, help="input rows; run the Python package on them")
    source.add_argument("--reference", type=Path, help="stored predictions to compare with")
    parser.add_argument("--checkpoint", help="fusealign-hub/1 safetensors (default: download from Zyphra/fusealign)")
    parser.add_argument("--device", help="Python device (default: cuda if available)")
    parser.add_argument("--save-python", type=Path, help="write the Python predictions here")
    args = parser.parse_args()

    cpp = {row["clip_id"]: row for row in read_jsonl(args.cpp)}
    if args.inputs:
        ref = python_reference(set(cpp), args.inputs, args.checkpoint, args.device)
        if args.save_python:
            with open(args.save_python, "w") as f:
                for row in ref.values():
                    f.write(json.dumps(row) + "\n")
    else:
        ref = {row["clip_id"]: row for row in read_jsonl(args.reference)}
    missing = [clip for clip in cpp if clip not in ref]
    if missing:
        sys.exit(f"{len(missing)} C++ clips have no reference row, e.g. {missing[0]}")

    words = identical = placed_agree = both_placed = clips_identical = 0
    edges = 0
    within = {1: 0, 5: 0, 20: 0}
    max_presence = None
    for clip, row in cpp.items():
        a, b = row["intervals"], ref[clip]["intervals"]
        if len(a) != len(b):
            sys.exit(f"{clip}: {len(a)} C++ words vs {len(b)} reference words")
        clips_identical += a == b
        for x, y in zip(a, b):
            words += 1
            identical += x == y
            placed_agree += (x is None) == (y is None)
            if x is not None and y is not None:
                both_placed += 1
                for u, v in zip(x, y):
                    edges += 1
                    for ms in within:
                        within[ms] += abs(u - v) <= ms / 1000 + 1e-9
        if "presence" in row and "presence" in ref[clip]:
            diff = max((abs(p - q) for p, q in zip(row["presence"], ref[clip]["presence"])), default=0.0)
            max_presence = diff if max_presence is None else max(max_presence, diff)

    def pct(n: int, d: int) -> str:
        return f"{100.0 * n / d:.3f}%" if d else "n/a"

    print(f"clips {len(cpp)} (identical {pct(clips_identical, len(cpp))}), words {words}")
    print(f"identical intervals {pct(identical, words)}, placed agreement {pct(placed_agree, words)}")
    print(f"edges {edges} over {both_placed} words placed on both sides: "
          + ", ".join(f"within {ms} ms {pct(n, edges)}" for ms, n in within.items()))
    if max_presence is not None:
        print(f"max |presence diff| {max_presence:.3g}")


if __name__ == "__main__":
    main()
