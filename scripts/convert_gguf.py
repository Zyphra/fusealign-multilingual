"""Convert the fusealign-hub safetensors checkpoint (plus Silero VAD) to a fusealign-gguf/1 file for ggml.

    uv run --with gguf==0.19.0 python scripts/convert_gguf.py [--checkpoint PATH] \
        [--outtype {f32,q8_0}] [-o OUT.gguf]

Tensors keep their PyTorch state-dict names under the prefixes ``model.``, ``text_encoder.`` and
``vad.`` (Silero's 16 kHz ``_model.*``), written as-is, so each ggml ``ne`` is the reversed torch
shape. f32: ``model.*`` F32, ``text_encoder.*`` BF16 (as stored), ``vad.*`` F32. q8_0: the same,
except Q8_0 for the transformer matmul weights (FuseAlign blocks and ``text_proj``, Dia
attention and MLP projections).
"""

import argparse
import json
from pathlib import Path
import re
import sys

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusealign_hub.aligner import (  # noqa: E402
    CHECKPOINT_FILE,
    CHECKPOINT_FORMAT,
    HF_REPO,
    MAX_SECONDS,
    PRESENCE_THRESHOLD,
)
from fusealign_hub.longform import MAX_TEXT_BYTES  # noqa: E402

GGUF_FORMAT = "fusealign-gguf/1"
N_MODEL, N_TEXT, N_VAD = 130, 98, 15
MODEL_ARGS_U32 = ("dim", "n_layers", "n_heads", "ffn_dim", "feat_dim", "patch_window", "patch_hop",
                  "n_upsample", "text_dim")
DIA_U32 = {"hidden_size": "hidden_size", "num_layers": "num_hidden_layers",
           "num_heads": "num_attention_heads", "head_dim": "head_dim",
           "intermediate_size": "intermediate_size", "vocab_size": "vocab_size",
           "max_position_embeddings": "max_position_embeddings"}
Q8_0_NAMES = re.compile(
    r"model\.blocks\.\d+\.(attn\.qkv|attn\.out|ffn\.w1|ffn\.w2|ffn\.w3)\.weight"
    r"|model\.text_proj\.weight"
    r"|text_encoder\.layers\.\d+\.(self_attention\.[qkvo]_proj|mlp\.gate_up_proj|mlp\.down_proj)\.weight"
)


def tensor_type(name: str, outtype: str):
    from gguf import GGMLQuantizationType as Q

    if outtype == "q8_0" and Q8_0_NAMES.fullmatch(name):
        return Q.Q8_0
    return Q.BF16 if name.startswith("text_encoder.") else Q.F32


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path,
                        help=f"fusealign-hub safetensors (default: {CHECKPOINT_FILE} from {HF_REPO})")
    parser.add_argument("--outtype", choices=("f32", "q8_0"), default="f32")
    parser.add_argument("-o", "--out", type=Path,
                        help="output path (default: dist/<checkpoint stem>-<outtype>.gguf)")
    args = parser.parse_args()

    import gguf
    from gguf import GGMLQuantizationType as Q
    import silero_vad
    from safetensors import safe_open
    from safetensors.torch import load_file

    checkpoint = args.checkpoint
    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = Path(hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT_FILE))
    out = args.out or Path("dist") / f"{checkpoint.stem}-{args.outtype}.gguf"

    with safe_open(str(checkpoint), framework="pt") as f:
        metadata = f.metadata() or {}
    if metadata.get("format") != CHECKPOINT_FORMAT:
        sys.exit(f"{checkpoint} is not a {CHECKPOINT_FORMAT} checkpoint")
    model_args = json.loads(metadata["model_args"])
    dia = json.loads(metadata["text_encoder_config"])

    state = load_file(str(checkpoint))
    tensors: dict[str, np.ndarray] = {}
    for key, value in state.items():
        if not key.startswith(("model.", "text_encoder.")):
            sys.exit(f"unexpected tensor {key} in {checkpoint}")
        tensors[key] = value.float().numpy()
    vad_state = silero_vad.load_silero_vad().state_dict()
    for key, value in vad_state.items():
        if key.startswith("_model."):
            tensors["vad." + key.removeprefix("_model.")] = value.float().numpy()

    counts = [sum(name.startswith(prefix) for name in tensors) for prefix in ("model.", "text_encoder.", "vad.")]
    if counts != [N_MODEL, N_TEXT, N_VAD]:
        sys.exit(f"expected {N_MODEL}/{N_TEXT}/{N_VAD} model/text_encoder/vad tensors, found {counts}")

    out.parent.mkdir(parents=True, exist_ok=True)
    writer = gguf.GGUFWriter(str(out), arch="fusealign")
    writer.add_name(checkpoint.stem)
    writer.add_string("fusealign.format", GGUF_FORMAT)
    for key in MODEL_ARGS_U32:
        writer.add_uint32(f"fusealign.{key}", int(model_args[key]))
    writer.add_float32("fusealign.rope_theta", float(model_args["rope_theta"]))
    writer.add_float32("fusealign.presence_threshold", PRESENCE_THRESHOLD)
    writer.add_uint32("fusealign.max_text_bytes", MAX_TEXT_BYTES)
    writer.add_float32("fusealign.max_seconds", MAX_SECONDS)
    for key, source in DIA_U32.items():
        writer.add_uint32(f"dia.{key}", int(dia[source]))
    writer.add_float32("dia.norm_eps", float(dia["norm_eps"]))
    writer.add_float32("dia.rope_theta", float(dia["rope_theta"]))
    writer.add_string("fusealign.source_checkpoint", checkpoint.name)
    writer.add_uint32("fusealign.step", int(metadata["step"]))

    for name, array in tensors.items():
        qtype = tensor_type(name, args.outtype)
        if qtype == Q.F32:
            writer.add_tensor(name, np.ascontiguousarray(array, dtype=np.float32))
        else:
            writer.add_tensor(name, gguf.quants.quantize(np.ascontiguousarray(array), qtype), raw_dtype=qtype)

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()

    # Read back: counts, names, shapes and types.
    reader = gguf.GGUFReader(str(out))
    if reader.fields["fusealign.format"].contents() != GGUF_FORMAT:
        sys.exit(f"{out}: format field did not round-trip")
    found = {t.name: t for t in reader.tensors}
    if found.keys() != tensors.keys():
        sys.exit(f"{out}: tensor names did not round-trip")
    for name, array in tensors.items():
        t = found[name]
        if t.tensor_type != tensor_type(name, args.outtype):
            sys.exit(f"{out}: {name} has type {t.tensor_type.name}")
        if tuple(int(n) for n in t.shape) != tuple(reversed(array.shape)):
            sys.exit(f"{out}: {name} has ne {tuple(t.shape)}, expected {tuple(reversed(array.shape))}")
    types: dict[str, int] = {}
    for t in reader.tensors:
        types[t.tensor_type.name] = types.get(t.tensor_type.name, 0) + 1
    print(f"{out}: {out.stat().st_size} bytes, {len(found)} tensors "
          f"(model {counts[0]}, text_encoder {counts[1]}, vad {counts[2]}), types {types}")


if __name__ == "__main__":
    main()
