"""Bundle a FuseAlign export and its Dia text encoder into one fusealign-hub safetensors file.

    uv run python scripts/build_checkpoint.py --export path/to/step_N.pt \
        --out dist/<name>.safetensors [--push Zyphra/fusealign]

The export is a ``torch.save`` dict with ``step``, ``model`` and ``model_args`` (from the training
code, not part of this repo). The FuseAlign weights stay fp32; the 98 Dia encoder tensors (from
Dia's first shard) are stored in bf16, as the aligner runs them. ``--push`` uploads the file to a
Hugging Face model repo (created public if it does not exist).
"""

import argparse
import hashlib
import json
from pathlib import Path
import sys

import torch
from huggingface_hub import HfApi, hf_hub_download
from safetensors import safe_open
from safetensors.torch import save_file

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from fusealign_hub.aligner import CHECKPOINT_FORMAT  # noqa: E402
from fusealign_hub.model import FuseAlign, FuseAlignArgs  # noqa: E402
from fusealign_hub.text_encoder import DiaTextEncoder  # noqa: E402

DIA_PREFIX = "model.encoder."
DIA_TENSORS = 98


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--export", type=Path, required=True, help="FuseAlign export (.pt)")
    parser.add_argument("--out", type=Path, required=True, help="output .safetensors path")
    parser.add_argument("--push", metavar="REPO_ID", help="upload to this Hugging Face model repo")
    args = parser.parse_args()

    export = torch.load(args.export, map_location="cpu", weights_only=True)
    if not {"step", "model", "model_args"} <= export.keys():
        sys.exit(f"{args.export} is not a FuseAlign export (needs step, model and model_args)")

    dia_repo = export["model_args"]["dia_repo"]
    config_path = hf_hub_download(repo_id=dia_repo, filename="config.json")
    shard_path = hf_hub_download(repo_id=dia_repo, filename="model-00001-of-00002.safetensors")
    with open(config_path, encoding="utf-8") as handle:
        enc_cfg = json.load(handle)["encoder_config"]
    with safe_open(shard_path, framework="pt", device="cpu") as shard:
        keys = [key for key in shard.keys() if key.startswith(DIA_PREFIX)]
        if len(keys) != DIA_TENSORS:
            sys.exit(f"expected {DIA_TENSORS} Dia encoder tensors in {shard_path}, found {len(keys)}")
        text_state = {
            key.removeprefix(DIA_PREFIX): shard.get_tensor(key).to(torch.bfloat16).contiguous()
            for key in keys
        }

    # Both strict loads must succeed before anything is written.
    FuseAlign(FuseAlignArgs(**export["model_args"])).load_state_dict(export["model"], strict=True)
    DiaTextEncoder(enc_cfg, text_state)

    tensors = {"model." + k: v.contiguous() for k, v in export["model"].items()} | {
        "text_encoder." + k: v for k, v in text_state.items()
    }
    metadata = {
        "format": CHECKPOINT_FORMAT,
        "model_args": json.dumps(export["model_args"], sort_keys=True),
        "text_encoder_config": json.dumps(enc_cfg, sort_keys=True),
        "text_encoder_repo": dia_repo,
        "step": str(export["step"]),
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    save_file(tensors, args.out, metadata=metadata)

    digest = hashlib.sha256()
    with open(args.out, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 24), b""):
            digest.update(block)
    print(f"{args.out}  {args.out.stat().st_size} bytes  sha256 {digest.hexdigest()}")

    if args.push:
        api = HfApi()
        api.create_repo(args.push, repo_type="model", exist_ok=True)
        api.upload_file(
            path_or_fileobj=str(args.out),
            path_in_repo=args.out.name,
            repo_id=args.push,
            commit_message=f"Add {args.out.name} (step {export['step']})",
        )
        print(f"uploaded to https://huggingface.co/{args.push}/blob/main/{args.out.name}")


if __name__ == "__main__":
    main()
