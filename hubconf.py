"""torch.hub entry point: torch.hub.load(<repo dir>, "fusealign", source="local")."""

dependencies = ["torch", "transformers", "safetensors", "huggingface_hub", "silero_vad"]

from fusealign_hub import load_aligner as _load_aligner


def fusealign(device=None, checkpoint=None, compile_model=False, batch_size=64):
    """FuseAlign multilingual CJK fine-tune (step 2500) as a ``FuseAlignAligner``.

    Downloads ``fusealign-multilingual-cjkft-step2500.safetensors`` from the Hugging Face repo
    ``Zyphra/fusealign`` unless ``checkpoint`` names a local file. ``device`` defaults to CUDA when
    available.

        aligner = torch.hub.load("Zyphra/fusealign-multilingual", "fusealign", trust_repo=True)
        words = aligner.align(wav, 16000, "she had your dark suit")
        # [WordAlignment(word="she", start=0.183, end=0.353, presence=..., placed=True), ...]
    """
    return _load_aligner(checkpoint=checkpoint, device=device,
                         compile_model=compile_model, batch_size=batch_size)
