"""Vectorized Silero VAD v5 (16 kHz) front-end for long-form cut points."""

from functools import lru_cache

import torch
from torch import nn
from torch.nn import functional as F

SAMPLE_RATE = 16000
VAD_BLOCK_FRAMES = 4096  # 512-sample frames per front-end call (~131 s); bounds memory on long recordings


@lru_cache(maxsize=1)
def _silero():
    """Silero VAD v5 16 kHz weights from the ``silero-vad`` package's JIT model:
    ``(weights, lstm, get_speech_timestamps_from_probs)``.

    Importing ``silero_vad`` runs ``torch.set_num_threads(1)`` (``silero_vad/model.py``), which
    would silently single-thread the rest of the process; the caller's count is put back.
    """
    threads = torch.get_num_threads()
    try:
        import silero_vad

        model = silero_vad.load_silero_vad()
    finally:
        torch.set_num_threads(threads)
    from_probs = getattr(silero_vad, "get_speech_timestamps_from_probs", None)
    if from_probs is None:
        raise RuntimeError(
            "silero-vad lacks get_speech_timestamps_from_probs; install silero-vad>=6.2.3"
        )
    weights = {
        key.removeprefix("_model."): value.detach().clone()
        for key, value in model.state_dict().items()
        if key.startswith("_model.")  # not "_model_8k."
    }
    lstm = nn.LSTM(128, 128)
    with torch.no_grad():
        for name in ("weight_ih", "weight_hh", "bias_ih", "bias_hh"):
            getattr(lstm, f"{name}_l0").copy_(weights[f"decoder.rnn.{name}"])
    lstm.requires_grad_(False)
    return weights, lstm.eval(), from_probs


@torch.inference_mode()
def silero_probs(wav: torch.Tensor) -> torch.Tensor:
    """Per-512-sample-frame speech probabilities of a 16 kHz CPU waveform.

    The JIT model's streaming loop (``get_speech_timestamps``) runs the whole network once per
    frame. Only its LSTM carries state: the STFT and encoder see one frame plus the previous
    frame's last 64 samples, so they run over every frame of a block at once and the LSTM steps
    through the sequence. Same weights and inputs (zero context first, zero-padded last frame);
    agrees with the JIT model to ~2e-6 on CPU, not bit for bit. ~20x faster on CPU; on CUDA it
    drifts to 2e-3, so it always runs on CPU.
    """
    weights, lstm, _ = _silero()
    n_frames = -(-wav.numel() // 512)
    features = []
    for first in range(0, n_frames, VAD_BLOCK_FRAMES):
        last = min(n_frames, first + VAD_BLOCK_FRAMES)
        lo = first * 512
        block = wav[lo:last * 512]
        frames = F.pad(block, (0, (last - first) * 512 - block.numel())).reshape(-1, 512)
        previous = wav[lo - 64:lo] if first else wav.new_zeros(64)
        context = torch.cat((previous.unsqueeze(0), frames[:-1, -64:]))
        x = F.pad(torch.cat((context, frames), dim=1), (0, 64), mode="reflect").unsqueeze(1)
        x = F.conv1d(x, weights["stft.forward_basis_buffer"], stride=128)
        x = (x[:, :129].square() + x[:, 129:].square()).sqrt()
        for index, stride in enumerate((1, 2, 2, 1)):
            x = F.relu(F.conv1d(
                x, weights[f"encoder.{index}.reparam_conv.weight"],
                weights[f"encoder.{index}.reparam_conv.bias"], stride=stride, padding=1,
            ))
        features.append(x.squeeze(-1))
    hidden, _ = lstm(torch.cat(features).unsqueeze(1))
    logits = F.conv1d(
        F.relu(hidden).permute(1, 2, 0),
        weights["decoder.decoder.2.weight"], weights["decoder.decoder.2.bias"],
    )
    return torch.sigmoid(logits).reshape(-1)


def speech_regions(wav: torch.Tensor) -> list[dict]:
    """``get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True)``,
    with the probabilities from ``silero_probs``."""
    _, _, from_probs = _silero()
    return from_probs(
        silero_probs(wav).tolist(), sampling_rate=SAMPLE_RATE, return_seconds=True,
        audio_length_samples=wav.numel(),
    )
