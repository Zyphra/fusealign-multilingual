"""FuseAlign word-level forced aligner: batching, long-form routing, the checkpoint loader.

FuseAlign (arXiv 2609.33650) is not a CTC model. A joint audio/text transformer reads the waveform
and the UTF-8 bytes of the transcript and predicts, for each word, a start and an end on a 1 ms
grid plus a presence probability. A word whose presence is below ``PRESENCE_THRESHOLD`` is "not
found" and takes the paper's uniform fallback interval (``fill_unplaced``).

Recordings over ``MAX_SECONDS`` (or transcripts over 1022 UTF-8 bytes) are handled as in the
paper's Appendix G.3: approximate per-word times (supplied by the caller, otherwise Whisper
large-v3 run once on the recording) route the words into 26 s cores with 2 s of context, Silero
VAD picks the cuts, and the chunk predictions are stitched.
"""

from dataclasses import dataclass
import json
import os
import unicodedata

import torch

from . import longform, vad
from .longform import MAX_TEXT_BYTES, fill_unplaced
from .model import FuseAlign, FuseAlignArgs, cast_autocast_weights
from .text_encoder import DiaTextEncoder

SAMPLE_RATE = 16000
MAX_SECONDS = 30.0
PRESENCE_THRESHOLD = 0.1
WHISPER_MODEL = "openai/whisper-large-v3"
HF_REPO = "Zyphra/fusealign"
CHECKPOINT_FILE = "fusealign-multilingual-cjkft-step2500.safetensors"
CHECKPOINT_FORMAT = "fusealign-hub/1"

# Padded audio per forward. Measured peak CUDA memory per process (H100, bf16, ~1 GB of it resident weights):
# 4.4 GB for 64 corpus-like spans (median 11 s), 5.4 GB for 32 x 30 s clips with ~1000-byte transcripts.
# At 120 s the forward was launch-bound (1.7x slower on corpus-like spans); past 960 s it gains nothing.
_BATCH_SAMPLES = 960 * SAMPLE_RATE


@dataclass(frozen=True)
class WordAlignment:
    """One transcript token. Times are in seconds.

    ``presence`` is the model's probability that the word is in the audio. ``placed`` is False
    when the word took the paper's uniform fallback (presence under ``PRESENCE_THRESHOLD``, audio
    under one patch, or a token no long-form chunk could hold) or is punctuation only, in which
    case it is zero-width at the previous token's end.
    """

    word: str
    start: float
    end: float
    presence: float
    placed: bool


def _is_wordy(token: str) -> bool:
    """True if ``token`` carries a letter or a digit, i.e. anything to say."""
    return any(unicodedata.category(ch)[0] in ("L", "N") for ch in token)


def _tokens(transcript) -> list[str]:
    if isinstance(transcript, str):
        return transcript.split()
    tokens = list(transcript)
    if any(not isinstance(t, str) or not t or any(ch.isspace() for ch in t) for t in tokens):
        raise ValueError("tokens must be non-empty strings without whitespace")
    return tokens


class FuseAlignAligner:
    """Word-level forced alignment with a FuseAlign network and its frozen Dia text encoder.

    ``batch_size`` clips (or long-form chunks) share one forward, length-sorted. Batches are not
    bit-identical to ``batch_size=1``, because bf16 GEMM rounding depends on the batch shape
    (Buckeye MAE 33.36 → 33.38 ms); set ``aligner.batch_size = 1`` for the exact per-clip outputs.
    """

    def __init__(self, model: FuseAlign, encoder: DiaTextEncoder, *, batch_size: int = 64,
                 whisper_model: str = WHISPER_MODEL):
        self.model = model
        self.encoder = encoder
        self.whisper_model = whisper_model
        self.batch_size = batch_size
        self._whisper = None

    @property
    def device(self) -> torch.device:
        return next(self.model.parameters()).device

    # ── model forward ────────────────────────────────────────────────────────

    @torch.inference_mode()
    def _forward_batch(self, items: list[tuple[torch.Tensor, list[str]]]):
        """``(intervals | None, presence)`` per word for each ``(wav, words)`` item.

        Items are 1-D float32 CPU waveforms at 16 kHz. They are stable-sorted by length and
        forwarded in groups of at most ``batch_size`` clips and ``_BATCH_SAMPLES`` padded
        samples, so each padded batch holds similar lengths; results come back in input order.
        An item without words gets ``([], [])`` and no forward.
        """
        results = [([], []) for _ in items]
        order = sorted(
            (i for i, (_, words) in enumerate(items) if words),
            key=lambda i: items[i][0].numel(),
        )
        groups, group = [], []
        for i in order:
            # Ascending lengths: this item's is the padded length of its group.
            if group and (len(group) == self.batch_size
                          or (len(group) + 1) * items[i][0].numel() > _BATCH_SAMPLES):
                groups.append(group)
                group = []
            group.append(i)
        if group:
            groups.append(group)
        for group in groups:
            for i, result in zip(group, self._forward_group([items[i] for i in group])):
                results[i] = result
        return results

    def _forward_group(self, items: list[tuple[torch.Tensor, list[str]]]):
        """One padded forward over mono 16 kHz clips: (intervals | None, presence) per word.

        Word spans exclude the two-token speaker/space prefix and count UTF-8
        bytes. Only each clip's valid frames participate in decoding.
        """
        model, encoder = self.model, self.encoder
        device = self.device
        id_rows, span_rows = [], []
        for wav, words in items:
            if wav.numel() < model.args.patch_window:
                raise ValueError("Waveform must contain at least one complete audio patch")
            transcript = bytearray()
            spans = []
            for index, word in enumerate(words):
                if index:
                    transcript.append(32)
                start = len(transcript)
                transcript.extend(word.encode("utf-8"))
                spans.append((start, len(transcript)))
            if len(transcript) > MAX_TEXT_BYTES:
                raise ValueError(
                    f"Use long-form alignment for transcripts over {MAX_TEXT_BYTES} UTF-8 bytes"
                )
            id_rows.append([1, 32, *transcript])
            span_rows.append(spans)

        batch = len(items)
        max_ids = max(len(row) for row in id_rows)
        max_words = max(len(row) for row in span_rows)
        wav = torch.zeros(batch, max(w.numel() for w, _ in items))
        text_ids = torch.zeros(batch, max_ids, dtype=torch.long)
        text_mask = torch.zeros(batch, max_ids, dtype=torch.bool)
        word_spans = torch.zeros(batch, max_words, 2, dtype=torch.long)
        word_mask = torch.zeros(batch, max_words, dtype=torch.bool)
        for row, ((clip, _), ids, spans) in enumerate(zip(items, id_rows, span_rows)):
            wav[row, :clip.numel()] = clip
            text_ids[row, :len(ids)] = torch.tensor(ids, dtype=torch.long)
            text_mask[row, :len(ids)] = True
            word_spans[row, :len(spans)] = torch.tensor(spans, dtype=torch.long)
            word_mask[row, :len(spans)] = True
        wav_lengths = torch.tensor([w.numel() for w, _ in items], dtype=torch.long)
        wav, wav_lengths, text_ids, text_mask, word_spans, word_mask = (
            t.to(device) for t in (wav, wav_lengths, text_ids, text_mask, word_spans, word_mask)
        )

        with torch.autocast(
            device_type=device.type, dtype=torch.bfloat16, enabled=device.type == "cuda"
        ):
            text_feats = encoder(text_ids, text_mask)
            # Dia stays bf16 on CPU too, but the trainable model is stored in fp32 there.
            if device.type != "cuda":
                text_feats = text_feats.float()
            predictions = model(
                wav=wav,
                wav_lengths=wav_lengths,
                text_feats=text_feats,
                text_mask=text_mask,
                word_spans=word_spans,
                word_mask=word_mask,
            )

        frames = torch.arange(predictions["start_logits"].shape[-1], device=device)
        beyond = frames[None, None, :] >= predictions["frame_lengths"][:, None, None]
        start = predictions["start_logits"].masked_fill(beyond, float("-inf")).argmax(dim=-1)
        end = predictions["end_logits"].masked_fill(beyond, float("-inf")).argmax(dim=-1)
        end = torch.maximum(end, start)
        decoded = torch.stack((start, end), dim=-1).cpu().tolist()
        # A forward can come back NaN for a clip (observed on some sub-second clips); its words are
        # then unplaced, and presence 0 says so where NaN would reach JSON as a non-standard literal.
        presence = predictions["presence_logits"].sigmoid().nan_to_num(0.0).cpu().tolist()
        results = []
        for row, spans in enumerate(span_rows):
            probs = presence[row][:len(spans)]
            results.append((
                [
                    (start_ms / 1000.0, end_ms / 1000.0) if p >= PRESENCE_THRESHOLD else None
                    for (start_ms, end_ms), p in zip(decoded[row], probs)
                ],
                probs,
            ))
        return results

    # ── approximate word times ───────────────────────────────────────────────

    def whisper_words(self, waveform: torch.Tensor, language: str | None = None) -> list[dict]:
        """Whisper word times for a 16 kHz mono waveform; the pipeline is loaded once.

        ``language`` is a Whisper language (``"en"``, ``"ja"``, ...); ``None`` auto-detects.
        """
        wav = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
        if wav.ndim != 1:
            raise ValueError("expected a mono 1-D waveform")
        if self._whisper is None:
            from transformers import pipeline

            device = self.device
            self._whisper = pipeline(
                "automatic-speech-recognition",
                model=self.whisper_model,
                dtype=torch.float16 if device.type == "cuda" else torch.float32,
                device=device,
                chunk_length_s=30,
                batch_size=16,
            )
        duration = wav.numel() / SAMPLE_RATE
        generate_kwargs = {"task": "transcribe"}
        if language is not None:
            generate_kwargs["language"] = language
        result = self._whisper(
            {"raw": wav.numpy(), "sampling_rate": SAMPLE_RATE},
            return_timestamps="word",
            generate_kwargs=generate_kwargs,
        )
        if self.device.type == "cuda":
            # Word-timestamp decoding peaks at tens of GB; the allocator would keep it cached and
            # starve other processes sharing the GPU.
            torch.cuda.empty_cache()
        words = []
        for chunk in result.get("chunks", []):
            start, end = chunk["timestamp"]
            # Missing word edges are bounded by the actual recording.
            words.append({
                "word": chunk["text"].strip(),
                "start": float(start if start is not None else 0.0),
                "end": float(end if end is not None else duration),
            })
        return words

    # ── alignment ────────────────────────────────────────────────────────────

    def intervals_batch(self, waveforms: list[torch.Tensor], word_lists: list[list[str]],
                        asr_word_lists: list[list[dict] | None] | None = None,
                        language: str | None = None):
        """Per item, ``(intervals | None, presence)`` per word for 16 kHz mono waveforms.

        Routing is the paper's, per item: under one patch every word is unplaced; up to
        ``MAX_SECONDS`` and 1022 transcript bytes one forward (shared with the other short
        items); otherwise long-form. ``asr_word_lists`` holds the approximate word times for
        long-form items (``None`` runs Whisper; an item's ``[]`` falls back to the VAD
        character-time map). ``language`` is Whisper's language for those runs (``None``
        auto-detects).
        """
        if asr_word_lists is None:
            asr_word_lists = [None] * len(waveforms)
        if not len(waveforms) == len(word_lists) == len(asr_word_lists):
            raise ValueError("waveforms, word lists and asr word lists differ in length")
        wavs = [torch.as_tensor(w, dtype=torch.float32, device="cpu") for w in waveforms]
        if any(wav.ndim != 1 for wav in wavs):
            raise ValueError("expected a mono 1-D waveform")
        results = [None] * len(wavs)
        short, short_at = [], []
        for i, (wav, words) in enumerate(zip(wavs, word_lists)):
            n = len(words)
            if not n:
                results[i] = ([], [])
            elif wav.numel() < self.model.args.patch_window:
                results[i] = ([None] * n, [0.0] * n)
            elif (wav.numel() <= MAX_SECONDS * SAMPLE_RATE
                    and len(" ".join(words).encode("utf-8")) <= MAX_TEXT_BYTES):
                short.append((wav, words))
                short_at.append(i)
            else:
                results[i] = self._long(wav, words, asr_word_lists[i], language)
        for i, result in zip(short_at, self._forward_batch(short)):
            results[i] = result
        return results

    @torch.inference_mode()
    def _long(self, wav: torch.Tensor, words: list[str], asr_words, language: str | None):
        """ASR anchors, VAD cores and context stitching."""
        if asr_words is None:
            asr_words = self.whisper_words(wav, language)
        duration = wav.numel() / SAMPLE_RATE
        speech_regions = vad.speech_regions(wav)
        approx = longform.approximate_times(words, asr_words, speech_regions, duration)
        # plan_chunks raises for words no chunk can hold, which would fail the whole recording for
        # one oversized token; they are left unplaced instead and take the uniform fallback.
        skipped = longform.unroutable(words, approx)
        keep = [i for i in range(len(words)) if i not in skipped]
        chunks = longform.plan_chunks([words[i] for i in keep], [approx[i] for i in keep],
                                      speech_regions, duration)
        for chunk in chunks:
            chunk["indices"] = [keep[i] for i in chunk["indices"]]
        items, offsets = [], []
        for chunk in chunks:
            start_seconds, end_seconds = chunk["audio"]
            start_sample = max(0, min(wav.numel(), round(start_seconds * SAMPLE_RATE)))
            end_sample = max(start_sample, min(wav.numel(), round(end_seconds * SAMPLE_RATE)))
            chunk_words = [words[index] for index in chunk["indices"]]
            items.append((wav[start_sample:end_sample], chunk_words))
            offsets.append(start_sample / SAMPLE_RATE)
        predictions, presences = [], []
        for offset, (ivs, probs) in zip(offsets, self._forward_batch(items)):
            predictions.append([
                None if iv is None else (iv[0] + offset, iv[1] + offset) for iv in ivs
            ])
            presences.append(probs)
        stitched = longform.stitch_predictions(len(words), chunks, predictions)

        presence = [0.0] * len(words)
        found = [False] * len(words)
        for chunk, placed, probs in zip(chunks, predictions, presences, strict=True):
            for index, iv, p in zip(chunk["indices"], placed, probs, strict=True):
                if stitched[index] is not None:
                    if not found[index] and iv == stitched[index]:
                        presence[index], found[index] = p, True
                else:
                    presence[index] = max(presence[index], p)
        return stitched, presence

    def align_batch(self, waveforms: list[torch.Tensor], sample_rate: int, transcripts,
                    asr_words: list[list[dict] | None] | None = None,
                    language: str | None = None) -> list[list[WordAlignment]]:
        """One list of ``WordAlignment`` per transcript, one per token, in order.

        A transcript is a ``str`` (tokens are ``transcript.split()``) or a list of tokens
        (pre-segmented text for ja/zh/yue/th). Waveforms are mono, resampled to 16 kHz when
        ``sample_rate`` differs. Tokens with a letter or digit go to the model; unplaced ones take
        the uniform fallback. Punctuation-only tokens are zero-width at the previous token's end.
        No snapping and no order repair. The clips share batched forwards. ``asr_words`` and
        ``language`` are as in ``intervals_batch``.
        """
        if asr_words is None:
            asr_words = [None] * len(waveforms)
        if not len(waveforms) == len(transcripts) == len(asr_words):
            raise ValueError("waveforms, transcripts and asr_words differ in length")
        out = [[] for _ in waveforms]
        prepared = []  # (index, tokens, wav, wordy)
        for i, (waveform, transcript) in enumerate(zip(waveforms, transcripts)):
            tokens = _tokens(transcript)
            if not tokens:
                continue
            wav = torch.as_tensor(waveform, dtype=torch.float32, device="cpu")
            if wav.ndim != 1:
                raise ValueError("expected a mono 1-D waveform")
            if sample_rate != SAMPLE_RATE:
                import torchaudio

                wav = torchaudio.functional.resample(wav, sample_rate, SAMPLE_RATE)
            wordy = [k for k, token in enumerate(tokens) if _is_wordy(token)]
            prepared.append((i, tokens, wav, wordy))

        results = self.intervals_batch(
            [wav for _, _, wav, _ in prepared],
            [[tokens[k] for k in wordy] for _, tokens, _, wordy in prepared],
            [asr_words[i] for i, _, _, _ in prepared],
            language,
        )
        for (i, tokens, wav, wordy), (ivs, probs) in zip(prepared, results):
            duration = wav.numel() / SAMPLE_RATE
            filled = fill_unplaced(ivs, duration)
            by_token = {
                k: (filled[j], probs[j], ivs[j] is not None) for j, k in enumerate(wordy)
            }

            first_start = filled[0][0] if filled else 0.0
            previous_end = None
            words = []
            for k, token in enumerate(tokens):
                if k in by_token:
                    (start, end), p, placed = by_token[k]
                    words.append(WordAlignment(token, start, end, p, placed))
                    previous_end = end
                else:
                    t = previous_end if previous_end is not None else first_start
                    words.append(WordAlignment(token, t, t, 0.0, False))
                    previous_end = t
            out[i] = words
        return out

    def align(self, waveform: torch.Tensor, sample_rate: int, transcript,
              asr_words: list[dict] | None = None,
              language: str | None = None) -> list[WordAlignment]:
        """``align_batch`` for one waveform."""
        return self.align_batch([waveform], sample_rate, [transcript], [asr_words], language)[0]


def load_aligner(checkpoint: str | os.PathLike | None = None,
                 device: str | torch.device | None = None, compile_model: bool = False,
                 batch_size: int = 64) -> FuseAlignAligner:
    """Load a ``fusealign-hub/1`` safetensors checkpoint (default: ``CHECKPOINT_FILE`` from the
    Hugging Face repo ``HF_REPO``) onto ``device`` (default: CUDA when available).

    ``compile_model`` wraps the network in ``torch.compile``; compiled outputs are not
    bit-identical to eager.
    """
    from safetensors import safe_open
    from safetensors.torch import load_file

    if checkpoint is None:
        from huggingface_hub import hf_hub_download

        checkpoint = hf_hub_download(repo_id=HF_REPO, filename=CHECKPOINT_FILE)
    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    with safe_open(checkpoint, framework="pt") as handle:
        metadata = handle.metadata() or {}
    if metadata.get("format") != CHECKPOINT_FORMAT:
        raise ValueError(
            f"{checkpoint} is not a {CHECKPOINT_FORMAT} checkpoint; "
            "build one with scripts/build_checkpoint.py"
        )
    model_state, text_state = {}, {}
    for key, value in load_file(checkpoint).items():
        if key.startswith("model."):
            model_state[key.removeprefix("model.")] = value
        elif key.startswith("text_encoder."):
            text_state[key.removeprefix("text_encoder.")] = value
        else:
            raise ValueError(f"unexpected tensor {key!r} in {checkpoint}")

    net = FuseAlign(FuseAlignArgs(**json.loads(metadata["model_args"])))
    net.load_state_dict(model_state, strict=True)
    net.to(device).eval()
    if device.type == "cuda":
        cast_autocast_weights(net)
    if compile_model:
        # Inductor fuses the RoPE/RMSNorm/mask elementwise chains and cuts launch overhead:
        # -20% forward on fresh corpus-like shapes at 960 s / 64 batches, MAE 39.41 -> 39.39 ms
        # on 400 Buckeye clips. Costs ~70 s of compilation on a cold cache (~11 s warm) and
        # ~30 s per new specialization; worth it only in long-running processes.
        # dynamic=True because every batch has new lengths. Inductor's coalesce tiling analysis
        # (default on in torch 2.11) fails an assertion in tiling_utils.get_pw_red_splits on
        # this graph's dynamic shapes; its own config notes it does not apply to them.
        from torch._inductor import config as inductor_config

        options = ({"triton.coalesce_tiling_analysis": False}
                   if hasattr(inductor_config.triton, "coalesce_tiling_analysis") else None)
        net.compile(dynamic=True, options=options)
    encoder = DiaTextEncoder(json.loads(metadata["text_encoder_config"]), text_state).to(device)
    return FuseAlignAligner(net, encoder, batch_size=batch_size)
