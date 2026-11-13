# fusealign-multilingual

Word-level forced alignment with FuseAlign, as a PyTorch package (`torch.hub` or pip) and a native C++ port on
ggml. FuseAlign ([Vaidya et al., 2026](https://arxiv.org/abs/2609.33650)) is not a CTC model. A joint audio/text
transformer reads the waveform and the UTF-8 bytes of the transcript and predicts, per word, a start and an end on
a 1 ms grid plus a presence probability. The weights are Zyphra's multilingual FuseAlign, fine-tuned on corrected
CJK labels (step 2,500); see [Model](#model).

## Usage

```python
import soundfile as sf
import torch

aligner = torch.hub.load("Zyphra/fusealign-multilingual", "fusealign", trust_repo=True, device="cuda")
wav, sr = sf.read("SA1.wav", dtype="float32")  # TIMIT's SA1 sentence
words = aligner.align(torch.from_numpy(wav), sr, "she had your dark suit in greasy wash water all year")
# [WordAlignment(word='she', start=0.183, end=0.353, presence=0.99..., placed=True), ...]
```

Or install the package:

```
pip install git+https://github.com/Zyphra/fusealign-multilingual.git
```

```python
from fusealign_hub import load_aligner

aligner = load_aligner(device="cuda")
```

Requirements: `torch`, `torchaudio`, `transformers` (>= 4.57, for Dia's encoder), `safetensors`, `huggingface_hub`
and `silero-vad` (>= 6.2.3). pip installs them; with `torch.hub`, install them yourself (`torchaudio` is imported
only when the input is not 16 kHz). The first load downloads one 1.34 GB file from the Hugging Face repo
[`Zyphra/fusealign`](https://huggingface.co/Zyphra/fusealign); Dia's 4.7 GB shard is not needed.
`checkpoint="path.safetensors"` (in `load_aligner` or `torch.hub.load`) loads a local file instead.

```python
# Several clips share length-sorted, padded forwards.
batches = aligner.align_batch([wav_a, wav_b], 16000, ["first transcript", "second one"])

# Raw model output for 16 kHz mono waveforms and pre-split words: per item,
# (intervals, presence) with None for words the model did not place.
[(intervals, presence)] = aligner.intervals_batch([wav16k], [["she", "had", "your"]])

# Pre-segmented tokens for Chinese, Japanese, Cantonese and Thai (no whitespace inside a token).
words = aligner.align(wav, 16000, ["我们", "今天", "去", "公园"])

# Long-form with known utterance times: per-token guide times for asr_words.
from fusealign_hub import guide_words
asr = guide_words([{"text": "hello there", "start": 0.4, "end": 1.1}, ...])
words = aligner.align(long_wav, 16000, transcript, asr_words=asr)
```

- A transcript is a string (tokens are `transcript.split()`) or a list of tokens. Callers segment
  CJK and Thai themselves; the model was trained on segmented words joined by spaces.
- Recordings over 30 s, or transcripts over 1022 UTF-8 bytes, are aligned long-form (paper
  Appendix G.3). Approximate word times route the words into 26 s cores with 2 s of context; Silero
  VAD picks the cuts; chunk predictions are stitched.
  - `asr_words=None` (default) runs Whisper large-v3 once on the recording. `language` is
    Whisper's language (`"en"`, `"ja"`, ...); `None` auto-detects.
  - `asr_words=[]` uses the VAD character-time map instead (no Whisper).
  - A list of `{"word", "start", "end"}` dicts supplies the anchors directly (`guide_words`).
  - A token no 1022-byte chunk can hold is left unplaced instead of failing the recording.

### Output fields

`WordAlignment(word, start, end, presence, placed)`, one per token, in order. Times are seconds.

| Field | Meaning |
| --- | --- |
| `start`, `end` | The model's interval, or the uniform fallback when `placed` is False |
| `presence` | Model probability that the word is in the audio (0 for punctuation and unroutable tokens) |
| `placed` | False when presence < 0.1, the clip is under one 512-sample patch, the token could not be routed, or the token is punctuation only |

Unplaced words split the gap between their placed neighbours equally (the paper's fallback); a
leading gap starts at 0 and a trailing gap ends at the clip's duration. Punctuation-only tokens
are zero-width at the previous token's end. There is no snapping and no order repair.

## Model

A reproduction of FuseAlign by Zyphra: the multilingual model (100,000 steps), fine-tuned 2,500 steps on CJK
junction-corrected labels. The network follows the paper: a 12-layer, 1024-wide joint transformer over 32 ms audio
patches and the frozen text encoder of [Dia](https://huggingface.co/nari-labs/Dia-1.6B-0626), with convolutional
refinement to a 1 ms grid and a presence head.

% of word boundaries off by more than 50 ms (the paper's asymmetric error), against
[Qwen3-ForcedAligner-0.6B](https://huggingface.co/Qwen/Qwen3-ForcedAligner-0.6B) on the same clips, transcripts and
references:

| Corpus | FuseAlign | Qwen3-FA |
| --- | ---: | ---: |
| TIMIT | 16.55 | 20.17 |
| Buckeye | 24.58 | 24.60 |
| AMI-silver | 49.38 | 58.22 |
| TIMIT-long | 17.25 | 20.04 |
| BZNSYP Mandarin (gold) | 15.94 | 28.04 |
| Seoul Corpus Korean (gold) | 21.18 | 33.58 |
| FLEURS Mandarin | 17.95 | 11.06 |
| FLEURS Japanese | 14.02 | 8.38 |
| FLEURS Korean | 8.39 | 11.53 |
| FLEURS Spanish | 19.66 | 11.71 |
| FLEURS Portuguese | 9.30 | 10.24 |
| FLEURS French | 13.15 | 12.35 |
| FLEURS German | 13.29 | 13.91 |
| FLEURS Russian | 12.74 | 11.18 |

- Qwen3-FA ran through qwen-asr 0.0.6's `Qwen3ForcedAligner.align` in bf16, one whole clip per call (the longest
  is 81 s, under its 180 s limit), on the words joined by spaces. A word spans the units qwen-asr cut from it (CJK
  characters, Japanese and Korean sub-words). Its boundaries fall on an 80 ms grid.
- FLEURS references are Montreal Forced Aligner output, not hand labels; BZNSYP's are hand-labelled.
- Seoul Corpus: spontaneous interviews of 40 speakers with hand-corrected word boundaries. Clips are the
  interviewee's turns, split at interviewer speech, unintelligible or redacted spans and laughed words, and packed
  into clips of at most 30 s, cut at pause midpoints (8,786 clips, 226,136 words, 31.9 h). The labeller behind the fine-tune's
  Korean labels has one 7.1 ms offset fitted on speakers s01–s20; on s21–s40 alone the rates are 20.63 (FuseAlign) and
  31.84 (Qwen3-FA).
- Boundaries off by more than 300 ms: Buckeye 1.10% (FuseAlign) vs 2.67% (Qwen3-FA), AMI-silver 16.31% vs 24.13%.

## Performance

- Clips and long-form chunks are stable-sorted by length and forwarded in padded groups of at most
  `batch_size` items (default 64) and 960 s of padded audio. Every mask is per sample, including
  the refinement heads' convolutions, so a clip's outputs do not depend on its batch companions
  beyond bf16 rounding. Batches are not bit-identical to `batch_size=1`, because bf16 GEMM
  rounding depends on the batch shape (Buckeye MAE 33.36 → 33.38 ms). `aligner.batch_size = 1`
  reproduces the per-clip outputs exactly.
- On CUDA the weights autocast would cast to bf16 on every forward are stored in bf16
  (bit-identical, ~200 fewer cast kernels per forward).
- Silero VAD runs vectorized: the STFT and encoder over every frame of a 131 s block at once, the
  LSTM over the sequence. ~20x faster than the streaming loop on CPU; it agrees with the JIT model
  to ~2e-6, not bit for bit.
- All chunks of a long recording go through one batched forward.
- `compile_model=True` wraps the network in `torch.compile(dynamic=True)`: -20% forward time on
  corpus-like shapes (MAE 39.41 → 39.39 ms on 400 Buckeye clips), at ~70 s of compilation on a cold
  cache (~11 s warm) and ~30 s per new specialization. Worth it only in long-running processes;
  compiled outputs are not bit-identical to eager (256 TIMIT clips: every edge within 20 ms of
  eager). Inductor's coalesce tiling analysis is turned off for this compile: on torch 2.11 it
  fails an assertion on the model's dynamic shapes.

## Use as a git submodule

```
git submodule add https://github.com/Zyphra/fusealign-multilingual.git third_party/fusealign-multilingual
uv add --editable ./third_party/fusealign-multilingual     # or: pip install -e third_party/fusealign-multilingual
```

Then `from fusealign_hub import load_aligner` as above, or, without installing it,
`torch.hub.load("third_party/fusealign-multilingual", "fusealign", source="local")`. Fresh clones of the
consumer need `git submodule update --init`.

The package pins only lower bounds; the consumer's own torch pin and index win. The package's
`[tool.uv]` constraint-dependencies apply only when it is the uv project itself.

## Checkpoint format

One safetensors file: `model.*` (FuseAlign, fp32) and `text_encoder.*` (the 98 Dia encoder
tensors, bf16). The metadata holds `format` (`fusealign-hub/1`), `model_args` and
`text_encoder_config` (JSON), `text_encoder_repo` and `step`. `scripts/build_checkpoint.py` builds it from a
training export (a `torch.save` dict with `step`, `model` and `model_args`; the training code is not part of
this repo) and Dia's first shard, and optionally uploads it:

```
uv run python scripts/build_checkpoint.py --export path/to/step_2500.pt \
  --out dist/fusealign-multilingual-cjkft-step2500.safetensors [--push Zyphra/fusealign]
```

## Native ggml version

`ggml/` is a C++17 port of the aligner on [ggml](https://github.com/ggml-org/ggml), the submodule
`ggml/ggml` pinned to v0.27.0 (`a05c9922`). It runs the FuseAlign network, the Dia text encoder,
Silero VAD and the long-form routing natively, on CPU or CUDA, without Python. It has no Whisper:
long-form recordings are routed by anchors the caller supplies, or else by the VAD character-time map
(the Python `asr_words=[]` path), which is much less accurate (below). Each clip or long-form chunk is
its own forward, as in Python with `batch_size=1`.

### Build and models

```
git submodule update --init ggml/ggml
cmake -B ggml/build -S ggml && cmake --build ggml/build -j                           # CPU
cmake -B ggml/build-cuda -S ggml -DGGML_CUDA=ON && cmake --build ggml/build-cuda -j  # CUDA
hf download Zyphra/fusealign fusealign-multilingual-cjkft-step2500-f32.gguf --local-dir models
```

CUDA builds target sm_90 (H100) unless `-DCMAKE_CUDA_ARCHITECTURES=...` says otherwise (for example `86` or `89`).

| GGUF | Size | Weights |
| --- | ---: | --- |
| `fusealign-multilingual-cjkft-step2500-f32.gguf` | 1.35 GB | FuseAlign F32, Dia BF16, VAD F32 (the safetensors numbers) |
| `fusealign-multilingual-cjkft-step2500-q8_0.gguf` | 0.51 GB | Transformer matmuls Q8_0 (FuseAlign blocks, `text_proj`, Dia attention and MLP); other Dia tensors BF16; the rest F32 |

Both are in [`Zyphra/fusealign`](https://huggingface.co/Zyphra/fusealign). `scripts/convert_gguf.py
--outtype {f32,q8_0} [-o OUT]` writes them from the safetensors checkpoint (default
`dist/<checkpoint stem>-<outtype>.gguf`; `uv run python scripts/convert_gguf.py ...`; `gguf` is in the dev
group). On CUDA the weights Python's autocast casts to bf16 run as BF16, everything else F32.

### CLI

```
ggml/build-cuda/fusealign-cli -m models/fusealign-multilingual-cjkft-step2500-f32.gguf \
  -a SA1.wav --text "she had your dark suit in greasy wash water all year"
# [{"word": "she", "start": 0.183, "end": 0.353, "presence": 0.99..., "placed": true}, ...]
ggml/build-cuda/fusealign-cli -m MODEL.gguf -a clip.wav --tokens '["我们", "今天", "去", "公园"]'
ggml/build-cuda/fusealign-cli -m MODEL.gguf --batch inputs.jsonl -o out.jsonl [--anchors asr.jsonl]
```

- **Single mode** prints one `{"word","start","end","presence","placed"}` per token. It takes any
  sample rate (resampled as `torchaudio.functional.resample` does), and the same fields and fallback
  rules as `align`.
- **Batch mode** reads JSON lines `{"clip_id","audio"}` plus one of:
  - `"text"` or `"tokens"`: the output row holds `"words"`, as in single mode.
  - `"words"`: the output row holds raw `"intervals"` (null when unplaced) and `"presence"`, as
    `intervals_batch` gives. The audio must be 16 kHz.

  A row that fails, including a line that is not valid JSON, is written as `{"clip_id","error"}` and the
  exit status is 1.
- **Anchors** for long-form audio are a row's `"anchors": [{"word","start","end"}]` (for example
  `guide_words` output), or `--anchors FILE` of JSON lines `{"clip_id","words":[...]}`, such as cached
  Whisper words.
- `-d auto|cpu|cuda|cuda:N` picks the device (default: CUDA if present); `-t N` sets CPU threads.

The C++ API is `ggml/src/fusealign.h`: `fusealign::Aligner` with `intervals(...)` and
`align(...)`, linked from the static library `fusealign`.

### Accuracy and speed

Measured on one H100 against the Python package's predictions on the same eval sets (Python on CUDA, one clip
per forward). Batches were the full eval sets; TIMIT-long used cached Whisper anchors.

| | TIMIT % >50 | BZNSYP % >50 | TIMIT-long % >50 | Edges within 20 ms (TIMIT / BZNSYP) | TIMIT clips/s |
| --- | ---: | ---: | ---: | ---: | ---: |
| Python | 16.55 ± 0.31 | 15.94 ± 0.26 | 17.25 ± 0.61 | — | 81 (b1), 950 (b64) |
| ggml f32, CUDA | 16.53 | 15.95 | 17.32 | 99.950% / 99.927% | 129 |
| ggml q8_0, CUDA | 16.60 | 15.97 | 17.35 | 99.903% / 99.777% | 135 |

- **Not bit-identical on CUDA.** About half the intervals are identical; the rest move by a few ms,
  because torch's bf16 rounding can't be reproduced. Python itself differs from its CUDA run by as
  much on CPU.
- **CPU** (f32, 16 threads): 0.55 s per TIMIT clip. On the first 500 TIMIT clips 99.977% of edges
  are within 20 ms of the Python predictions; on the first 200, 99.97% are within 5 ms of Python's
  CPU path.
- **q8_0** is a third of the size with the same scores, but no faster on an H100.
- **Long-form without anchors (VAD character map):** TIMIT-long % >50 is 30.04 (Python's same path
  30.07), against 17.32 with Whisper anchors. Supply anchors whenever you have them.

### ggml tests

```
ctest --test-dir ggml/build                                    # long-form, text, WAV and resampling unit tests
uv run --with soundfile python ggml/tests/host/host_parity.py   # VAD, long-form, resampling, WAV, Unicode vs Python
uv run --with soundfile python ggml/tests/model/model_parity.py --help  # network vs Python on your clips
```

`host_parity.py --eval-dir DIR` adds real recordings (layout in its docstring); without it, only the seeded
random, synthetic and Unicode cases run.

## Tests

```
uv run pytest -q
```

The development environment (`uv sync`) is locked to Python 3.12, torch and torchaudio 2.9.1 (PyPI's cu128
build), transformers 4.57.3, safetensors 0.7.0 and huggingface-hub 0.36.0, through `[tool.uv]
constraint-dependencies`. Those constraints do not apply to consumers.

## License

Apache-2.0 ([LICENSE](LICENSE)). Third-party components:

| Component | Where | License |
| --- | --- | --- |
| Dia text encoder weights ([nari-labs/Dia-1.6B-0626](https://huggingface.co/nari-labs/Dia-1.6B-0626)) | `text_encoder.*` in the checkpoint and GGUFs | Apache-2.0 |
| [Silero VAD](https://github.com/snakers4/silero-vad) weights | `vad.*` in the GGUFs; the `silero-vad` package in Python | MIT |
| [ggml](https://github.com/ggml-org/ggml) | `ggml/ggml` submodule | MIT |
| [nlohmann/json](https://github.com/nlohmann/json) | `ggml/vendor/json.hpp` | MIT |
| [Whisper large-v3](https://huggingface.co/openai/whisper-large-v3) | downloaded at runtime for long-form alignment without anchors | Apache-2.0 |

## Citation

FuseAlign is described in:

```bibtex
@article{vaidya2026fusealign,
  title   = {FuseAlign: Forced Alignment in the Wild},
  author  = {Vaidya, Mithilesh and Bailey, Stephen and Badam, Sumukh and Bendel, Matthew and He, Xingzhe},
  journal = {arXiv preprint arXiv:2609.33650},
  year    = {2026}
}
```
