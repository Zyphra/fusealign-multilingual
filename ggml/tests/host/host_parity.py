"""Parity of the ggml port's host-side pieces with the Python package.

Drives ``test-host-parity`` (JSON lines over stdin/stdout) and compares it with ``fusealign_hub.longform``,
``fusealign_hub.vad``, ``str.split``/``_is_wordy``, ``torchaudio.functional.resample`` and ``soundfile.read`` on
seeded random cases, every code point and synthetic WAVs. Longform and text results must be bit-identical, VAD
probabilities within 1e-5, speech regions identical, resampling within 1e-6, WAV samples identical.

    cmake -B ggml/build -S ggml && cmake --build ggml/build -j --target test-host-parity
    uv run --with soundfile python ggml/tests/host/host_parity.py [--eval-dir DIR]

``--eval-dir`` adds real recordings: DIR/timit/inputs_ref.jsonl and DIR/bznsyp/inputs_ref.jsonl (rows with an
``audio`` WAV path; ``--wavs`` of each go through read_wav and resampling), and DIR/timit_long/inputs_ref.jsonl
(rows ``{"clip_id", "audio", "words"}``, 16 kHz mono) with DIR/timit_long/asr_whisper.jsonl (rows
``{"clip_id", "words": [{"word", "start", "end"}]}``) for VAD probabilities, speech regions and long-form
planning. Without it, those cases are skipped.

``--gguf`` defaults to dist/fusealign-multilingual-ns4-step60000-f32.gguf; when that file is missing a GGUF holding
only the 15 ``vad.*`` tensors is written from ``silero_vad`` into the temporary directory (needs ``gguf``).
"""

import argparse
import json
import math
from pathlib import Path
import random
import subprocess
import sys
import tempfile
import time
import unicodedata

import numpy as np
import torch

from fusealign_hub import longform, vad
from fusealign_hub.aligner import _is_wordy

REPO = Path(__file__).resolve().parents[3]
DEFAULT_GGUF = REPO / "dist" / "fusealign-multilingual-ns4-step60000-f32.gguf"
DEFAULT_BIN = REPO / "ggml" / "build" / "tests" / "host" / "test-host-parity"


class Driver:
    def __init__(self, binary):
        self.proc = subprocess.Popen([str(binary)], stdin=subprocess.PIPE, stdout=subprocess.PIPE, text=True,
                                     encoding="utf-8", bufsize=1)

    def __call__(self, op, **request):
        self.proc.stdin.write(json.dumps({"op": op, **request}) + "\n")
        self.proc.stdin.flush()
        line = self.proc.stdout.readline()
        if not line:
            raise RuntimeError("test-host-parity exited")
        response = json.loads(line)
        if "error" in response:
            if response["error"] == "ValueError":
                return ["ValueError", response["message"]]
            raise RuntimeError(f"{op}: {response['message']}")
        return response["ok"]

    def close(self):
        self.proc.stdin.close()
        self.proc.wait()


def python(fn, *args):
    try:
        return fn(*args)
    except ValueError as error:
        return ("ValueError", str(error))


def as_json(value):
    """Python results in the driver's JSON shape (tuples -> lists, sets -> sorted lists, region dicts -> pairs)."""
    if isinstance(value, set):
        return sorted(value)
    if isinstance(value, dict) and set(value) == {"start", "end"}:
        return [float(value["start"]), float(value["end"])]
    if isinstance(value, dict):
        return {k: as_json(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [as_json(v) for v in value]
    return value


def identical(a, b):
    """Bit-identical JSON values: floats compared by their bits (so 0.0 != -0.0), ints and floats kept apart."""
    if isinstance(a, float) or isinstance(b, float):
        return type(a) is type(b) and (a == b and math.copysign(1, a) == math.copysign(1, b) or a != a and b != b)
    if isinstance(a, list) and isinstance(b, list):
        return len(a) == len(b) and all(identical(x, y) for x, y in zip(a, b))
    if isinstance(a, dict) and isinstance(b, dict):
        return a.keys() == b.keys() and all(identical(a[k], b[k]) for k in a)
    return a == b and type(a) is type(b)


def floats(value):
    """Interval results hold floats; Python may carry ints that came in as ints (e.g. fill_unplaced of int input)."""
    if isinstance(value, list):
        return [floats(v) for v in value]
    if isinstance(value, int) and not isinstance(value, bool):
        return float(value)
    return value


class Tally:
    def __init__(self):
        self.counts = {}
        self.failures = []

    def check(self, name, ok, detail=""):
        passed, total = self.counts.get(name, (0, 0))
        self.counts[name] = (passed + bool(ok), total + 1)
        if not ok and len(self.failures) < 30:
            self.failures.append(f"{name}: {detail}"[:2000])


# ----------------------------------------------------------------------------------------------- random cases

ALPHABET_WORDS = ["she", "Had", "YOUR", "dark", "suit", "in", "greasy", "wash", "water", "all", "year", "don't",
                  "a", "an", "the", "I", "1999", "3rd", "x", "co-op", "U.S.A.", "naïve", "café", "İstanbul",
                  "\u212aelvin", "Straße", "中文", "日本語", "한국어", "ภาษาไทย", "Ωμέγα", "ΣΊΣΥΦΟΣ", "—", "...", ",",
                  "“quoted”", "🙂", "e\u0301", "ﬁne", "Ⅻ", "٣٤", "", "ab\u200bcd"]


def random_word(rng):
    roll = rng.random()
    if roll < 0.6:
        return rng.choice(ALPHABET_WORDS[:-2] if rng.random() < 0.95 else ALPHABET_WORDS)
    if roll < 0.9:
        return "".join(rng.choice("abcdefghijklmnopqrstuvwxyzABC0123456789'-") for _ in range(rng.randint(1, 12)))
    if roll < 0.97:
        return "".join(chr(rng.choice([rng.randint(0x4E00, 0x9FFF), rng.randint(0x0E01, 0x0E2E),
                                        rng.randint(0x00C0, 0x024F), rng.randint(0x0391, 0x03C9)]))
                       for _ in range(rng.randint(1, 30)))
    return rng.choice(["x", "ห", "中"]) * rng.randint(200, 700)


def grid_time(rng, hi):
    roll = rng.random()
    if roll < 0.4:
        return round(rng.uniform(0, hi), 2)
    if roll < 0.6:
        return float(rng.randint(0, int(hi)))
    return rng.uniform(0, hi)


def random_regions(rng, duration):
    regions = []
    for _ in range(rng.randint(0, 25)):
        start = grid_time(rng, duration * 1.1) - (1.0 if rng.random() < 0.05 else 0.0)
        end = start + rng.choice([0.0, rng.uniform(0, 8), round(rng.uniform(0.1, 3), 1), -0.5])
        regions.append([start, end])
    if rng.random() < 0.5:
        regions.sort()
    return regions


def random_asr(rng, words, duration):
    asr = []
    t = rng.uniform(0, 2)
    for word in words:
        roll = rng.random()
        if roll < 0.08:
            continue
        if roll < 0.14:
            asr.append({"word": random_word(rng), "start": t, "end": t + 0.2})
            t += 0.2
        step = rng.choice([rng.uniform(0.05, 0.6), round(rng.uniform(0.02, 0.6), 2), 0.0])
        if rng.random() < 0.03:
            step += rng.uniform(10, 40)  # trips the 10 s gate
        start = t if rng.random() < 0.9 else max(0.0, t - rng.uniform(0, 2))
        end = start + rng.uniform(0, 0.5)
        entry = {"word": word.upper() if rng.random() < 0.1 else word, "start": start, "end": end}
        if rng.random() < 0.03:
            entry[rng.choice(["start", "end"])] = None
        asr.append(entry)
        t += step
    if rng.random() < 0.1:
        rng.shuffle(asr)
    if rng.random() < 0.1:
        asr = []
    return asr


def approx_case(rng):
    n = rng.choice([1, 2, 3, rng.randint(1, 30), rng.randint(30, 150)])
    words = [random_word(rng) for _ in range(n)]
    if rng.random() < 0.03:
        words = [""] * n
    duration = rng.choice([rng.uniform(1, 40), rng.uniform(30, 200), round(rng.uniform(1, 100), 3)])
    return words, random_asr(rng, words, duration), random_regions(rng, duration), duration


def random_approx(rng, n, duration):
    roll = rng.random()
    if roll < 0.4:  # sorted starts on a coarse grid (ties)
        starts = sorted(round(rng.uniform(0, duration), 1) for _ in range(n))
    elif roll < 0.7:
        starts = sorted(rng.uniform(0, duration) for _ in range(n))
    else:  # unsorted / overlapping word maps
        starts = [rng.uniform(0, duration) for _ in range(n)]
    return [[s, s + rng.choice([0.0, rng.uniform(0, 1.5), rng.uniform(0, 30)])] for s in starts]


def unroutable_case(rng):
    n = rng.randint(0, 60)
    starts = [float(rng.randint(0, 6)) for _ in range(n)]
    if rng.random() < 0.5:
        starts.sort()
    words = [rng.choice(["a", "y" * rng.randint(1, 200), "x" * rng.randint(900, 1100), "ห" * rng.randint(1, 400),
                         random_word(rng)]) for _ in range(n)]
    return words, [[s, s + 1.0] for s in starts]


def plan_case(rng):
    duration = rng.choice([rng.uniform(5, 30), rng.uniform(30, 120), rng.uniform(100, 400), 26.0, 52.0])
    n = rng.randint(1, 400)
    if rng.random() < 0.5:
        words, asr, regions, _ = approx_case(rng)
        words = words[:n] or ["word"]
        approx = longform.approximate_times(words, asr, regions, duration)
    else:
        words = [rng.choice(["w" * rng.randint(1, 60), random_word(rng)]) for _ in range(n)]
        words = [w if len(w.encode()) <= 1022 or rng.random() < 0.05 else w[:10] or "a" for w in words]
        regions = random_regions(rng, duration)
        approx = random_approx(rng, n, duration)
    if rng.random() < 0.05:
        approx = approx[:-1]
    return words, approx, regions, duration


def stitch_case(rng):
    n_words = rng.randint(0, 40)
    chunks, predictions = [], []
    for _ in range(rng.randint(0, 6) if n_words else 0):
        a = grid_time(rng, 100)
        b = a + rng.choice([26.0, rng.uniform(0.5, 26)])
        lo = rng.randint(0, n_words)
        hi = rng.randint(lo, n_words)
        chunks.append({"core": [a, b], "audio": [max(0.0, a - 2.0), b + 2.0], "indices": list(range(lo, hi))})
        placed = []
        for _ in range(hi - lo):
            if rng.random() < 0.2:
                placed.append(None)
            else:
                start = rng.choice([a + rng.uniform(-4, 30), round(a + rng.uniform(-4, 30), 2)])
                placed.append([start, start + rng.choice([0.0, rng.uniform(0, 1)])])
        predictions.append(placed)
    if len(chunks) > 1 and rng.random() < 0.3:  # identical predictions in two chunks: ties keep the first
        chunks[1] = dict(chunks[0], core=chunks[1]["core"])
        predictions[1] = predictions[0]
    return n_words, chunks, predictions


def fill_case(rng):
    n = rng.randint(0, 30)
    out = []
    for _ in range(n):
        if rng.random() < 0.5:
            out.append(None)
        else:
            start = grid_time(rng, 30)
            out.append([start, start + rng.uniform(-1, 3)])
    return out, rng.choice([rng.uniform(1, 40), float(rng.randint(1, 40))])


SPACES = [chr(c) for c in range(0x110000) if chr(c).isspace()]
TEXT_CHARS = list("abcXYZ019.,'-") + ["中", "ห", "é", "İ", "🙂", "\u200b", "\u2060", "\ufeff", "\u0301", "—"]


def text_case(rng):
    return "".join(rng.choice(SPACES) if rng.random() < 0.3 else rng.choice(TEXT_CHARS)
                   for _ in range(rng.randint(0, 40)))


def token_case(rng):
    pool = TEXT_CHARS + SPACES + [chr(rng.randint(0, 0x10FFFF)) for _ in range(20)]
    pool = [c for c in pool if not 0xD800 <= ord(c) <= 0xDFFF]
    return ["".join(rng.choice(pool) for _ in range(rng.randint(0, 6))) for _ in range(rng.randint(1, 30))]


def randomized(drive, tally, rng, cases):
    for _ in range(cases["approximate_times"]):
        words, asr, regions, duration = approx_case(rng)
        want = as_json(python(longform.approximate_times, words, asr, regions, duration))
        got = drive("approximate_times", words=words, asr_words=asr, speech_regions=regions, duration=duration)
        tally.check("approximate_times", identical(floats(got), floats(want)), f"{words!r} {asr!r} {regions!r} {duration!r}")
    for _ in range(cases["unroutable"]):
        words, approx = unroutable_case(rng)
        want = as_json(longform.unroutable(words, approx))
        got = drive("unroutable", words=words, approx=approx)
        tally.check("unroutable", identical(got, want), f"{words!r} {approx!r}")
    for _ in range(cases["plan_chunks"]):
        words, approx, regions, duration = plan_case(rng)
        want = as_json(python(longform.plan_chunks, words, approx, regions, duration))
        got = drive("plan_chunks", words=words, approx=approx, speech_regions=regions, duration=duration)
        tally.check("plan_chunks", identical(floats(got), floats(want)), f"{len(words)} words, {duration!r}: {got!r:.300} vs {want!r:.300}")
    for _ in range(cases["stitch_predictions"]):
        n_words, chunks, predictions = stitch_case(rng)
        want = as_json(longform.stitch_predictions(n_words, chunks, predictions))
        got = drive("stitch_predictions", n_words=n_words, chunks=chunks, predictions=predictions)
        tally.check("stitch_predictions", identical(floats(got), floats(want)), f"{n_words} {chunks!r} {predictions!r}")
    for _ in range(cases["fill_unplaced"]):
        intervals, duration = fill_case(rng)
        want = as_json(longform.fill_unplaced(intervals, duration))
        got = drive("fill_unplaced", intervals=intervals, duration=duration)
        tally.check("fill_unplaced", identical(floats(got), floats(want)), f"{intervals!r} {duration!r}")
    for _ in range(cases["split"]):
        text = text_case(rng)
        tally.check("split", drive("split", text=text) == text.split(), repr(text))
    for _ in range(cases["is_wordy"]):
        tokens = token_case(rng)
        got = drive("is_wordy", tokens=tokens)
        tally.check("is_wordy", got["is_wordy"] == [_is_wordy(t) for t in tokens]
                    and got["has_whitespace"] == [any(ch.isspace() for ch in t) for t in tokens], repr(tokens))


def codepoints(drive, tally):
    got = drive("codepoints")
    import re

    wordy, space, key = [], [], []
    for cp in range(0x110000):
        if 0xD800 <= cp <= 0xDFFF:
            wordy.append("-"), space.append("-"), key.append("-")
            continue
        ch = chr(cp)
        wordy.append("1" if _is_wordy(ch) else "0")
        space.append("1" if ch.isspace() else "0")
        key.append(re.sub(r"[^0-9a-z]", "", ch.lower()) or ".")
    for name, want in (("is_wordy", wordy), ("isspace", space), ("key", key)):
        want = "".join(want)
        bad = [i for i, (a, b) in enumerate(zip(got[name], want)) if a != b]
        tally.check(f"codepoints.{name}", got[name] == want, f"{len(bad)} differ, first U+{bad[0]:04X}" if bad else "length")


# ------------------------------------------------------------------------------------------------- audio cases

def write_f32(path, array):
    np.ascontiguousarray(array, dtype=np.float32).tofile(path)


def build_vad_gguf(path):
    import gguf
    import silero_vad

    threads = torch.get_num_threads()
    model = silero_vad.load_silero_vad()
    torch.set_num_threads(threads)
    writer = gguf.GGUFWriter(str(path), "fusealign")
    for key, value in model.state_dict().items():
        if key.startswith("_model."):
            writer.add_tensor("vad." + key.removeprefix("_model."), value.detach().numpy().astype(np.float32))
    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()


def resample_cases(drive, tally, rng, tmp, real):
    import torchaudio

    worst = 0.0
    lengths = [1, 7, 160, 441, 1000, 16000, 44100 * 3 + 17]
    pairs = [(8000, 16000), (22050, 16000), (44100, 16000), (48000, 16000), (16000, 8000)]
    signals = []
    for n in lengths:
        signals.append(("random", torch.from_numpy(np.random.default_rng(rng.randint(0, 2**31)).uniform(-1, 1, n)
                                                   .astype(np.float32))))
    for path in real:
        import soundfile

        signals.append((Path(path).name, torch.from_numpy(soundfile.read(path, dtype="float32")[0])))
    for name, x in signals:
        for orig, new in pairs:
            want = torchaudio.functional.resample(x, orig, new).numpy()
            write_f32(tmp / "in.f32", x.numpy())
            got_n = drive("resample", **{"in": str(tmp / "in.f32"), "n": x.numel(), "orig": orig, "new": new,
                                         "out": str(tmp / "out.f32")})["n"]
            got = np.fromfile(tmp / "out.f32", dtype=np.float32)
            ok = got_n == want.size and got.shape == want.shape
            diff = float(np.abs(got - want).max()) if ok and want.size else 0.0
            worst = max(worst, diff)
            tally.check("resample", ok and diff <= 1e-6, f"{name} {orig}->{new}: n {got_n} vs {want.size}, diff {diff}")
    return worst


def read_wav_cases(drive, tally, rng, tmp, paths):
    import soundfile

    synthetic = []
    noise = np.random.default_rng(rng.randint(0, 2**31)).uniform(-1, 1, (3001, 2)).astype(np.float64)
    noise[0] = [1.0, -1.0]
    for fmt, subtype in (("WAV", "PCM_U8"), ("WAV", "PCM_16"), ("WAV", "PCM_24"), ("WAV", "PCM_32"), ("WAV", "FLOAT"),
                         ("WAV", "DOUBLE"), ("WAVEX", "PCM_16"), ("WAVEX", "PCM_24"), ("WAVEX", "FLOAT")):
        for channels in (1, 2):
            path = tmp / f"synthetic_{fmt}_{subtype}_{channels}.wav"
            soundfile.write(path, noise[:, :channels], 22050 if channels == 2 else 16000, subtype=subtype, format=fmt)
            synthetic.append(path)
    for path in [*synthetic, *paths]:
        data, sr = soundfile.read(path, dtype="float32", always_2d=True)
        got = drive("read_wav", path=str(path), out=str(tmp / "wav.f32"))
        samples = np.fromfile(tmp / "wav.f32", dtype=np.float32)
        want = data.reshape(-1)
        ok = (got["sample_rate"] == sr and got["channels"] == data.shape[1] and samples.shape == want.shape
              and np.array_equal(samples.view(np.uint32), want.view(np.uint32)))
        tally.check("read_wav", ok, f"{path}")


def timit_long(drive, tally, tmp, gguf_path, eval_dir):
    import soundfile

    rows = [json.loads(line) for line in (eval_dir / "timit_long" / "inputs_ref.jsonl").read_text().splitlines()]
    asr = {r["clip_id"]: r["words"] for r in map(json.loads, (eval_dir / "timit_long" / "asr_whisper.jsonl")
                                                  .read_text().splitlines())}
    worst, speed = 0.0, []
    for row in rows:
        wav, sr = soundfile.read(row["audio"], dtype="float32")
        assert sr == 16000 and wav.ndim == 1
        got_wav = drive("read_wav", path=row["audio"], out=str(tmp / "wav.f32"))
        same_wav = np.array_equal(np.fromfile(tmp / "wav.f32", dtype=np.float32).view(np.uint32), wav.view(np.uint32))
        tally.check("read_wav", same_wav and got_wav["sample_rate"] == 16000, row["audio"])

        x = torch.from_numpy(wav)
        want_probs = vad.silero_probs(x).numpy()
        write_f32(tmp / "in.f32", wav)
        got = drive("vad_probs", **{"in": str(tmp / "in.f32"), "n": wav.size, "out": str(tmp / "probs.f32"),
                                    "gguf": str(gguf_path)})
        probs = np.fromfile(tmp / "probs.f32", dtype=np.float32)
        ok = probs.shape == want_probs.shape
        diff = float(np.abs(probs - want_probs).max()) if ok else math.inf
        worst = max(worst, diff)
        speed.append(got["seconds"] / (wav.size / 16000) * 80.0)
        tally.check("vad_probs", ok and diff <= 1e-5, f"{row['clip_id']}: diff {diff}")

        want_regions = as_json(vad.speech_regions(x))
        got_regions = drive("speech_regions", **{"in": str(tmp / "in.f32"), "n": wav.size, "gguf": str(gguf_path)})
        tally.check("speech_regions", identical(floats(got_regions), floats(want_regions)),
                    f"{row['clip_id']}: {got_regions} vs {want_regions}")

        duration = wav.size / 16000
        for anchors in (asr[row["clip_id"]], []):
            approx = longform.approximate_times(row["words"], anchors, want_regions, duration)
            skipped = longform.unroutable(row["words"], approx)
            keep = [i for i in range(len(row["words"])) if i not in skipped]
            chunks = longform.plan_chunks([row["words"][i] for i in keep], [approx[i] for i in keep], want_regions,
                                          duration)
            for chunk in chunks:
                chunk["indices"] = [keep[i] for i in chunk["indices"]]
            want = as_json({"approx": approx, "skipped": skipped, "chunks": chunks})
            got = drive("long_plan", words=row["words"], asr_words=anchors, speech_regions=got_regions, duration=duration)
            tally.check("long_plan" + ("[whisper]" if anchors else "[vad]"), identical(floats(got), floats(want)),
                        row["clip_id"])
    return worst, speed


def vad_edge_cases(drive, tally, rng, tmp, gguf_path):
    """Lengths around frame and block boundaries (Python's 4096-frame blocks, the port's 2048)."""
    worst = 0.0
    for n in (1, 511, 512, 513, 2047 * 512 + 100, 2048 * 512, 4097 * 512 + 1):
        x = (np.random.default_rng(rng.randint(0, 2**31)).standard_normal(n) * 0.1).astype(np.float32)
        want = vad.silero_probs(torch.from_numpy(x)).numpy()
        write_f32(tmp / "in.f32", x)
        drive("vad_probs", **{"in": str(tmp / "in.f32"), "n": n, "out": str(tmp / "probs.f32"), "gguf": str(gguf_path)})
        got = np.fromfile(tmp / "probs.f32", dtype=np.float32)
        ok = got.shape == want.shape
        diff = float(np.abs(got - want).max()) if ok else math.inf
        worst = max(worst, diff)
        tally.check("vad_probs", ok and diff <= 1e-5, f"random n={n}: diff {diff}")
    return worst


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bin", type=Path, default=DEFAULT_BIN)
    parser.add_argument("--gguf", type=Path, default=None)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--scale", type=float, default=1.0, help="multiplies the randomized case counts")
    parser.add_argument("--wavs", type=int, default=300, help="TIMIT and BZNSYP WAVs each for read_wav")
    parser.add_argument("--eval-dir", type=Path, help="real-recording inputs (layout above); skipped when absent")
    args = parser.parse_args()

    rng = random.Random(args.seed)
    torch.manual_seed(args.seed)
    tally = Tally()
    drive = Driver(args.bin)
    base = {"approximate_times": 800, "unroutable": 400, "plan_chunks": 600, "stitch_predictions": 400,
            "fill_unplaced": 400, "split": 300, "is_wordy": 300}
    cases = {k: max(1, int(v * args.scale)) for k, v in base.items()}

    with tempfile.TemporaryDirectory(prefix="host_parity_") as tmp:
        tmp = Path(tmp)
        gguf_path = args.gguf or DEFAULT_GGUF
        if not gguf_path.exists():
            gguf_path = tmp / "vad.gguf"
            build_vad_gguf(gguf_path)
        print(f"VAD weights: {gguf_path}", flush=True)

        t0 = time.time()
        randomized(drive, tally, rng, cases)
        codepoints(drive, tally)
        print(f"longform/text: {time.time() - t0:.1f} s", flush=True)

        wavs, speed, vad_worst = [], [], 0.0
        if args.eval_dir:
            timit_rows, bzn_rows = (
                [json.loads(line) for line in (args.eval_dir / name / "inputs_ref.jsonl").read_text().splitlines()]
                for name in ("timit", "bznsyp"))
            wavs = ([r["audio"] for r in rng.sample(timit_rows, args.wavs)]
                    + [r["audio"] for r in rng.sample(bzn_rows, args.wavs)])
        read_wav_cases(drive, tally, rng, tmp, wavs)
        resample_worst = resample_cases(drive, tally, rng, tmp, wavs[:2] + wavs[-2:])
        if args.eval_dir:
            vad_worst, speed = timit_long(drive, tally, tmp, gguf_path, args.eval_dir)
        else:
            print("no --eval-dir: real-recording cases skipped", flush=True)
        vad_worst = max(vad_worst, vad_edge_cases(drive, tally, rng, tmp, gguf_path))
    drive.close()

    for name, (passed, total) in sorted(tally.counts.items()):
        print(f"{name:28s} {passed:6d}/{total:<6d} {'ok' if passed == total else 'FAIL'}")
    randomized_total = sum(tally.counts[k][1] for k in base)
    print(f"randomized longform/text cases: {randomized_total}")
    print(f"VAD probs max abs diff: {vad_worst:.3g}; resample max abs diff: {resample_worst:.3g}")
    if speed:
        print(f"VAD seconds per 80 s of audio (C++, CPU): median {np.median(speed):.3f}, max {max(speed):.3f}")
    for failure in tally.failures:
        print("FAIL", failure)
    sys.exit(0 if all(p == t for p, t in tally.counts.values()) else 1)


if __name__ == "__main__":
    main()
