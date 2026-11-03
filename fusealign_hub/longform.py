"""Long-form routing (FuseAlign paper, arXiv:2609.33650, Appendix G.3): approximate word times,
26 s cores with 2 s of context cut at silences, and stitching. ``fill_unplaced`` is the paper's
uniform fallback for unplaced words (Section 3).
"""

from bisect import bisect_left, bisect_right
from difflib import SequenceMatcher
import re

_CORE_SECONDS = 26.0
_CONTEXT_SECONDS = 2.0
MAX_TEXT_BYTES = 1022


def _oversized_start_group(words, approx):
    previous = None
    size = 0
    for word, (start, _) in zip(words, approx):
        size = size + 1 if start == previous else 0
        size += len(word.encode("utf-8"))
        if size > MAX_TEXT_BYTES:
            return True
        previous = start
    return False


def unroutable(words, approx) -> set[int]:
    """Indices of the words no chunk can hold: one over ``MAX_TEXT_BYTES``, and every word of a
    run sharing one approximate start (no time cut separates them) whose text, once those are
    left out, is over it too. ``plan_chunks`` raises for either; ``_long`` leaves them unplaced."""
    out = {i for i, word in enumerate(words) if len(word.encode("utf-8")) > MAX_TEXT_BYTES}
    run, size, previous = [], 0, None
    for i, (word, (start, _)) in enumerate(zip(words, approx)):
        if i in out:
            continue
        length = len(word.encode("utf-8"))
        if run and start == previous:
            run.append(i)
            size += 1 + length
        else:
            if size > MAX_TEXT_BYTES:
                out.update(run)
            run, size = [i], length
        previous = start
    if size > MAX_TEXT_BYTES:
        out.update(run)
    return out


def _speech_intervals(speech_regions, duration):
    """Clip and union speech regions before measuring speech or its complement."""
    intervals = []
    for region in speech_regions:
        if isinstance(region, dict):
            start, end = region["start"], region["end"]
        else:
            start, end = region
        start, end = max(0.0, float(start)), min(duration, float(end))
        if end > start:
            intervals.append((start, end))
    merged = []
    for start, end in sorted(intervals):
        if merged and start <= merged[-1][1]:
            merged[-1] = (merged[-1][0], max(merged[-1][1], end))
        else:
            merged.append((start, end))
    return merged


def _speech_fallback(words, speech_regions, duration):
    regions = _speech_intervals(speech_regions, duration)
    if not regions:
        return [(0.0, 0.0) for _ in words]
    cumulative = [0.0]
    for start, end in regions:
        cumulative.append(cumulative[-1] + end - start)
    lengths = [len(word) for word in words]
    if not sum(lengths):
        lengths = [1] * len(words)
    total_chars = sum(lengths)

    def recording_time(speech_time, *, is_end):
        # At a speech-region seam, starts belong to the next region and ends
        # to the preceding one, so a word ending at a seam excludes silence.
        search = bisect_left if is_end else bisect_right
        index = max(0, min(len(regions) - 1, search(cumulative, speech_time) - 1))
        return regions[index][0] + speech_time - cumulative[index]

    times = []
    chars = 0
    for length in lengths:
        start = recording_time(cumulative[-1] * chars / total_chars, is_end=False)
        chars += length
        end = recording_time(cumulative[-1] * chars / total_chars, is_end=True)
        times.append((start, max(start, end)))
    return times


def approximate_times(words, asr_words, speech_regions, duration):
    """Match gated ASR anchors, interpolate, or use a VAD character-time map.

    ``asr_words`` contains dictionaries with ``word``, ``start``, and ``end``;
    VAD regions may be dictionaries or ``(start, end)`` pairs in seconds.
    """
    if not words:
        return []
    asr = [
        word for word in asr_words
        if word.get("start") is not None and word.get("end") is not None
    ]
    key = lambda word: re.sub(r"[^0-9a-z]", "", word.lower())
    matcher = SequenceMatcher(
        None, [key(word) for word in words], [key(word["word"]) for word in asr],
        autojunk=False,
    )
    candidates = [
        (target + offset, source + offset)
        for target, source, size in matcher.get_matching_blocks()
        for offset in range(size)
    ]
    anchors = []
    for target, source in candidates:
        start, end = float(asr[source]["start"]), float(asr[source]["end"])
        if not anchors or start - anchors[-1][2] <= 10.0 + 0.5 * (target - anchors[-1][0] - 1):
            anchors.append((target, start, end))
    minimum = 1 if len(words) <= 2 else 2
    if len(anchors) < minimum or len(anchors) < 0.5 * len(candidates):
        return _speech_fallback(words, speech_regions, duration)

    times = [None] * len(words)
    for index, start, end in anchors:
        times[index] = (start, end)

    def spread(lo, hi, start, end):
        count = hi - lo
        for offset in range(count):
            times[lo + offset] = (
                start + (end - start) * offset / count,
                start + (end - start) * (offset + 1) / count,
            )

    first, last = anchors[0], anchors[-1]
    spread(0, first[0], float(asr[0]["start"]), first[1])
    for left, right in zip(anchors, anchors[1:]):
        spread(left[0] + 1, right[0], left[2], right[1])
    spread(last[0] + 1, len(words), last[2], float(asr[-1]["end"]))
    previous_start = float("-inf")
    result = []
    for start, end in times:
        start = max(start, previous_start)
        result.append((start, max(start, end)))
        previous_start = start
    if _oversized_start_group(words, result):
        # Time cuts cannot separate words that share one approximate start; a run of them over
        # MAX_TEXT_BYTES falls back to the VAD character-time map.
        return _speech_fallback(words, speech_regions, duration)
    return result


def _cut_tiers(approx, speech_regions, duration):
    silences = []
    previous_end = 0.0
    for start, end in _speech_intervals(speech_regions, duration):
        if start - previous_end >= 0.5:
            silences.append((start + previous_end) / 2.0)
        previous_end = end
    if duration - previous_end >= 0.5:
        silences.append((duration + previous_end) / 2.0)

    # Test silence midpoints against the union of approximate word intervals.
    # Keeping cumulative maximum ends also handles overlapping word maps.
    ordered = sorted(approx)
    starts, furthest_ends = [], []
    furthest = float("-inf")
    for start, end in ordered:
        starts.append(start)
        furthest = max(furthest, end)
        furthest_ends.append(furthest)
    safe_silences = []
    for point in silences:
        index = bisect_right(starts, point) - 1
        if index < 0 or furthest_ends[index] < point:
            safe_silences.append(point)
    word_gaps = sorted({(left[1] + right[0]) / 2.0 for left, right in zip(approx, approx[1:])})
    return safe_silences, silences, word_gaps


def plan_chunks(words, approx, speech_regions, duration):
    """Return cores, context audio ranges, and original word indices.

    Cores are at most 26 seconds and audio adds at most two seconds on either
    side. Oversized transcripts first lose context and then split their core
    recursively, preserving time-based word ownership.
    """
    if not words:
        return []
    if len(words) != len(approx):
        raise ValueError("Word and approximate-time counts differ")
    starts = [float(interval[0]) for interval in approx]
    byte_prefix = [0]
    for word in words:
        length = len(word.encode("utf-8"))
        if length > MAX_TEXT_BYTES:
            raise ValueError("A single word exceeds the 1022-byte transcript limit")
        byte_prefix.append(byte_prefix[-1] + length)

    def text_bytes(lo, hi):
        return byte_prefix[hi] - byte_prefix[lo] + max(0, hi - lo - 1)

    # Equal approximate starts cannot be separated by any time-based cut.
    if _oversized_start_group(words, approx):
        raise ValueError("Words sharing an approximate start exceed the 1022-byte limit")

    tiers = _cut_tiers(approx, speech_regions, duration)
    cores = []
    a = 0.0
    while duration - a > _CORE_SECONDS:
        b = a + _CORE_SECONDS
        for tier in tiers:
            right = bisect_right(tier, b)
            if right and tier[right - 1] > a:
                b = tier[right - 1]
                break
        cores.append((a, b))
        a = b
    cores.append((a, float(duration)))

    def route(core):
        a, b = core
        audio = (max(0.0, a - _CONTEXT_SECONDS), min(duration, b + _CONTEXT_SECONDS))
        # The first/last cores also own starts on the recording's outer edges.
        owned_lo = 0 if a == 0.0 else bisect_left(starts, a)
        owned_hi = len(words) if b == duration else bisect_left(starts, b)
        lo, hi = owned_lo, owned_hi

        def overlaps(index):
            start, end = approx[index]
            return start < audio[1] and end > audio[0]

        while lo > 0 and overlaps(lo - 1):
            lo -= 1
        while hi < len(words) and overlaps(hi):
            hi += 1
        if text_bytes(lo, hi) > MAX_TEXT_BYTES:
            lo, hi = owned_lo, owned_hi
        if text_bytes(lo, hi) <= MAX_TEXT_BYTES:
            return {"core": core, "audio": audio, "indices": list(range(lo, hi))}
        return None

    result = []
    pending = list(reversed(cores))
    while pending:
        core = pending.pop()
        chunk = route(core)
        if chunk is not None:
            result.append(chunk)
            continue
        a, b = core
        midpoint = (a + b) / 2.0
        cut = midpoint
        for tier in tiers:
            lo, hi = bisect_right(tier, a), bisect_left(tier, b)
            if lo < hi:
                near = bisect_left(tier, midpoint, lo, hi)
                choices = tier[max(lo, near - 1):min(hi, near + 1)]
                cut = min(choices, key=lambda point: (abs(point - midpoint), -point))
                break
        if not a < cut < b:
            raise ValueError("Cannot split an oversized transcript at distinct recording times")
        pending.append((cut, b))
        pending.append((a, cut))
    return result


def stitch_predictions(n_words, chunks, predictions):
    """Select one placed recording-time prediction for each original word.

    An in-core center beats every out-of-core center; among in-core centers,
    prefer the largest margin to an edge. Otherwise prefer the smallest
    distance to the core. Ties retain the first prediction in chunk order.
    """
    result = [None] * n_words
    if not n_words:
        return result
    priorities = [None] * n_words
    for chunk, placed in zip(chunks, predictions, strict=True):
        a, b = chunk["core"]
        for index, interval in zip(chunk["indices"], placed, strict=True):
            if interval is None:
                continue
            start, end = interval
            center = (start + end) / 2.0
            if a <= center <= b:
                priority = (1, min(center - a, b - center))
            else:
                priority = (0, -max(a - center, center - b))
            if priorities[index] is None or priority > priorities[index]:
                priorities[index] = priority
                result[index] = (start, end)
    return result


def fill_unplaced(intervals, duration):
    """Fill each maximal null run with equal parts of its surrounding gap.

    A leading gap begins at zero; a trailing gap ends at ``duration``. With no
    placed words the entire clip is divided equally. Placed entries are kept
    verbatim, without clamping, reordering, or correcting overlapping intervals.
    """
    filled = list(intervals)
    index = 0
    while index < len(filled):
        if filled[index] is not None:
            index += 1
            continue
        lo = index
        while index < len(filled) and filled[index] is None:
            index += 1
        left = filled[lo - 1][1] if lo else 0.0
        right = filled[index][0] if index < len(filled) else duration
        width = (right - left) / (index - lo)
        for offset in range(index - lo):
            filled[lo + offset] = (left + offset * width,
                                  right if lo + offset + 1 == index
                                  else left + (offset + 1) * width)
    return filled


def guide_words(utterances, words_of=str.split) -> list[dict]:
    """Turn timed utterances into the per-token ``{"word", "start", "end"}``
    list that ``approximate_times`` consumes.

    Token ``i`` of ``n`` spans ``[start + i/n*(end-start), start + (i+1)/n*(end-start)]``:
    the paper's interpolation between anchors, with the utterance edges as the
    anchors. Utterances with a missing time or no tokens are skipped.
    """
    out = []
    for utterance in utterances:
        start, end = utterance.get("start"), utterance.get("end")
        tokens = words_of(utterance["text"])
        if start is None or end is None or not tokens:
            continue
        n = len(tokens)
        for i, token in enumerate(tokens):
            out.append({
                "word": token,
                "start": start + i / n * (end - start),
                "end": start + (i + 1) / n * (end - start),
            })
    return out
