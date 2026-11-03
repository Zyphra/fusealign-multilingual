import unittest

import torch

from fusealign_hub import fill_unplaced, guide_words, longform
from tiny import tiny_aligner


class FillUnplacedTests(unittest.TestCase):
    def test_null_runs_split_their_gap_and_placed_stay_verbatim(self):
        got = fill_unplaced([None, (1, 2), None, None, (5, 6), None], 8)
        self.assertEqual(got, [(0, 1), (1, 2), (2, 3.5), (3.5, 5), (5, 6), (6, 8)])

    def test_all_unplaced_divides_the_clip_equally(self):
        self.assertEqual(fill_unplaced([None] * 3, 3), [(0, 1), (1, 2), (2, 3)])


class GuideWordsTests(unittest.TestCase):
    def test_tokens_interpolate_across_the_utterance_and_untimed_are_skipped(self):
        got = guide_words([
            {"text": "a b c", "start": 0, "end": 3},
            {"text": "d", "start": None, "end": 4},
        ])
        self.assertEqual(
            [(g["word"], g["start"], g["end"]) for g in got],
            [("a", 0, 1), ("b", 1, 2), ("c", 2, 3)],
        )


class LongFormTests(unittest.TestCase):
    def test_over_30s_is_chunked_and_every_word_lands_inside_the_recording(self):
        aligner = tiny_aligner()
        torch.manual_seed(1)
        wav = torch.randn(35 * 16000)
        transcript = " ".join(f"word{i}" for i in range(60))
        words = aligner.align(wav, 16000, transcript, asr_words=[])
        self.assertEqual([w.word for w in words], transcript.split())
        for w in words:
            self.assertTrue(0 <= w.start <= w.end <= 35.0, w)

    def test_a_word_no_chunk_can_hold_is_left_unplaced_not_fatal(self):
        """One 1,500-byte token used to fail the whole recording."""
        aligner = tiny_aligner()
        torch.manual_seed(1)
        wav = torch.randn(35 * 16000)
        junk = "ห" * 500  # 1,500 UTF-8 bytes
        tokens = [f"word{i}" for i in range(30)] + [junk] + [f"word{i}" for i in range(30, 60)]
        words = aligner.align(wav, 16000, " ".join(tokens), asr_words=[])
        self.assertEqual([w.word for w in words], tokens)
        self.assertIs(words[30].placed, False)
        self.assertEqual(words[30].presence, 0.0)
        for w in words:  # a fallback fill between overlapping neighbours may be inverted
            self.assertTrue(0 <= min(w.start, w.end) and max(w.start, w.end) <= 35.0, w)


class UnroutableTests(unittest.TestCase):
    def test_oversized_word_and_oversized_equal_start_run_only(self):
        big = "x" * 1023
        words = ["a", big, "b"] + ["y" * 100] * 11 + ["c"]
        approx = [(0.0, 1.0), (1.0, 2.0), (2.0, 3.0)] + [(5.0, 6.0)] * 11 + [(7.0, 8.0)]
        # 11 x 100 bytes + 10 spaces = 1,110 bytes share one start: no time cut separates them.
        self.assertEqual(longform.unroutable(words, approx), {1, *range(3, 14)})

    def test_equal_start_run_that_fits_is_kept(self):
        words = ["y" * 100] * 10
        self.assertEqual(longform.unroutable(words, [(5.0, 6.0)] * 10), set())


if __name__ == "__main__":
    unittest.main()
