import unittest

import torch

from tiny import tiny_aligner


class TinyModelTests(unittest.TestCase):
    def setUp(self):
        self.aligner = tiny_aligner()
        self.wav = torch.randn(32000)

    def test_one_alignment_per_token_with_punctuation_zero_width(self):
        words = self.aligner.align(self.wav, 16000, ", hello , world —")
        self.assertEqual([w.word for w in words], [",", "hello", ",", "world", "—"])
        lead, hello, comma, world, dash = words
        self.assertEqual((lead.start, lead.end), (hello.start, hello.start))
        self.assertEqual((comma.start, comma.end), (hello.end, hello.end))
        self.assertEqual((dash.start, dash.end), (world.end, world.end))
        for w in (lead, comma, dash):
            self.assertIs(w.placed, False)
            self.assertEqual(w.presence, 0.0)
        for w in (hello, world):
            self.assertTrue(0 <= w.start <= w.end <= 2.0)
            self.assertTrue(0 <= w.presence <= 1)

    def test_audio_under_one_patch_is_all_unplaced_across_the_clip(self):
        words = self.aligner.align(torch.randn(160), 16000, "hello world")
        self.assertEqual([w.placed for w in words], [False, False])
        self.assertEqual([(w.start, w.end) for w in words], [(0.0, 0.005), (0.005, 0.01)])

    def test_stereo_waveform_is_rejected(self):
        with self.assertRaises(ValueError):
            self.aligner.intervals_batch([torch.randn(2, 16000)], [["a"]])

    def test_a_batch_gives_each_clip_what_aligning_it_alone_gives(self):
        # Singleton forwards make the equality exact; B>1 numerics are PaddedBatchTests'.
        self.aligner.batch_size = 1
        clips = [torch.randn(48000), torch.randn(160), torch.randn(16000),
                 torch.randn(16000), torch.randn(24000)]
        texts = ["a b c", "x y", "one", "", ", ."]
        got = self.aligner.align_batch(clips, 16000, texts)
        want = [self.aligner.align(c, 16000, t) for c, t in zip(clips, texts)]
        self.assertEqual(got, want)

    def test_a_token_with_whitespace_is_rejected(self):
        with self.assertRaises(ValueError):
            self.aligner.align(self.wav, 16000, ["hello", "big world"])


if __name__ == "__main__":
    unittest.main()
