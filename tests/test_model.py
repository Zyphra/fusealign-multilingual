import unittest

import torch

from fusealign_hub import FuseAlign, FuseAlignArgs


class PaddedBatchTests(unittest.TestCase):
    def test_a_clip_scores_the_same_alone_and_beside_a_longer_one(self):
        torch.manual_seed(0)
        args = FuseAlignArgs(dim=64, n_layers=1, n_heads=4, ffn_dim=128, feat_dim=16, text_dim=32)
        model = FuseAlign(args).eval()
        clips = [torch.randn(16000), torch.randn(40000)]
        counts = [3, 5]
        bank = torch.randn(2, 10, 32)

        def run(rows):
            # Word k is the single byte [2k, 2k+1), as in "x x x"; row r's text has 2n-1 bytes.
            ids = [2 + 2 * counts[r] - 1 for r in rows]
            text_mask = torch.zeros(len(rows), max(ids), dtype=torch.bool)
            text_feats = torch.zeros(len(rows), max(ids) - 2, 32)
            word_spans = torch.zeros(len(rows), max(counts[r] for r in rows), 2, dtype=torch.long)
            word_mask = torch.zeros(len(rows), word_spans.shape[1], dtype=torch.bool)
            wav = torch.zeros(len(rows), max(clips[r].numel() for r in rows))
            for i, r in enumerate(rows):
                text_mask[i, :ids[i]] = True
                text_feats[i, :ids[i] - 2] = bank[r, :ids[i] - 2]
                for k in range(counts[r]):
                    word_spans[i, k] = torch.tensor([2 * k, 2 * k + 1])
                word_mask[i, :counts[r]] = True
                wav[i, :clips[r].numel()] = clips[r]
            with torch.no_grad():
                return model(
                    wav=wav,
                    wav_lengths=torch.tensor([clips[r].numel() for r in rows]),
                    text_feats=text_feats, text_mask=text_mask,
                    word_spans=word_spans, word_mask=word_mask,
                )

        alone, beside = run([0]), run([0, 1])
        frames = int(alone["frame_lengths"][0])
        self.assertEqual(frames, int(beside["frame_lengths"][0]))
        for name in ("start_logits", "end_logits"):
            torch.testing.assert_close(
                beside[name][0, :3, :frames], alone[name][0, :3, :frames], atol=1e-5, rtol=0
            )
        torch.testing.assert_close(
            beside["presence_logits"][0, :3], alone["presence_logits"][0, :3], atol=1e-5, rtol=0
        )
