import math
import unittest
from unittest import mock

import torch
from torch.nn import functional as F

from fusealign_hub import vad


class SileroTests(unittest.TestCase):
    def test_batched_probabilities_match_the_streaming_model_across_block_seams(self):
        import silero_vad

        generator = torch.Generator().manual_seed(0)
        wav = 0.05 * torch.randn(159700, generator=generator)  # last 512-frame is partial
        t = torch.arange(40000, 100000) / 16000
        wav[40000:100000] += 0.3 * torch.sin(2 * math.pi * 220 * t) * (1 + torch.sin(2 * math.pi * 3 * t))

        threads = torch.get_num_threads()
        try:
            model = silero_vad.load_silero_vad()
        finally:
            torch.set_num_threads(threads)
        model.reset_states()
        with torch.inference_mode():
            expected = torch.tensor([
                model(F.pad(window, (0, 512 - window.numel())), 16000).item()
                for window in wav.split(512)
            ])

        with mock.patch.object(vad, "VAD_BLOCK_FRAMES", 37):
            torch.testing.assert_close(vad.silero_probs(wav), expected, atol=1e-4, rtol=0)
        self.assertEqual(
            vad.speech_regions(wav),
            silero_vad.get_speech_timestamps(wav, model, sampling_rate=16000, return_seconds=True),
        )


if __name__ == "__main__":
    unittest.main()
