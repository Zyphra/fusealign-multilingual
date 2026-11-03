"""A tiny random FuseAlign with a stub text encoder, for tests that need no checkpoint."""

import torch
from torch import nn

from fusealign_hub import FuseAlign, FuseAlignAligner, FuseAlignArgs

TINY_ARGS = dict(dim=64, n_layers=1, n_heads=4, ffn_dim=128, feat_dim=16, text_dim=32)


class StubEncoder(nn.Module):
    def forward(self, ids, mask):
        return torch.zeros(ids.shape[0], ids.shape[1] - 2, 32, dtype=torch.bfloat16)


def tiny_aligner() -> FuseAlignAligner:
    torch.manual_seed(0)
    return FuseAlignAligner(FuseAlign(FuseAlignArgs(**TINY_ARGS)).eval(), StubEncoder())
