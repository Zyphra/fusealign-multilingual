"""Frozen Dia UTF-8 text features, excluding the two-token ``[S1] `` prefix."""

import torch
from torch import nn


class DiaTextEncoder(nn.Module):
    """Dia's text encoder built from its config and the 98 encoder tensors, stored in bf16.

    ``config`` is Dia's ``encoder_config``; ``state`` is keyed as ``DiaEncoder`` expects
    (``model.encoder.`` stripped). Inputs contain the speaker/space prefix and right padding. The
    returned bf16 features retain transcript padding but omit the two prefix positions. This
    module remains frozen and in evaluation mode even when its parent enters training mode.
    """

    def __init__(self, config: dict, state: dict[str, torch.Tensor]):
        from transformers.models.dia.configuration_dia import DiaEncoderConfig
        from transformers.models.dia.modeling_dia import DiaEncoder

        super().__init__()
        self.encoder = DiaEncoder(DiaEncoderConfig(**config))
        self.encoder.to(dtype=torch.bfloat16)
        self.encoder.load_state_dict(
            {key: value.to(torch.bfloat16) for key, value in state.items()}, strict=True
        )
        self.requires_grad_(False)
        self.eval()

    def train(self, mode: bool = True):
        return super().train(False)

    @torch.no_grad()
    def forward(self, text_ids: torch.Tensor, text_mask: torch.Tensor) -> torch.Tensor:
        if text_ids.shape[-1] > 1024:
            raise ValueError(
                f"Dia accepts at most 1024 tokens including the prefix; "
                f"received {text_ids.shape[-1]}"
            )
        return self.encoder(
            input_ids=text_ids, attention_mask=text_mask.long()
        ).last_hidden_state[:, 2:]
