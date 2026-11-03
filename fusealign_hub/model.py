"""FuseAlign network (inference only).

Parameter names match the training code's state dicts, so exports load with ``strict=True``.
Deviation from the training code: ``RefinementHead`` masks each sample's padded frames, so a padded
batch scores every clip as it would alone.
"""

from dataclasses import dataclass

import torch
from torch import nn
from torch.nn import functional as F


class RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        normalized = x.float()
        normalized = normalized * torch.rsqrt(
            normalized.square().mean(dim=-1, keepdim=True) + self.eps
        )
        return normalized.to(x.dtype) * self.weight.to(x.dtype)


class RotaryEmbedding(nn.Module):
    """RoPE with real-valued positions, shared across the joint blocks."""

    def __init__(self, head_dim: int, theta: float):
        super().__init__()
        inv_freq = theta ** (-torch.arange(0, head_dim, 2, dtype=torch.float32) / head_dim)
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, positions: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        # Keep fractional text positions and their phases in fp32 under autocast.
        with torch.autocast(device_type=positions.device.type, enabled=False):
            angles = positions.float().unsqueeze(-1) * self.inv_freq.float()
            angles = torch.cat((angles, angles), dim=-1).unsqueeze(1)
            return angles.cos(), angles.sin()


def _apply_rope(
    x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor
) -> torch.Tensor:
    first, second = x.chunk(2, dim=-1)
    rotated = torch.cat((-second, first), dim=-1)
    return x * cos + rotated * sin


class JointAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int):
        super().__init__()
        self.n_heads = n_heads
        self.head_dim = dim // n_heads
        self.qkv = nn.Linear(dim, 3 * dim, bias=False)
        self.out = nn.Linear(dim, dim, bias=False)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        batch_size, sequence_length, dim = x.shape
        qkv = self.qkv(x).reshape(
            batch_size, sequence_length, 3, self.n_heads, self.head_dim
        )
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        cos, sin = cos.to(q.dtype), sin.to(q.dtype)
        q = _apply_rope(q, cos, sin)
        k = _apply_rope(k, cos, sin)
        # True means an allowed key. Padded queries still see valid keys, so
        # no query has an all-masked softmax; blocks zero their outputs below.
        attended = F.scaled_dot_product_attention(q, k, v, attn_mask=key_mask)
        attended = attended.transpose(1, 2).reshape(batch_size, sequence_length, dim)
        return self.out(attended)


class SwiGLU(nn.Module):
    def __init__(self, dim: int, ffn_dim: int):
        super().__init__()
        self.w1 = nn.Linear(dim, ffn_dim, bias=False)
        self.w3 = nn.Linear(dim, ffn_dim, bias=False)
        self.w2 = nn.Linear(ffn_dim, dim, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w2(F.silu(self.w1(x)) * self.w3(x))


class JointBlock(nn.Module):
    def __init__(self, dim: int, n_heads: int, ffn_dim: int):
        super().__init__()
        self.attn_norm = RMSNorm(dim)
        self.attn = JointAttention(dim, n_heads)
        self.ffn_norm = RMSNorm(dim)
        self.ffn = SwiGLU(dim, ffn_dim)

    def forward(
        self,
        x: torch.Tensor,
        key_mask: torch.Tensor,
        padding_mask: torch.Tensor,
        cos: torch.Tensor,
        sin: torch.Tensor,
    ) -> torch.Tensor:
        x = x + self.attn(self.attn_norm(x), key_mask, cos, sin)
        x = x + self.ffn(self.ffn_norm(x))
        return x.masked_fill(padding_mask, 0)


class RefinementHead(nn.Sequential):
    def __init__(self, dim: int, feat_dim: int, n_upsample: int):
        layers = [nn.Conv1d(dim, feat_dim, kernel_size=1)]
        for _ in range(n_upsample):
            layers.extend(
                [
                    nn.ConvTranspose1d(
                        feat_dim, feat_dim, kernel_size=4, stride=2, padding=1
                    ),
                    nn.GELU(),
                ]
            )
        layers.extend(
            [
                nn.Conv1d(feat_dim, feat_dim, kernel_size=5, padding=2),
                nn.GELU(),
                nn.Conv1d(feat_dim, feat_dim, kernel_size=1),
            ]
        )
        super().__init__(*layers)

    def forward(self, x: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        """Zero each sample's frames past ``lengths`` after every convolution.

        Unlike the plain ``nn.Sequential`` used in training: without the masking, conv bias in a
        padded batch's padded frames reaches the last valid frames of shorter samples through the
        upsampling and kernel-5 convolutions, so a clip's end-of-clip logits would depend on its
        batch companions. A no-op for an unpadded batch.
        """
        for layer in self:
            x = layer(x)
            if isinstance(layer, nn.ConvTranspose1d):
                lengths = lengths * 2
            if not isinstance(layer, nn.GELU):
                frames = torch.arange(x.shape[-1], device=x.device)
                x = x.masked_fill(frames[None, None, :] >= lengths[:, None, None], 0)
        return x


def _word_mlp(dim: int, output_dim: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(dim, dim), nn.GELU(), nn.Linear(dim, output_dim))


class FuseAlign(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.args = args
        self.audio_patch = nn.Conv1d(
            1, args.dim, kernel_size=args.patch_window, stride=args.patch_hop
        )
        self.text_proj = nn.Linear(args.text_dim, args.dim)
        self.mod_emb = nn.Parameter(torch.empty(2, args.dim))
        nn.init.normal_(self.mod_emb, mean=0.0, std=0.02)
        self.rope = RotaryEmbedding(args.dim // args.n_heads, args.rope_theta)
        self.blocks = nn.ModuleList(
            JointBlock(args.dim, args.n_heads, args.ffn_dim)
            for _ in range(args.n_layers)
        )
        self.final_norm = RMSNorm(args.dim)
        self.start_refinement = RefinementHead(args.dim, args.feat_dim, args.n_upsample)
        self.end_refinement = RefinementHead(args.dim, args.feat_dim, args.n_upsample)
        self.start_mlp = _word_mlp(args.dim, args.feat_dim)
        self.end_mlp = _word_mlp(args.dim, args.feat_dim)
        self.presence_mlp = _word_mlp(args.dim, 1)
        nn.init.constant_(self.presence_mlp[-1].bias, args.presence_bias_init)

    @staticmethod
    def _standardize(wav: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        sample_mask = torch.arange(wav.shape[1], device=wav.device)[None, :] < lengths[:, None]
        with torch.autocast(device_type=wav.device.type, enabled=False):
            samples = wav.float().masked_fill(~sample_mask, 0)
            counts = lengths[:, None].float()
            mean = samples.sum(dim=1, keepdim=True) / counts
            centered = (samples - mean).masked_fill(~sample_mask, 0)
            std = (centered.square().sum(dim=1, keepdim=True) / counts).sqrt()
            return centered / (std + 1e-5)

    def forward(
        self,
        wav: torch.Tensor,
        wav_lengths: torch.Tensor,
        text_feats: torch.Tensor,
        text_mask: torch.Tensor,
        word_spans: torch.Tensor,
        word_mask: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        audio = self.audio_patch(self._standardize(wav, wav_lengths).unsqueeze(1))
        audio = audio.transpose(1, 2)
        audio_lengths = (wav_lengths - self.args.patch_window) // self.args.patch_hop + 1
        audio_positions = torch.arange(audio.shape[1], device=wav.device)
        audio_mask = audio_positions[None, :] < audio_lengths[:, None]

        text = self.text_proj(text_feats)
        byte_mask = text_mask[:, 2:].bool()
        byte_positions = torch.arange(text.shape[1], device=wav.device)
        text_lengths = byte_mask.sum(dim=1)
        text_scale = audio_lengths.float() / (text_lengths - 1).clamp_min(1).float()
        text_positions = byte_positions[None, :].float() * text_scale[:, None]
        text_positions = text_positions.masked_fill(~byte_mask, 0)
        audio_rope_positions = audio_positions[None, :].expand(wav.shape[0], -1).float()
        audio_rope_positions = audio_rope_positions.masked_fill(~audio_mask, 0)
        positions = torch.cat((audio_rope_positions, text_positions), dim=1)
        cos, sin = self.rope(positions)
        cos, sin = cos.to(audio.dtype), sin.to(audio.dtype)

        # Rectangular modality padding does not change the valid joint sequence:
        # attention masks out those keys, and RoPE uses per-sample positions.
        joint = torch.cat((audio + self.mod_emb[0], text + self.mod_emb[1]), dim=1)
        valid_mask = torch.cat((audio_mask, byte_mask), dim=1)
        padding_mask = ~valid_mask.unsqueeze(-1)
        joint = joint.masked_fill(padding_mask, 0)
        key_mask = valid_mask[:, None, None, :]
        for block in self.blocks:
            joint = block(joint, key_mask, padding_mask, cos, sin)
        joint = self.final_norm(joint).masked_fill(padding_mask, 0)
        audio_out = joint[:, :audio.shape[1]].transpose(1, 2)
        text_out = joint[:, audio.shape[1]:]

        membership = (
            (byte_positions[None, None, :] >= word_spans[:, :, 0, None])
            & (byte_positions[None, None, :] < word_spans[:, :, 1, None])
            & byte_mask[:, None, :]
            & word_mask[:, :, None]
        )
        pooling = membership.to(text_out.dtype)
        pooling = pooling / pooling.sum(dim=-1, keepdim=True).clamp_min(1)
        word_vectors = torch.bmm(pooling, text_out)
        start_features = self.start_refinement(audio_out, audio_lengths)
        end_features = self.end_refinement(audio_out, audio_lengths)

        with torch.autocast(device_type=wav.device.type, enabled=False):
            word_vectors = word_vectors.float()
            start_logits = torch.einsum(
                "bwf,bft->bwt", self.start_mlp(word_vectors), start_features.float()
            )
            end_logits = torch.einsum(
                "bwf,bft->bwt", self.end_mlp(word_vectors), end_features.float()
            )
            presence_logits = self.presence_mlp(word_vectors).squeeze(-1)
        return {
            "start_logits": start_logits,
            "end_logits": end_logits,
            "presence_logits": presence_logits,
            "frame_lengths": audio_lengths * (2 ** self.args.n_upsample),
        }


@dataclass
class FuseAlignArgs:
    dim: int = 1024
    n_layers: int = 12
    n_heads: int = 16
    ffn_dim: int = 4096
    feat_dim: int = 256
    patch_window: int = 512
    patch_hop: int = 256
    n_upsample: int = 4
    text_dim: int = 1024
    rope_theta: float = 1e4
    presence_bias_init: float = 2.0
    dia_repo: str = "nari-labs/Dia-1.6B-0626"


def cast_autocast_weights(net: FuseAlign) -> None:
    """Store in bf16 the weights CUDA autocast casts to bf16 on every forward.

    Bit-identical (autocast rounds the same fp32 values the same way) and drops ~200 cast
    kernels per forward. RMSNorm weights, ``mod_emb`` and the word MLPs, which run with
    autocast disabled, stay fp32.
    """
    fp32 = {id(layer) for mlp in (net.start_mlp, net.end_mlp, net.presence_mlp) for layer in mlp}
    for module in net.modules():
        if isinstance(module, (nn.Linear, nn.Conv1d, nn.ConvTranspose1d)) and id(module) not in fp32:
            module.to(torch.bfloat16)
