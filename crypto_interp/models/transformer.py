"""Minimal transformer mirroring Nanda et al.'s grokking setup.

Architecture choices follow the paper:
  - 1 transformer block
  - No LayerNorm (cleaner linear-algebra picture for interp)
  - ReLU MLP, hidden dim 4 * d_model
  - Hook points at every internal activation for caching

Forward pass on input [a, b, =]:
  embed -> pos_embed -> attn (residual) -> mlp (residual) -> unembed
We take logits at the final position only.
"""

from dataclasses import dataclass

import einops
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


class HookPoint(nn.Module):
    """Identity module that exposes a named slot for caching/hooking activations."""

    def __init__(self):
        super().__init__()
        self.name: str = ""
        self._handles: list = []

    def give_name(self, name: str) -> None:
        self.name = name

    def add_hook(self, hook):
        """Hook signature: hook(activation, name) -> activation or None."""

        def wrapper(module, inputs, output):
            return hook(output, self.name)

        h = self.register_forward_hook(wrapper)
        self._handles.append(h)

    def remove_hooks(self) -> None:
        for h in self._handles:
            h.remove()
        self._handles = []

    def forward(self, x):
        return x


class Embed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_E = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_model))

    def forward(self, x):
        # x: (batch, seq) of long indices
        return einops.rearrange(self.W_E[:, x], "d b s -> b s d")


class Unembed(nn.Module):
    def __init__(self, d_vocab: int, d_model: int):
        super().__init__()
        self.W_U = nn.Parameter(torch.randn(d_model, d_vocab) / np.sqrt(d_vocab))

    def forward(self, x):
        return x @ self.W_U


class PosEmbed(nn.Module):
    def __init__(self, n_ctx: int, d_model: int):
        super().__init__()
        self.W_pos = nn.Parameter(torch.randn(n_ctx, d_model) / np.sqrt(d_model))

    def forward(self, x):
        return x + self.W_pos[: x.shape[-2]]


class Attention(nn.Module):
    def __init__(self, d_model: int, num_heads: int, d_head: int, n_ctx: int):
        super().__init__()
        self.W_K = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_Q = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_V = nn.Parameter(torch.randn(num_heads, d_head, d_model) / np.sqrt(d_model))
        self.W_O = nn.Parameter(torch.randn(d_model, d_head * num_heads) / np.sqrt(d_model))
        self.register_buffer("mask", torch.tril(torch.ones(n_ctx, n_ctx)))
        self.d_head = d_head
        self.hook_k = HookPoint()
        self.hook_q = HookPoint()
        self.hook_v = HookPoint()
        self.hook_z = HookPoint()
        self.hook_attn = HookPoint()
        self.hook_attn_pre = HookPoint()

    def forward(self, x):
        # x: (batch, seq, d_model)
        k = self.hook_k(torch.einsum("ihd,bsd->bish", self.W_K, x))
        q = self.hook_q(torch.einsum("ihd,bsd->bish", self.W_Q, x))
        v = self.hook_v(torch.einsum("ihd,bsd->bish", self.W_V, x))
        # attn_scores[b, i, query, key]
        scores_pre = torch.einsum("bisd,bitd->bits", k, q)
        seq = x.shape[-2]
        scores_masked = scores_pre - 1e10 * (1 - self.mask[:seq, :seq])
        attn = self.hook_attn(F.softmax(self.hook_attn_pre(scores_masked / np.sqrt(self.d_head)), dim=-1))
        # z[b, i, query, head_dim]
        z = self.hook_z(torch.einsum("bisd,bits->bitd", v, attn))
        z_flat = einops.rearrange(z, "b i s d -> b s (i d)")
        out = torch.einsum("df,bsf->bsd", self.W_O, z_flat)
        return out


class MLP(nn.Module):
    def __init__(self, d_model: int, d_mlp: int):
        super().__init__()
        self.W_in = nn.Parameter(torch.randn(d_mlp, d_model) / np.sqrt(d_model))
        self.b_in = nn.Parameter(torch.zeros(d_mlp))
        self.W_out = nn.Parameter(torch.randn(d_model, d_mlp) / np.sqrt(d_model))
        self.b_out = nn.Parameter(torch.zeros(d_model))
        self.hook_pre = HookPoint()
        self.hook_post = HookPoint()

    def forward(self, x):
        pre = self.hook_pre(torch.einsum("md,bsd->bsm", self.W_in, x) + self.b_in)
        post = self.hook_post(F.relu(pre))
        return torch.einsum("dm,bsm->bsd", self.W_out, post) + self.b_out


class Block(nn.Module):
    def __init__(self, d_model: int, d_mlp: int, d_head: int, num_heads: int, n_ctx: int):
        super().__init__()
        self.attn = Attention(d_model, num_heads, d_head, n_ctx)
        self.mlp = MLP(d_model, d_mlp)
        self.hook_attn_out = HookPoint()
        self.hook_mlp_out = HookPoint()
        self.hook_resid_pre = HookPoint()
        self.hook_resid_mid = HookPoint()
        self.hook_resid_post = HookPoint()

    def forward(self, x):
        x = self.hook_resid_mid(x + self.hook_attn_out(self.attn(self.hook_resid_pre(x))))
        x = self.hook_resid_post(x + self.hook_mlp_out(self.mlp(x)))
        return x


@dataclass
class TransformerConfig:
    d_vocab: int
    d_model: int = 128
    d_mlp: int = 512
    num_heads: int = 4
    d_head: int = 32
    n_ctx: int = 3
    num_layers: int = 1


class Transformer(nn.Module):
    def __init__(self, cfg: TransformerConfig):
        super().__init__()
        self.cfg = cfg
        self.embed = Embed(cfg.d_vocab, cfg.d_model)
        self.pos_embed = PosEmbed(cfg.n_ctx, cfg.d_model)
        self.blocks = nn.ModuleList(
            [Block(cfg.d_model, cfg.d_mlp, cfg.d_head, cfg.num_heads, cfg.n_ctx)
             for _ in range(cfg.num_layers)]
        )
        self.unembed = Unembed(cfg.d_vocab, cfg.d_model)
        for name, module in self.named_modules():
            if isinstance(module, HookPoint):
                module.give_name(name)

    def forward(self, x):
        x = self.embed(x)
        x = self.pos_embed(x)
        for block in self.blocks:
            x = block(x)
        return self.unembed(x)

    def hook_points(self):
        return [m for _, m in self.named_modules() if isinstance(m, HookPoint)]

    def remove_all_hooks(self):
        for hp in self.hook_points():
            hp.remove_hooks()

    def cache_all(self, cache: dict):
        def save(t, name):
            cache[name] = t.detach()

        for hp in self.hook_points():
            hp.add_hook(save)


if __name__ == "__main__":
    p = 113
    cfg = TransformerConfig(d_vocab=p + 1)
    model = Transformer(cfg)
    x = torch.tensor([[0, 1, p], [5, 7, p]], dtype=torch.long)
    out = model(x)
    print("Input shape:", x.shape, "Output shape:", out.shape)
    print("Hook points:", [hp.name for hp in model.hook_points()])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"Total parameters: {n_params:,}")
