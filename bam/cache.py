"""StateCache module for BAM (bolt-on attention memory).

A single small cross-attention module attached at one midpoint layer of
Falcon Mamba. Its only job is to let the model retrieve hidden states
captured at previous `[CACHE]` positions when generating the token right
after a `[CACHE]`.

Design notes:
- Writes are detached from the computation graph (keys/values come from
  hidden states produced by frozen Mamba layers, so there is no reason to
  route gradients through them during training).
- Reads use gradients through W_Q / W_out / gate so only the cache-module
  parameters are trained.
- The cache itself is a bounded ring buffer of at most `max_entries`
  projected (key, value) pairs.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class StateCache(nn.Module):
    def __init__(self, d_model: int = 4096, d_attn: int = 256, max_entries: int = 32):
        super().__init__()
        self.d_model = d_model
        self.d_attn = d_attn
        self.max_entries = max_entries

        self.W_Q = nn.Linear(d_model, d_attn, bias=False)
        self.W_K = nn.Linear(d_model, d_attn, bias=False)
        self.W_V = nn.Linear(d_model, d_attn, bias=False)
        self.W_out = nn.Linear(d_attn, d_model, bias=False)
        nn.init.zeros_(self.W_out.weight)

        self.keys: list[torch.Tensor] = []
        self.values: list[torch.Tensor] = []

    def reset(self) -> None:
        self.keys = []
        self.values = []

    @staticmethod
    def _ensure_1d(x: torch.Tensor) -> torch.Tensor:
        """Accept (d,) or (1, d) / (b, d) with b == 1 and return (d,)."""
        if x.dim() == 1:
            return x
        if x.dim() == 2 and x.size(0) == 1:
            return x.squeeze(0)
        raise ValueError(f"StateCache expected a single vector, got shape {tuple(x.shape)}")

    @torch.no_grad()
    def write(self, h: torch.Tensor) -> None:
        """Append a single (key, value) pair derived from hidden state `h`."""
        h_vec = self._ensure_1d(h.detach())
        w_dtype = self.W_K.weight.dtype
        k = self.W_K(h_vec.to(w_dtype))
        v = self.W_V(h_vec.to(w_dtype))
        self.keys.append(k)
        self.values.append(v)
        if len(self.keys) > self.max_entries:
            self.keys.pop(0)
            self.values.pop(0)

    def read(self, h: torch.Tensor) -> torch.Tensor:
        """Return `h + sigmoid(gate) * W_out(attn(Q(h), K, V))`.

        Handles mixed dtypes: computation runs in the module's dtype, result
        is cast back to the input dtype before returning.
        """
        h_vec = self._ensure_1d(h)
        if not self.keys:
            return h_vec

        w_dtype = self.W_Q.weight.dtype
        q = self.W_Q(h_vec.to(w_dtype))
        keys = torch.stack(self.keys, dim=0)
        values = torch.stack(self.values, dim=0)

        attn = F.scaled_dot_product_attention(
            q.view(1, 1, 1, -1),
            keys.view(1, 1, -1, self.d_attn),
            values.view(1, 1, -1, self.d_attn),
        ).view(-1)

        out = self.W_out(attn)
        return h_vec + out.to(h_vec.dtype)
