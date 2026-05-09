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
    def __init__(
        self,
        d_model: int = 4096,
        d_attn: int = 256,
        max_entries: int = 32,
        route_gate: str = "off",
    ):
        super().__init__()
        if route_gate not in {"off", "scalar", "vector"}:
            raise ValueError(f"route_gate must be off, scalar, or vector; got {route_gate!r}")
        self.d_model = d_model
        self.d_attn = d_attn
        self.max_entries = max_entries
        self.route_gate = route_gate

        self.W_Q = nn.Linear(d_model, d_attn, bias=False)
        self.W_K = nn.Linear(d_model, d_attn, bias=False)
        self.W_V = nn.Linear(d_model, d_attn, bias=False)
        if route_gate == "scalar":
            self.W_route_gate = nn.Linear(d_model, 1, bias=False)
            nn.init.zeros_(self.W_route_gate.weight)
        elif route_gate == "vector":
            self.W_route_gate = nn.Linear(d_model, d_model, bias=False)
            nn.init.zeros_(self.W_route_gate.weight)
        self.W_out = nn.Linear(d_attn, d_model, bias=False)
        nn.init.zeros_(self.W_out.weight)
        # Gate controls how much of the delta to apply; zero-init → sigmoid(0)=0.5 at start.
        self.W_gate = nn.Linear(d_model, 1, bias=False)
        nn.init.zeros_(self.W_gate.weight)

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

    def route(self, h: torch.Tensor) -> torch.Tensor:
        """Optionally gate hidden states before Q/K/V projection.

        The multiplier is identity-initialized: zero route-gate weights produce
        ``2 * sigmoid(0) == 1``, so the ablation starts equivalent to no route
        gate and learns deviations during training.
        """
        if self.route_gate == "off":
            return h
        return h * (2.0 * torch.sigmoid(self.W_route_gate(h)))

    @torch.no_grad()
    def write(self, h: torch.Tensor) -> None:
        """Append a single (key, value) pair derived from hidden state `h`."""
        h_vec = self._ensure_1d(h.detach())
        w_dtype = self.W_K.weight.dtype
        routed = self.route(h_vec.to(w_dtype))
        k = self.W_K(routed)
        v = self.W_V(routed)
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
        q = self.W_Q(self.route(h_vec.to(w_dtype)))
        keys = torch.stack(self.keys, dim=0)
        values = torch.stack(self.values, dim=0)

        attn = F.scaled_dot_product_attention(
            q.view(1, 1, 1, -1),
            keys.view(1, 1, -1, self.d_attn),
            values.view(1, 1, -1, self.d_attn),
        ).view(-1)

        gate = torch.sigmoid(self.W_gate(h_vec.to(w_dtype)))  # (1,) scalar
        out = self.W_out(attn) * gate
        return h_vec + out.to(h_vec.dtype)
