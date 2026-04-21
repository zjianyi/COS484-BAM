"""BAMGenerator: autoregressive generation with a bolt-on StateCache.

Fixes two issues in the prototype shown in `architecture.md`:
  1. Properly threads Falcon Mamba's recurrent state across steps via
     `cache_params: MambaCache`, so each forward processes only the newest
     token. Without this the generator would silently re-run the whole
     prefix each step and destroy the linear-inference property.
  2. Keeps the StateCache hooks enabled across steps, controlled by a
     `previous_token_was_cache` flag so the cache is read only when the
     model just emitted a [CACHE] token, and written only on the following
     step (when the hidden state that "just finished a reasoning step" is
     produced while consuming that [CACHE] token as input).
"""

from __future__ import annotations

from typing import Optional

import torch

try:
    from transformers.cache_utils import MambaCache
except ImportError:
    try:
        from transformers.models.mamba.modeling_mamba import MambaCache  # type: ignore
    except ImportError:  # transformers >=5.x uses DynamicCache for Mamba
        from transformers.cache_utils import DynamicCache as MambaCache  # type: ignore


class BAMGenerator:
    def __init__(
        self,
        model,
        tokenizer,
        cache_module,
        cache_layer_idx: Optional[int] = None,
        cache_token: str = "[CACHE]",
    ):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache_module
        config = model.config
        if cache_layer_idx is None:
            cache_layer_idx = config.num_hidden_layers // 2
        self.cache_layer_idx = cache_layer_idx
        self.cache_token_id = tokenizer.convert_tokens_to_ids(cache_token)

        self.previous_token_was_cache = False
        self.captured_hidden: Optional[torch.Tensor] = None
        self.num_writes = 0

        base = _get_backbone(model)
        layers = base.layers
        if cache_layer_idx + 1 >= len(layers):
            raise ValueError(
                f"cache_layer_idx={cache_layer_idx} is too late; need a layer after it for injection"
            )

        self._capture_handle = layers[cache_layer_idx].register_forward_hook(self._capture_hook)
        self._inject_handle = layers[cache_layer_idx + 1].register_forward_pre_hook(
            self._inject_hook, with_kwargs=False
        )

    def _capture_hook(self, module, inputs, output):
        hidden = output[0] if isinstance(output, tuple) else output
        self.captured_hidden = hidden[:, -1:, :].detach()
        return output

    def _inject_hook(self, module, args):
        if not self.previous_token_was_cache or self.captured_hidden is None:
            return None
        if not args:
            return None
        x = args[0]
        h_last = self.captured_hidden.squeeze(0).squeeze(0)
        enriched = self.cache.read(h_last)
        x = x.clone()
        x[:, -1, :] = enriched
        return (x,) + args[1:]

    def reset(self) -> None:
        self.cache.reset()
        self.previous_token_was_cache = False
        self.captured_hidden = None
        self.num_writes = 0

    def close(self) -> None:
        self._capture_handle.remove()
        self._inject_handle.remove()

    @torch.no_grad()
    def generate(
        self,
        prompt_input_ids: torch.Tensor,
        max_new_tokens: int = 2048,
        eos_token_id: Optional[int] = None,
    ) -> list[int]:
        """Generate up to `max_new_tokens` new tokens.

        Args:
          prompt_input_ids: (1, T0) tensor on the model's device.
          max_new_tokens: hard cap on generated tokens.
          eos_token_id: stop token id (defaults to tokenizer.eos_token_id).

        Returns the list of newly generated token ids (not including the prompt).
        """
        if eos_token_id is None:
            eos_token_id = self.tokenizer.eos_token_id

        device = prompt_input_ids.device
        dtype = next(self.model.parameters()).dtype

        self.reset()

        cache_params = MambaCache(config=self.model.config)

        out = self.model(
            input_ids=prompt_input_ids,
            cache_params=cache_params,
            use_cache=True,
            return_dict=True,
        )
        next_token = out.logits[:, -1, :].argmax(dim=-1)
        generated: list[int] = []

        for _ in range(max_new_tokens):
            tok_id = int(next_token.item())

            if self.previous_token_was_cache and self.captured_hidden is not None:
                self.cache.write(self.captured_hidden.squeeze(0).squeeze(0))
                self.num_writes += 1

            self.previous_token_was_cache = tok_id == self.cache_token_id

            generated.append(tok_id)
            if tok_id == eos_token_id:
                break

            cur_input = next_token.view(1, 1)
            out = self.model(
                input_ids=cur_input,
                cache_params=cache_params,
                use_cache=True,
                return_dict=True,
            )
            next_token = out.logits[:, -1, :].argmax(dim=-1)

        return generated


def _get_backbone(model):
    """Return the FalconMamba backbone whether the model is a PeftModel or not."""
    if hasattr(model, "backbone"):
        return model.backbone
    base = getattr(model, "base_model", None)
    if base is not None:
        inner = getattr(base, "model", base)
        if hasattr(inner, "backbone"):
            return inner.backbone
        if hasattr(inner, "model") and hasattr(inner.model, "backbone"):
            return inner.model.backbone
    raise AttributeError("Could not locate `.backbone` on the given model")
