"""One-shot discovery script: prints the Falcon Mamba module tree.

Use this on the H100 once to confirm layer paths and hidden dimensions
before running `train_cache.py`. Expected output for
`tiiuae/falcon-mamba-7b-instruct`:
  num_hidden_layers: 64
  hidden_size: 4096
  ... layer path `backbone.layers.{i}` ...
"""

import torch
from transformers import AutoConfig, AutoModelForCausalLM

MODEL_NAME = "tiiuae/falcon-mamba-7b-instruct"


def main() -> None:
    cfg = AutoConfig.from_pretrained(MODEL_NAME)
    print("num_hidden_layers:", cfg.num_hidden_layers)
    print("hidden_size:", cfg.hidden_size)
    print("intermediate_size:", getattr(cfg, "intermediate_size", None))
    print("state_size:", getattr(cfg, "state_size", None))

    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    for name, module in model.named_modules():
        print(name, type(module).__name__)


if __name__ == "__main__":
    main()
