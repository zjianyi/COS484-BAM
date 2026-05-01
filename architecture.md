Great — 0.96 F1 on cache placement means the model reliably produces `[CACHE]` at the right locations. The SFT is done. Now we build the actual BAM cache module.

Let me lay out exactly what you need to implement, in order.

**Step 1: Figure out the internal structure of Falcon Mamba 7B.**

Before writing any cache code, you need to know the exact module names and hidden state shapes inside the model. Run this:

```python
from transformers import AutoModelForCausalLM

model = AutoModelForCausalLM.from_pretrained(
    "tiiuae/falcon-mamba-7b-instruct",
    torch_dtype=torch.bfloat16,
    device_map="auto",
)

# Print all layers
for name, module in model.named_modules():
    print(name, type(module).__name__)
```

You need to identify three things: what the layers are called (probably something like `model.backbone.layers.0`, `model.backbone.layers.1`, etc.), how many there are (likely 64), and the hidden dimension D (likely 4096). Also find the exact layer index for the midpoint — that's where the cache hooks go.

**Step 2: Implement the StateCache module.**

```python
import torch
import torch.nn as nn
import torch.nn.functional as F

class StateCache(nn.Module):
    def __init__(self, d_model=4096, d_attn=256, max_entries=32):
        super().__init__()
        self.W_q = nn.Linear(d_model, d_attn, bias=False)
        self.W_k = nn.Linear(d_model, d_attn, bias=False)
        self.W_v = nn.Linear(d_model, d_attn, bias=False)
        self.W_out = nn.Linear(d_attn, d_model, bias=False)
        self.gate = nn.Parameter(torch.zeros(1))
        self.max_entries = max_entries
        self.reset()
    
    def reset(self):
        self.keys = []
        self.values = []
    
    def write(self, h):
        k = self.W_k(h.detach())
        v = self.W_v(h.detach())
        self.keys.append(k)
        self.values.append(v)
        if len(self.keys) > self.max_entries:
            self.keys.pop(0)
            self.values.pop(0)
    
    def read(self, h):
        if len(self.keys) == 0:
            return h
        
        q = self.W_q(h)
        keys = torch.stack(self.keys)
        values = torch.stack(self.values)
        
        attn = F.scaled_dot_product_attention(
            q.unsqueeze(0),
            keys.unsqueeze(0),
            values.unsqueeze(0),
        ).squeeze(0)
        
        out = self.W_out(attn)
        return h + torch.sigmoid(self.gate) * out
```

Note the `h.detach()` on the write — this stops gradients from flowing back through the cache during training, keeping the training loop simple. The read path has gradients flowing through W_Q, W_out, and the gate, which is what we want to train.

**Step 3: Hook it into the generation loop.**

This is the trickiest part. You need to intercept the hidden state at the midpoint layer during autoregressive generation. The timing issue we discussed: you don't know a token is `[CACHE]` until after the forward pass, so the cache interaction happens on the *next* step when that token is fed as input.

```python
class BAMGenerator:
    def __init__(self, model, tokenizer, cache_module, cache_layer_idx=32):
        self.model = model
        self.tokenizer = tokenizer
        self.cache = cache_module
        self.cache_layer_idx = cache_layer_idx
        self.CACHE_TOKEN_ID = tokenizer.convert_tokens_to_ids("[CACHE]")
        
        self.previous_token_was_cache = False
        self.captured_hidden = None
        
        # Hook to capture hidden state at cache layer
        def capture_hook(module, input, output):
            # output shape: [batch, seq_len, d_model]
            self.captured_hidden = output[:, -1:, :]
            return output
        
        # Hook to inject retrieved state into next layer's input
        def inject_hook(module, args):
            if self.previous_token_was_cache and self.captured_hidden is not None:
                x = args[0]
                enriched = self.cache.read(self.captured_hidden.squeeze(0))
                x = x.clone()
                x[:, -1:, :] = enriched.unsqueeze(0)
                return (x,) + args[1:]
            return args
        
        layer = model.backbone.layers[cache_layer_idx]
        next_layer = model.backbone.layers[cache_layer_idx + 1]
        layer.register_forward_hook(capture_hook)
        next_layer.register_forward_pre_hook(inject_hook)
    
    def generate(self, prompt, max_tokens=2048):
        input_ids = self.tokenizer(prompt, return_tensors="pt").input_ids.cuda()
        self.cache.reset()
        self.previous_token_was_cache = False
        
        generated = []
        
        for step in range(max_tokens):
            with torch.no_grad():
                outputs = self.model(input_ids)
            
            next_token = outputs.logits[:, -1, :].argmax(dim=-1)
            
            # If THIS token is [CACHE], write to cache
            # (captured_hidden was set by the hook during forward pass)
            if self.previous_token_was_cache and self.captured_hidden is not None:
                self.cache.write(self.captured_hidden.squeeze(0))
            
            # Track whether current output is [CACHE] for next step
            self.previous_token_was_cache = (next_token.item() == self.CACHE_TOKEN_ID)
            
            generated.append(next_token.item())
            input_ids = next_token.unsqueeze(0)
            
            if next_token.item() == self.tokenizer.eos_token_id:
                break
        
        return self.tokenizer.decode(generated, skip_special_tokens=False)
```

**Step 4: Verify the hooks work before training anything.**

Load your SFT model, attach the cache module (with random weights), and generate a few solutions. Check:

```python
cache_module = StateCache(d_model=4096, d_attn=256).cuda().bfloat16()
generator = BAMGenerator(model, tokenizer, cache_module, cache_layer_idx=32)

response = generator.generate("Solve: What is 347 * 23?")
print(response)
print(f"Cache entries written: {len(cache_module.keys)}")
```

You want to see: the model still generates coherent text (hooks didn't break anything), `[CACHE]` tokens appear (SFT is working), and `cache entries written` matches the number of `[CACHE]` tokens in the output. The cache module has random weights so the reads won't help yet — you're just verifying the plumbing.

**Step 5: Train the cache module.**

Freeze everything except the StateCache parameters. This is a separate training phase from the SFT — you're training the cache to be useful, not teaching the model to produce `[CACHE]`.

This is the part that needs the most thought about how to set up the training loop, because you're training a module that operates during generation, but training requires teacher-forced forward passes. You have two options:

**Option A (simpler): Train on the SFT data with the same teacher-forcing approach.** Run the full sequence through Mamba, extract hidden states at all `[CACHE]` positions at layer 32, do the cache cross-attention with causal masking, inject the results, continue the forward pass through layers 33-64, and compute the standard language modeling loss. The cache module learns to produce retrievals that reduce the LM loss. This is the approach I'd start with.

**Option B (harder but cleaner): Train on generation quality directly.** Generate solutions with the cache active, check if accuracy improved over no-cache baseline. This is closer to RL and much harder to implement.

Go with Option A. If the cache module learns to reduce the LM loss at positions following `[CACHE]` tokens, it's learning to retrieve useful information.

**Step 6: Evaluate — produce the BAM scaling curves.**

Once the cache module is trained, run the exact same CoT scaling experiments from Phase 1. Same models, same K values, same benchmarks. But now you have four lines on the plot:

- Falcon Mamba (vanilla, no cache)
- Falcon Mamba + BAM (with trained cache)
- Codestral Mamba (vanilla, no cache)
- Llama-3-8B-Instruct (Transformer baseline)

The key comparison: does the BAM line sit above the vanilla Mamba line at large K values? Does the gap widen as K increases? That's your result.