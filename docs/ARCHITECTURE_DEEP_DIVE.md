# BAM Architecture Deep Dive

This document explains the current BAM/StateCache architecture in this repo.
It reflects the BABILong implementation in `bam/train_babilong.py`,
`bam/train_babilong_loss.py`, `bam/cache.py`, and `bam/generator.py`.

## High-Level Idea

BAM adds a small trainable cross-attention memory module to a frozen
FalconMamba 7B backbone.

FalconMamba is an SSM-style sequence model, so it does not naturally keep a
Transformer-style KV cache over the full context. That is good for linear-time
scaling, but bad when the model needs to recall specific facts from a long
context. BAM tries to recover a small amount of explicit retrieval without
turning the model into full quadratic attention.

The key design is selective memory:

1. Insert identical `[CACHE]` tokens at selected positions: after important
   passage facts, and once after the question immediately before the answer
   token during training.
2. Run FalconMamba up to the cache layer and capture the hidden state at every
   `[CACHE]` position.
3. For every captured cache hidden state, compute both a key/value pair with
   `W_K`/`W_V` and a query vector with `W_Q`.
4. At the cache layer, compute attention scores `Q @ K.T / sqrt(d_attn)` across
   the cache positions, then apply a strict causal mask so each `[CACHE]` token
   can attend only to earlier `[CACHE]` tokens.
5. Add the retrieved value mixture as a gated residual delta at cache positions
   that have earlier cache positions available.
6. Continue through the remaining frozen FalconMamba layers.

The base model is frozen. Only the small StateCache module is trained.

## Components

### Frozen Backbone

The backbone is:

```text
tiiuae/falcon-mamba-7b-instruct
```

The code loads it with:

```python
AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.bfloat16,
    device_map="cuda",
)
```

In the current BABILong scripts, the backbone has:

- `num_hidden_layers = 64`
- `hidden_size = 4096`
- cache injection layer `CACHE_LAYER_IDX = -2`, which resolves to layer 62

Layer 62 means BAM injects after most of the model has already processed the
sequence, with only one remaining Mamba block plus final norm/output projection
after the injected delta.

### StateCache Module

Defined in `bam/cache.py`:

```python
class StateCache(nn.Module):
    def __init__(self, d_model=4096, d_attn=256, max_entries=32):
        self.W_Q = nn.Linear(d_model, d_attn, bias=False)
        self.W_K = nn.Linear(d_model, d_attn, bias=False)
        self.W_V = nn.Linear(d_model, d_attn, bias=False)
        self.W_out = nn.Linear(d_attn, d_model, bias=False)
        self.W_gate = nn.Linear(d_model, 1, bias=False)
```

Default BABILong settings:

```text
d_model = 4096
d_attn = 256
max_entries = 64
```

The trainable parameters are:

- `W_Q`: query projection from current hidden state
- `W_K`: key projection from cached hidden states
- `W_V`: value projection from cached hidden states
- `W_out`: output projection back to model hidden size
- `W_gate`: scalar gate deciding how much of the cache delta to add

The saved checkpoint files, such as `bam/cache_babilong_loss.pt`, contain only
these StateCache weights plus a small config dictionary. They do not contain
FalconMamba weights and they do not contain per-example runtime memory entries.

## The Computation

Let:

```text
h_i in R^4096
```

be the FalconMamba hidden state at a `[CACHE]` token position after layer 62.

For every `[CACHE]` position, BAM builds:

```text
q_i = W_Q h_i
k_i = W_K stopgrad(h_i)
v_i = W_V stopgrad(h_i)
```

The stop-gradient on `k_i` and `v_i` means the cache does not backpropagate
through the frozen backbone states used as writes. Gradients still flow through
`W_K` and `W_V`, but not back into FalconMamba.

For a query cache position `i`, BAM can attend only to earlier cache positions:

```text
j < i
```

The attention scores are:

```text
score(i, j) = q_i dot k_j / sqrt(d_attn)
```

Then:

```text
a_i = softmax(score(i, j) over earlier j)
r_i = sum_j a_i,j v_j
delta_i = W_out r_i * sigmoid(W_gate h_i)
h'_i = h_i + delta_i
```

`h'_i` replaces the original hidden state at that `[CACHE]` position. The
modified sequence then continues through the remaining frozen layers.

## Why This Preserves Linear Scaling

Full self-attention over a sequence of length `n` costs roughly `O(n^2)`.

BAM does not attend over all tokens. It attends only over selected `[CACHE]`
positions. If the number of cache positions is `k`, the cache attention cost is:

```text
O(k^2)
```

For the main BABILong experiments, loss-triggered placement uses:

```python
TOP_K = 4
```

So the cache attention is effectively constant-size:

```text
O(4^2) = O(16)
```

The Mamba backbone still processes the sequence linearly. This is the core
architectural claim: BAM adds bounded retrieval without replacing the SSM with
full-context attention.

## Cache Token Placement

BAM depends on deciding where `[CACHE]` tokens should go. This repo has two
current placement strategies.

### Regex-Triggered Placement

Implemented in `bam/train_babilong.py`.

This strategy inserts `[CACHE]` after BABILong entity-movement facts, such as:

```text
John went to the kitchen. [CACHE]
```

The regex is intended to capture facts like:

```text
X went to Y.
X travelled to Y.
X moved to Y.
```

The final question gets a query cache token:

```text
Where is John? [CACHE] kitchen
```

This is strong but task-specific.

### Loss-Triggered Placement

Implemented in `bam/train_babilong_loss.py`.

This strategy avoids task-specific rules. It runs the frozen model over the
passage and computes per-token next-token negative log likelihood. High-loss
positions are interpreted as places where the model is surprised, and therefore
where writing memory may help.

The script picks:

```python
TOP_K = 4
```

highest-loss positions per passage, inserts `[CACHE]` after them, then appends
the question plus a final query `[CACHE]`.

Because the backbone is frozen, these positions are precomputed once before
training and reused for all epochs.

## Training Path

The BABILong training scripts use teacher forcing, not autoregressive rollout.

For one training sequence:

```text
passage with write [CACHE] tokens
question [CACHE] answer
```

the training forward does this:

1. Embed the full token sequence.
2. Run FalconMamba layers up to and including `layer_idx`.
3. Find all `[CACHE]` token positions.
4. Build K/V writes from all cache positions.
5. Build Q reads at all cache positions.
6. Apply a strict causal mask so each cache token only reads earlier cache
   tokens.
7. Add the gated retrieved delta at eligible cache positions.
8. Run the remaining frozen FalconMamba layers.
9. Compute cross-entropy loss only on the answer token after the final
   `[CACHE]`.

The focused loss is intentional:

```text
CE(logits[last_cache_pos], answer_token)
```

This directly asks: did the final query cache token retrieve enough information
to predict the answer?

## Why Layer 62

Early versions tried middle-layer injection. The current notes identify a
failure mode: if the cache is injected around layer 32, gradients from the loss
must pass through roughly 32 frozen Mamba layers before reaching the cache
module. That fixed backward path was empirically poor.

Layer 62 leaves only one frozen block after injection. This makes the cache
delta much closer to the final logits and gives a cleaner training signal.

That is why the main scripts use:

```python
CACHE_LAYER_IDX = -2
```

which resolves to:

```text
64 + (-2) = 62
```

## Initialization

`W_out` is zero-initialized:

```python
nn.init.zeros_(self.W_out.weight)
```

This means the cache initially produces no residual delta, so the model starts
as the frozen backbone behavior. Training then grows `W_out` away from zero.

`W_gate` is also zero-initialized:

```python
nn.init.zeros_(self.W_gate.weight)
```

So the initial gate value is:

```text
sigmoid(0) = 0.5
```

However, because `W_out` starts at zero, the actual initial delta is still zero.

## Runtime Generation

`bam/generator.py` contains `BAMGenerator`, which is the autoregressive runtime
path.

The generator registers two hooks:

1. A forward hook on the cache layer to capture the latest hidden state.
2. A pre-forward hook on the next layer to replace the latest hidden state with
   the cache-enriched hidden state when appropriate.

It also threads FalconMamba's recurrent generation cache with `MambaCache` or
`DynamicCache`, depending on the installed Transformers version. This matters
because otherwise generation would repeatedly rerun the whole prefix.

The timing is subtle:

1. The model emits a `[CACHE]` token.
2. On the next generation step, that `[CACHE]` token is consumed as input.
3. The hidden state produced while consuming it is captured at the cache layer.
4. If this is a write position, it is written to StateCache.
5. If this is a query position and previous entries exist, the cache read
   enriches the hidden state before the next layer.

The BABILong training scripts do not use this hook-based generator for the main
training loop. They implement the same cache math directly over full sequences,
which is simpler and faster for teacher forcing.

## What Gets Saved

Training saves checkpoints like:

```text
bam/cache_babilong.pt
bam/cache_babilong_loss.pt
bam/cache_babilong_loss.ep0.pt
...
```

Each checkpoint has the form:

```python
{
    "state_dict": {
        "W_Q.weight": ...,
        "W_K.weight": ...,
        "W_V.weight": ...,
        "W_out.weight": ...,
        "W_gate.weight": ...,
    },
    "config": {
        "d_model": hidden_size,
        "d_attn": D_ATTN,
        "max_entries": MAX_ENTRIES,
        "cache_layer_idx": layer_idx,
    },
}
```

FalconMamba weights live in the HuggingFace cache, not in these checkpoint
files. Runtime cache entries are temporary and reset between examples.

## Important Ablation Knobs

The main architecture ablations are:

- Placement policy: regex vs loss-triggered vs random
- Layer index: layer 32 vs layer 62
- `TOP_K`: 1 vs 4 vs 8 cache writes
- Gate: learned `W_gate` vs no gate
- `D_ATTN`: cache bottleneck size
- `MAX_ENTRIES`: ring-buffer capacity

When running ablations, keep the eval protocol fixed. Change one architectural
variable at a time and give every run a distinct `CACHE_OUT_PATH`.

## Mental Model

Think of StateCache as a tiny external memory attached near the end of
FalconMamba:

```text
tokens
  -> embeddings
  -> FalconMamba layers 0..62
  -> selected [CACHE] states become Q/K/V
  -> causal attention over previous selected states
  -> gated residual delta added at [CACHE] states
  -> FalconMamba layer 63
  -> final norm
  -> LM head
  -> answer token
```

It is not a separate retriever, not a vector database, and not a full attention
layer over the whole context. It is a learned residual adapter that lets a small
number of selected hidden states communicate with later selected hidden states.
