# Intake — what I need to reason about your inference path

Goal: trace one prompt through your actual stack, understand what your current
vLLM config already does, and put a defensible number on what speculative
decoding would buy.

Nothing here needs to be answered all at once. Sections **D** and **E** are the
ones that decide the answer; **A**–**C** are cheap to collect; **F**–**H** shape
which options are admissible at all.

---

## A. Model identity

| # | Question | How to get it |
|---|---|---|
| A1 | Exact model id / local path | the `--model` arg, or `model` in your server config |
| A2 | Is it a fine-tune of a base model, and of which one? | matters for whether an off-the-shelf draft model exists |
| A3 | Weight dtype as served | `bf16`, `fp16`, `fp8`, or a quantized checkpoint |
| A4 | Is it already quantized? Which method? | `--quantization awq / gptq / fp8 / none` |

```bash
# A1, A3, A4 — from the model directory
cat $MODEL_PATH/config.json
```

The fields I actually read out of `config.json`:

```
hidden_size            -> d
num_hidden_layers      -> L
num_attention_heads    -> h
num_key_value_heads    -> h_kv     (GQA ratio; drives KV cache size)
head_dim               -> d_h      (may be absent; then d / h)
intermediate_size      -> m
max_position_embeddings
rms_norm_eps           -> confirms RMSNorm rather than LayerNorm
hidden_act             -> confirms SwiGLU
torch_dtype
rope_scaling           -> whether context was extended
quantization_config    -> present if the checkpoint is pre-quantized
architectures          -> e.g. LlamaForCausalLM / Qwen2ForCausalLM / MistralForCausalLM
```

## B. Tokenizer

| # | Question | Why it matters |
|---|---|---|
| B1 | Which tokenizer, and is it the model's own? | vocab size sets the unembedding cost and the draft-model compatibility requirement |
| B2 | Are you applying a chat template? | templates add fixed tokens to every request — free prefix-cache material, or free waste |
| B3 | Is the prompt built by string concatenation? | stray whitespace at a prefix boundary silently kills prefix-cache hits |

```bash
python -c "
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('$MODEL_PATH')
print('class      :', type(t).__name__)
print('vocab      :', len(t))
print('bos/eos/pad:', t.bos_token, t.eos_token, t.pad_token)
print('chat tmpl  :', bool(t.chat_template))
"
```

**A speculative-decoding blocker to rule out early:** a draft model must share the
*exact* tokenizer with the target. If it does not, spec decoding is off the table
until you find or train one that does.

## C. Serving configuration

| # | Question |
|---|---|
| C1 | vLLM version, and V0 or V1 engine? |
| C2 | The full launch command or `EngineArgs` — verbatim |
| C3 | `tensor_parallel_size`, `pipeline_parallel_size` |
| C4 | `max_model_len`, `max_num_seqs`, `max_num_batched_tokens` |
| C5 | `gpu_memory_utilization` |
| C6 | `enable_prefix_caching` — on or off? |
| C7 | `enforce_eager` — if true, CUDA graphs are disabled and every decode step pays Python/launch overhead |
| C8 | `kv_cache_dtype` — `auto` or `fp8` |
| C9 | Are you using the OpenAI-compatible server, `LLM.generate`, or the async engine? |
| C10 | Sampling params per request: `temperature`, `top_p`, `max_tokens`, `n`, `stop` |
| C11 | Any guided/structured decoding already enabled? (`guided_json`, `response_format`) |

```bash
pip show vllm | head -3
# and paste the launch command verbatim, e.g.
# vllm serve $MODEL --tensor-parallel-size 2 --max-num-seqs 256 ...
```

Also useful: the **startup log**. vLLM prints the derived KV cache size and the
number of GPU blocks, which tells me your real concurrency ceiling:

```
# grep the server log for these lines
GPU blocks: ..., CPU blocks: ...
Maximum concurrency for ... tokens per request: ...x
```

## D. Workload shape — the section that decides everything

This is the part I cannot guess, and it dominates the arithmetic.

| # | Question | Unit |
|---|---|---|
| D1 | Median and p95 **prompt** length | tokens, not characters |
| D2 | Median and p95 **output** length | tokens |
| D3 | Requests per second at peak, and at typical load | req/s |
| D4 | Is traffic bursty or steady? | shape |
| D5 | How many sequences are actually running concurrently? | see E2 |
| D6 | How much of the prompt is a **fixed prefix** (system prompt, policy text, few-shot examples)? | tokens, and % of D1 |
| D7 | Is that prefix byte-identical across requests? | yes/no — decides prefix caching |
| D8 | Is it a single-turn call, or a conversation with growing history? | |

```python
# D1, D2, D6 — run over a sample of real traffic (a few thousand rows)
from transformers import AutoTokenizer
import numpy as np, json
tok = AutoTokenizer.from_pretrained(MODEL_PATH)

n_in  = np.array([len(tok(p).input_ids) for p in prompts])
n_out = np.array([len(tok(o).input_ids) for o in outputs])
for name, a in [("prompt", n_in), ("output", n_out)]:
    print(f"{name:7} median {np.median(a):7.1f}  p95 {np.percentile(a,95):7.1f}  max {a.max():6d}")
print("ratio n_in / n_out (median):", np.median(n_in) / np.median(n_out))
```

That last ratio alone predicts the prefill/decode split. Your measured 5 / 95
implies roughly `n_in ≈ 8 · n_out` at batch 1 on A100-class hardware — so I want
to see whether the real numbers match, because a mismatch means the bottleneck is
something other than raw arithmetic (scheduling, queueing, or padding).

## E. Current measurements

| # | Question | Where |
|---|---|---|
| E1 | How was the 5% / 95% split measured? | profiler, vLLM metrics, or wall-clock estimate? |
| E2 | Average number of running sequences | `vllm:num_requests_running` |
| E3 | Queue depth | `vllm:num_requests_waiting` |
| E4 | KV cache utilization | `vllm:gpu_cache_usage_perc` |
| E5 | Prefix cache hit rate | `vllm:gpu_prefix_cache_hit_rate`, or grep the log |
| E6 | TTFT and TPOT / ITL percentiles | `vllm:time_to_first_token_seconds`, `vllm:time_per_output_token_seconds` |
| E7 | End-to-end p50 / p95 latency as the caller sees it | your own instrumentation |
| E8 | GPU utilization and achieved memory bandwidth | `nvidia-smi dmon`, or Nsight |

```bash
# E2-E6 — if the metrics endpoint is up
curl -s localhost:8000/metrics | grep -E \
  'num_requests_(running|waiting)|cache_usage|prefix_cache|time_to_first_token|time_per_output_token' \
  | grep -v '^#'

# E5 also appears in the server log
grep -i 'prefix cache' server.log | tail -20
```

**Why E2 and E4 matter most.** If `num_requests_running` sits well below
`max_num_seqs` while `num_requests_waiting` is zero, you are not batch-limited —
you are simply not being offered enough concurrent work, and every decode step is
reading the full weights to serve very few tokens. That is the cheapest possible
fix and it costs nothing to check.

## F. Hardware

| # | Question |
|---|---|
| F1 | GPU model and count per replica |
| F2 | Number of replicas |
| F3 | Interconnect if TP > 1 (NVLink vs PCIe) |
| F4 | Is anything else sharing the GPU? |

```bash
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv
nvidia-smi topo -m
```

## G. The safety filter itself

This is what decides whether speculative decoding is even the right lever.

| # | Question |
|---|---|
| G1 | The **exact output schema** — field names, types, and whether values are enums or free text |
| G2 | A handful of real input/output pairs |
| G3 | Does the model emit any reasoning or explanation before the JSON? |
| G4 | Is the output ever long, or is it always a small fixed dictionary? |
| G5 | How is validity enforced today — prompting, retries, or a grammar? |
| G6 | Is the system prompt fixed, or does it vary per request/tenant? |
| G7 | What is the quality bar, and how is it measured? (needed before proposing a smaller model) |

Please paste G1 verbatim, including the JSON example shown to the model.

**Why:** if every value is drawn from a small enum, most output positions are
grammatically forced, and constrained decoding with jump-ahead *removes* those
decode steps rather than merely accelerating them. That would beat speculative
decoding, be lossless, and require no draft model. I cannot tell which applies
without seeing the schema.

## H. Constraints

| # | Question |
|---|---|
| H1 | Latency SLO — is it p50 or p99, and what number? |
| H2 | Is the filter on the critical path of a user-facing request, or offline/batch? |
| H3 | Are outputs required to be bit-identical to today's, or is a small quality delta acceptable? |
| H4 | Can the model be swapped, or is this specific checkpoint fixed? |
| H5 | Can the prompt be changed, or is it locked by another team? |
| H6 | Is there budget for training a draft model / an EAGLE head? |
| H7 | Any compliance constraint on which models may be used? |

H3 is the one that gates the most: speculative decoding is output-distribution
preserving, quantization is not, and swapping models certainly is not.

---

## Minimum viable set

If gathering all of the above is too slow, these seven answers unblock the
analysis on their own:

1. `config.json` (A)
2. The vLLM launch command and version (C1, C2)
3. Median prompt and output token counts (D1, D2)
4. Fixed-prefix length and whether it is byte-identical (D6, D7)
5. `num_requests_running` and `gpu_cache_usage_perc` under real load (E2, E4)
6. GPU model (F1)
7. The output schema, verbatim (G1)

---

## What I will produce from it

- A per-request time budget: tokenize / prefill / decode / detokenize, with the
  arithmetic shown rather than asserted.
- A read on which of your current flags are helping, which are inert, and which
  are actively costing you (`enforce_eager` being the usual culprit).
- An expected speedup range for speculative decoding, derived from your schema's
  predictability rather than from a generic benchmark number — plus an honest
  statement of when it would *slow you down*, which happens at high batch size.
- A ranked list of alternatives with effort estimates, so the comparison is
  against the real menu rather than against doing nothing.
