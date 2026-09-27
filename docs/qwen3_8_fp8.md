# Qwen3.8-27B-FP8

The checkpoint uses the `qwen3_5` architecture: 64 language layers (48 gated
delta-net and 16 full-attention layers) and a 27-layer vision transformer.
The deployment reuses `models/qwen3_5`; only its graph, weight format and
precision policy differ from the NVFP4 deployment.

## Preparation

Run from the repository root. No checkpoint is downloaded.

```bash
uv venv --allow-existing --python /usr/bin/python3.12 ~/workspace/envs/qwen3.8-fp8
uv pip install --python ~/workspace/envs/qwen3.8-fp8/bin/python \
  -r tools/requirements-qwen3_8_fp8.txt
~/workspace/envs/qwen3.8-fp8/bin/python tools/convert_qwen3_8_fp8_to_gguf.py
~/workspace/envs/qwen3.8-fp8/bin/python tools/qwen3_8_fp8_infer.py
~/workspace/envs/qwen3.8-fp8/bin/python -m unittest discover -s tools -p 'test_qwen3_8_fp8.py'
```

Both tools default to
`~/workspace/MultimodalModels/QwenModels/Qwen3.8-27B-FP8`.
Use `--model` to override it. The exporter writes separate language and
vision/projector files to `model/qwen3_8_fp8`; `--output-dir` changes that
directory. It refuses to overwrite an existing export.

On the current AArch64 installation, `uv pip check` flags NVIDIA
`nvidia-cusparselt-cu13` 0.8.1's internal `manylinux2014_sbsa` wheel tag.
Its shared library is an AArch64 ELF; dense inference does not use cuSPARSELt.
Do not retag installed wheels to hide this metadata warning.

### Numerical contract

* Original E4M3FN bytes and their original 128x128 scales are preserved.
  The language GGUF uses the existing Omni-Runtime extension type **43**,
  not a standard upstream ggml type. Every FP8 tensor has a separate FP32
  `<projection>.weight_scale_inv` tensor, stored in `[output/128, input/128]`
  order (GGUF dimension order is reversed). Generic gguf-py/llama.cpp readers
  do not implement this extension.
* Activations are dynamically quantized per token and 128 input values:
  `scale = max(absmax, 1e-12) / 448`, round-to-nearest-even E4M3FN, saturation
  at 448. Scale evaluation multiplies by the FP32 reciprocal of 448,
  matching PyTorch scalar division (a different rounding order changes ties).
  Each block dot is summed exactly, rounded once to FP32, and then multiplied
  by the activation and weight scales with FP32 block accumulation.
  CUDA prefill decomposes FP8 values into three balanced base-128 digits and
  uses exact INT8 tensor MMA dots with INT32 accumulators combined in INT64; decode
  and OpenCL use INT64 sums. The independent Python reference uses FP64
  dots: E4M3FN values are multiples of 1/512, and a 128-value dot needs at most
  43 integer bits, fitting exactly in FP64. This avoids BF16 midpoint changes
  caused by different FP32 reduction orders. It does not convert or
  cache the model weights as F16, Q8 or NVFP4.
  Recurrent output projections additionally restore the original logical
  block-summation order with `block_reduction_groups`. Reordering value heads
  and their scales alone preserves real arithmetic, but changes FP32 block
  accumulation and can change BF16 rounding.
  CUDA quantizes activations once per projection into capture-resident scratch,
  rather than repeating quantization for every output channel/tile. Prefill
  stages all three signed digits once per K block in padded shared memory.
  Decode distributes independent 128-value dots across four warps per output
  channel, then accumulates the scaled partials in the original logical order
  using a bounded 256-block window.
  Its activation workspace retains the exact scaled integer code once per
  quantized activation instead of decoding that byte again for every output
  channel. A shared 256-entry integer codebook decodes streamed weights;
  eight-lane subgroups use aligned 16-byte loads, with byte-safe unaligned views.
  On Blackwell, aligned loads reuse the existing streaming GEMV helper to avoid
  allocating one-use weights in L1 and request 256-byte L2 prefetches.
  Prefill retains byte-sized activation storage.
  The scratch cache is isolated by output allocation, view offset and batch
  shape, allocated before graph capture and included in workspace accounting.
* The deployment uses the checkpoint's BF16 activation/output dtype and BF16
  KV cache. Scaled FP8 block accumulation, normalization statistics, convolution and
  recurrent state use FP32. Attention products, softmax and weighted sums also
  use FP32 boundaries (`fp32_sdpa` in the independent reference), with an
  explicitly selected CUDA memory-efficient SDPA backend and expanded GQA
  keys/values. Automatic backend selection is not part of this contract: on
  identical cached-decode operands, PyTorch's math and efficient backends
  produced seven different BF16 attention values; native matched efficient.
  The BF16
  CUDA path reproduces the framework's TF32x3 tensor products, 128-key online
  softmax reductions and separately rounded epilogues; probabilities are not
  rounded to BF16. Causal 256-value language heads share this implementation
  with noncausal 72-value vision heads having complete 128-key tiles.
  Existing F16/F32 attention paths are unchanged.
  Grouped BF16 decode with a supplied `decode_partials` buffer first computes
  independent 128-key QK tiles into that existing FP32 scratch. Output partitions
  reuse these scores instead of recomputing QK eight times. Online softmax, PV
  products and final normalization still traverse keys in exactly the same order;
  there is no split-K softmax merge. Scratch must hold
  `cached_token_count * query_head_count` floats and is rewritten on replay.
  Calls without scratch retain fused attention; no allocation or public API is added.
  For two through eight query heads per KV head, the score consumer uses two
  warps and sixteen 16-value output partitions. Larger groups retain four
  warps and eight 32-value partitions. Both preserve the same key reduction order.
  Zero-centered RMSNorm weights remain FP32 after
  adding one; the norm is rounded only after multiplying those weights.
  Gated RMSNorm instead preserves its BF16 norm/weight multiplication and
  FP32 SiLU gate, as in Transformers. Other nonquantized weights are uploaded
  as BF16 without changing the FP8 GGUF storage.
  Contiguous CUDA BF16 RMSNorm widths from 128 through 5120, divisible by four,
  use four independent vector accumulators and a row-count-dependent reduction
  width. Mean scaling and epsilon addition retain separate FP32 rounding;
  fusing them changes BF16 outputs. Fused residual RMSNorm first rounds the
  residual sum to BF16 and uses the same reduction.
  FP8 visual linear projections use BF16 checkpoint biases and BF16 output
  descriptors (`is_float_bias: false`), selecting the first cuBLASLt heuristic.
  Describing the same bias values as FP32 selects a different reduction/epilogue
  and changes BF16 midpoint rounding. Other graphs retain the existing default
  FP32-bias policy and autotuning; cached plans distinguish the bias policies.
  Patch convolution instead rounds its BF16 result before adding bias,
  matching the checkpoint's convolution implementation.
  Unbiased BF16 linear projections use cuBLAS GEMM for single-token decode
  as well as prefill. The existing custom warp GEMV uses a different FP32
  reduction: identical real LM-head operands produced 23 differing BF16 logits.
  A non-checkpoint 5120x4096 regression also distinguishes the two reductions
  and exercises captured replay. Direct BF16 calls explicitly bind the same
  workspace as captured calls; changing workspace availability can select a
  different GEMM. Legacy F16/F32 single-row kernels remain unchanged.
  Recurrent beta/sigmoid gates and language RoPE products preserve their
  separate BF16 rounding boundaries.
* The exporter applies Qwen3.5's zero-centered RMSNorm offset and reorders
  recurrent V heads and both scale axes together. Original BF16 `A_log`
  values are retained as FP32 `blk.<layer>.ssm_a_log`; the negative exponential
  is evaluated on the GPU rather than rounded by a CPU exponential at export.
  Earlier FP8 exports containing only `ssm_a` must be regenerated into an
  empty directory before using this graph. The loader rejects the old tensor
  registry rather than interpreting negative decay constants as logarithms.
  NVFP4's existing `ssm_a` contract is unchanged.
  MTP tensors are not used in autoregressive decoding.
* FP8 opts into the CUDA `is_chunked` delta-net path with 128-value heads.
  Prefill uses 64-token chunks, contiguous-four-value FP32 L2 reductions,
  an FP32 Sklansky prefix scan, FP32 BLAS products and unit-triangular solves.
  As in PyTorch's CUDA scan, at most 32 batch rows scan all 64 values together;
  larger batches scan two 32-value segments, adding the previous segment's
  carry before the second scan. Using the wrong geometry changes FP32 state.
  The CUDA solver preserves the reference's separate dot/subtract order for
  at most eight batches and eight-row blocked substitutions for larger batches.
  Cached single-token decode retains separate FP32 decay, update and four-warp
  reduction boundaries. Scratch buffers are allocated before CUDA graph capture and
  isolated by output allocation and batch shape. These settings are not enabled
  for NVFP4 or Qwen-VL. OpenCL explicitly rejects this numerical mode.
* The explicit chat template matches the existing native user/assistant
  template (no inserted system message or thinking prefix). Image dimensions
  are rounded up to multiples of 32; bicubic RGB preprocessing uses `[-1,1]`
  normalization without letterboxing, Pillow's 22-bit bicubic coefficient
  precision and exact merger GELU. These are FP8-only graph settings.
  The FP8 path uses the JPEG decoder
  already used by the Qwen-VL deployment. The NVFP4 path remains unchanged.
  BF16 patch convolution uses temporal/spatial order and channels padded to 16.
  Position interpolation preserves separate weighted-tap products. Vectorized
  Welford LayerNorm preserves the reference's mean-merge FMA boundary.
  Visual rotary frequencies use reciprocal positive powers and unfused products.

The independent Python reference loads safetensors, not GGUF. Its FP8
projections use PyTorch FP8 rounding and explicit scaled block multiplication;
it never invokes native runtime kernels. The script derives nonquantized
module exclusions from the checkpoint index because prefix-based exclusions
such as `mlp.gate` can incorrectly exclude `mlp.gate_proj` in Transformers.
Pure PyTorch reference inference is intentionally slower than deployment.
Prefill uses batches of at most 512 tokens, split at image boundaries, with
Transformers' own cache and model modules. This matches the deployed scheduling:
different prefill batch shapes can change FP8 rounding and greedy token choices
even between two PyTorch runs.

The custom `fp32_sdpa` implementation must register **both** its attention
function and the standard SDPA mask factory. Without mask registration,
Transformers returns no causal mask for this custom name. Cached multi-token
prefill then truncates historical KV to the current query length and applies
an incorrectly aligned causal mask. The corrected independent reference uses
causal masks covering the complete cached prefix; its JSON records
`attention_mask: sdpa_causal_with_cache` and
`attention_backend: efficient_attention_fp32`. GQA keys/values are explicitly
expanded before SDPA, so cached decode cannot silently fall back to the math
backend. Missing causal masks for cached multi-token queries raise an error
instead of discarding historical KV. The corrected mask and explicit backend
were approved before independently regenerating the FP8 fixtures. Older FP8
fixtures without this contract must be regenerated with the corrected Python
script, not with native outputs. Qwen-VL and NVFP4 references are not regenerated.

It saves `test/inference/qwen3_8_27b_fp8.{txt,tokens,prompt.tokens,json}`.
The C++ example compares **all generated token IDs, the full decoded text,
the prompt token IDs and streaming output**, without a similarity threshold,
truncation or updating the reference from C++ results. Defaults are the demo
image, `Describe this image.`, greedy decoding and a 512-token generation
budget. Custom reference arguments do not change the fixed regression test;
use `--reference-output` outside that fixture for experiments.

## Operator inventory

| Required operation | Implementation |
| --- | --- |
| Block E4M3FN W8A8 linear, per-token activation scales | New `kernels/fp8_kernel` CUDA/OpenCL implementation and parity/replay test |
| Unquantized linear/embedding | Existing `matmul_kernel`, `quantized_kernel`; optional FP32 linear bias and captured BF16 projection |
| RMSNorm, residual RMSNorm, LayerNorm | Existing `layernorm_kernel` |
| Causal convolution, gated delta rule, sigmoid gate, partial MRoPE | Existing `delta_net_kernel` |
| Split gated Q, attention, KV writes | Existing `gated_attention_kernel`, `gqa_attention_kernel`, `kv_cache_kernel` |
| SwiGLU, residual add, greedy argmax, position advance | Existing `swiglu_kernel`, `add_kernel`, `argmax_kernel`, `sampling_kernel` |
| Bicubic image patches, paired patch embedding, grid position/RoPE, GELU/bias/merger | Existing `image_kernel`, `vision_encoder_kernel` |

No separate FP8 model class or duplicate model pipeline is needed.
The model graphs follow the current directory layout:
`models/qwen3_5/graphs/qwen3.8-27b-fp8/qwen3.8-27b-fp8.yaml`,
`models/qwen3_5/graphs/qwen3.8-27b-nvfp4/qwen3.8-27b-nvfp4.yaml`, and
`models/qwen3_vl/graphs/qwen3-vl-8b-fp16.yaml`.
NVFP4 and Qwen-VL numerical graph settings remain unchanged.

## Build and regression tests

```bash
bazel clean
./script/build.sh
./script/run_test.sh
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
./script/run_test.sh --target //example:qwen3.8_27B_nvfp4_inference_test --test-output all
./script/run_test.sh --target //example:qwen3.8_27B_fp8_inference_test --test-output all
```

Model files are local runfiles, not checked into git. The Python-generated
reference fixtures must exist before building/running the FP8 example.
The FP8 example alone links cuBLAS/cuBLASLt from the specified Python environment
through `@qwen_fp8_blas`; CUDA runtime and the two legacy model targets retain
their system toolkit linkage. On this host, cuBLAS 13.1.1 matches the independent
BF16 gate projections, while system cuBLAS 13.0.2 differs on identical operands.
Applications embedding the FP8 engine must use the same BLAS dependency rather
than replacing global CUDA libraries or setting a process-wide toolkit override.
Model tests run exclusively to avoid overlapping large GPU allocations.
The test script retains parallel compilation but serializes local test binaries:
CUDA/OpenCL backend initialization and small allocation tests also share the GPU.

## Exact-decode follow-up (2026-11-18)

The committed `d8c2b94` production kernels measured 7.82 and 7.61
decode tokens/s in two unprofiled runs (124.62 and 124.53 prefill
tokens/s). These samples supersede the older baseline for evaluating
this pass; they do not establish an improvement between commits.

The nominal GB10 bandwidth of 273 GB/s divided by 26.876 GB of
FP8 codes/scales and BF16 output weights per token gives **10.16
tokens/s**. The previously measured 234–236 GB/s streaming rate
gives **8.72–8.79 tokens/s** for those weights alone. Neither bound
accounts for KV reads, recurrent state, arithmetic, or launch gaps.
NVFP4 reads fewer weight bytes; its prefill also uses a different
compute path than the six-INT8-MMA reconstruction used for exact FP8.
Its throughput is therefore not a target for an unchanged FP8 path.

A fresh Nsight Systems CUDA-node trace isolated the 454 launches of
the 1,477-node single-token graph (excluding prefill and vision):

| Decode kernel family | Time per token, ms |
| --- | ---: |
| FP8 GEMV | 109.032 |
| BF16 output head (cuBLAS) | 14.283 |
| Grouped BF16 attention, both stages | 4.103 |
| Delta-net update | 3.536 |
| BF16 RMS normalization | 1.168 |
| Other BF16 cuBLAS projections | 0.600 |
| FP8 activation quantization | 0.478 |

FP8 GEMV alone accounts for about 81% of profiled decode wall time.
Its weight bytes divided by kernel time imply about 223 GB/s of
effective throughput, not a DRAM-counter reading. At the measured
streaming rate, those GEMVs alone would still require about 103 ms
per token. Thus eliminating their remaining gap to the streaming
reference would save only about 6 ms per profiled token with the
rest of the graph unchanged. The BF16 output head is the next
largest target, but changing its reduction order can change logits.

Three exactness-preserving candidates were tested: a 256-thread GEMV
for large aligned projections, packed weight-NaN checks with a
block-level activation-NaN flag, and native warp integer sums of the
high and low parts of the exact 64-bit dot. All passed the strict
model-output test, but complete decode runs measured 7.59, 7.57,
and 7.41 tokens/s respectively. The wider GEMV improved several
rotating-buffer microbenchmarks but not whole-model throughput.
None established a repeatable gain over the control, so no candidate
was retained in production. The expanded NaN regression exercises
both weight-NaN signs across GEMV lanes and isolates a NaN activation
row in one-, two-, and eight-row decode and nine-row prefill.

After a clean build (63 targets), all 21 test targets and the three
individual model commands passed. The final unprofiled FP8 run matched
all 455 output token IDs and measured 7.49 decode and 123.59 prefill
tokens/s. NVFP4 decode measured 11.86 before and 11.77 after; Qwen3-VL
decode measured 15.21/15.18 before and 15.06/15.11 after for its two
workloads. These are individual samples with identical production
kernels, not evidence of either a speedup or a change in kernel performance.

## Bandwidth-limited follow-up (2026-11-17)

The baseline is the complete working tree after the score-reuse pass below.
It was preserved in a runnable source snapshot before editing, with local
model files accessed through symlinks. Baseline FP8 measured **7.32 decode
tokens/s**, with the same 455-token reference.

Only two execution changes survived measurement and exact-output checks:

* **Streaming FP8 weight loads:** reuse `matmul_kernel::LoadGemvWeight<true>`
  rather than introducing another load helper. On Blackwell its
  `L1::no_allocate.L2::256B` policy avoids filling L1 with one-use weights;
  activation reuse and the exact integer codebook are unchanged. Older
  architectures retain ordinary vector loads through the existing helper.
* **Compact ordered attention:** for grouped score consumers with at most
  eight heads per KV head, use two warps and 16-value output tiles. Qwen's
  four KV heads now launch 64 blocks instead of 32, exposing work to all
  48 GB10 SMs. No key partition, new scratch buffer or softmax merge is added.
  Groups of 9–16 retain the previous four-warp path. The regression now
  explicitly covers both sides of the 8/9-head boundary.

The full-output development run measured **7.42 decode tokens/s** and
124.16 prefill tokens/s. The CUDA-node trace measured 2.959 ms/token for
ordered softmax/PV, versus 3.453 ms in the preceding trace, a **14.3%**
reduction in this kernel family. Score preparation still costs 1.203 ms/token.
FP8 GEMVs remain dominant at **110.793 ms/token**, followed by BF16 BLAS
at **14.825 ms/token**. These traces are not substitutes for alternating
unprofiled model measurements.

Several faster-looking alternatives were deliberately rejected:

* Splitting the 5120x248320 BF16 output projection reduced about 15 ms to
  10.6–10.9 ms, but changed 29–51 BF16 results across three independent
  synthetic inputs. Padding the input to multiple rows changed hundreds.
  Neither method preserves the existing BLAS reduction contract.
* Exact INT8 tensor-MMA decode preserved results, but took approximately
  0.73 ms instead of 0.36 ms for a cache-cold 5120x17408 projection.
* Fixed-shape GEMV specialization did not show a stable enough benefit to
  justify the extra dispatch and compiled variants; it was removed.

The weight inventory and bandwidth limits below still apply: **10.158
tokens/s** is the optimistic rated-bandwidth ceiling, and **8.72–8.79
tokens/s** is the measured weight-only streaming reference. 7.42 tokens/s
is about **84%** of the latter, not proof of physical saturation. No
checkpoint conversion, approximate accumulation, output-fixture update,
global CUDA-library override or system performance setting was used.

### Follow-up verification and matched A/B results

The final `bazel clean`, `./script/build.sh` and default
`./script/run_test.sh` passed: **63 build targets / 451 actions**, **21/21
test targets**, and **217 kernel cases without skips**. The three requested
individual model commands were run twice on the final code; all passed their
unchanged full-output assertions. Memcheck reported **0 errors** across ten
FP8/attention cases, and compact-attention Racecheck reported **0 errors /
0 warnings**.

Two serial pairs used identical individual commands on the preserved
baseline (A) and final source (B), including complete generations, not shortened
token budgets. Decode throughput is shown as **A / B**:

| Case | Pair 1 | Pair 2 | Mean A / B |
| --- | ---: | ---: | ---: |
| Qwen-VL, 64-token run | 14.91 / 14.86 | 14.76 / 14.83 | 14.835 / 14.845 |
| Qwen-VL, 352-token reused graph | 14.90 / 14.84 | 14.75 / 14.80 | 14.825 / 14.820 |
| NVFP4, 399-token run | 11.71 / 11.70 | 11.63 / 11.64 | 11.670 / 11.670 |
| FP8, 455-token run | 7.32 / 7.41 | 7.32 / 7.40 | 7.320 / 7.405 |

The incremental FP8 decode gain is **1.16%**. The full-suite FP8 run measured
7.42 tokens/s; its result is reported separately from the matched individual
pairs. This is a small bandwidth-limited improvement, not a new large speedup.
FP8 mean prefill was 123.67 / 123.40 tokens/s; no prefill acceleration is claimed.

Legacy decode means differ by at most **0.07%**. Mean prefill throughput
was 3131.985 / 3127.245 for the first Qwen-VL run, 3708.035 / 3716.870
for its reused graph, and 581.365 / 579.385 for NVFP4: changes within
**0.35%**, with no legacy kernel or numerical-policy change.
First-run Qwen-VL vision timings varied from 1021–1086 ms on A and
1000–1162 ms on B; B's two-sample mean was 2.6% higher. The clean full-suite
Qwen-VL first run also had a separate 4658.22 ms vision / 2387.21 ms prefill
timing outlier, while its reused run measured 590.27 / 739.89 ms. These
first-run measurements are not silently mixed into the matched individual
throughput comparison or treated as proof of a zero-regression guarantee.
All numerical checks passed, including these runs.

## GB10 decode roofline and score reuse (2026-11-17)

This pass uses the existing working tree, including the three earlier passes,
as its baseline. No checkpoint, reference, precision policy, prompt, generation
budget or BLAS dependency was changed.

### Physical limit versus an attainable target

[NVIDIA specifies 273 GB/s for GB10's shared LPDDR5x memory](https://docs.nvidia.com/dgx/dgx-spark/hardware.html).
Reading the actual GGUF tensor directory, rather than estimating from the
model's name, gives the following principal per-token traffic:

| Data | Bytes |
| --- | ---: |
| 400 FP8 projection matrices | 24,326,963,200 |
| Original FP32 block scales | 5,939,200 |
| BF16 output projection | 2,542,796,800 |
| Total | 26,875,699,200 |

Thus `273e9 / 26875699200 = 10.158 tokens/s`, or **98.45 ms/token**,
is an optimistic weight-streaming ceiling for this batch-one implementation.
It excludes KV, small nonquantized projections, recurrent-state reads/writes,
activation traffic, computation and synchronization. It is not a universal
limit for batching, lossless compression or speculative execution.

A separate CUDA read-only bandwidth microbenchmark used a **1 GiB** buffer
initialized with nonconstant integers, 16-byte coalesced loads, a checksum
output, 32 passes, and 128/256/512/1024 blocks of 256 threads. The buffer
exceeds GB10's measured **24 MiB L2**. CUDA-event measurements were
236.32/234.47/236.02/235.56 GB/s, implying a same-host, weight-only ceiling
of **8.72–8.79 tokens/s**, before the other work above. This is a measured
bandwidth reference, not proof that no other access pattern could be faster.
No clocks, power settings, driver permissions or other users' processes were changed.

NVFP4 streams fewer weight bytes and uses a different arithmetic contract.
Its 12.18 tokens/s is not an attainable target for this unchanged FP8
weight-streaming path. Similarly, prefill's arithmetic intensity and exact
six-product INT8 reconstruction differ from NVFP4; advertised sparse FP4
FLOPS do not predict exact FP8 prefill or batch-one decode.

### Trace-driven implementation

Nsight Systems CUDA-node traces isolate all **454 resident decode steps**
from vision, prefill and graph preparation:

| GPU work per decode token | Baseline | Optimized |
| --- | ---: | ---: |
| 400 FP8 GEMVs | 113.866 ms | 111.160 ms |
| 97 BF16 BLAS projections, including LM head | 14.925 ms | 14.909 ms |
| 16 attention layers, including score preparation | 9.370 ms | 4.640 ms |
| 48 recurrent updates | 3.430 ms | 3.414 ms |
| RMSNorm kernels | 1.372 ms | 1.353 ms |
| 400 activation quantizations | 0.549 ms | 0.550 ms |

The GEMV uses four warps per output, eight lanes per 128-value dot, 16-byte
aligned weight loads and a shared exact integer codebook. Unaligned views
remain byte-safe. INT64 dot reconstruction, NaN propagation and ordered FP32
block scaling/addition are unchanged. Alternatives were compared using eight
rotating weight buffers, not just repeated warm-cache execution.

Grouped BF16 attention previously recomputed the same QK scores for eight
output partitions. Two compile-time stages of the **same existing kernel**
now write independent QK tiles to the existing `decode_partials` allocation
and consume them in original key order. The first stage parallelizes QK
tiles; it does not partition softmax or combine independently normalized
outputs. The second preserves every online maximum, denominator, rescaling,
TF32 product and final epilogue boundary. In the trace, score preparation costs
1.187 ms/token and ordered softmax/PV costs 3.453 ms/token together, reducing
attention time by **50.5%**.

There is no new operator, model implementation, allocation cache, configuration
flag or persistent expanded-weight copy. Vision/prefill and calls without
scratch retain the fused implementation. Qwen-VL and NVFP4 do not select the
FP8 GEMV or BF16 attention policies.

### Full-output measurements and limits

The unprofiled requested FP8 command, with all 455 generated tokens:

| Metric | Baseline | Optimized |
| --- | ---: | ---: |
| Vision | 2111.05 ms | 2094.20 ms |
| Prefill | 22576.63 ms / 122.52 tokens/s | 22277.61 ms / 124.16 tokens/s |
| Decode | 65332.31 ms / 6.95 tokens/s | 61676.44 ms / 7.36 tokens/s |

Decode throughput improves **5.9%**. The profiled pair independently measured
6.89 / 7.27 tokens/s. Small vision/prefill differences should not be attributed
to a new prefill algorithm. Initial JIT/graph preparation is not decode time.

7.36 tokens/s is approximately **84% of the measured weight-only bandwidth
ceiling**, and **72% of the rated-bandwidth ceiling**. Remaining FP8 GEMVs
stream about 219 GB/s in the trace; BF16 BLAS and non-weight kernels still cost
time. This result approaches the observed streaming limit but **does not
establish physical saturation**. Nsight Compute counters remain unavailable
(`ERR_NVGPUCTRPERM`); CUDA-node timings and bandwidth measurements must not be
presented as measured hardware-counter utilization.

Focused regressions compare every encoded FP8 pair in prefill and decode,
unaligned/8-byte-aligned/16-byte-aligned weights, changed-input replay, NaNs,
cancellation and ordered reductions across the 256-block window. Attention
compares staged captured execution against fused direct execution for head
ratios 1/2/6/16/17, changed queries, 128-key boundaries, 3072 keys, the final
4097th cache slot and reset. Undersized score scratch is rejected explicitly.
The independent SDPA digests, full generated tokens, prompt tokens, decoded
text and streaming assertions remain unchanged.

### Final clean-build verification

After the final source/test edits, `bazel clean` and `./script/build.sh`
passed for **63 targets / 451 actions**. The default `./script/run_test.sh`
passed **21/21 test targets**, including **217 kernel cases with no skips**.
All three explicitly requested individual model commands then passed:
Qwen-VL 5/5, NVFP4 3/3 and FP8 3/3.

The full-suite FP8 measurement was **123.66 prefill / 7.33 decode tokens/s**.
The final individual run measured **123.50 / 7.28 tokens/s**, with vision
2133.66 ms and decode 62328.82 ms. Across the three unprofiled validations,
decode was **7.28–7.36 tokens/s**, versus 6.95 at baseline: approximately
**4.8–5.9% improvement**, not a claim based solely on the fastest sample.
Every run retained all 455 reference tokens and complete text.

Legacy-model individual measurements, baseline / final:

| Case | Vision (ms) | Prefill (tokens/s) | Decode (tokens/s) |
| --- | ---: | ---: | ---: |
| Qwen-VL, 64-token run | 1053.73 / 1068.83 | 3170.81 / 3108.07 | 14.89 / 14.95 |
| Qwen-VL, 352-token reused graph | 592.77 / 594.28 | 3748.07 / 3706.21 | 14.87 / 14.94 |
| NVFP4, 399-token run | 2063.42 / 2090.06 | 588.59 / 580.26 | 11.69 / 11.66 |

The existing legacy kernels and numerical policies are unchanged. Exact
output equality is verified; these finite shared-host timing samples do
**not** prove literal zero performance regression. In particular, final legacy
prefill samples are 1.1–2.0% lower, and NVFP4 decode is 0.26% lower. These
differences are reported rather than hidden by selecting faster samples.

Final-source Memcheck passed all ten focused FP8/resident-attention cases
with **0 errors**, including the new eight-byte-offset weight view. Racecheck
of shared codebook and staged attention replay reported **0 errors / 0 warnings**.
No reference fixture was regenerated or assertion relaxed.

## Original integration validation

The specified Python environment independently generated **455 tokens** from
**2766 prompt tokens**. Native inference matches every prompt/generated token
ID, the complete decoded text and streaming output. No comparison was relaxed,
and no fixture was generated from native results. This verifies the fixed
demo-image regression under the numerical contract above, not arbitrary inputs
or different attention backends.

The final `bazel clean` and `./script/build.sh` completed successfully for
**63 targets** (451 actions). The complete `./script/run_test.sh` passed
**21/21 test targets**, and the specified environment passed all **nine**
Python reference/converter tests. Each requested model command was also rerun
individually:

| Model | Passing cases | Full-text/token result |
| --- | --- | --- |
| Qwen-VL | 5/5 | Unchanged references, including the 352-token reused-graph run |
| NVFP4 | 3/3 | Unchanged 399-token reference |
| FP8 | 3/3 | All 455 generated tokens, full text, prompt and streaming match |

Diagnostic CUDA graph snapshots matched all 64 language layers across all
eight prefill batches, including raw FP32 recurrent state, and all 64 layers
of the first decode step. All **14,090,240** final image-embedding values
matched the independent reference. Direct execution was not assumed equivalent
to captured execution: workspace-dependent BLAS selection was checked in the
captured graphs. Temporary production trace hooks were removed before the
final build and tests.

Focused regressions cover BF16 single-row reduction and captured replay,
BF16/FP32 bias-plan separation, optional graph-policy binding, batch-dependent
delta-net scan geometry, cached causal masks and efficient-only GQA attention.
The shared model pipeline is retained; Qwen-VL and NVFP4 fixture files and
system BLAS linkage are unchanged.

### First optimization pass (2026-11-16)

Nsight Systems attributed **45.9%** of baseline GPU kernel time to FP8
prefill and **45.0%** to FP8 GEMV. The bottleneck was not token streaming:
GEMV repeated activation quantization for every output channel, and prefill
reloaded/decomposed weights and synchronized for each of nine FP16 digit
products in every 128-value block.

The optimized CUDA path:

* Quantizes each activation block once per projection, including graph replay.
* Stages all three signed base-256 digits once per K block. INT8
  `m16n8k32` MMA halves the K-step count compared with FP16 `m16n8k16`.
  INT32 digit-dot accumulators cannot overflow: each contains at most
  `128 * 128 * 128 = 2^21` in magnitude. Reconstructing the nine digit
  products in INT64 preserves the original exact-dot contract.
* Uses 64x32 prefill tiles and 144-byte shared-memory row strides to reduce
  register pressure and bank conflicts.
* Computes decode block dots using four warps per output channel. A bounded
  32-block shared-memory window preserves sequential FP32 block accumulation,
  including reordered recurrent heads; it does not introduce split-K summation.

No weights, fixtures, token budgets, batch boundaries, attention policy or
precision settings changed. A model-sized 5120x17408 projection additionally
matched the original CUDA kernel bit for bit at 1, 8, 9, 65 and 512 input rows.
Kernel regressions cover the GEMV/prefill boundary, partial tiles, more than
32 K blocks, head-block ordering, activation/weight NaNs, catastrophic
cancellation and captured replay with changed inputs. FP32 reference checks
now use exact equality and explicitly separate the reference's multiply/add
rounding boundaries instead of allowing host FMA contraction.

Same-host measurements with the unchanged 2766-token prompt and all 455
generated tokens:

| Metric | Baseline | Optimized |
| --- | --- | --- |
| Vision | 3079.51 ms | 3102.96 ms |
| Prefill | 120352.00 ms / 22.98 tokens/s | 34145.62 ms / 81.01 tokens/s |
| Decode | 140798.29 ms / 3.22 tokens/s | 83862.99 ms / 5.41 tokens/s |
| Generation wall time | 265534.73 ms | 123109.08 ms |

This is **3.52x prefill throughput**, **1.68x decode throughput**, and
**2.16x end-to-end speedup**. The baseline was collected with CUDA node
tracing; its timings reproduce the original unprofiled report. The optimized
unprofiled full-suite run independently measured 80.97 prefill tokens/s and
5.42 decode tokens/s. This is not a claim of peak hardware throughput:
preserving exact integer block dots costs more than approximate native FP8
accumulation, and single-token decoding still streams the model's FP8 weights.

### First-pass legacy-model A/B regression

The final clean build passed all **63 build targets** (451 actions), and
`./script/run_test.sh` passed **21/21 test targets**, including all **214**
kernel cases without skips. All three individual model commands above passed.
Prompt/generated tokens, full text and streaming remain exact; no fixture was
regenerated.

For legacy performance, an isolated copy restored the original modified
source files from commit `c8936f8`; model files were read through symlinks,
not copied or modified. Three serial A/B rounds alternated baseline/modified
order (A/B, B/A, A/B). Every round ran the complete original strict-output
tests, not a shortened generation. All decode samples below are tokens/s;
each cell is **baseline / modified**:

| Model / case | Round 1 | Round 2 | Round 3 | Mean |
| --- | --- | --- | --- | --- |
| Qwen-VL, 64-token run | 14.94 / 14.87 | 14.76 / 14.80 | 14.73 / 14.73 | 14.810 / 14.800 |
| Qwen-VL, 352-token reused graph | 14.93 / 14.84 | 14.75 / 14.79 | 14.72 / 14.72 | 14.800 / 14.783 |
| NVFP4, 399-token run | 11.70 / 11.66 | 11.62 / 11.64 | 11.60 / 11.60 | 11.640 / 11.633 |

Mean prefill throughput was 3153.17 / 3134.05 tokens/s for the first Qwen-VL
run, 3722.18 / 3707.04 for its reused graph, and 580.25 / 579.00 for NVFP4.
Decode mean differences are below **0.12%** and prefill differences below
**0.61%**, smaller than the observed run-to-run variation. There is no
observed material legacy performance regression in this A/B measurement;
a shared host cannot guarantee identical timings under arbitrary concurrent
loads. The FP8 dispatch, scratch storage and kernels are not used by either
legacy model.

## Second optimization pass: all three stages

The baseline for this pass is the working tree at the start of the second
optimization request, including the first pass above, not commit `c8936f8`.
An isolated source snapshot retained that exact baseline for A/B tests.

### Remaining bottlenecks and changes

* **Vision:** the FP8 checkpoint's vision graph still computes in BF16 with
  FP32 attention boundaries. Its 11008 image patches are not a short language
  prompt. The 27 full-attention layers repeatedly reloaded each query fragment
  for four independent score tiles and every 128-key step. The kernel now
  reuses each query fragment across score tiles and caches each vision query
  tile in padded shared memory. The original TF32 MMA sequence, softmax
  reductions, FP32 probabilities and BF16 output rounding are unchanged.
* **Prefill:** the old 64x32 exact FP8 tile required 240 registers per thread
  in the standalone CUDA build. Smaller 16x32/32x32 tiles reduce wasted rows
  and register pressure. Four-byte weight/code loads and packed digit stores
  replace scalar byte traffic, with a separate safe path for unaligned weight
  views. Tensor fragments are reused across digit products. Two INT32 radix
  limbs replace repeated INT64 arithmetic; their absolute bounds are below
  `2^31`, including the largest E4M3FN codes. The exact dot is reconstructed
  as `high * 65536 + low` in INT64 before the original FP32 rounding/scaling.
* **Decode:** FP8 GEMV processes four output channels per block, uses vector
  loads and a 256-block ordered reduction window, and bypasses integer
  division for ordinary contiguous K blocks. Reordered recurrent-head
  projections keep their original logical block order. Attention no longer
  pads one real query to two groups of 16 query rows. Four independent
  output-dimension partitions increase parallelism without splitting KV
  history, changing softmax or merging differently ordered partial sums.
* **Synchronization:** Racecheck identified a warp-level read/write hazard
  between reading the previous softmax denominator and lane zero updating
  it. An explicit warp barrier now separates these operations.

No new public operator, model class, persistent weight representation,
configuration flag or model-specific execution pipeline was added. The new
attention policy is selected only by the existing BF16 graph path; Qwen-VL
and NVFP4 retain their F16/F32 kernels and numerical policies.

Standalone model-sized CUDA comparisons retained bit-identical outputs.
Representative final-source timings (previous / current) were 0.459 / 0.401 ms
for a single-row 5120x17408 FP8 projection, 24.85 / 17.21 ms for a 512-row
17408x5120 projection, 94.46 / 62.11 ms for 11008-token vision attention,
and 0.953 / 0.436 ms for 3072-key decode attention. These are isolated-kernel
measurements, not end-to-end throughput claims.

### Final stage measurements

The unprofiled individual command, with the unchanged 2766-token prompt and
455-token generation, measured:

| Metric | Previous version | This pass |
| --- | --- | --- |
| Vision | 3081.94 ms | 2436.90 ms |
| Prefill | 33846.44 ms / 81.72 tokens/s | 27104.73 ms / 102.05 tokens/s |
| Decode | 83695.36 ms / 5.42 tokens/s | 69760.40 ms / 6.51 tokens/s |
| Generation wall time | 121832.63 ms | 100772.06 ms |

This pass reduces vision latency by **20.93%**, increases prefill throughput
by **24.87%**, increases decode throughput by **19.98%**, and reduces generation
wall time by **17.29%**. Against the originally reported 265699.77 ms generation,
the cumulative wall-time speedup is **2.64x**, not a claim that the hardware
limit has been reached.

The final full-suite run independently measured 2434.71 ms vision,
102.56 prefill tokens/s and 6.55 decode tokens/s. The individually profiled
run measured 2442.72 ms vision, 101.77 prefill tokens/s and 6.45 decode tokens/s.
Its GPU trace attributes 55.5% of kernel time to FP8 GEMV, about 25.0% to FP8
prefill, 5.0% to single-query BF16 attention and 1.9% to vision BF16 attention.
The unchanged BF16 BLAS projections account for another 7.0%. These remaining
costs are reported rather than treating an isolated kernel speedup as a model
speedup.

### Numerical and safety validation

The final `bazel clean`, `./script/build.sh` and `./script/run_test.sh` passed:
**63 build targets**, **21/21 test targets** and **215 kernel cases**, without
skips. All three individually requested model commands also passed, including
all 455 FP8 output tokens, complete text, prompt tokens and streaming checks.
The reference files and generation budgets were not modified.

Expanded FP8 regressions cover aligned and unaligned weights, 8/9/16/17-row
dispatch boundaries, 32-row tails, changed-input graph replay, and 260 K blocks
crossing the decode reduction window with reordered heads. An added BF16
attention regression replays resident cache lengths across 128-key boundaries,
the final cache entry and reset. Existing independent SDPA digests remain
unchanged. Targeted Compute Sanitizer Memcheck passed with **0 errors**;
Racecheck passed with **0 errors and 0 warnings** after the warp barrier fix.

Nsight Systems tracing remained available. Nsight Compute hardware counters
were not accessible (`ERR_NVGPUCTRPERM`); no driver permissions, GPU clocks or
shared-system configuration were changed. These measurements therefore do
not certify saturation of every hardware resource. In particular, single-token
decode streams tens of GB of weights, and exact 128-value integer-dot/FP32
rounding is more expensive than approximate native FP8 accumulation. The
device's peak low-precision arithmetic rate is not an achievable end-to-end
tokens/s target under this unchanged numerical contract.

### Second-pass legacy-model A/B regression

Three complete serial A/B rounds used the preserved previous working tree
and final code in order A/B, B/A, A/B. All original model assertions passed
in every run. Model files were accessed through symlinks, never modified.
All decode samples below are tokens/s; each cell is **previous / current**:

| Model / case | Round 1 | Round 2 | Round 3 | Mean |
| --- | --- | --- | --- | --- |
| Qwen-VL, 64-token run | 14.77 / 14.73 | 14.68 / 14.70 | 14.69 / 14.68 | 14.713 / 14.703 |
| Qwen-VL, 352-token reused graph | 14.75 / 14.71 | 14.67 / 14.69 | 14.68 / 14.67 | 14.700 / 14.690 |
| NVFP4, 399-token run | 11.60 / 11.59 | 11.56 / 11.57 | 11.57 / 11.56 | 11.577 / 11.573 |

Mean prefill throughput was 3170.50 / 3172.52 tokens/s for the first Qwen-VL
run, 3749.06 / 3720.24 for its reused graph, and 580.78 / 582.06 for NVFP4.
Mean vision latency was 998.96 / 998.89 ms for the first Qwen-VL run,
591.72 / 593.97 ms for its reused graph, and 2084.27 / 2077.58 ms for NVFP4.
The largest decode mean difference is **0.07%**; the largest prefill mean
difference is **0.77%**, and the largest vision mean difference is **0.39%**.
The reused Qwen-VL prefill sample is slightly lower in all three pairs; these
measurements should not be described as literally zero timing regression.
No material legacy regression was observed at the one-percent scale, but a
shared host and these finite samples cannot establish a strict zero-regression
guarantee under arbitrary loads. Numerical output equality is verified.

## Third optimization pass: exact data reuse

The baseline for this pass is the working tree at the start of the third
request, including the second pass. A separate source snapshot was used for
baseline executions; model files and Python reference fixtures were unchanged.

### FP8 arithmetic and scheduling

Prefill now represents the exact integer `512 * DecodeFp8(code)` as three
balanced base-128 digits. The first two digits are in `[-64, 63]`, so every
pairwise digit sum still fits signed INT8. Three diagonal INT8 MMA products
and three pairwise-sum products recover all cross terms: for example,
`cross01 = dot(a0 + a1, b0 + b1) - dot(a0, b0) - dot(a1, b1)`.
This replaces nine digit products with six, without approximating the dot.

The two reconstructed INT32 limbs are combined as `high * 16384 + low` in
INT64, then rounded to FP32 and scaled at the original 128-value boundary.
Exhaustive enumeration of all 65536 encoded pairs, including the integer
intermediates for NaN codes, bounds a 128-value low limb by 67108864 and
the high limb by 471859200. Both fit INT32. NaN codes still propagate as NaN.

A 256-entry shared lookup expands bytes to packed digits; byte permutations
assemble the tensor operands. It replaces repeated exponent decoding in
every output tile. Linear thread-block IDs are mapped to adjacent row tiles
of one weight tile, improving weight reuse without narrowing the original
launch-grid limits. Small row tiles and explicit unaligned-weight handling
are retained. A tested 64-row alternative used 214 registers per thread and
was slower for important projections, so it was rejected.

Decode quantization retains exact integer activations in transient INT32
scratch storage, with a separate NaN sentinel. The existing per-output/shape
workspace cache owns the allocation before graph capture. Streamed weights
remain in their original FP8 representation and are decoded directly.
There is no persistent expanded-weight cache or checkpoint conversion.

### Vision and grouped decode attention

The attention probability matrix still has 128 logical keys per step, but
its shared-memory row stride is 136 floats. This separates the banks used by
adjacent eight-lane row groups. Four rows perform softmax concurrently, while
retaining the original per-lane sum order and four ordered denominator groups.
Inactive padded rows are zeroed and do not run softmax; warp collectives use
the appropriate eight-lane active mask.

The 72-dimension vision query/key product omits the original all-zero padded
MMA steps. Residue/full-tile accumulation boundaries are preserved. The
FP32 probability representation, compensated TF32 value products and final
BF16 rounding are unchanged.

For single-query BF16 attention with 2–16 query heads per KV head, those
query heads now occupy independent rows of one MMA tile. Eight disjoint
output-dimension partitions provide parallelism. Each head keeps its own
maximum, denominator and output accumulator, and traverses KV blocks in the
original order. Other head ratios retain the independent-head path. This
does not partition the KV reduction or introduce a differently ordered merge.

An intermediate policy looked faster in warm, isolated attention benchmarks
but increased real resident-cache decode attention to about 1.04 ms per
launch. Whole-model profiling caught that regression. Grouped reuse reduced
it to about 0.60 ms in the measured trace before the final prefill scheduling
change. Isolated kernel results were not accepted as proof of model speed.

### Interpreting the hardware ceiling

[NVIDIA's DGX Spark hardware specifications](https://docs.nvidia.com/dgx/dgx-spark/hardware.html)
list **273 GB/s** memory bandwidth; the advertised **1 PFLOP** is sparse FP4
compute, not exact FP8 batch-one decode throughput.

The unchanged GGUF header contains 400 FP8 matrices totaling
**24326963200 code bytes**, plus **5939200 scale bytes**, per traversal.
The 5120x248320 output projection is executed in BF16 and adds
**2542796800 weight bytes**. Embedding lookup does not read the entire
embedding table and is not counted twice here.

For this uncompressed, single-request execution, streaming just these
weights at the rated bandwidth already takes about **98.4 ms/token**
(roughly **10.2 tokens/s**), before KV reads, exact integer reconstruction,
attention, recurrent state, norms and launch overhead. This is a simplified
streaming roofline, not a proof of saturation or a universal bound for
different compression, batching or speculative-execution algorithms.

### Regression coverage

The added FP8 test checks all 65536 encoded input/weight combinations against
an independent scalar reference, including subnormals, signed codes and NaNs,
at different packed-byte positions on CUDA and OpenCL. Existing cancellation,
rounding-boundary, aligned/unaligned, changed-input replay and reordered-head
tests remain exact. Resident BF16 decode replay now covers head ratios
1, 2, 6, 16 and 17, cache-tile boundaries, the final cache entry and reset.
The cache-position buffer is explicitly initialized before capture.

### Third-pass final measurements and verification

The final unprofiled requested FP8 command retained the same 2766 prompt
tokens, 455 generated tokens, complete text and streaming assertions:

| Metric | Third-pass baseline | Final code |
| --- | --- | --- |
| Vision | 2455.48 ms | 2130.83 ms |
| Prefill | 26962.24 ms / 102.59 tokens/s | 22747.88 ms / 121.59 tokens/s |
| Decode | 69756.74 ms / 6.51 tokens/s | 67205.21 ms / 6.76 tokens/s |
| Generation wall time | 100677.09 ms | 94789.58 ms |

Vision latency decreased **13.22%**, prefill throughput increased **18.53%**,
decode throughput increased **3.80%**, and generation wall time decreased
**5.85%**. The cumulative generation-wall speedup over the initially reported
265699.77 ms is **2.80x**. These measurements do not include initial model
loading as generation time and do not claim hardware saturation.

Final-source standalone comparisons were bit-identical: 17.14 / 13.84 ms
for a 512-row 5120x17408 FP8 projection, 17.26 / 14.20 ms for a 512-row
17408x5120 projection, and 61.01 / 49.19 ms for 11008-patch vision attention.
The whole-model result above, not these isolated speedups, is the outcome.

After the final scheduling change, `bazel clean`, the complete build script,
the complete test script and all three individually requested model commands
were rerun successfully: **63 build targets**, **21/21 test targets** and
**216 kernel cases** passed. Targeted Memcheck reported **0 errors**;
Racecheck reported **0 errors and 0 warnings**. No reference fixture,
generation budget, public operation or legacy numerical policy was changed.

Three final serial legacy A/B rounds used order A/B, B/A, A/B. Every original
model assertion passed in every execution. Decode samples are tokens/s,
with each cell showing **previous / current**:

| Model / case | Round 1 | Round 2 | Round 3 | Mean |
| --- | --- | --- | --- | --- |
| Qwen-VL, 64-token run | 14.66 / 14.61 | 14.55 / 14.59 | 14.53 / 14.51 | 14.580 / 14.570 |
| Qwen-VL, 352-token reused graph | 14.64 / 14.58 | 14.55 / 14.57 | 14.53 / 14.51 | 14.573 / 14.553 |
| NVFP4, 399-token run | 11.52 / 11.49 | 11.49 / 11.48 | 11.48 / 11.49 | 11.497 / 11.487 |

Mean prefill throughput was 3155.29 / 3156.01 tokens/s for the first Qwen-VL
run, 3720.57 / 3711.09 for its reused graph, and 577.86 / 578.07 for NVFP4.
The largest decode mean decline was **0.14%**; the largest prefill mean
decline was **0.26%**.

Vision samples in ms were 1213.85, 1006.72, 1004.86 / 1020.66, 996.54,
1006.26 for the first Qwen-VL run; 600.22, 603.63, 602.23 / 601.64, 600.77,
601.24 for its reused graph; and 2098.39, 2111.30, 2110.09 / 2119.44,
2113.54, 2104.72 for NVFP4. The first baseline Qwen-VL vision sample was
an outlier and is retained here rather than omitted. Its median was
1006.72 / 1006.26 ms; no speedup is inferred from the outlier. NVFP4 mean
vision latency increased **0.28%**. The measurements show no material
legacy regression at the one-percent scale, not a guarantee of literally
zero timing difference on a shared host.

## Fourth pass: FP8 versus mixed NVFP4 decode roofline

The comparison is not between two identical four-bit/eight-bit execution
paths. The NVFP4 GGUF uses NVFP4 for its 192 feed-forward matrices but
Q8_0 for 305 other two-dimensional tensors, including its output head.
NVFP4 stores 64 weights in 36 bytes; Q8_0 stores 32 in 34 bytes.
The FP8 model instead traverses 400 E4M3 matrices, and its F32 GGUF output
head is converted to BF16 at loading time. Both models retain their
original, different golden generations: 455 FP8 tokens and 399 NVFP4
tokens. Total generation times alone are therefore not a same-length
comparison.

Read-only GGUF-header accounting, checked against the loader's actual
runtime types, gives the following dense traversal estimates. The
embedding table is excluded because generation reads a lookup row, not
the entire table. Convolution weights remain F32.

| Runtime weight category | FP8 bytes/token | NVFP4/Q8_0 bytes/token |
| --- | ---: | ---: |
| FP8 codes / NVFP4 feed-forward weights | 24326963200 | 9625927680 |
| Other Q8_0 matrices, including output | 0 | 9041018880 |
| BF16 output head | 2542796800 | 0 |
| BF16 recurrent alpha/beta projections | 47185920 | Included above |
| F32 convolution weights | 7864320 | 7864320 |
| FP8 block scales | 5939200 | 0 |
| Total | 26930749440 | 18674810880 |

At the manufacturer's **273 GB/s** rating, these traversals alone take
**98.65 ms/token** and **68.41 ms/token**, respectively: optimistic
weight-streaming ceilings of **10.14** and **14.62 tokens/s**.
The dominant-weight-only estimate in the preceding section is slightly
more optimistic because it excludes the small recurrent projections.
These are not attainable end-to-end guarantees: KV/state traffic,
activations, normalization, exact arithmetic, memory efficiency and
dispatch all consume additional time. They also assume the current
uncompressed, batch-one, non-speculative execution; a different algorithm
can change the traffic model.

Consequently, 12.18 tokens/s is not a physically realistic target for the
current FP8 weight-streaming path: its counted weights would require
about **328 GB/s** even before other work. Conversely, observing about
7 tokens/s does not establish saturation. The rated-bandwidth ceiling
still leaves measurable headroom.

The prefill difference has an additional arithmetic explanation.
NVFP4 prefill prepares BF16 operands and calls cuBLAS tensor-core GEMM.
FP8 prefill preserves the original exact 128-element block dot and
block-by-block FP32 rounding with six integer MMA products plus
reconstruction. Replacing it with an ordinary approximate FP8/BF16 GEMM
would not preserve that numerical contract and was not used as a shortcut.

### Profile-driven changes

The fourth-pass baseline whole-model CUDA-node trace attributed 56.9% of
kernel time to FP8 GEMV, 22.1% to the main FP8 prefill kernel, 7.7% to
BF16 GEMV, and 4.7% to grouped decode attention. These percentages include
vision, prefill and cancellation checks and are not decode-only fractions.

Restricting the trace to the 454 greedy single-token decode steps gives
**50.650 s** for FP8 GEMV, **6.723 s** for BF16 GEMV,
**4.250 s** for attention, **1.546 s** for recurrent decode and
**0.621 s** for RMS normalization. Activation quantization takes only
**0.252 s**. Thus repeatedly optimizing quantization or CPU token printing
cannot remove the dominant latency. FP8 code bytes divided by GEMV time
correspond to about **218 GB/s** (80% of rated bandwidth), not a direct
DRAM-counter measurement. The BF16 category includes the output head
and small recurrent projections.

Decode now uses 128-thread CTAs with two output columns. Within each
warp, independent 16-lane subgroups process separate 128-value blocks.
Eight weight codes are fetched with one aligned 64-bit vector load per
lane; unaligned views retain a byte-safe path. Four blocks per column
are evaluated concurrently, with one fewer shuffle stage than the
previous 32-lane reduction. NaN ballots are subgroup-local, including
partially populated final warps. The bounded shared partial window,
logical-to-physical head mapping, exact INT64 dot, separate scale
multiplications and sequential logical-block accumulation are unchanged.
No new persistent weight copy, public option, approximation or legacy
kernel policy is introduced.

Several plausible alternatives were measured and rejected: smaller
four/eight-lane groups, wider shape-specific warp configurations,
loop unrolling, cache streaming hints, native FP8 conversion followed
by exact integer accumulation, and finer attention output partitions.
Warm microbenchmark gains were not treated as model gains. In particular,
cached small matrices can report code-byte throughput above 273 GB/s;
that is not a measurement of sustained DRAM bandwidth.

The additional regression checks odd three-block subgroup tails and a
259-block reduction spanning the shared-memory window, seven-way
reordering, aligned vector loads, FP32 rounding and isolation of a NaN
in the final block from adjacent output columns.

### Fourth-pass verified outcome

The final source passed a fresh `bazel clean`, the complete build script
and complete test script: **63 build targets**, **21/21 test targets**,
including **217 kernel cases**. The three explicitly requested individual
model commands also passed in every A/B round. FP8 retained all 455
generated token IDs, full text, prompt-token and streaming assertions;
legacy golden outputs, cancellation and invalid-request checks were
unchanged. Memcheck reported **0 errors** and Racecheck **0 errors /
0 warnings** for all nine FP8 cases.

Three serial, unprofiled A/B rounds used order A/B, B/A, A/B. A is the
unchanged third-pass source snapshot; B is the final fourth-pass source.
Every sample is retained below; entries show **A / B**:

| Round | Vision, ms | Prefill, tokens/s | Decode, tokens/s | Generation wall, ms |
| --- | --- | --- | --- | --- |
| 1 | 2139.06 / 2128.13 | 122.02 / 122.18 | 6.92 / 6.95 | 92126.58 / 92862.74 |
| 2 | 2128.13 / 2131.40 | 122.08 / 122.18 | 6.84 / 6.93 | 92558.53 / 91794.59 |
| 3 | 2132.86 / 2132.35 | 122.03 / 122.07 | 6.84 / 6.92 | 92652.95 / 91997.33 |
| Mean | 2133.35 / 2130.63 | 122.04 / 122.14 | 6.867 / 6.933 | 92446.02 / 92218.22 |

Mean decode latency decreased from **66129.06 to 65485.53 ms**;
throughput increased approximately **0.97%**. Mean generation wall time
decreased **0.25%**. Vision and prefill were not changed by this pass;
their approximately 0.1% differences are not evidence of an optimization.
The first current wall-time sample was slower and is not discarded.
This is a small, measured decode improvement, **not achievement of the
10.14-token/s optimistic ceiling**.

A final back-to-back A/B CUDA-node profile at the end of validation
measured FP8 GEMV at **52.729 / 51.903 s** across the same 181600 decode
launches, a **1.57%** reduction. Model decode was **66906.95 /
66208.29 ms**. BF16 GEMV remained **6.774 / 6.779 s** and attention
**4.261 / 4.286 s**. An earlier, temporally separated final-source trace
measured FP8 GEMV at 51.843 s versus the initial baseline's 50.650 s;
it did not show a speedup. This is why the matched unprofiled rounds
and back-to-back traces, rather than distant profile totals, determine
the outcome. The remaining large costs have not disappeared.

Legacy decode throughput in the same three unprofiled rounds:

| Model / case | Round 1 | Round 2 | Round 3 | Mean |
| --- | --- | --- | --- | --- |
| Qwen-VL, 64 tokens | 15.14 / 14.90 | 14.82 / 14.86 | 14.76 / 14.74 | 14.907 / 14.833 |
| Qwen-VL, 352 tokens, reused graph | 15.12 / 14.90 | 14.80 / 14.85 | 14.77 / 14.74 | 14.897 / 14.830 |
| NVFP4, 399 tokens | 11.76 / 11.71 | 11.69 / 11.69 | 11.65 / 11.65 | 11.700 / 11.683 |

Legacy prefill means were **3117.09 / 3113.95**, **3696.52 / 3695.44**
and **581.57 / 581.24 tokens/s**, respectively. Their complete samples
were 3116.06, 3134.02, 3101.20 / 3095.08, 3137.01, 3109.77;
3720.43, 3701.20, 3667.92 / 3686.64, 3712.40, 3687.27; and
582.83, 579.52, 582.37 / 580.72, 580.58, 582.41.

Legacy vision samples, in the same order, were 1045.72, 980.90, 986.67 /
989.22, 986.42, 1051.22 ms; 585.64, 593.74, 591.66 / 591.90, 589.65,
596.75 ms; and 2077.31, 2083.78, 2080.18 / 2082.23, 2081.35, 2080.75 ms.
Largest mean decode decline was **0.49%**, prefill decline **0.10%**,
and vision latency increase **0.45%**. These measurements do not show a
material regression at the one-percent scale, but they do not prove
literally zero performance difference on a shared machine.
