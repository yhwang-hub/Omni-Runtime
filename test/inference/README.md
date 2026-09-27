# Inference validation data

This directory is the only repository location used by inference tests for
images and evaluation manifests.

- `demo.jpeg`: required Qwen3-VL image input.
- `bus.jpg`: shared YOLO and Qwen3-VL smoke image.
- `qwen3_vl_accuracy_dataset.jsonl`: local caption/VQA evaluation manifest.
- `qwen3_8_27b_llama.txt`: Qwen3.8-27B greedy reference captured on NVIDIA
  GB10 with the command below. The final file newline is a fixture delimiter,
  not a generated token; the test removes exactly that delimiter. The three
  console newlines printed by `generate_response()` and `main()` are excluded.

```bash
cd /home/wangyh/workspace/llama.cpp
./build-omni/bin/llama-mtmd-cli \
  --model /home/wangyh/workspace/Omni-Runtime/model/qwen3_8_nvfp4/Qwen3.8-27B-NVFP4-Q8_0.gguf \
  --mmproj /home/wangyh/workspace/Omni-Runtime/model/qwen3_8_nvfp4/mmproj-Qwen3.8-27B-F16.gguf \
  --image /home/wangyh/workspace/Omni-Runtime/test/inference/demo.jpeg \
  --prompt "Describe this image." --gpu-layers all --ctx-size 8192 \
  --n-predict 512 --temp 0 --perf --log-verbosity 4
```

The acceptance test compares the entire generated text, requires EOS, and
checks the measured GB10 baseline: vision 2452 ms, prefill 500.44 tokens/s,
decode 12.22 tokens/s. These are hardware-specific acceptance thresholds,
not portable unit-test performance limits. Native CUDA/OpenCL inference passes
full text equality for this input, including the thinking text, whitespace and
punctuation. With asynchronous quantized prefill staging, two complete native
CUDA runs measured 573.81 and 571.71 prefill tokens/s, exceeding 500.44.
Decode measured 10.99 and 10.96 tokens/s and remains below 12.22; OpenCL also
remains below its acceptance thresholds. The complete acceptance test therefore
**still fails**, rather than relaxing the criteria. A fresh reference-command
run measured 486.68 prefill and 11.35 decode tokens/s and reproduced the unchanged
fixture byte for byte after removing the CLI-only blank lines.

An additional single-token CUDA capture-dependency optimization measured
11.11/11.08 decode tokens/s against a same-version serial control of 10.97;
the clean-build run measured 567.69 prefill and 11.08 decode tokens/s.
A refreshed reference-command run measured 489.94/11.35 tokens/s and again
matched the unchanged fixture. The decode threshold is **still not met**.
`OMNI_PARALLEL_LINEAR=0` disables the independent-projection scheduling without
changing the model, reference output or arithmetic kernels. Speculative decoding
is not used.

With column-group Q8 quantization and physical FP16 KV storage, the final
clean-build CUDA run measured 579.38 prefill and 11.51 decode tokens/s; the
complete test-suite run measured 583.80/11.54. The same-round reference command
measured 487.62/11.33 and again matched the unchanged fixture byte for byte.
Although both native CUDA rates exceed this fresh reference measurement,
decode is **still below the required 12.22**, so the strict acceptance test
still fails. CUDA/OpenCL full text, 399 tokens, EOS and state-recovery checks
pass; OpenCL performance remains below all three acceptance thresholds.
The final clean build succeeded, and 15/16 test targets passed, including all
193 enabled kernel cases and all five Qwen3-VL inference cases. The only
default-backend failure is the Qwen3.8 decode throughput assertion.

Each JSONL record contains an image, prompt, task type, and either
`required_keywords` for caption recall or `answers` for normalized VQA exact
match. Large public datasets are intentionally not committed. Place their
converted JSONL manifests and images below `test/inference/datasets/<name>` so
the same evaluator can consume them without changing runtime code.
