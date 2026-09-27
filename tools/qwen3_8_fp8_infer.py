#!/usr/bin/env python3
"""Independent W8A8 reference: E4M3FN, 128x128 weights, per-token 128-value scales.

Activations use checkpoint BF16; block accumulation and recurrent state use FP32.
The PyTorch reference never loads GGUF or calls Omni-Runtime kernels.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from types import MethodType

import torch
import transformers
from PIL import Image
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AutoConfig, AutoProcessor, Qwen3_5ForConditionalGeneration
from transformers.integrations.finegrained_fp8 import FP8Linear
from transformers.masking_utils import AttentionMaskInterface, sdpa_mask


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = Path("/home/wangyh/workspace/MultimodalModels/QwenModels/Qwen3.8-27B-FP8")


def fp32_attention(module, query, key, value, attention_mask, **kwargs):
    """Use one FP32 SDPA backend for vision, prefill and cached grouped decode."""
    if attention_mask is not None and attention_mask.dtype != torch.bool:
        attention_mask = attention_mask.float()
    is_causal = (
        query.shape[-2] > 1 and attention_mask is None
        and kwargs.get("is_causal", getattr(module, "is_causal", True))
    )
    if is_causal and key.shape[-2] != query.shape[-2]:
        raise ValueError("Cached prefill requires an explicit causal mask covering historical KV")
    groups = query.shape[1] // key.shape[1]
    if groups > 1:
        key = key.repeat_interleave(groups, dim=1)
        value = value.repeat_interleave(groups, dim=1)
    with sdpa_kernel(SDPBackend.EFFICIENT_ATTENTION):
        output = torch.nn.functional.scaled_dot_product_attention(
            query.float(), key.float(), value.float(), attn_mask=attention_mask,
            dropout_p=kwargs.get("dropout", 0.0), is_causal=is_causal,
            scale=kwargs.get("scaling"),
        )
    return output.transpose(1, 2).to(query.dtype), None


def register_fp32_attention() -> None:
    AttentionInterface.register("fp32_sdpa", fp32_attention)
    AttentionMaskInterface.register("fp32_sdpa", sdpa_mask)


def fp8_linear(module: FP8Linear, inputs: torch.Tensor) -> torch.Tensor:
    """Explicit block accumulation, independent of native CUDA or hub kernels."""
    shape = inputs.shape
    flattened = inputs.reshape(-1, shape[-1]).float()
    if not torch.isfinite(flattened).all():
        raise ValueError("FP8 projection inputs contain nonfinite values")
    rows, inner = flattened.shape
    if inner % 128 or module.weight.shape[0] % 128:
        raise ValueError("This checkpoint requires complete 128x128 FP8 blocks")
    blocks = flattened.reshape(rows, inner // 128, 128)
    scales = blocks.abs().amax(dim=-1).clamp_min(1e-12) / 448.0
    codes = (blocks / scales.unsqueeze(-1)).clamp(-448, 448).to(torch.float8_e4m3fn)
    output = torch.zeros(rows, module.weight.shape[0], device=inputs.device, dtype=torch.float32)
    # Every E4M3FN value is an integer / 512. A 128-value dot needs at most
    # 43 integer bits, so FP64 evaluates it exactly before the single FP32 rounding.
    for block in range(inner // 128):
        start = block * 128
        weight = module.weight[:, start:start + 128].double()
        partial = (codes[:, block].double() @ weight.T).float()
        weight_scale = module.weight_scale_inv[:, block].float().repeat_interleave(128)
        output.add_(partial * scales[:, block, None] * weight_scale[None, :])
    if module.bias is not None:
        output.add_(module.bias)
    return output.reshape(*shape[:-1], module.weight.shape[0]).to(inputs.dtype)


def prepare_inputs(processor, image_path: Path | None, prompt: str) -> dict:
    # This explicit template preserves the established native Qwen3.5 UX;
    # it intentionally does not insert a system message or a thinking prefix.
    text = "<|im_start|>user\n"
    images = None
    if image_path is not None:
        with Image.open(image_path) as source:
            image = source.convert("RGB")
            width = (image.width + 31) // 32 * 32
            height = (image.height + 31) // 32 * 32
            images = [image.resize((width, height), Image.Resampling.BICUBIC)]
        text += "<|vision_start|><|image_pad|><|vision_end|>"
    text += prompt + "<|im_end|>\n<|im_start|>assistant\n"
    return processor(text=[text], images=images, do_resize=False, return_tensors="pt")


def prefill_ranges(total: int, image_offset: int, image_tokens: int) -> list[tuple[int, int]]:
    if total <= 0 or image_tokens < 0:
        raise ValueError("Invalid prompt or image token count")
    boundaries = [0, total]
    if image_tokens:
        if image_offset < 0 or image_offset + image_tokens > total:
            raise ValueError("Invalid image token range")
        boundaries.extend((image_offset, image_offset + image_tokens))
    boundaries = sorted(set(boundaries))
    return [
        (first, min(first + 512, end))
        for start, end in zip(boundaries, boundaries[1:])
        for first in range(start, end, 512)
    ]


def greedy_generate(model, inputs: dict, max_new_tokens: int, end_ids: set[int]) -> list[int]:
    """Keep PyTorch's model/cache, with the native deployment's prefill boundaries."""
    ids = inputs["input_ids"]
    embeds = model.get_input_embeddings()(ids)
    image_positions = torch.where(ids[0] == model.config.image_token_id)[0]
    if "pixel_values" in inputs:
        features = model.get_image_features(
            inputs["pixel_values"], inputs["image_grid_thw"], return_dict=True
        ).pooler_output
        features = torch.cat(features).to(embeds)
        mask, _ = model.model.get_placeholder_mask(ids, inputs_embeds=embeds, image_features=features)
        embeds = embeds.masked_scatter(mask, features)
    positions = model.model.compute_3d_position_ids(
        input_ids=ids, image_grid_thw=inputs.get("image_grid_thw"), inputs_embeds=embeds,
        attention_mask=inputs["attention_mask"], mm_token_type_ids=inputs.get("mm_token_type_ids"),
    )
    if positions is None:
        positions = torch.arange(ids.shape[1], device=ids.device).view(1, 1, -1).expand(3, 1, -1)
    ranges = prefill_ranges(
        ids.shape[1], image_positions[0].item() if image_positions.numel() else -1,
        image_positions.numel(),
    )
    cache = None
    for first, end in ranges:
        output = model.model.language_model(
            inputs_embeds=embeds[:, first:end], position_ids=positions[:, :, first:end],
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
    next_position = positions.max().item() + 1
    generated = []
    for step in range(max_new_tokens):
        token = model.lm_head(output.last_hidden_state[:, -1:]).argmax(-1).item()
        generated.append(token)
        if token in end_ids or step + 1 == max_new_tokens:
            break
        output = model.model.language_model(
            input_ids=torch.tensor([[token]], device=ids.device),
            position_ids=torch.full((3, 1, 1), next_position + step, device=ids.device),
            past_key_values=cache, use_cache=True, return_dict=True,
        )
        cache = output.past_key_values
    return generated


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--image", type=Path, default=ROOT / "test/inference/demo.jpeg")
    parser.add_argument("--text-only", action="store_true")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--reference-output", type=Path, default=ROOT / "test/inference/qwen3_8_27b_fp8")
    args = parser.parse_args()
    if not args.prompt or args.max_new_tokens <= 0:
        parser.error("A nonempty prompt and positive --max-new-tokens are required")
    if not torch.cuda.is_available():
        raise RuntimeError("The FP8 reference requires a CUDA device")
    torch.manual_seed(0)
    torch.set_num_threads(4)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.allow_tf32 = False
    config = AutoConfig.from_pretrained(args.model, local_files_only=True)
    weights = json.loads((args.model / "model.safetensors.index.json").read_text())["weight_map"]
    # Some releases match exclusion prefixes ("mlp.gate") against gate_proj.
    # Derive exact nonquantized modules from the checkpoint, not those prefixes.
    config.quantization_config["modules_to_not_convert"] = [
        name.removesuffix(".weight") for name in weights
        if name.endswith(".weight") and name.removesuffix(".weight") + ".weight_scale_inv" not in weights
    ]
    register_fp32_attention()
    model = Qwen3_5ForConditionalGeneration.from_pretrained(
        args.model, config=config, dtype=torch.bfloat16, device_map="cuda", local_files_only=True,
        attn_implementation="fp32_sdpa",
    ).eval()
    count = 0
    for module in model.modules():
        if isinstance(module, FP8Linear):
            if not torch.isfinite(module.weight.float()).all():
                raise RuntimeError("The checkpoint contains nonfinite FP8 weights")
            module.forward = MethodType(fp8_linear, module)
            count += 1
    expected_count = sum(
        name.endswith(".weight_scale_inv") and not name.startswith("mtp.") for name in weights
    )
    if count == 0 or count != expected_count:
        raise RuntimeError(f"Loaded {count} FP8 projections; checkpoint requires {expected_count}")
    print(f"FP8 projections: {count}; torch={torch.__version__}; transformers={transformers.__version__}",
          flush=True)
    processor = AutoProcessor.from_pretrained(args.model, local_files_only=True)
    inputs = prepare_inputs(processor, None if args.text_only else args.image, args.prompt).to("cuda")
    torch.cuda.synchronize()
    start = time.perf_counter()
    with torch.inference_mode():
        token_ids = greedy_generate(
            model, inputs, args.max_new_tokens,
            {model.config.text_config.eos_token_id, processor.tokenizer.convert_tokens_to_ids("<|im_end|>")},
        )
    torch.cuda.synchronize()
    elapsed_ms = (time.perf_counter() - start) * 1000
    text = processor.decode(token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)
    print(text)
    print(f"Generation: {elapsed_ms:.2f} ms; generated tokens: {len(token_ids)}")
    reference = {
        "model": str(args.model), "quantization": "fp8_e4m3fn_block_128", "compute_dtype": "bfloat16",
        "prefill_batch_tokens": 512,
        "fp8_block_dot": "exact_then_fp32",
        "torch": torch.__version__, "transformers": transformers.__version__, "seed": 0,
        "do_sample": False, "attention": "fp32_sdpa", "prompt": args.prompt,
        "attention_mask": "sdpa_causal_with_cache",
        "attention_backend": "efficient_attention_fp32",
        "max_new_tokens": args.max_new_tokens, "image": None if args.text_only else str(args.image),
        "prompt_token_ids": inputs["input_ids"][0].tolist(),
        "image_grid_thw": inputs["image_grid_thw"].tolist() if "image_grid_thw" in inputs else [],
        "generated_token_ids": token_ids, "text": text, "generation_time_ms": elapsed_ms,
    }
    output = args.reference_output
    output.parent.mkdir(parents=True, exist_ok=True)
    output.with_suffix(".json").write_text(json.dumps(reference, ensure_ascii=False, indent=2) + "\n")
    output.with_suffix(".txt").write_text(text, encoding="utf-8")
    output.with_suffix(".tokens").write_text(" ".join(map(str, token_ids)) + "\n", encoding="ascii")
    output.with_suffix(".prompt.tokens").write_text(
        " ".join(map(str, reference["prompt_token_ids"])) + "\n", encoding="ascii"
    )


if __name__ == "__main__":
    main()
