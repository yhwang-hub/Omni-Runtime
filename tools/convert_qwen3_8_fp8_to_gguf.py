#!/usr/bin/env python3
"""Export Qwen3.5-architecture FP8 checkpoints without requantizing their weights."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import gguf
import numpy as np
import torch
from safetensors import safe_open


MODEL_PATH = Path("/home/wangyh/workspace/MultimodalModels/QwenModels/Qwen3.8-27B-FP8")
ROOT = Path(__file__).resolve().parents[1]
FP8_TYPE = 43  # Omni-Runtime's documented GGUF E4M3FN extension.


def reorder_heads(tensor: torch.Tensor, axis: int, key_heads: int, head_size: int) -> torch.Tensor:
    shape = list(tensor.shape)
    value_heads = shape[axis] // head_size
    expanded = shape[:axis] + [key_heads, value_heads // key_heads, head_size] + shape[axis + 1:]
    return tensor.reshape(expanded).transpose(axis, axis + 1).reshape(shape).contiguous()


def transform_language(name: str, tensor: torch.Tensor, config: dict, is_scale: bool) -> torch.Tensor:
    """Keep the runtime's tiled V-head order, including both FP8 scale axes."""
    if "linear_attn." in name:
        key_heads = config["linear_num_key_heads"]
        head_size = config["linear_value_head_dim"]
        unit = 128 if is_scale else 1
        if ".in_proj_qkv." in name or ".conv1d." in name:
            first = 2 * key_heads * config["linear_key_head_dim"] // unit
            tensor = torch.cat((
                tensor[:first],
                reorder_heads(tensor[first:], 0, key_heads, head_size // unit),
            ))
        elif ".in_proj_z." in name:
            tensor = reorder_heads(tensor, 0, key_heads, head_size // unit)
        elif any(key in name for key in (".in_proj_a.", ".in_proj_b.", ".A_log", ".dt_bias")):
            tensor = reorder_heads(tensor, 0, key_heads, 1)
        elif ".out_proj." in name:
            tensor = reorder_heads(tensor, 1, key_heads, head_size // unit)
    if name.endswith("norm.weight") and not name.endswith("linear_attn.norm.weight"):
        tensor = tensor.float() + 1.0
    return tensor.squeeze() if ".conv1d." in name else tensor


def add_tokenizer(writer: gguf.GGUFWriter, source: Path, vocabulary_size: int) -> None:
    tokenizer = json.loads((source / "tokenizer.json").read_text())
    vocabulary = [f"[PAD{index}]" for index in range(vocabulary_size)]
    types = [int(gguf.TokenType.UNUSED)] * vocabulary_size
    for token, index in tokenizer["model"]["vocab"].items():
        vocabulary[index] = token
        types[index] = int(gguf.TokenType.NORMAL)
    for token in tokenizer["added_tokens"]:
        vocabulary[token["id"]] = token["content"]
        types[token["id"]] = int(gguf.TokenType.CONTROL if token["special"] else gguf.TokenType.USER_DEFINED)
    merges = tokenizer["model"]["merges"]
    writer.add_tokenizer_model("gpt2")
    writer.add_tokenizer_pre("qwen35")
    writer.add_token_list(vocabulary)
    writer.add_token_types(types)
    writer.add_token_merges([" ".join(pair) if isinstance(pair, list) else pair for pair in merges])
    special = {token["content"]: token["id"] for token in tokenizer["added_tokens"]}
    writer.add_bos_token_id(special["<|endoftext|>"])
    writer.add_eos_token_id(special["<|im_end|>"])
    writer.add_add_bos_token(False)
    writer.add_chat_template((source / "chat_template.jinja").read_text())


def add_metadata(language: gguf.GGUFWriter, vision: gguf.GGUFWriter, config: dict, source: Path) -> None:
    text = config["text_config"]
    visual = config["vision_config"]
    language.add_name(source.name)
    language.add_string("omni.quantization", "fp8_e4m3fn_block_128")
    integers = {
        "context_length": text["max_position_embeddings"],
        "block_count": text["num_hidden_layers"],
        "embedding_length": text["hidden_size"],
        "feed_forward_length": text["intermediate_size"],
        "attention.head_count": text["num_attention_heads"],
        "attention.head_count_kv": text["num_key_value_heads"],
        "attention.key_length": text["head_dim"],
        "rope.dimension_count": int(text["head_dim"] * text["partial_rotary_factor"]),
        "ssm.conv_kernel": text["linear_conv_kernel_dim"],
        "ssm.state_size": text["linear_key_head_dim"],
        "ssm.group_count": text["linear_num_key_heads"],
        "ssm.time_step_rank": text["linear_num_value_heads"],
        "full_attention_interval": text["full_attention_interval"],
    }
    for key, value in integers.items():
        language.add_uint32("qwen35." + key, value)
    language.add_float32("qwen35.rope.freq_base", text["rope_parameters"]["rope_theta"])
    language.add_float32("qwen35.attention.layer_norm_rms_epsilon", text["rms_norm_eps"])
    language.add_array("qwen35.rope.dimension_sections", text["rope_parameters"]["mrope_section"] + [0])
    language.add_array("qwen35.attention.recurrent_layers", [
        kind == "linear_attention" for kind in text["layer_types"]
    ])
    add_tokenizer(language, source, text["vocab_size"])
    vision.add_name(source.name + " vision")
    vision.add_string("clip.projector_type", "qwen3vl_merger")
    for key, value in {
        "image_size": math.isqrt(visual["num_position_embeddings"]) * visual["patch_size"],
        "patch_size": visual["patch_size"],
        "embedding_length": visual["hidden_size"],
        "feed_forward_length": visual["intermediate_size"],
        "block_count": visual["depth"],
        "attention.head_count": visual["num_heads"],
        "projection_dim": visual["out_hidden_size"],
        "spatial_merge_size": visual["spatial_merge_size"],
    }.items():
        vision.add_uint32("clip.vision." + key, value)
    vision.add_float32("clip.vision.attention.layer_norm_epsilon", text["rms_norm_eps"])


def vision_name(name: str, mapping: gguf.TensorNameMap) -> str:
    name = name.replace("model.visual.", "visual.", 1)
    if name.startswith("visual.merger."):
        suffix = name.removeprefix("visual.merger.")
        for old, new in (("norm.", "v.post_ln."), ("linear_fc1.", "mm.0."), ("linear_fc2.", "mm.2.")):
            if suffix.startswith(old):
                return new + suffix.removeprefix(old)
    result = mapping.get_name(name, try_suffixes=(".weight", ".bias"))
    if result is None:
        raise ValueError(f"Unmapped vision tensor: {name}")
    return result


def convert(source: Path, output: Path) -> None:
    config = json.loads((source / "config.json").read_text())
    quantization = config["quantization_config"]
    if (config["model_type"] != "qwen3_5" or quantization["quant_method"] != "fp8"
            or quantization["fmt"] != "e4m3" or quantization["weight_block_size"] != [128, 128]
            or quantization["activation_scheme"] != "dynamic"):
        raise ValueError("Expected Qwen3.5 E4M3FN dynamic FP8 with 128x128 weight blocks")
    output.mkdir(parents=True, exist_ok=True)
    paths = [output / "Qwen3.8-27B-FP8.gguf", output / "mmproj-Qwen3.8-27B-F32.gguf"]
    if any(path.exists() for path in paths):
        raise FileExistsError("Output GGUF already exists; choose an empty output directory")
    language = gguf.GGUFWriter(paths[0], "qwen35", use_temp_file=True)
    vision = gguf.GGUFWriter(paths[1], "clip", use_temp_file=True)
    add_metadata(language, vision, config, source)
    text = config["text_config"]
    language_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.QWEN35, text["num_hidden_layers"])
    vision_map = gguf.get_tensor_name_map(gguf.MODEL_ARCH.MMPROJ, config["vision_config"]["depth"])
    weights = json.loads((source / "model.safetensors.index.json").read_text())["weight_map"]
    names_written: set[str] = set()
    for shard in sorted(set(weights.values())):
        with safe_open(source / shard, framework="pt", device="cpu") as tensors:
            for name in sorted(tensors.keys()):
                if name.startswith("mtp."):
                    continue
                tensor = tensors.get_tensor(name)
                is_visual = name.startswith("model.visual.")
                is_scale = name.endswith(".weight_scale_inv")
                if is_visual and is_scale:
                    raise ValueError("Quantized vision tensors are not supported by this checkpoint exporter")
                if not is_visual:
                    base = name.removesuffix("_scale_inv") if is_scale else name
                    base = base.replace("model.language_model.", "model.", 1)
                    if base.endswith(".dt_bias"):
                        base = base.removesuffix(".dt_bias") + ".dt_proj.bias"
                    mapped = language_map.get_name(base, try_suffixes=(".weight", ".bias"))
                    if mapped is None:
                        raise ValueError(f"Unmapped language tensor: {name}")
                    if name.endswith(".A_log"):
                        mapped += "_log"
                    if is_scale:
                        mapped = mapped.removesuffix(".weight") + ".weight_scale_inv"
                    if tensor.dtype == torch.float8_e4m3fn:
                        scale_name = name.removesuffix(".weight") + ".weight_scale_inv"
                        if scale_name not in weights:
                            raise ValueError(f"Missing FP8 scale: {scale_name}")
                        expected = [(dimension + 127) // 128 for dimension in tensor.shape]
                        if list(tensors.get_slice(scale_name).get_shape()) != expected:
                            raise ValueError(f"Invalid FP8 scale shape: {scale_name}")
                    tensor = transform_language(name, tensor, text, is_scale)
                    writer = language
                else:
                    mapped = vision_name(name, vision_map)
                    writer = vision
                if name == "model.visual.patch_embed.proj.weight":
                    if tensor.shape[2] != 2:
                        raise ValueError("Expected two temporal patches")
                    vision.add_tensor("v.patch_embd.weight", tensor[:, :, 0].float().contiguous().numpy())
                    vision.add_tensor("v.patch_embd.weight.1", tensor[:, :, 1].float().contiguous().numpy())
                    continue
                if mapped in names_written:
                    raise ValueError(f"Duplicate GGUF tensor: {mapped}")
                names_written.add(mapped)
                if tensor.dtype == torch.float8_e4m3fn:
                    raw = tensor.contiguous().view(torch.uint8).numpy()
                    # gguf-py intentionally does not assign this extension a standard ggml type.
                    writer.add_tensor(mapped, raw.view(np.int8), raw_dtype=FP8_TYPE)
                else:
                    array = tensor.float().contiguous().numpy()
                    if not np.isfinite(array).all() or (is_scale and not (array > 0).all()):
                        raise ValueError(f"Invalid tensor values: {name}")
                    writer.add_tensor(mapped, array)
        print(f"Exported {shard}", flush=True)
    for writer in (language, vision):
        writer.write_header_to_file()
        writer.write_kv_data_to_file()
        writer.write_tensors_to_file(progress=True)
        writer.close()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", type=Path, default=MODEL_PATH)
    parser.add_argument("--output-dir", type=Path, default=ROOT / "model/qwen3_8_fp8")
    args = parser.parse_args()
    torch.set_num_threads(4)
    convert(args.model, args.output_dir)


if __name__ == "__main__":
    main()
