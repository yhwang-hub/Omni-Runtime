#!/usr/bin/env python3
"""Convert the pi0.5-base safetensors checkpoint to a self-contained GGUF file.

The converter is intentionally offline: activate the pi0.5 environment before
running it.  It exports the four model blocks (SigLIP, multimodal projector,
PaliGemma VLM and Gemma action expert), flow-matching heads, and the local BPE
tokenizer so the C++ runtime does not depend on HuggingFace at inference time.
"""

from __future__ import annotations

import argparse
import json
import pathlib
import sys
from typing import Any

import gguf
import numpy as np
from safetensors import safe_open

CKPT_DIR = pathlib.Path("/root/workspace/pi0.5/pi05_base")
DEFAULT_OUTPUT_DIR = pathlib.Path("/root/workspace/pi0.5")
ARCH = "pi05"

VLM = {
    "width": 2048,
    "depth": 18,
    "mlp_dim": 16384,
    "num_heads": 8,
    "num_kv_heads": 1,
    "head_dim": 256,
}
EXPERT = {
    "width": 1024,
    "depth": 18,
    "mlp_dim": 4096,
    "num_heads": 8,
    "num_kv_heads": 1,
    "head_dim": 256,
}
VISION = {
    "width": 1152,
    "depth": 27,
    "mlp_dim": 4304,
    "num_heads": 16,
    "head_dim": 72,
    "patch_size": 14,
    "image_size": 224,
    "num_positions": 256,
    "projection_dim": 2048,
}
VOCAB_SIZE = 257152
RMS_EPS = 1.0e-6
LN_EPS = 1.0e-6
ROPE_FREQ_BASE = 10000.0
ACTION_DIM = 32
ACTION_HORIZON = 50
MAX_STATE_DIM = 32
NUM_INFERENCE_STEPS = 10
MIN_PERIOD = 0.004
MAX_PERIOD = 4.0
TOKENIZER_MAX_LENGTH = 200

PREFIX = "paligemma_with_expert."
VLM_PREFIX = PREFIX + "paligemma.model.language_model."
VISION_PREFIX = PREFIX + "paligemma.model.vision_tower.vision_model."
MM_PREFIX = PREFIX + "paligemma.model.multi_modal_projector."
EXPERT_PREFIX = PREFIX + "gemma_expert.model."
LM_HEAD = PREFIX + "paligemma.lm_head.weight"


def _map_tensor_name(key: str) -> str | None:
    if key.startswith(VLM_PREFIX):
        return "vlm." + key[len(VLM_PREFIX) :]
    if key.startswith(VISION_PREFIX):
        return "vision." + key[len(VISION_PREFIX) :]
    if key.startswith(MM_PREFIX):
        return "mm." + key[len(MM_PREFIX) :]
    if key.startswith(EXPERT_PREFIX):
        return "expert." + key[len(EXPERT_PREFIX) :]
    if key in (LM_HEAD, PREFIX + "gemma_expert.lm_head.weight"):
        return None
    return key


def _add_gemma_metadata(writer: gguf.GGUFWriter, prefix: str, config: dict[str, int]) -> None:
    writer.add_uint32(f"{prefix}.embedding_length", config["width"])
    writer.add_uint32(f"{prefix}.block_count", config["depth"])
    writer.add_uint32(f"{prefix}.feed_forward_length", config["mlp_dim"])
    writer.add_uint32(f"{prefix}.attention.head_count", config["num_heads"])
    writer.add_uint32(f"{prefix}.attention.head_count_kv", config["num_kv_heads"])
    writer.add_uint32(f"{prefix}.attention.key_length", config["head_dim"])
    writer.add_uint32(f"{prefix}.attention.value_length", config["head_dim"])
    writer.add_float32(f"{prefix}.attention.layer_norm_rms_epsilon", RMS_EPS)
    writer.add_float32(f"{prefix}.rope.freq_base", ROPE_FREQ_BASE)


def _add_metadata(writer: gguf.GGUFWriter) -> None:
    writer.add_string("general.name", "pi05_base")
    _add_gemma_metadata(writer, f"{ARCH}.vlm", VLM)
    writer.add_uint32(f"{ARCH}.vlm.vocab_size", VOCAB_SIZE)
    _add_gemma_metadata(writer, f"{ARCH}.expert", EXPERT)

    writer.add_uint32(f"{ARCH}.vision.embedding_length", VISION["width"])
    writer.add_uint32(f"{ARCH}.vision.block_count", VISION["depth"])
    writer.add_uint32(f"{ARCH}.vision.feed_forward_length", VISION["mlp_dim"])
    writer.add_uint32(f"{ARCH}.vision.attention.head_count", VISION["num_heads"])
    writer.add_uint32(f"{ARCH}.vision.attention.key_length", VISION["head_dim"])
    writer.add_uint32(f"{ARCH}.vision.patch_size", VISION["patch_size"])
    writer.add_uint32(f"{ARCH}.vision.image_size", VISION["image_size"])
    writer.add_uint32(f"{ARCH}.vision.num_positions", VISION["num_positions"])
    writer.add_uint32(f"{ARCH}.vision.projection_dim", VISION["projection_dim"])
    writer.add_float32(f"{ARCH}.vision.attention.layer_norm_epsilon", LN_EPS)

    writer.add_uint32(f"{ARCH}.action_dim", ACTION_DIM)
    writer.add_uint32(f"{ARCH}.action_horizon", ACTION_HORIZON)
    writer.add_uint32(f"{ARCH}.max_state_dim", MAX_STATE_DIM)
    writer.add_uint32(f"{ARCH}.num_inference_steps", NUM_INFERENCE_STEPS)
    writer.add_uint32(f"{ARCH}.tokenizer_max_length", TOKENIZER_MAX_LENGTH)
    writer.add_float32(f"{ARCH}.min_period", MIN_PERIOD)
    writer.add_float32(f"{ARCH}.max_period", MAX_PERIOD)


def _add_tokenizer(writer: gguf.GGUFWriter, tokenizer_dir: pathlib.Path) -> None:
    tokenizer_path = tokenizer_dir / "tokenizer.json"
    if not tokenizer_path.is_file():
        raise FileNotFoundError(f"tokenizer.json not found: {tokenizer_path}")
    tokenizer: dict[str, Any] = json.loads(tokenizer_path.read_text())
    model = tokenizer.get("model", {})
    if model.get("type") != "BPE":
        raise ValueError(f"pi0.5 tokenizer must be BPE, got {model.get('type')!r}")

    vocab_map: dict[str, int] = model.get("vocab", {})
    vocabulary = [""] * (max(vocab_map.values()) + 1)
    for token, token_id in vocab_map.items():
        vocabulary[token_id] = token
    writer.add_tokenizer_model("gpt2")
    writer.add_token_list(vocabulary)
    writer.add_token_merges(model.get("merges", []))
    writer.add_bos_token_id(2)
    writer.add_eos_token_id(1)
    writer.add_pad_token_id(0)
    writer.add_unk_token_id(3)
    writer.add_add_bos_token(True)
    writer.add_add_eos_token(False)


def _to_numpy(tensor: Any, dtype: str) -> np.ndarray:
    array = tensor.detach().cpu().numpy()
    if dtype == "f16":
        return np.ascontiguousarray(array.astype(np.float16))
    return np.ascontiguousarray(array.astype(np.float32))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--src", default=str(CKPT_DIR / "model.safetensors"))
    parser.add_argument("--tokenizer-dir", default=str(CKPT_DIR / "tokenizer"))
    parser.add_argument("--out-dir", default=str(DEFAULT_OUTPUT_DIR))
    parser.add_argument("--dtype", choices=("f32", "f16"), default="f32")
    args = parser.parse_args()

    source = pathlib.Path(args.src)
    output_dir = pathlib.Path(args.out_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    output = output_dir / f"pi05_base_{args.dtype}.gguf"
    if not source.is_file():
        print(f"[ERROR] checkpoint not found: {source}", file=sys.stderr)
        return 1

    print(f"source : {source}")
    print(f"output : {output}")
    print(f"dtype  : {args.dtype}")
    writer = gguf.GGUFWriter(str(output), ARCH)
    _add_metadata(writer)
    _add_tokenizer(writer, pathlib.Path(args.tokenizer_dir))

    exported = 0
    with safe_open(str(source), framework="pt", device="cpu") as reader:
        keys = list(reader.keys())
        for key in keys:
            name = _map_tensor_name(key)
            if name is None:
                continue
            writer.add_tensor(name, _to_numpy(reader.get_tensor(key), args.dtype))
            exported += 1
        writer.add_tensor(
            "vlm.token_embd.weight",
            _to_numpy(reader.get_tensor(LM_HEAD), args.dtype),
        )
        exported += 1

    writer.write_header_to_file()
    writer.write_kv_data_to_file()
    writer.write_tensors_to_file()
    writer.close()
    print(f"[OK] exported {exported} tensors to {output} ({output.stat().st_size / 2**30:.2f} GiB)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
