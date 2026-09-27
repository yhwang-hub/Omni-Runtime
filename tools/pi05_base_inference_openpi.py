#!/usr/bin/env python
"""End-to-end inference for pi0.5 base (pi05_base) using **openpi's own model code**.

与 ``pi05_base_inference.py``（基于 LeRobot ``PI05Policy``）功能一致，但这里直接使用
openpi 官方仓库的 PyTorch 模型 ``openpi.models_pytorch.pi0_pytorch.PI0Pytorch`` 进行推理。

背景与做法
----------
* 权重目录 ``/root/workspace/pi0.5/pi05_base/model.safetensors`` 本身就是由 openpi 的
  ``examples/convert_jax_model_to_pytorch.py`` 导出的 **openpi-PyTorch 命名** 权重
  （812 个 key 与 ``PI0Pytorch`` 完全对应），因此可以被 openpi 模型直接加载。
* openpi 的 PyTorch 实现依赖它定制的 transformers（adaRMS 补丁，官方要求 4.53.2），
  与本 venv 的 transformers 5.3.0 不兼容；且 ``openpi.models.gemma`` 会拉起 jax。
  ``_openpi_runtime/openpi_bootstrap.py`` 以**隔离、非破坏**的方式解决：用 uv 缓存里的
  transformers 4.57.6 + huggingface_hub 0.36.2 叠加 openpi 的 transformers_replace，
  并用无 jax 的 stub 顶替 gemma / image_tools（详见该文件注释）。

运行：
    source /root/workspace/tether/.venv-pi05-cu118/bin/activate
    python /root/workspace/zelos/Omni-Runtime/tools/pi05_base_inference_openpi.py
"""

from __future__ import annotations

import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import dataclasses
import argparse
import pathlib
import sys
import types

import numpy as np
from PIL import Image

# --- 1) 启动隔离运行时（必须在 import torch / transformers / openpi 之前）---------
_RUNTIME_CANDIDATES = (
    pathlib.Path(__file__).resolve().parent / "_openpi_runtime",
    pathlib.Path("/root/workspace/openpi/_openpi_runtime"),
)
for _RUNTIME in _RUNTIME_CANDIDATES:
    if (_RUNTIME / "openpi_bootstrap.py").is_file():
        sys.path.insert(0, str(_RUNTIME))
        break
else:
    raise FileNotFoundError("openpi_bootstrap.py not found in _openpi_runtime")
import openpi_bootstrap  # noqa: E402

openpi_bootstrap.bootstrap()

import torch  # noqa: E402
from safetensors.torch import load_file  # noqa: E402
from transformers import AutoTokenizer  # noqa: E402

from openpi.models_pytorch.pi0_pytorch import PI0Pytorch  # noqa: E402

# ----------------------------------------------------------------------------
CKPT_DIR = "/root/workspace/pi0.5/pi05_base"
CKPT = os.path.join(CKPT_DIR, "model.safetensors")
TOKENIZER_DIR = os.path.join(CKPT_DIR, "tokenizer")
INPUT_DIR = pathlib.Path("/root/workspace/zelos/Omni-Runtime/test/inference/pi05_example_inputs")

MAX_STATE_DIM = 32
ACTION_DIM = 32
ACTION_HORIZON = 50
NUM_INFERENCE_STEPS = 10
LIBERO_ACTION_DIM = 7
IMAGE_KEYS = ("base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb")


@dataclasses.dataclass
class PI0PytorchConfig:
    """PI0Pytorch.__init__ 只读取以下字段（其余 openpi 训练配置在推理时用不到）。"""
    pi05: bool = True                          # π0.5：state 走离散 token + adaRMS
    paligemma_variant: str = "gemma_2b"        # VLM 主干
    action_expert_variant: str = "gemma_300m"  # 动作专家
    dtype: str = "float32"
    action_dim: int = ACTION_DIM
    action_horizon: int = ACTION_HORIZON
    pytorch_compile_mode = None


def load_image_chw_minus1_1(path: pathlib.Path) -> torch.Tensor:
    """PNG -> float32 CHW，范围 [-1,1]（openpi 推理时 preprocess 不再做归一化，
    需调用方给到 SigLIP 期望的 [-1,1]）。"""
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0  # [0,1] HWC
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # CHW
    return chw * 2.0 - 1.0  # [-1,1]


def build_prompt_and_state():
    raw_state = np.load(INPUT_DIR / "state.npy").astype(np.float32)  # (8,)
    state = np.zeros(MAX_STATE_DIM, dtype=np.float32)
    state[: raw_state.shape[0]] = raw_state  # 补零到 32
    task = (INPUT_DIR / "prompt.txt").read_text().strip()

    # π0.5 提示词构造，与 openpi tokenizer.py / lerobot 处理步骤完全一致：
    #   state 离散化到 256 个 bin，拼进语言提示
    cleaned = task.strip().replace("_", " ").replace("\n", " ")
    discretized = np.digitize(state, bins=np.linspace(-1, 1, 256 + 1)[:-1]) - 1
    state_str = " ".join(map(str, discretized))
    full_prompt = f"Task: {cleaned}, State: {state_str};\nAction: "
    return full_prompt, state, raw_state, task


def format_action_vector(values: np.ndarray) -> str:
    """Format one action vector like the C++ inference test."""
    flat = np.asarray(values, dtype=np.float32).reshape(-1)
    formatted = [f"{0.0 if abs(float(value)) < 5e-5 else float(value):.4f}" for value in flat]
    return "[" + " ".join(formatted) + "]"


def build_observation(device):
    imgs = {k: load_image_chw_minus1_1(INPUT_DIR / f"{k}.png") for k in IMAGE_KEYS}
    images = {k: v[None].to(device) for k, v in imgs.items()}  # (1,3,224,224)
    # LIBERO：base + 左腕为真实相机(mask=1)，右腕为占位(mask=0)
    image_masks = {
        "base_0_rgb": torch.tensor([True], device=device),
        "left_wrist_0_rgb": torch.tensor([True], device=device),
        "right_wrist_0_rgb": torch.tensor([False], device=device),
    }

    full_prompt, state, raw_state, task = build_prompt_and_state()
    tok = AutoTokenizer.from_pretrained(TOKENIZER_DIR)
    enc = tok(full_prompt, max_length=200, truncation=True, padding="max_length",
              padding_side="right", return_tensors="pt")
    tokenized_prompt = enc["input_ids"].to(device)
    tokenized_prompt_mask = enc["attention_mask"].to(dtype=torch.bool, device=device)

    state_t = torch.from_numpy(state)[None].to(device)  # (1,32)

    obs = types.SimpleNamespace(
        images=images,
        image_masks=image_masks,
        state=state_t,
        tokenized_prompt=tokenized_prompt,
        tokenized_prompt_mask=tokenized_prompt_mask,
        token_ar_mask=None,
        token_loss_mask=None,
    )
    return obs, full_prompt, raw_state, task, tokenized_prompt_mask.sum().item()


def load_model(device):
    model = PI0Pytorch(PI0PytorchConfig())
    sd = load_file(CKPT)
    missing, unexpected = model.load_state_dict(sd, strict=False)
    # π0.5/gemma 绑定权重：语言输入嵌入 embed_tokens 与 lm_head 共享，
    # checkpoint 只存了 paligemma.lm_head.weight，这里绑定回 embed_tokens。
    tie_key = "paligemma_with_expert.paligemma.lm_head.weight"
    emb = model.paligemma_with_expert.paligemma.model.language_model.embed_tokens
    with torch.no_grad():
        emb.weight.copy_(sd[tie_key])
    missing = [m for m in missing if "embed_tokens" not in m]
    assert not missing and not unexpected, f"missing={missing}, unexpected={unexpected}"
    return model.to(device).eval()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=pathlib.Path,
        default=INPUT_DIR / "predicted_actions_openpi.npy",
        help="path for the float32 action block (.npy)",
    )
    args = parser.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("=" * 78)
    print("π0.5 (pi05_base) 推理示例  |  openpi 原生 PI0Pytorch")
    print("=" * 78)
    print(f"device        : {device}")
    print(f"transformers  : {__import__('transformers').__version__} (隔离运行时)")
    print(f"checkpoint    : {CKPT}")

    print("\n[1/4] 用 openpi PI0Pytorch 加载权重 ...")
    model = load_model(device)
    n_params = sum(p.numel() for p in model.parameters())
    print(f"      参数量        : {n_params/1e9:.2f} B")
    print(f"      pi05          : {model.pi05}  (state 走离散 token + adaRMS)")
    print(f"      action_horizon: {model.config.action_horizon}")
    print(f"      去噪步数      : {NUM_INFERENCE_STEPS}  (flow-matching)")

    print("\n[2/4] 构造真实输入（LIBERO 观测）...")
    obs, full_prompt, raw_state, task, n_tok = build_observation(device)
    print(f'      语言指令 task : "{task}"')
    print(f"      原始状态(8维) : {np.array2string(raw_state, precision=3)}")
    print(f"      π0.5 提示词   : {full_prompt[:96]}... (共 {n_tok} 个有效 token)")
    for k in IMAGE_KEYS:
        t = obs.images[k]
        print(f"      {k:18s}: shape={tuple(t.shape)} range=[{t.min():.2f},{t.max():.2f}] "
              f"mask={bool(obs.image_masks[k].item())}")

    print("\n[3/4] 采样固定噪声（保证结果可复现）并运行 flow-matching 推理 ...")
    noise_path = INPUT_DIR / "_verify_noise.npy"
    if noise_path.is_file():
        noise = torch.from_numpy(np.load(noise_path)).to(device=device, dtype=torch.float32)
    else:
        gen = torch.Generator(device=device).manual_seed(0)
        noise = torch.randn((1, ACTION_HORIZON, ACTION_DIM), generator=gen,
                            dtype=torch.float32, device=device)

    with torch.no_grad():
        actions = model.sample_actions(device, obs, noise=noise, num_steps=NUM_INFERENCE_STEPS)

    actions = actions.detach().float().cpu()
    first = actions[0, 0]

    print("\n" + "-" * 78)
    print("推理结果")
    print("-" * 78)
    print(f"完整动作块 shape          : {tuple(actions.shape)}  = (batch, {ACTION_HORIZON}, {ACTION_DIM})")
    print(f"\n第 1 步动作 (模型输出全 {ACTION_DIM} 维; LIBERO 取前 {LIBERO_ACTION_DIM} 维):")
    print("  " + format_action_vector(first.cpu().numpy()))
    print(f"\nLIBERO 有效动作 (前 {LIBERO_ACTION_DIM} 维: Δx Δy Δz Δroll Δpitch Δyaw gripper):")
    print("  " + format_action_vector(first[:LIBERO_ACTION_DIM].cpu().numpy()))

    out = args.output
    out.parent.mkdir(parents=True, exist_ok=True)
    np.save(out, actions.numpy())
    print(f"\n完整动作块已保存到: {out}")

    assert actions.shape == (1, ACTION_HORIZON, ACTION_DIM)
    assert torch.isfinite(actions).all(), "动作出现 NaN/Inf"
    print("\n[OK] openpi 原生推理完成，输出为有限实数，形状符合 π0.5 规范。")


if __name__ == "__main__":
    main()
