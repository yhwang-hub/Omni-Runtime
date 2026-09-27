#!/usr/bin/env python
"""End-to-end inference example for the pi0.5 base model (pi05_base).

本脚本演示如何用 LeRobot 的 ``PI05Policy`` 加载 π0.5 base 权重，并对一帧
真实结构的 LIBERO 观测（第三人称相机 + 手腕相机 + 本体状态 + 语言指令）
运行推理，得到机器人动作。

Checkpoint 格式说明
-------------------
``/root/workspace/pi0.5/pi05_base`` 是 **LeRobot 版 π0.5** 权重目录：
    config.json                 -> PI05Config (type=pi05)
    model.safetensors           -> PaliGemma(2B) + action-expert(300M) 权重
    policy_preprocessor.json    -> 预处理流水线定义
    policy_postprocessor.json   -> 后处理流水线定义
    tokenizer/                  -> PaliGemma 分词器（离线可用）

因此使用 LeRobot 的 ``PI05Policy`` / ``make_pre_post_processors`` 加载，
与 openpi 参考实现在数学上等价（LeRobot 的 pi05 实现逐行对齐 openpi）。

运行方式
--------
    source /root/workspace/tether/.venv-pi05-cu118/bin/activate
    python /root/workspace/zelos/Omni-Runtime/tools/pi05_base_inference.py
"""

from __future__ import annotations

# --- 离线设置：本机无外网，且 checkpoint 自带分词器，必须走离线路径 --------
import os

os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import pathlib

import numpy as np
import torch
from PIL import Image

from transformers import AutoTokenizer

from lerobot.policies.pi05 import PI05Policy
from lerobot.policies.pi05.processor_pi05 import Pi05PrepareStateTokenizerProcessorStep
from lerobot.processor import (
    AddBatchDimensionProcessorStep,
    DeviceProcessorStep,
    NormalizerProcessorStep,
    PolicyProcessorPipeline,
    RenameObservationsProcessorStep,
    TokenizerProcessorStep,
    UnnormalizerProcessorStep,
)
from lerobot.processor.converters import (
    batch_to_transition,
    policy_action_to_transition,
    transition_to_batch,
    transition_to_policy_action,
)
from lerobot.utils.constants import (
    POLICY_POSTPROCESSOR_DEFAULT_NAME,
    POLICY_PREPROCESSOR_DEFAULT_NAME,
)


def build_processors(cfg, device, tokenizer_dir):
    """手动搭建 π0.5 的前/后处理流水线。

    这里不使用 ``make_pre_post_processors(..., pretrained_path=CKPT)`` 直接从
    checkpoint 的 JSON 反序列化，原因是该 checkpoint 由较新版本的 LeRobot 导出，
    其中的步骤名 ``relative_actions_processor`` 在当前安装的 lerobot 0.5.1 中
    已改名为 ``delta_actions_processor``，直接加载会触发 KeyError。

    因此按 ``processor_pi05.make_pi05_pre_post_processors`` 的逻辑逐步重建，
    并做两处离线适配：
      * 分词器：传入本地已加载的 AutoTokenizer 对象（避免联网下载 paligemma）；
      * device：预处理结束搬到 GPU，后处理结束搬回 CPU。
    base 模型的 checkpoint 未附带归一化统计量，NormalizerProcessorStep 在无
    stats 时对该特征直接透传（identity），与官方 README 的 base 用法一致。
    """
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_dir)

    input_steps = [
        RenameObservationsProcessorStep(rename_map={}),
        AddBatchDimensionProcessorStep(),
        NormalizerProcessorStep(
            features={**cfg.input_features, **cfg.output_features},
            norm_map=cfg.normalization_mapping,
            stats=None,  # base 模型无 stats -> 透传
        ),
        Pi05PrepareStateTokenizerProcessorStep(max_state_dim=cfg.max_state_dim),
        TokenizerProcessorStep(
            tokenizer=tokenizer,  # 直接给对象，绕过联网
            max_length=cfg.tokenizer_max_length,
            padding_side="right",
            padding="max_length",
        ),
        DeviceProcessorStep(device=str(device)),
    ]
    output_steps = [
        UnnormalizerProcessorStep(
            features=cfg.output_features, norm_map=cfg.normalization_mapping, stats=None
        ),
        DeviceProcessorStep(device="cpu"),
    ]

    preprocess = PolicyProcessorPipeline(
        steps=input_steps,
        name=POLICY_PREPROCESSOR_DEFAULT_NAME,
        to_transition=batch_to_transition,
        to_output=transition_to_batch,
    )
    postprocess = PolicyProcessorPipeline(
        steps=output_steps,
        name=POLICY_POSTPROCESSOR_DEFAULT_NAME,
        to_transition=policy_action_to_transition,
        to_output=transition_to_policy_action,
    )
    return preprocess, postprocess

# ----------------------------------------------------------------------------
# 路径配置
# ----------------------------------------------------------------------------
CKPT_DIR = "/root/workspace/pi0.5/pi05_base"
TOKENIZER_DIR = os.path.join(CKPT_DIR, "tokenizer")  # 用 checkpoint 自带分词器（离线）
INPUT_DIR = pathlib.Path("/root/workspace/zelos/Omni-Runtime/test/inference/pi05_example_inputs")

# π0.5 的模型状态/动作维度（不足会补零到该维度）
MAX_STATE_DIM = 32
# LIBERO 真实动作维度：6 DoF 末端位姿增量 + 1 维夹爪
LIBERO_ACTION_DIM = 7


def load_image_chw01(path: pathlib.Path) -> torch.Tensor:
    """读取 PNG -> float32、[0,1]、CHW 的张量。

    LeRobot 数据集内部即以 float32/[0,1]/CHW 存储图像，PI05Policy 的
    ``_preprocess_images`` 会再把 [0,1] 线性映射到 SigLIP 期望的 [-1,1]。
    """
    arr = np.asarray(Image.open(path).convert("RGB"), dtype=np.float32) / 255.0  # (H,W,C)
    chw = torch.from_numpy(arr).permute(2, 0, 1).contiguous()  # (C,H,W)
    return chw


def build_observation() -> dict:
    """从磁盘上的真实输入文件构造一帧观测（未加 batch 维的扁平 dict）。

    键名必须与 ``config.json`` 中的 input_features 完全一致：
        observation.images.base_0_rgb / left_wrist_0_rgb / right_wrist_0_rgb
        observation.state
        task  (语言指令，进入 complementary_data)
    """
    base = load_image_chw01(INPUT_DIR / "base_0_rgb.png")
    left_wrist = load_image_chw01(INPUT_DIR / "left_wrist_0_rgb.png")
    right_wrist = load_image_chw01(INPUT_DIR / "right_wrist_0_rgb.png")

    raw_state = np.load(INPUT_DIR / "state.npy").astype(np.float32)  # (8,)
    # 补零到 32 维（π0.5 的最大状态维度），与训练时的 pad 行为一致。
    state = np.zeros(MAX_STATE_DIM, dtype=np.float32)
    state[: raw_state.shape[0]] = raw_state
    state_t = torch.from_numpy(state)

    prompt = (INPUT_DIR / "prompt.txt").read_text().strip()

    return {
        "observation.images.base_0_rgb": base,
        "observation.images.left_wrist_0_rgb": left_wrist,
        "observation.images.right_wrist_0_rgb": right_wrist,
        "observation.state": state_t,
        "task": prompt,
    }, prompt, raw_state


def main() -> None:
    torch.manual_seed(0)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print("=" * 78)
    print("π0.5 (pi05_base) 推理示例  |  LeRobot PI05Policy")
    print("=" * 78)
    print(f"device        : {device}")
    print(f"checkpoint    : {CKPT_DIR}")
    print(f"tokenizer     : {TOKENIZER_DIR}")
    print(f"input dir     : {INPUT_DIR}")

    # ------------------------------------------------------------------
    # 1) 加载策略（VLM 主干 PaliGemma-2B + 动作专家 Gemma-300M）
    # ------------------------------------------------------------------
    print("\n[1/4] 加载 PI05Policy 权重 ...")
    policy = PI05Policy.from_pretrained(CKPT_DIR)
    policy = policy.to(device).eval()
    policy.reset()  # 清空/初始化内部 action 队列
    cfg = policy.config
    n_params = sum(p.numel() for p in policy.parameters())
    print(f"      参数量        : {n_params/1e9:.2f} B")
    print(f"      dtype         : {cfg.dtype}")
    print(f"      chunk_size    : {cfg.chunk_size}  (一次预测的动作步数)")
    print(f"      n_action_steps: {cfg.n_action_steps}")
    print(f"      去噪步数      : {cfg.num_inference_steps}  (flow-matching)")

    # ------------------------------------------------------------------
    # 2) 构造前/后处理流水线
    #    - tokenizer_processor: 覆盖为本地分词器目录（离线）
    #    - device_processor   : 把张量搬到目标 device
    # ------------------------------------------------------------------
    print("\n[2/4] 构造前/后处理流水线 ...")
    preprocess, postprocess = build_processors(cfg, device, TOKENIZER_DIR)

    # ------------------------------------------------------------------
    # 3) 读取真实输入并预处理
    # ------------------------------------------------------------------
    print("\n[3/4] 读取真实输入并预处理 ...")
    frame, prompt, raw_state = build_observation()
    print(f'      语言指令 task : "{prompt}"')
    print(f"      原始状态(8维) : {np.array2string(raw_state, precision=3)}")
    for k, v in frame.items():
        if isinstance(v, torch.Tensor):
            print(f"      {k:38s}: shape={tuple(v.shape)} dtype={v.dtype} "
                  f"range=[{v.min():.3f},{v.max():.3f}]")

    batch = preprocess(frame)
    print("      预处理后进入模型的关键张量:")
    for key in ["observation.images.base_0_rgb", "observation.state",
                "observation.language.tokens", "observation.language.attention_mask"]:
        if key in batch and isinstance(batch[key], torch.Tensor):
            t = batch[key]
            print(f"        {key:40s}: shape={tuple(t.shape)} dtype={t.dtype}")

    # ------------------------------------------------------------------
    # 4) 推理
    #    predict_action_chunk -> 整个动作序列 (B, n_action_steps, 32)
    #    select_action        -> 队列中取出的单步动作 (B, 32)，与 README 一致
    # ------------------------------------------------------------------
    print("\n[4/4] 运行推理 (flow-matching 去噪) ...")
    with torch.inference_mode():
        action_chunk = policy.predict_action_chunk(batch)  # 归一化空间的完整动作块
        policy.reset()
        single_action = policy.select_action(batch)        # (B, 32)
        single_action = postprocess(single_action)         # 反归一化 + 搬回 CPU

    action_chunk = action_chunk.detach().float().cpu()
    single_action = single_action.detach().float().cpu()
    if single_action.ndim == 2:
        single_action = single_action[0]

    print("\n" + "-" * 78)
    print("推理结果")
    print("-" * 78)
    print(f"完整动作块 shape          : {tuple(action_chunk.shape)}  "
          f"= (batch, n_action_steps={cfg.n_action_steps}, action_dim={cfg.max_action_dim})")
    print(f"单步动作 shape            : {tuple(single_action.shape)}")

    libero_action = single_action[:LIBERO_ACTION_DIM]
    print(f"\n第 1 步动作 (模型输出全 {cfg.max_action_dim} 维; LIBERO 只取前 {LIBERO_ACTION_DIM} 维, "
          f"其余 {cfg.max_action_dim-LIBERO_ACTION_DIM} 维对应训练时的填充槽位, 推理时按 LIBERO 约定丢弃):")
    print("  " + np.array2string(single_action.numpy(), precision=4, suppress_small=True,
                                  max_line_width=120))
    print(f"\nLIBERO 有效动作 (前 {LIBERO_ACTION_DIM} 维: Δx Δy Δz Δroll Δpitch Δyaw gripper):")
    print("  " + np.array2string(libero_action.numpy(), precision=4, suppress_small=True))

    # 保存输出，便于复核
    out_path = INPUT_DIR / "predicted_actions.npy"
    np.save(out_path, action_chunk.numpy())
    print(f"\n完整动作块已保存到: {out_path}")

    # 简单的数值健康检查
    assert action_chunk.shape[1] == cfg.n_action_steps
    assert action_chunk.shape[2] == cfg.max_action_dim
    assert torch.isfinite(action_chunk).all(), "动作中出现 NaN/Inf"
    print("\n[OK] 推理完成，输出为有限实数，形状符合 π0.5 规范。")


if __name__ == "__main__":
    main()
