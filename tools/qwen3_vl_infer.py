"""Deterministic FP16 reference for the native Qwen3-VL inference test."""

import argparse
import json
from pathlib import Path

import modelscope
import torch
import transformers
from modelscope import AutoProcessor, Qwen3VLForConditionalGeneration
from PIL import Image


def main():
    root = Path(__file__).resolve().parents[1]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model",
        default="/home/wangyh/workspace/MultimodalModels/QwenModels/Qwen3-VL-8B-Instruct",
    )
    parser.add_argument("--image", type=Path, default=root / "test/inference/demo.jpeg")
    parser.add_argument("--prompt", default="Describe this image.")
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--reference-output", type=Path)
    args = parser.parse_args()
    if args.max_new_tokens <= 0:
        parser.error("--max-new-tokens must be positive")

    print(f"PyTorch: {torch.__version__}, Transformers: {transformers.__version__}, "
          f"ModelScope: {modelscope.__version__}")
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        args.model, dtype=torch.float16, device_map="auto", attn_implementation="sdpa"
    ).eval()
    processor = AutoProcessor.from_pretrained(args.model)
    with Image.open(args.image) as image:
        messages = [{"role": "user", "content": [
            {"type": "image", "image": image.convert("RGB")},
            {"type": "text", "text": args.prompt},
        ]}]
        inputs = processor.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=True,
            return_dict=True, return_tensors="pt",
        ).to(model.device)

    timings = {"vit": [], "language": []}
    starts = {}

    def before(name):
        def hook(module, arguments):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            starts[name] = event
        return hook

    def after(name):
        def hook(module, arguments, output):
            event = torch.cuda.Event(enable_timing=True)
            event.record()
            timings[name].append((starts[name], event))
        return hook

    handles = []
    for name, module in (("vit", model.model.visual),
                         ("language", model.model.language_model)):
        handles.append(module.register_forward_pre_hook(before(name)))
        handles.append(module.register_forward_hook(after(name)))
    with torch.inference_mode():
        generated = model.generate(
            **inputs, max_new_tokens=args.max_new_tokens, do_sample=False,
            top_k=None, top_p=None, temperature=None,
        )
    torch.cuda.synchronize()
    for handle in handles:
        handle.remove()
    token_ids = generated[0, inputs["input_ids"].shape[1]:].tolist()
    text = processor.decode(
        token_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False
    )
    elapsed = {name: [start.elapsed_time(end) for start, end in events]
               for name, events in timings.items()}
    print(text)
    print(f"VIT: {sum(elapsed['vit']):.3f} ms, "
          f"prefill: {elapsed['language'][0]:.3f} ms, "
          f"decode: {sum(elapsed['language'][1:]):.3f} ms")
    if args.reference_output is not None:
        reference = {
            "model": args.model, "dtype": "float16", "do_sample": False,
            "attention": "sdpa", "torch": torch.__version__,
            "transformers": transformers.__version__,
            "prompt": args.prompt, "max_new_tokens": args.max_new_tokens,
            "prompt_token_ids": inputs["input_ids"][0].tolist(),
            "image_grid_thw": inputs["image_grid_thw"].tolist(),
            "generated_token_ids": token_ids, "text": text,
        }
        args.reference_output.with_suffix(".json").write_text(
            json.dumps(reference, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        args.reference_output.with_suffix(".txt").write_text(text, encoding="utf-8")
        args.reference_output.with_suffix(".tokens").write_text(
            " ".join(map(str, token_ids)) + "\n", encoding="ascii"
        )


if __name__ == "__main__":
    main()
