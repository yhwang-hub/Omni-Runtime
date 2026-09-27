# Inference validation data

This directory is the only repository location used by inference tests for
images and evaluation manifests.

- `demo.jpeg`: required Qwen3-VL image input.
- `bus.jpg`: shared YOLO and Qwen3-VL smoke image.
- `qwen3_vl_accuracy_dataset.jsonl`: local caption/VQA evaluation manifest.

Each JSONL record contains an image, prompt, task type, and either
`required_keywords` for caption recall or `answers` for normalized VQA exact
match. Large public datasets are intentionally not committed. Place their
converted JSONL manifests and images below `test/inference/datasets/<name>` so
the same evaluator can consume them without changing runtime code.
