# Omni-Runtime Bazel 构建与测试指南

本文说明如何直接使用 Bazel 完成 Omni-Runtime 的编译和测试，以及如何使用 `script/build.sh` 与 `script/run_test.sh`。

模块边界、公共图像预处理、已实现的 backend 中立 projection 并行及后续扩展方案见
[架构与并行执行方案](architecture_and_parallel_execution.md)。

默认启用 backend 的 projection 并行策略，业务代码无需区分 CUDA/OpenCL。
同版本串行 A/B 对照可在测试命令前设置 `OMNI_PARALLEL_LINEAR=0`；设为 `1`
或不设置则启用并行。CUDA 多行 projection 首次遇到新 shape 会进行耗时选择，
这部分初始化开销计入首次 prefill；评估稳态性能时需要比较后续复用执行。

公共图像预处理回归可单独运行：

```bash
./script/run_test.sh --target //Omni-Runtime/inference/common/image_preprocessor:image_preprocessor_test
```

算子代码使用仓库根目录 `.clang-format`（与 graph 的两空格缩进、100 列及右侧指针对齐一致）。
CUDA/OpenCL 源码、runtime 头文件和算子测试可统一检查：

```bash
find Omni-Runtime/kernels -type f \( -name '*.h' -o -name '*.cuh' -o -name '*.cl' -o -name '*.cc' \) \
  -print0 | xargs -0 clang-format --dry-run --Werror
```

## 1. 直接使用 Bazel

所有命令都应在仓库根目录执行：

```bash
cd /home/wangyh/workspace/Omni-Runtime
```

Bazel 通过 `BUILD.bazel` 发现目标。目标格式为：

```text
//<package>:<target>
```

例如：

```text
//Omni-Runtime/kernel:kernel
//Omni-Runtime/kernels:kernels_tests
//example:qwen3_vl_inference_test
```

### 1.1 编译单个目标

```bash
bazel build //Omni-Runtime/kernel:kernel
```

执行内容：

- 解析 `MODULE.bazel` 和依赖；
- 编译 `Omni-Runtime/kernel` 下的 C++ 源码；
- 生成静态库和动态库；
- 产物出现在：

```text
bazel-bin/Omni-Runtime/kernel/libkernel.a
bazel-bin/Omni-Runtime/kernel/libkernel.so
```

### 1.2 编译所有目标

```bash
bazel build //...
```

执行内容：

- 遍历整个 workspace；
- 编译所有 `cc_library`、`cc_test`、`cc_binary`；
- 不运行测试。

### 1.3 指定编译模式

```bash
bazel build --compilation_mode=opt //...
bazel build --compilation_mode=dbg //...
```

作用：

| 模式 | 作用 | 适用场景 |
|---|---|---|
| `opt` | 默认优化模式，使用 `-O2`/`-O3` | 发布、性能测试 |
| `dbg` | 带调试信息，不启用强优化 | GDB 调试、断点分析 |

### 1.4 指定并行任务数

```bash
bazel build --jobs=8 //...
```

作用：

- 控制 Bazel 同时执行多少个编译、链接 action；
- 提高构建速度；
- 任务数过高时可能导致内存不足。

### 1.5 继续执行而不是遇错停止

```bash
bazel build --keep_going //...
```

作用：

- 遇到编译错误时继续构建其他 target；
- 适合一次性收集多个错误。

### 1.6 查看详细编译命令

```bash
bazel build --subcommands //...
```

作用：

- 打印每个 action 的真实命令，例如 g++ 编译参数；
- 排查 include path、宏定义、链接参数时使用。

### 1.7 清理 Bazel 输出

```bash
bazel clean
```

清除当前 workspace 的 build output。

更彻底的清理：

```bash
bazel clean --expunge
```

作用：

- 删除 Bazel output base；
- 清除 build cache、test cache、external repository cache；
- 下一次构建会重新下载依赖、重新编译全部内容。

注意：

- `bazel clean --expunge` 删除的是真实 output base；
- workspace 根目录下的 `bazel-bin`、`bazel-out`、`bazel-testlogs`、`bazel-Omni-Runtime` 是 convenience symlink；
- 这些 symlink 可能残留，后续由 `script/build.sh --clean` 一并删除。

### 1.8 运行所有测试

```bash
bazel test //...
```

作用：

- 先编译测试目标；
- 然后执行所有测试；
- 输出每个 target 的通过/失败状态。

### 1.9 只运行 kernel 测试

```bash
bazel test //Omni-Runtime/kernels:kernels_tests
```

该目标定义在：

```text
Omni-Runtime/kernels/BUILD.bazel
```

它会收集所有：

```text
Omni-Runtime/kernels/**/*_test.cc
```

并运行其中 18 个 GoogleTest 用例，包括：

- `AddKernelTest`
- `LayernormKernelTest`
- `AdarmsKernelTest`
- `SwigluKernelTest`
- `ArgmaxKernelTest`
- `EmbeddingKernelTest`
- `FlowStepKernelTest`
- `GatedAddKernelTest`
- `MatmulKernelTest`
- `FlashAttentionKernelTest`
- `MhaKernelTest`
- `GqaAttentionKernelTest`
- `MropeKernelTest`
- `GatedAttentionKernelTest`
- `KvCacheKernelTest`
- `FusedFfnKernelTest`
- `ImageKernelTest`
- `VisionEncoderKernelTest`

### 1.10 控制测试输出

```bash
bazel test //Omni-Runtime/kernels:kernels_tests --test_output=all
```

常用取值：

| 值 | 作用 |
|---|---|
| `summary` | 只显示摘要 |
| `errors` | 只在失败时输出错误 |
| `all` | 输出完整测试日志 |

推荐排查失败时使用：

```bash
bazel test //Omni-Runtime/kernels:kernels_tests --test_output=all
```

### 1.11 运行单个测试套件或单个用例

只运行 `AddKernelTest`：

```bash
bazel test //Omni-Runtime/kernels:kernels_tests \
  --test_filter=AddKernelTest.*
```

只运行一个具体用例：

```bash
bazel test //Omni-Runtime/kernels:kernels_tests \
  --test_filter=AddKernelTest.ComputesQwen3VlActivationAdd
```

### 1.12 查看全部测试用例名

```bash
bazel run //Omni-Runtime/kernels:kernels_tests -- --gtest_list_tests
```

### 1.13 直接运行测试二进制

```bash
bazel run //Omni-Runtime/kernels:kernels_tests
```

也可以带过滤条件：

```bash
bazel run //Omni-Runtime/kernels:kernels_tests -- \
  --gtest_filter=AddKernelTest.*
```

### 1.14 查看测试日志

终端实时查看：

```bash
bazel test //Omni-Runtime/kernels:kernels_tests \
  --test_output=all
```

查看落盘日志：

```bash
less bazel-testlogs/Omni-Runtime/kernels/kernels_tests/test.log
```

查看 XML 结果：

```bash
less bazel-testlogs/Omni-Runtime/kernels/kernels_tests/test.xml
```

### 1.15 给测试进程传递环境变量

CUDA JIT 测试支持调试环境变量：

```bash
bazel test //Omni-Runtime/kernels:kernels_tests \
  --test_output=all \
  --test_env=OMNI_JIT_DEBUG=1 \
  --test_env=OMNI_JIT_PRINT_COMPILER_COMMAND=1
```

作用：

- `OMNI_JIT_DEBUG=1`：开启 JIT 调试信息、PTX/SASS dump；
- `OMNI_JIT_PRINT_COMPILER_COMMAND=1`：打印 NVCC 编译命令。

### 1.16 生成 XML 测试报告

```bash
bazel test //Omni-Runtime/kernels:kernels_tests \
  --test_output=all \
  --test_arg=--gtest_output=xml:/tmp/kernels_tests.xml
```

查看：

```bash
less /tmp/kernels_tests.xml
```

### 1.17 清理 Bazel symlink

Bazel 会在 workspace 根目录生成四个 convenience symlink：

```text
bazel-Omni-Runtime
bazel-bin
bazel-out
bazel-testlogs
```

它们指向：

```text
~/.cache/bazel/_bazel_wangyh/<hash>/...
```

如果只想删除这些链接：

```bash
for link in bazel-bin bazel-out bazel-testlogs bazel-Omni-Runtime; do
  if [[ -L "${link}" ]]; then
    rm "${link}"
  fi
done
```

注意这里使用 `[[ -L ... ]]`，只删除 symlink，避免误删普通目录。

## 2. 使用 `script/build.sh`

脚本位置：

```text
script/build.sh
```

功能：

- 设置 CUDA 环境变量；
- 调用 Bazel 编译指定 target；
- 可选清理 Bazel output base；
- 可选清理 workspace 根目录下的 Bazel symlink。

查看帮助：

```bash
./script/build.sh --help
```

### 2.1 常用命令

```bash
./script/build.sh
```

等价于：

```bash
bazel build --compilation_mode=opt --jobs=<CPU核数> //...
```

执行后：

- 编译所有目标；
- 生成或更新 `bazel-bin`、`bazel-out`、`bazel-testlogs`、`bazel-Omni-Runtime`；
- 产物保存在 Bazel output base 中。

```bash
./script/build.sh --mode dbg
```

等价于：

```bash
bazel build --compilation_mode=dbg --jobs=<CPU核数> //...
```

适用调试。

```bash
./script/build.sh --target //Omni-Runtime/kernel:kernel
```

只编译 kernel 库。

```bash
./script/build.sh --target //Omni-Runtime/kernels:kernels_tests
```

只编译 kernel 测试目标，但不会运行测试。

```bash
./script/build.sh --jobs 4
```

限制并行任务数为 4，适合内存受限环境。

```bash
./script/build.sh --keep-going
```

遇到错误后继续编译其他目标。

```bash
./script/build.sh --verbose
```

显示 Bazel 每个编译 action 的真实命令。

### 2.2 `--clean` 的具体行为

执行：

```bash
./script/build.sh --clean
```

脚本会依次执行：

1. `bazel clean --expunge`

   清除内容：

   - output base；
   - build cache；
   - test cache；
   - external repository cache；
   - test logs；
   - 编译产物；
   - JIT 测试产物所在的 execroot。

2. 删除 workspace 根目录下的四个 Bazel convenience symlink：

   ```text
   bazel-Omni-Runtime
   bazel-bin
   bazel-out
   bazel-testlogs
   ```

   删除规则：

   - 只有目标存在且是 symlink 时才删除；
   - 如果是普通目录，脚本不会删除，避免误删源码；
   - 删除时会打印：

     ```text
     Removing Bazel convenience symlink: bazel-bin
     ```

3. 清理完成后继续执行构建。

因此：

```bash
./script/build.sh --clean
```

实际过程是：

- `bazel clean --expunge` 删除真实 output base；
- 随后立即删除 workspace 根目录下这四个 symlink；
- 脚本继续执行构建；
- 构建完成后 Bazel 会重新生成这四个 symlink，并指向新的 output base。

如果你只想清理而不立即重建，可以执行：

```bash
bazel clean --expunge
```

然后手动删除这四个 symlink，或使用后续构建让 Bazel 重新创建它们。

如果只想 clean，不想立刻完整重建，可以执行：

```bash
bazel clean --expunge
```

再手动删除四个链接。

## 3. 使用 `script/run_test.sh`

脚本位置：

```text
script/run_test.sh
```

功能：

- 设置 CUDA 环境；
- 调用 Bazel 编译并运行测试；
- 支持测试过滤；
- 支持输出模式控制；
- 支持 clean。

查看帮助：

```bash
./script/run_test.sh --help
```

### 3.1 常用命令

```bash
./script/run_test.sh
```

等价于：

```bash
bazel test --compilation_mode=opt \
  --jobs=<CPU核数> \
  --test_output=errors \
  //...
```

作用：

- 编译所有测试目标；
- 运行全部测试；
- 失败时输出错误日志。

```bash
./script/run_test.sh --test-output all
```

等价于：

```bash
bazel test --test_output=all //...
```

会显示完整测试日志，包括：

- 每个测试开始/结束；
- kernel JIT 编译日志；
- stdout/stderr；
- 失败断言。

仅显式指定单个 `//example:qwen3_vl_inference_test`，且输出模式为 `all` 或 `streamed` 时，
脚本使用 `bazel run` 执行同一个 GoogleTest 测试程序，实时转发不带换行的文本片段。
该方式每次执行测试，不使用测试结果缓存；断言和失败退出码仍然有效。
`--test-filter` 在此模式下转换为程序参数 `--gtest_filter`。
其他目标及全量测试仍使用 `bazel test`，不改变原有测试发现和日志行为。
通用 `--test-output streamed` 对应 `bazel test --test_output=streamed --nocache_test_results`。

```bash
./script/run_test.sh --test-filter AddKernelTest.*
```

等价于：

```bash
bazel test --test_filter=AddKernelTest.* //...
```

只运行 `AddKernelTest` 套件。

```bash
./script/run_test.sh \
  --target //Omni-Runtime/kernels:kernels_tests \
  --test-filter AddKernelTest.ComputesQwen3VlActivationAdd \
  --test-output all
```

等价于：

```bash
bazel test //Omni-Runtime/kernels:kernels_tests \
  --test_filter=AddKernelTest.ComputesQwen3VlActivationAdd \
  --test_output=all
```

只运行一个具体测试用例，并输出完整日志。

```bash
./script/run_test.sh --target //Omni-Runtime/memory:memory_test
```

只运行 memory 测试。

```bash
./script/run_test.sh --mode dbg
```

以调试模式构建并运行测试。

```bash
./script/run_test.sh --jobs 4
```

限制并行任务数为 4。

```bash
./script/run_test.sh --keep-going
```

某个测试失败后继续运行其他测试。

```bash
./script/run_test.sh --verbose
```

显示测试二进制和编译 action 的详细命令。

### 3.2 `run_test.sh --clean` 的行为

当前 `run_test.sh` 也支持：

```bash
./script/run_test.sh --clean
```

它会：

```bash
bazel clean --expunge
```

清除 Bazel output base 和相关 cache。

需要注意：

> 当前 `run_test.sh --clean` 只清理 output base，不会额外删除 workspace 根目录下的四个 bazel symlink。若需要同时删除这四个链接，使用：
>
> ```bash
> ./script/build.sh --clean
> ```
>
> 因为四个 symlink 由 `build.sh --clean` 统一处理。

## 4. 推荐工作流

### 4.1 日常开发

```bash
./script/run_test.sh --test-output all
```

适合观察完整输出和 JIT 日志。

### 4.2 快速验证

```bash
./script/run_test.sh
```

只显示失败错误，速度较快。

### 4.3 调试单个 kernel

```bash
./script/run_test.sh \
  --target //Omni-Runtime/kernels:kernels_tests \
  --test-filter AddKernelTest.* \
  --test-output all
```

### 4.4 完整重建

```bash
./script/build.sh --clean
./script/run_test.sh --test-output all
```

`--clean` 会清除 Bazel cache 和 workspace 根目录下的四个 bazel symlink，然后重新完整编译。

### 4.5 查看 Qwen3-VL 推理输出

除 CUDA Toolkit/cuBLAS 外，Qwen3-VL 的 JPEG 解码还需要系统 libjpeg 开发包。在 Debian/Ubuntu 上：

```bash
sudo apt-get install libjpeg-dev
```

cuBLAS 通过 Bazel CUDA toolchain 的 `@local_cuda//:cublas` 引入，不需要手动指定库目录。

```bash
./script/run_test.sh \
  --target //example:qwen3_vl_inference_test \
  --test-output all
```

输出参考文本：

```text
This is a heartwarming and serene photograph capturing a moment of connection between a woman and her dog on a beach at sunset.

**Main Subjects:**
- A woman with long, dark hair, wearing a black and white plaid shirt and dark pants. She is sitting cross-legged in the sand, smiling warmly as
```

测试使用 **FP16 + greedy decoding**，不再使用旧的 `qwen3_vl_8b_expected.txt`：

- 第一轮严格比较 64 个 token ID 和解码文本。
- 第二轮复用同一个 pipeline，以 1024 为生成上限；当前参考生成 352 个 token（包括 EOS），再次严格比较所有 token ID 和文本。
- 两轮均在回调中即时输出新增文本并 flush，不再在结束后重复打印完整文本。
- 检查图复用后的 GPU 内存预算不变。独立的 runtime 回归还检查 generation reset 不释放 KV cache、workspace 和 deepstack buffer。

示例分为精度/流式输出、正常取消与恢复、拒绝重入与回调异常三个测试，共享一次初始化的 pipeline。
负向测试捕获并严格断言预期错误日志，成功运行时不展示这些刻意触发的错误。
实际调用中的重入/异常仍会明确报错；回调异常仅由产生错误的 `TextStream` 记录一次。

参考数据为 `test/inference/qwen3_vl_8b_fp16*` 和 `qwen3_vl_8b_fp16_long*`。
对应 JSON 保存 Python 版本、输入 token、图像网格和生成参数，便于排查参考环境差异。

### 4.6 运行和更新 Python FP16 参考

```bash
source /home/wangyh/workspace/envs/qwen3_vl_env/bin/activate

# 保持脚本默认的 1024-token 上限。
python tools/qwen3_vl_infer.py

# 仅在有意更新参考时运行以下导出命令。
python tools/qwen3_vl_infer.py --max-new-tokens 64 \
  --reference-output test/inference/qwen3_vl_8b_fp16
python tools/qwen3_vl_infer.py --max-new-tokens 1024 \
  --reference-output test/inference/qwen3_vl_8b_fp16_long
```

脚本显式使用 `dtype=torch.float16`、`do_sample=False` 和 SDPA，覆盖模型 generation config 中的采样默认值。
导出包含 `.txt`、`.tokens` 和 `.json`，不要只修改文本来绕过 token 精度检查。
当前参考环境为 PyTorch `2.13.0+cu130`、Transformers `5.17.0`、ModelScope `1.40.1`。

### 4.7 Qwen3-VL 性能与精度回归

原链路的主要问题及对应处理：

| 节点 | 原因 | 当前实现 |
|---|---|---|
| ViT | CPU resize、低吞吐 attention、重复合并重排/拷贝 | libjpeg 解码；与 Torchvision uint8 antialiased bicubic 对齐的 GPU resize/归一化/patch 融合；分块 Tensor Core attention，Q fragment 常驻寄存器；直接复用已合并顺序的张量 |
| prefill | 逐元素 GEMM、重复 JIT 源码处理 | 后端持久化 BLAS handle，FP32 累加；Tensor Core causal/GQA attention；缓存已加载 JIT kernel |
| decode | 访存受限的矩阵向量乘、整份词表 logits 回传、多余的最后一步 | 对齐向量加载的 GEMV；分区 attention；GPU argmax，仅回传 token ID；EOS/长度到达后不再计算无用 token |
| 生命周期 | 重建 graph、运行中分配 KV cache/workspace | 初始化构图并 Prepare；按 `max_context_tokens` 一次性预分配共享 workspace 和 KV cache，Generate 仅重置状态；超容量明确报错，不自动扩容 |

精度修复还包括 FP16 权重/归一化、interleaved MRoPE、低精度中间结果舍入、vision merge 顺序、merger 的精确 GELU，以及归约共享 scratch 复用前缺少屏障导致的竞争。
模型业务代码不直接调用 CUDA API；BLAS、同步和分配在 backend，GPU 算子细节在 JIT/kernel 层。

首轮实现的历史数据如下（后续 Blackwell 优化见 4.8）。输入为默认 demo 图和提示词，视觉 token 数 2752，prefill token 数 2766：

| 运行方式 | ViT | prefill | decode |
|---|---:|---:|---:|
| 第一轮，64 tokens，包含进程内首次 JIT 加载/编译 | 5628.69 ms | 4360.09 ms | 5473.02 ms |
| 同 pipeline 第二轮，1024 上限，352 tokens（含 EOS） | 1858.01 ms | 1106.90 ms | 26162.40 ms |

第二轮 decode 对应 351 次后续生成，约 13.42 token/s；不能直接与第一轮不同长度的 decode 总耗时比较。
这里的“图复用”是框架 inference graph 复用，不是 CUDA Graph capture。
C++ 在该历史轮次采用同步后的 wall time，ViT 包含图像读取/解码等 CPU 工作；首次调用也包含 lazy JIT。
现行 C++ event 计时口径见 4.9，不能将两种口径直接混比。
Python 脚本使用 visual/language_model 的 CUDA event，未覆盖 CPU 图像预处理、LM head 和采样，因此两者不是同口径的端到端 benchmark。

首轮实现的 Nsight Systems 观测：

- GPU kernel 累计时间中，GEMV 约 73.81%，分块 attention 约 10.01%，decode 分区 attention 约 8.68%，BLAS GEMM 约 4.87%。
- 从第一次预处理 kernel 到最后一次 argmax kernel，观测到的 `cudaMalloc` / `cudaFree` 调用均为 **0**；分配集中在初始化，释放集中在销毁。
- 重复 LayerNorm 归约用例通过 Compute Sanitizer racecheck，报告 0 errors / 0 warnings。

当前 decode 的主要限制已转为权重访存带宽；ViT attention 和 kernel 调度仍有优化空间。
这些结果证明了正确性和本机可复现的改进，**不代表已经证明达到硬件理论上限**。

### 4.8 DGX Spark / Blackwell 热点优化

本轮在 GB10（SM 12.1，48 SM，24 MiB L2）上继续优化，保持相同 FP16 greedy 参考、图像分辨率和生成长度。
没有改为 FP8/INT8，没有减少图像 token，也没有缩短输出来获取“加速”。

#### 分层和生命周期

- **模型层**只负责层次结构、张量和权重选择。现行版本统一通过 `Backend::Compute` 和
  `Backend::LinearMatmul` 调用计算；kernel runtime、launch geometry 和设备分派已收拢到后端，见 4.10。
- **CUDA backend**负责持久化 cuBLASLt handle、32 MiB workspace、计时 event 和 GEMM plan。
  `LinearPlan` 只在 `CudaBackend` 中使用，收拢为其私有结构；所有 CUDA/cuBLAS API、资源销毁和错误日志均留在后端。
  backend 自有 compute workspace 通过 `compute_workspace_bytes()` 计入模型显存统计。
- **GEMM plan**按 dtype、M/N/K 缓存，使用 row-major layout、FP32 累加，并禁止低精度 split-K reduction。
  第一次遇到形状时，在预分配 workspace 中比较至多 8 个 heuristic 候选，每个执行 3 次，用 GPU event 选择计划。
  后续相同形状、不同权重或输入直接复用；非对齐输入和 FP32 继续使用原 BLAS 路径。
  workspace/event 只在 `InitializeCompute()` 分配；形状计划是 CPU 对象，不在 Generate 中重新分配 GPU buffer。
  首次形状选优成本计入首次推理，不能拿首次耗时和稳态耗时混比。
- **kernel 层**在 `matmul_kernel/tensor_core.cuh` 复用 FP16/BF16 的 PTX `m16n8k16` 原语，避免依赖 WMMA 未公开的 fragment 元素布局。
  JIT runtime 根据设备能力启用 SM80+ tensor 路径；保留通用算子路径，不把 SM121 判断写进模型层。
  这些是 warp-level MMA 优化，不是 tcgen05/TMA 或 CUDA Graph capture。

#### 热点变更

| 热点 | 原实现的问题 | 当前方案 |
|---|---|---|
| ViT/prefill attention | 每轮将 score、softmax probability、rescale 后的 O 在 shared memory 与 register 间往返 | Q、score、softmax 和 O 常驻寄存器；只缓存 K/V；SMEM padding 减少 bank conflict，使用 online softmax，保持低精度 probability 的既有舍入语义 |
| ViT attention tile | Q/K tile 未同时考虑带宽、寄存器占用和 SM occupancy | D≤80 使用 Q128/K32、256 threads；D≤128 的大维度路径使用 Q64/K64、128 threads。vision 路径比较 Q64/K32、Q64/K64、Q64/K128 和 Q128/K32 后采用后者 |
| 双帧 patch embedding | 每一个输出元素启动一个 block，重复加载相同 patch/weight，约 1268 万个 blocks/图 | 共享 64×32 input/weight tiles，用 Tensor Core 同时累加两组权重；两个时间片保持独立输入权重、FP32 累加，最后才加 bias 并转换到输出 dtype |
| decode attention | 每个 Q head 的 warp 串行遍历 128 个 key；GQA 重复读取 K/V | 通用路径并行计算 partition 的 logits；Q/KV=4 且 D128 的低精度路径共享 KV tile，Tensor Core 计算 QK，再以 FP32 probability/accumulator 计算 PV，不引入 FP16 softmax probability 舍入 |
| prefill/ViT GEMM | 默认 BLAS 选择未针对实际矩阵形状测量 | 后端缓存并测速的 row-major cuBLASLt 计划，保持 FP32 累加和固定 workspace |

GEMV 已经接近实测带宽；streaming-load 改动在独立比较中没有稳定收益，因此没有保留，也没有加入额外权重副本或低精度权重格式。

#### 测量与硬件边界

无 profiler、同 pipeline 第二轮（1024 上限，352 tokens 含 EOS）的一次测量：

| 阶段 | 用户提供的本轮基线 | 优化后 | 耗时降低 |
|---|---:|---:|---:|
| ViT | 1842.63 ms | 751.61 ms | 59.2% |
| prefill | 1102.59 ms | 824.31 ms | 25.2% |
| decode | 24924.30 ms | 24663.00 ms | 1.0% |

不同运行的 decode 约 24.2–24.7 秒，受共享内存带宽、时钟和系统负载影响。
decode 总量的提升较小，不能把 attention kernel 的局部倍数等同于整个 decode 的倍数。
首次 64-token 推理仍包含 lazy JIT/形状选优成本；一轮 clean build 后约为 ViT 4675.08 ms、prefill 4218.83 ms、decode 5252.89 ms。

Nsight Systems 对相同默认输入的算子平均时间（不是 end-to-end 阶段时间）：

| kernel | 优化前 | 优化后 |
|---|---:|---:|
| ViT attention，D72 | 55.108 ms | 18.586 ms |
| prefill attention，D128 | 8.156 ms | 3.125 ms |
| decode partition attention | 209.083 µs | 85.766 µs |
| paired patch embedding | 73.709 ms | 1.781 ms |

当前 decode 的主要边界是 **每个 token 需要重新读取约 15.136 GB 的稠密 FP16 矩阵权重**：

```text
bytes/token =
  2 × [36 × (2×4096² + 2×4096×1024 + 3×4096×12288) + 151936×4096]
  = 15,136,194,560 bytes
```

这已远大于 24 MiB L2。按 [DGX Spark 官方规格](https://www.nvidia.com/en-us/products/workstations/dgx-spark/)的 273 GB/s，
351 次后续 decode 的理想权重读取下界约为 **19.46 秒**，还没有计入 KV cache、计算、同步和调度。
本机独立大缓冲区测量使用非恒定输入、超 L2 的 151936×4096 FP16 矩阵、4 次预热和 40 次计时：
GEMV 约 **238.5 GB/s**，同大小 device-to-device copy 的读写合计约 **236.0 GB/s**。
使用实测 GEMV 速率估算，权重读取本身就需要约 **22.28 秒**；24 秒量级的完整 decode 已接近该带宽参照，
但这不是所有算子达到理论峰值的证明。Nsight Compute 硬件计数器受当前用户权限限制，未取得 SM/DRAM 峰值利用率。

继续获得数量级 decode 加速需要改变权重流量或生成算法，例如低比特权重、批处理或可验证的推测解码；
不能仅通过普通 FP16 kernel 调优承诺数倍提升。这些改变没有在本轮悄悄启用。

#### 验证

```bash
./script/build.sh --clean
./script/build.sh
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
./script/run_test.sh --test-output all
```

该轮全量为 11 个测试目标、82 个用例。短 64-token 和完整 352-token 参考的 token ID、文本均完全相等。
FP16/BF16 回归覆盖 Q/K/V tile 尾部、奇数 head dimension、2/4 组 GQA、KV 分区边界、causal source limit、
尖锐 softmax、双帧 patch 以及 cuBLASLt 计划对不同输入的复用和非对齐 fallback。
新增路径通过 Compute Sanitizer memcheck（0 errors）及 attention/patch racecheck（0 errors / 0 warnings）。
最终 Nsight trace 在首个预处理 kernel 到最后 argmax kernel 之间观测到 **0 次 `cudaMalloc/cudaFree`**，包括首次 GEMM 形状选优。
再次扫描模型及 inference 层，没有直接 CUDA API 或 `dim3` 使用；GPU buffer 的拥有式分配仍集中在初始化，运行中的 buffer view 不拥有内存。

### 4.9 流式输出与后端 event 计时

`TextStream` 的实现和单元测试位于 `Omni-Runtime/inference/common/text_stream/`，
作为公共推理组件独立构建，不依赖 tokenizer、模型或 GPU 后端。
Tokenizer 负责 token 到字节的还原，`TextStream` 负责字节到完整 UTF-8 文本的增量交付；
Qwen3-VL decode 直接依赖 `//Omni-Runtime/inference/common/text_stream:text_stream`。
独立回归无需加载模型或图像数据：

```bash
./script/run_test.sh \
  --target //Omni-Runtime/inference/common/text_stream:text_stream_test \
  --test-output all
```

`GenerationRequest::stream_callback` 在生成过程中接收**新增文本**，不是累积的整段文本：

```cpp
GenerationRequest request;
request.prompt = "Describe this image.";
request.image_path = "test/inference/demo.jpeg";
request.max_new_tokens = 64;
request.stream_callback = [](const std::string &chunk) {
  std::cout << chunk << std::flush;
  return true;
};
GenerationResult result;
const bool is_successful = pipeline.Generate(request, &result);
```

- 第一个 token 也会流式输出；EOS 保留在结果 token ID 中，但不交给回调。
- Qwen byte-level BPE 的 token 可以拆开一个 UTF-8 字符。公共 `TextStream` 保留未完成的字节，
  仅下发完整 UTF-8；正常结束时无法补全或非法的字节使用 U+FFFD，与非流式返回文本采用同一逻辑。
  因此成功且未取消时，所有回调文本拼接严格等于 `result.text`。
- 回调返回 `false` 表示正常用户中止：不再计算下一个 token，`Generate` 返回 `true`，
  `result.is_cancelled` 为 `true`。结果保留已生成的 token ID 和已交付的完整文本；
  取消时尚未组成字符的尾部字节不再下发。
- 回调抛异常会明确记录错误并使 `Generate` 失败，不将异常当成正常取消。
  回调不得重入同一 pipeline 的 `Generate` 或 `Init`；入口会明确拒绝。不同线程也应串行使用同一 pipeline。
- 无回调时继续正常推理，不依赖控制台输出。decode 节点同时负责增量解码，删除了重复的全量 detokenize 节点。

**控制台传输也必须支持增量输出**：Bazel 的 `--test_output=all` 在测试结束后才展示日志；
本机 Bazel 8.4.1 的 `--test_output=streamed` 仍会缓冲没有换行的片段。
因此单独运行 Qwen3-VL 示例并指定 `all` 或 `streamed` 时，脚本改用 `bazel run`，
不向生成文本插入额外换行，也不绕过 GoogleTest 断言。
直接执行 `bazel test` 或运行全量测试时，不保证控制台按 token 即时显示。

阶段时间不再使用 `std::chrono`：

| 指标 | event 区间 |
|---|---|
| ViT | CPU 读取/JPEG 解码和网格检查之后，图像上传之前，到最后一次视觉投影完成 |
| prefill | prompt/position 构建之后，从输入上传到 LM head 和 GPU argmax 完成 |
| decode | 每次后续语言步分别记录并求和；不含首个 prefill token，也不含用户流式回调 |

`GenerationResult` 同时提供以下统计：

- `input_image_size` / `resized_image_size`：原始图像和实际 resize 后的宽、高，单位 pixels；
  图像解码与网格计算时记录，不根据固定 demo 尺寸填充。纯文本请求两者均为零。
- `prefill_token_count`：展开视觉占位后真正输入语言模型的总 token 数，包括视觉和文本。
  `prefill_tokens_per_second()` 为 `1000 × prefill_token_count / prefill_time_ms`。
- `decode_token_count`：成功执行且计时的后续语言步数，不含 prefill 已生成的首 token；
  包括生成 EOS 的一步。`decode_tokens_per_second()` 为 `1000 × decode_token_count / decode_time_ms`。
  单 token 或首 token 后取消时，没有后续 decode 步；零耗时返回零速率，不除以零。

默认图像实际为 2048 × 1365，resize 后为 2048 × 1376，prefill 总输入 2766 tokens。
复用图生成完整参考时的一次实测报告如下；它不是跨进程冷启动或端到端 wall time：

```text
Model infer Time:
  VIT:
    Input image (W x H): 2048 x 1365 pixels
    Resized image (W x H): 2048 x 1376 pixels
    Time: 590.58 ms
  Prefill:
    Input tokens: 2766
    Time: 771.65 ms
    Throughput: 3584.54 tokens/s
  Decode:
    Generated token IDs: 352
    Timed decode tokens: 351 (first token is generated by prefill)
    Time: 24525.85 ms
    Throughput: 14.31 tokens/s
```

`backend::EventTimer` 在初始化时创建可复用事件；CUDA 使用 `cudaEventRecord/cudaEventElapsedTime`，
OpenCL 队列开启 profiling，使用 marker event 的 `CL_PROFILING_COMMAND_END` 时间戳。
所有平台 API 均在 backend。当前计算实际使用默认 compute stream/queue，timer 也记录在这条流上，
删除了模型 runtime 中原先未被计算使用的额外 stream，避免在错误流上计时。
GPU argmax 的主机 token 回传发生在语言计时终点之后；callback 刻意延迟不会进入 decode event 区间。

**event 区间不是 kernel duration 的简单求和**：同一流的两次时间戳之间仍可能有 CPU 提交、
首次 JIT/形状选优造成的设备空闲。冷启动与复用图的稳态时间仍要区分。
C++ 指标还包括 LM head/argmax，不能直接与 Python 仅包围 visual/language_model 的 event 混比。

### 4.10 统一后端计算边界

依赖方向为：

```text
models/qwen3_vl
  -> backend.h / backend_compute_define.h / memory
  -> backend_cuda.cc 或 backend_opencl.cc
  -> 内部 JIT backend / kernel runtime
  -> .cuh 算子
```

- `ComputeOperation` 使用有类型的 operation variant 表达 embedding、norm、RoPE、KV cache、attention、
  residual、激活、图像/patch 等操作，只含张量地址和语义维度，不含 CUDA 类型、block/grid 或 JIT 配置。
  精度模板分派、GEMV/GEMM 选择、设备能力与 V4 路由都由 CUDA backend 决定。
- `Backend::InitializeCompute()` 统一初始化 JIT、持久 BLAS handle、workspace 和选优 event。
  `BackendKind::kAuto` 的平台选择在 factory 内完成。模型不再直接构造 CUDA backend。
  OpenCL 当前支持内存、stream/event 与独立 JIT 测试，**尚未实现这些模型计算操作**；
  请求模型 compute 时会明确报错，不伪装成功或回落到空计算。
- 原 `kernel/backend/cuda`、`kernel/backend/opencl` 和 CUDA JIT adapter 已分别迁入
  `backend/cuda_jit`、`backend/opencl_jit`。它们是后端内部编译/装载实现，不向模型暴露；
  没有为了形式上的“单文件”把编译器、driver loader 和资源管理塞成一个巨型源文件。
  `kernel` 只保留无平台 API 的编译缓存/配置/工具；旧 `//Omni-Runtime/kernel:kernel` 和
  `//Omni-Runtime/kernel:opencl_jit_test` 标签保留兼容入口。
- JIT 所需 `.cuh` 和公共 `.h` 都作为 kernels library 的运行时数据分发，后端测试不依赖源码工作目录碰巧可见。
- 两个模型 executor 删除了重复的权重查找和矩阵投影 helper、GPU 精度模板及 launch 参数。
  权重查找、标量类型/字节数和投影入口复用 runtime 的业务 helper；GPU 工作区仍只在初始化分配。
- attention 的 MMA/online softmax 数学主体由 `flash_attention_v2_kernel/attention_impl.cuh` 持有，
  V4 在自己的算子实现中提供异步 staging 策略，GQA 多 token 路径直接调用 V2 runtime；
  不再通过 common attention/tensor 文件承载算法，也不复制整份数学实现。

### 4.11 V1 / V2 / SM120 V4 的实现范围与优化记录

参考资料：

1. [FlashAttention V1：IO-aware exact attention](https://arxiv.org/abs/2205.14135)。
2. [FlashAttention V2：parallelism and work partitioning](https://arxiv.org/abs/2307.08691)。
3. 官方 [FlashAttention-4 CuTeDSL README](https://github.com/Dao-AILab/flash-attention/blob/e9d8944c855489e03c66262d07ba4c81d9d8dd31/flash_attn/cute/README.md)。
4. 官方 [SM120 forward 的硬件适配](https://github.com/Dao-AILab/flash-attention/blob/e9d8944c855489e03c66262d07ba4c81d9d8dd31/flash_attn/cute/flash_fwd_sm120.py)。

官方 SM120 路径明确包含 DGX Spark：与数据中心 Blackwell 不同，SM120/SM121 使用
`mma.sync.aligned.m16n8k16`，不是 `tcgen05`，共享内存预算按 99 KiB 处理。
本仓库据此实现**原生 CUDA 的 SM120/SM121 FA4 路径**，没有复制官方 CuTeDSL 源码，
也不宣称它是官方实现的逐行移植或完整功能等价替代。
本次范围为 dense forward、FP16/BF16、causal/noncausal 和 GQA；未实现反向、变长接口、FP8、
SM100/SM103 的数据中心专用 FA4 路径。

| 版本 | 实现 | 支持范围 |
|---|---|---|
| V1 | Q4/K16 分块，共享 K/V，FP32 stable online softmax 和 PV；不物化全局 N×N score | FP32/FP16/BF16，D≤256；保留原 head-major API，并增加公共 token-major attention API |
| V2 | sliced-Q warp 分工，Q/score/P/O 常驻寄存器，FP32 MMA 累加，同步 K/V staging | SM80+，FP16/BF16，D≤128，支持尾部维度/尾部 token |
| V4 | SM120/121 能力检查，16-byte `cp.async`，零填充及非对齐尾部回退，按形状选择异步单/双缓冲 | FP16/BF16，D≤128；Qwen ViT 和多 token prefill 使用此路径 |

单 token decode 保留 FP32 probability/PV 的专门分区 GQA kernel，不用通用大 Q tile 强行替换。
FP32、其他架构或不支持的模型形状由 backend 保留原有计算路径；直接请求不支持的 V2/V4 类型或设备会失败。

#### 单 kernel 调优：实际 Qwen 输入形状

同一进程预热后用 event 统计 20 次平均值。ViT 为 Q11008/H16/D72，非 causal；
prefill 为 Q2766/QH32/KVH8/D128，causal。数据为 FP16，未降低精度或减少 token。

| 路径/实验 | ViT attention | prefill attention | 结论 |
|---|---:|---:|---|
| V2，同步 staging，K32 / K64 | 19.169 ms | 2.577 ms | 比较基线 |
| V4，异步单缓冲，K32 / K64 | 14.165 ms | 1.757 ms | 异步加载有收益 |
| V4，异步双缓冲，K32 / K64 | 12.970 ms | 2.779 ms | D128 的 K64 双缓冲需要约 68 KiB，实测更慢，拒绝统一开启 |
| V4，K64 双缓冲 / K32 单缓冲 | **12.290 ms** | **1.539 ms** | 最终采用，分别缩短约 35.9% / 40.3% |
| V4，K64 单缓冲 / K32 双缓冲 | 19.068 ms | 1.834 ms | 均不如各自最终方案 |

另一次尝试把 inner causal mask 提升为 64-bit 运算后，K64 ViT kernel 退化到约 17.55 ms。
最终在主机入口检查索引范围，GPU 内层继续使用安全的 32-bit mask；保留溢出输入回归。
最终 D72 使用 Q128/K64、双缓冲、45056 bytes SMEM；
D128 使用 Q64/K32、单缓冲、17408 bytes SMEM。
这说明 tile、staging 数和整数代码生成都需要实测，不能仅凭“更新的架构”统一加大 tile。

复现调优（两个 dtype 的性能用例默认禁用，不将性能波动变成正确性测试失败）：

```bash
./script/run_test.sh --target //Omni-Runtime/kernels:kernels_tests \
  --test-filter '*FlashAttention*' --test-output all
bazel-bin/Omni-Runtime/kernels/kernels_tests \
  --gtest_also_run_disabled_tests \
  --gtest_filter='FlashAttentionV4Test/0.DISABLED_BenchmarksModelShapes'
```

benchmark 的 variant：2=V2；41=K32/K64 单缓冲；42=K32/K64 双缓冲；
43=D128 K32 双缓冲；44=D128 K32 单缓冲；45=D72 K64 双缓冲；46=D72 K64 单缓冲。

#### 模型结果与验证

同一默认输入、同一 pipeline 第二次生成，352 个 token（含 EOS），**两组均为当前 C++ event 口径**：

| 路径 | ViT | prefill | decode |
|---|---:|---:|---:|
| 改造后、接入 V4 前的事件基线 | 746.785 ms | 830.559 ms | 23587.2 ms |
| 最终 V4 路径，清理重建后的回归 | **592.418 ms** | **767.461 ms** | 24321.2 ms |

该次 ViT 缩短约 20.7%，prefill 缩短约 7.6%。decode kernel 未因 V4 改写，
观察到约 23.6–24.3 秒的运行波动；本轮**不能宣称 decode 提速或端到端总时延稳定降低**。
Nsight 中 GEMV 占全部 GPU kernel 时间约 84.3%，继续受到 FP16 权重流量限制，参见 4.8 的带宽分析。

最终实际模型 Nsight trace，而非仅算子 benchmark，确认：

- ViT 执行 `BlackwellAttentionImpl<half,80,64,2>` 54 次（两张图各 27 层），平均约 12.976 ms。
- 图像 prefill 执行 `BlackwellAttentionImpl<half,128,32,1>` 72 次（两次各 36 层），平均约 1.599 ms。
  另有文本-only 的短 prefill，不能把它们混入长 prompt 平均数制造“加速”。
- 上述 kernel 的 local memory 为 0；ViT/语言分别使用 128/149 registers per thread。
- 从首个 resize/preprocess 到最后 argmax，观测到 `cudaMalloc`、`cudaFree` 均为 **0 次**。

接入 V4 时的历史验证使用 4.8 的四条脚本命令，全部通过：

- 12 个测试标签（含一个兼容 alias），对应 **11 个实际测试程序、96 个启用用例**。
  另有 2 个按需启用的性能用例，不计入默认回归数量。
- 64-token 流式结果与 Python FP16 token/text 严格相等；复用图的完整 352-token 非流式结果也严格相等。
  额外检查首 token/单 token、正常中止、回调异常、拒绝重入、失败后再次生成及 GPU 内存预算不变。
- 流式单元测试覆盖 UTF-8 跨 token、四字节字符、非法/未完成字节替换、取消及异常。
- V1/V2/V4 与独立 CPU double-precision softmax 参考比较，覆盖 FP32/FP16/BF16、GQA、causal、
  奇数维度、tile 尾部、单 query、强负 logits 及非法分组/维度/溢出位置。
- Compute Sanitizer memcheck：10 个 attention/GQA 用例，**0 errors**。
  V1/V4 racecheck：5 个用例，**0 errors / 0 warnings**。
- 模型目录和平台无关的 kernel 编译工具中没有 CUDA/OpenCL API；模型也没有 kernel-runtime 类型或 launch geometry。

流式报告修复后的本轮回归：

```bash
./script/build.sh --clean
./script/build.sh
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
./script/run_test.sh --jobs 1 --test-output all
```

- 13 个测试标签（含一个兼容 alias），对应 **12 个实际测试程序、101 个启用用例**，全部通过。
  另有 2 个默认禁用的性能用例；数量按测试 XML 去重统计。
- 两轮流式输出均与 Python FP16 的 token ID、文本严格相等，无重复整段打印或预期外错误日志。
  对脚本 stdout 的带时间戳读取分别观测到 64 次、352 次增量到达，跨度约 4.44 秒、24.70 秒，
  验证的是没有换行时的实际交付，而不仅是回调次数。
- 图像尺寸、2766 个 prefill token、63/351 个 decode 计时步以及速率计算均有断言；
  独立单元测试覆盖默认值、实际计时步与总生成数的区别、零耗时。
- 脚本的 10 项参数/传输/退出码检查通过，覆盖输出模式、单/多目标、
  GoogleTest filter 转换及失败时不输出成功提示。

`TextStream` 迁移到公共组件后，再次执行上述 clean/build/模型/全量测试命令，全部通过。
4 个流式用例迁入独立的 `text_stream_test`，tokenizer 测试保留其原有 2 个用例；
当前共 14 个测试标签（含一个兼容 alias）、13 个实际测试程序、101 个启用用例，
另有 2 个默认禁用的性能用例。迁移未改变 UTF-8 处理、回调、取消、异常或模型输出行为。

### 4.12 算子归属整理与第二轮 CUDA 热点优化

#### 逐类扫描与取舍

本轮扫描所有设备算子实现，并先对真实的 Qwen3-VL 64/352-token 生成采集 Nsight CUDA trace。
修改前 GPU kernel 时间约 85.0% 为 `LinearGemvImpl`，4.1% 为 grouped decode attention。
不能把 attention 或未接入模型的算子微基准加速直接当作整模型加速。

| 算子类别 | 扫描结果与处理 |
|---|---|
| GEMV | 对齐路径已有 16-byte 向量访存和 warp 归约；128/256/512-thread block 扫描没有稳定实质提升，保留现有主路径和 FP32 累加顺序。新增非对齐/非整向量长度的 warp GEMV，不再依赖外部提供 GEMM grid |
| 通用 GEMM/linear fallback | 原 16×16 每线程单输出；新增每线程 4 输出的 16×64 register tile，复用左矩阵并共享两种 RHS 布局的数学实现。小行数保留 16×16，避免并行块数不足；模型多行投影仍由 backend 使用 cuBLAS/cuBLASLt |
| MHA、通用 GQA fallback、V1 | 删除逐 key 多次 block 同步归约，采用 V1 的 Q4/K16 分块 online softmax；两 lane 协作一个 QK dot；shared memory 按实际 head dimension 对齐后申请，而不是固定预留 D256 |
| grouped decode attention | K/V 使用 16-byte 合并加载；每个 lane 连续处理 4 个 V 元素，用 8-byte shared load 减少指令。保持每个输出维度沿 token 的 FP32 FMA 顺序及 FP32 probability/PV；未对齐输入明确走标量 split-decode |
| V2、V4、视觉 patch embedding | 保留 MMA、寄存器 Q/score/P/O 和已调优的 V4 tile/staging；将数学实现、异步搬运和 MMA 原语分别归入对应算子，不改成另一份低精度近似实现 |
| fused FFN | 两层投影均改为 warp-per-neuron，连续读取权重，复用一个 dot helper；保留 FP32 中间激活。自动生成合法 launch 和 shared-memory 大小，不要求调用方手填 |
| norm/adaRMS | 已采用 FP32 block 归约；保留 reduction scratch 的必要屏障和 Qwen RMSNorm 的 FP16 中间舍入 |
| add、gated add、flow step、SwiGLU、embedding、KV cache、gated query split | 保留合并访问/grid-stride 实现；模型 trace 中不是主要耗时，不以改变舍入、额外 kernel 或 speculative fusion 换取表面加速 |
| MRoPE、视觉位置/merge/bias/GELU、图像预处理、argmax | 保留与参考一致的位置布局、激活及 uint8 bicubic 舍入；argmax 保持最小索引 tie-break。没有把严格 resize 或 FP16 舍入替换成不等价的近似操作 |

这是一轮经过实测的热点优化，不是“所有 dtype、形状、设备均已达到理论最优”的声明。
例如 fused FFN 的单 CTA/row 和 shared-memory 容量仍限制小 batch/超大维度的适用性；
大型模型投影使用后端 BLAS，而不是强行改用通用 SIMT fallback。

#### 算子文件归属

- `flash_attention_kernel/flash_attention_kernel.cuh`：V1 分块标量 attention，MHA/GQA fallback 复用它。
- `gqa_attention_kernel/attention_define.h`：跨 attention 入口使用的参数及校验；
  `decode_attention.cuh`：单 token 分区计算、grouped decode 和 merge。
- `flash_attention_v2_kernel/attention_impl.cuh`：MMA/online softmax 数学主体和同步 staging；
  `mma_attention_runtime.h`：按类型、tile 和 shared-memory 大小生成 launch。
- `flash_attention_v4_kernel/flash_attention_v4_kernel.cuh`：Blackwell 异步 staging；
  V4 runtime 自己负责设备能力检查与 tile 选择，不让 V2 launcher 判断其他算子版本。
- `matmul_kernel/tensor_core.cuh`：矩阵乘的 FP16/BF16 MMA 原语，由 attention 和视觉 patch 投影复用。
- common 目录仅保留基础类型转换和归约工具。旧 attention/tensor 文件以及重复的 GQA MMA wrapper 已移除；
  Bazel 同步分发算子目录中的 `.h/.cuh`，确保 JIT 在 runfiles 内也能找到完整 include 链。

#### 单算子实测

DGX Spark GB10/SM121，NVCC `-O3 -std=c++20 -arch=sm_121`，非恒定输入，
预热后使用 CUDA event；测量期间不同时运行其他测试。下表比较相同 dtype、形状和数学操作，
不是包含 JIT/模型加载的 wall time，也不是与官方 FlashAttention 的比较：

| 算子/形状 | 修改前 | 修改后 | 说明 |
|---|---:|---:|---|
| FP16 grouped decode，QH32/KVH8/D128，KV257 | 20.645 us | 约 10.3–11.8 us | 对齐输入，128-token 分区 |
| FP16 grouped decode，同上，KV2816 | 40.538 us | 18.512 us | 约 2.19× |
| FP32 MHA，H8/T512/D128，noncausal | 2.894 ms | 0.906 ms | 约 3.19× |
| FP32 GQA fallback，QH8/KVH2/T512/D128 | 2.793 ms | 0.959 ms | 约 2.91× |
| FP32 V1，同 MHA 形状 | 1.902 ms | 0.939 ms | 约 2.03× |
| FP32 fused FFN，R64/I256/F512/O256 | 217.438 us | 61.729 us | 约 3.52×；不是 Qwen 模型当前调用的 FFN 路径 |
| FP32 GEMM，M128/K4096/N4096 | 2.751 ms | 1.881 ms | 约 31.7% 缩短 |
| FP32 GEMM，M8/K4096/N4096 | 0.401 ms | 0.404 ms | 保留小行数 tile，没有可宣称的稳定提速 |

一次被拒绝的调整：将 V2 同步 staging 改为每线程标量连续搬运后，
FP16 prefill attention 从约 2.577 ms 退化到 4.476 ms，ViT 从约 19.169 ms 退化到 20.873 ms。
最终恢复每线程连续 8 元素的 vector-friendly 搬运；回归得到 V2 ViT 19.117 ms、prefill 2.577 ms。
V4 最终 tile 同轮为 12.367 ms / 1.547 ms，BF16 各 variant 也通过。
不能仅凭看起来“更合并”的源码判断编译后的访存和吞吐。

#### 精度和边界验证

- 新增 FP32/FP16/BF16 GEMM 双 RHS 布局、M/N/K tile 尾部与非对齐指针回归。
- 扩展 GEMV 的长度 1/31/32/37/40/257、非对齐地址及输出尾部，使用空 launch options 验证 runtime 自行生成配置。
- grouped decode 覆盖 127/128/129/257 token 分区边界、2/4 组 GQA、causal 截断及非对齐输入，
  验证向量路径的合法分派，而不是仅测试对齐大矩阵。
- 新增 MHA head-major 奇数维度/尾 tile，以及 FP32/FP16/BF16 fused FFN 多行/非整 warp 维度；
  保留所有原有精度阈值和独立 CPU 参考。

#### 真实模型 trace 与脚本回归

同一默认输入、64/352-token 测试、相同 Nsight 参数，修改前后的 grouped decode 均调用 14904 次：
平均 kernel 时间从 **85.325 us 降至 57.213 us（缩短约 32.9%）**，
累计从 1271.690 ms 降至 852.701 ms。真实模型确实执行了优化路径，不是仅在微基准中生效。

复用图第二轮的 backend event 报告如下。它只是一组同条件观测，不应解释成消除了温度、频率和运行波动：

| 阶段 | 修改前 | 修改后 |
|---|---:|---:|
| ViT | 584.08 ms | 587.76 ms |
| prefill | 767.28 ms | 760.82 ms |
| decode，351 个计时步 | 24589.98 ms | 24359.77 ms |
| decode throughput | 14.27 tokens/s | 14.41 tokens/s |

本轮没有改写主 GEMV 的对齐数学路径，其 trace 累计时间反而波动约 +0.8%；
单组观测中整段 decode 仅缩短约 0.94%。因此**不宣称 ViT/prefill 稳定提速、
decode 数倍提速或整模型已达到硬件极限**。FP16 权重带宽仍占主导，见 4.8 的流量下界。
新 trace 中 V4 的 ViT/prefill kernel 调用数仍为 54/72，从首个预处理到最后 argmax，
CUDA runtime 的 `cudaMalloc/cudaFree` 调用仍为 0。

验证仍使用用户指定的三条命令及全量脚本：

```bash
./script/build.sh --clean
./script/build.sh
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
./script/run_test.sh --jobs 1 --test-output all
```

- 模型的 64/352-token ID、文本、流式回调、取消/异常恢复和图复用检查通过。
- 默认回归共 14 个标签、13 个实际测试程序、108 个启用用例；另有 2 个默认禁用的性能用例。
- Compute Sanitizer memcheck：22 个 attention/GQA/MHA/GEMV/GEMM/FFN 用例，0 errors。
  racecheck：15 个用例，0 errors / 0 warnings。恢复 V2 staging 后，额外对 V2/GQA 的
  4 个用例分别重新执行 memcheck/racecheck，均为 0 errors / 0 warnings。
- 两个默认禁用的 FP16/BF16 V4 tile benchmark 显式启用后通过。
- 模型及 inference 层没有 CUDA/OpenCL API 泄漏，旧 common attention/tensor include 无残留。

### 4.13 常驻 decode 状态、后端执行计划与 Blackwell GEMV

#### 执行架构

语言层通过同一个 `LanguageLauncher` 描述计算，避免复制整套 prefill/decode 数学链路：

- Prefill 直接调用统一后端接口；初始化时将单 token decode 描述为固定的
  `ComputeOperation` 序列，提前绑定权重、工作区和算子参数。
- `Backend::CreateExecutionPlan` 接收后端无关的计算序列和结果回传区域。
  CUDA 实现在 `backend_cuda.*` 中完成 JIT 准备、私有非阻塞 stream、
  CUDA Graph capture/instantiate/upload、固定 pinned readback 和计时 event 的创建。
  准备阶段只编译，不执行模型，也不修改 KV cache；所有资源在初始化时创建。
- Kernel 的 JIT `LaunchScope` 仅在后端内部设置准备模式或 capture stream，
  并由执行计划保留 kernel 的共享所有权，避免另一个后端初始化并清理 JIT 缓存后，
  已捕获的 graph 引用被卸载的 kernel。退出 scope 会恢复先前状态。
- Prefill 完成后，首次采样结果已在 GPU 上。每个 decode step 直接将该设备 token
  用作 embedding 输入；graph 尾部 argmax 覆盖 token，推进三轴 position 和 KV 写入位置。
  CPU 仅维护容量检查所需的镜像计数，不再每步构造并上传 token/position 容器。
- 每步在计算终点记录 event，再异步下载 4-byte token，最后只等待一次 stream。
  计算 event 终点仍在下载之前；tokenizer、流式回调和回传不计入 decode event。
  EOS、达到生成长度、取消和回调异常都不会额外执行下一步。

框架 graph 与 CUDA Graph 是不同层次的复用。新 decode 热路径不再重新构造计算描述、
查找字符串权重、格式化 JIT 源码或逐 kernel 提交。OpenCL 的完整推理和 prepared sequence
已在后续双后端实现中补齐，见 4.14；它不等同于 CUDA Graph replay。

#### 动态长度与精度

CUDA Graph 固定的是存储地址和最大 launch 容量，不是实际 KV 长度：

- 专用 grouped/split decode attention 从设备状态读取实际长度；超过有效长度的 CTA
  整块返回，merge 仅遍历有效 partition，既不读取过期 partial，也不改变有效归约顺序。
- 当前使用配置容量对应的固定 partition grid，而非每 128 token 捕获一份 graph。
  这使初始化资源保持简单，并避免长度节点逐步更新；极大容量、很短上下文时会有
  无效 CTA 的调度成本，不能宣称该策略对所有上下文长度都最优。
- 新增 `ResidualRmsNorm` 保留 residual add 和 RMSNorm 内部的 FP16 舍入。
  `NormalizeRopeCache` 将 Q/K head norm、MRoPE、KV 写入融合，复用原归约和旋转数学，
  保留归一化、权重相乘、sin/cos、两个旋转乘积及最终相加的低精度舍入点。
- `Q=1` 继续使用专用 decode attention，ViT/prefill 的多 query FA4 路径不变。

真实模型 trace 中，每 token 的 kernel 数由 **652 降至 509**，减少约 21.9%。
Warm decode 区间没有 H2D，也没有 CUDA runtime malloc/free；每步仅一次 4-byte D2H、
一次 graph launch 和一次结果等待。API 的等待时长包含 GPU 计算，不能另算成同步浪费。

#### GEMV 实验与保留方案

`BlackwellGemvImpl<T, kInner>` 针对 SM120/121、对齐低精度权重和 K=4096/12288，
使用 16-byte vector load、固定形状循环展开及
`ld.global.L1::no_allocate.L2::256B` 流式权重读取，避免污染输入向量缓存。
仍然采用 FP32 FMA、原每 lane 累加和 warp 归约顺序，不改变 FP16 模型精度。
这是一条针对 GB10 调优的 CUDA GEMV 路径，不是将 GEMV 包装成 Tensor Core GEMM。

GPU 内部 event 的独立实验使用非恒定输入、64 MiB cache flush、轮换测试顺序、
5 次预热、50 次计时及中位数；不将 CPU JIT cache-hit dispatch 空隙算成 kernel 收益：

| FP16 GEMV K x N | 原通用 kernel | Blackwell 候选 | 观察 |
|---|---:|---:|---|
| 4096 x 4096 | 0.1576 ms | 0.1410 ms | 约 10.5% 缩短 |
| 4096 x 1024 | 0.0532 ms | 0.0450 ms | 小矩阵对 cache flush 和调度敏感 |
| 4096 x 12288 | 0.4503 ms | 0.4247 ms | 约 5.7% 缩短 |
| 12288 x 4096 | 0.4595 ms | 0.4287 ms | 约 6.7% 缩短 |
| 4096 x 151936 | 5.1873 ms | 5.1853 ms | 无稳定收益，LM head 保留通用路径 |

还比较了 128/256/512 threads、128/256-byte L2 预取、不同 L1 cache 策略。
没有把所有微小差异都变成新的运行配置。gate/up/SwiGLU 融合候选在真实 FFN 形状上
没有收益，已移除，没有在模型中保留更复杂但更慢的实现。

默认禁用的 `MatmulKernelTest.DISABLED_ProfilesBlackwellStreamingGemv` 可显式运行：

```bash
bazel test -c opt //Omni-Runtime/kernels:kernels_tests \
  --test_filter='*DISABLED_ProfilesBlackwellStreamingGemv' \
  --test_arg=--gtest_also_run_disabled_tests --test_output=all
```

该回归 benchmark 的 `dispatch_plus_*` 输出包含主机提交空隙，与上表 GPU 内部
event 的 kernel-only 结果不是同一计时口径，不能直接混用。

#### 整模型对照与验证

同机串行运行原代码和最终代码，各取三次独立调用中的第二轮完整生成；
每轮均为 2766 prefill token、352 generated ID、351 decode step，精度断言全部通过：

| Warm 指标 | 原代码三次中位数 | 最终代码三次中位数 |
|---|---:|---:|
| ViT | 583.55 ms | 593.82 ms |
| Prefill | 760.07 ms | 765.53 ms |
| Decode | 24123.47 ms | 23432.96 ms |
| Decode throughput | 14.55 tokens/s | 14.98 tokens/s |

本组 decode 时间缩短约 **2.86%**，吞吐增加约 **2.95%**。最终三次 decode 为
23509.06/23394.47/23432.96 ms。更早的单次原代码也曾达到 15.37 tokens/s，
其他新代码运行约 15.07–15.40 tokens/s；时钟、温度和共享系统负载会影响结果，
因此不能把少量百分点观测表述为数倍加速或硬件峰值证明，也不宣称 ViT/prefill 提速。
主要 FP16 权重字节数没有改变，带宽边界仍适用。

- 用户要求的 clean build、build、模型测试，以及全量测试脚本均通过。
- 全量为 14 个标签、13 个实际程序、120 个启用用例；另有 3 个默认禁用 benchmark。
- 模型新增 decode 后取消、重新生成的状态复位验证；64/352-token ID、文本和流式行为
  仍严格匹配 FP16 reference，不修改 reference 来迁就新实现。
- FP16/BF16 覆盖动态长度 127/128/129/255/256/257/384/385、固定容量中的无效 NaN
  partial、状态复位、grouped/scalar split；融合算子覆盖 FP32/FP16/BF16 和别名输出。
- 定向 memcheck/racecheck 各 10 个 kernel 用例，0 errors / 0 hazards；
  后端 graph 准备、重复执行及 JIT cache 清理后的生命周期测试额外 memcheck 通过。
- 模型、inference、memory 和 graph 层扫描未发现 CUDA/OpenCL API。

推测解码作为独立算法阶段评估：候选多 token 的 target 验证有机会分摊权重读取，
但还需要指定 draft 模型、测量接受率和验证 KV 回滚；多 token GEMM 的浮点路径也需要
重新证明严格 greedy 对齐。本轮没有引入 draft、量化或改变采样行为，不能据此承诺收益。

### 4.14 CUDA / OpenCL 统一算子与完整推理

#### 分层与扩展约定

- 模型只描述 `ComputeOperation`，持有 `Buffer` / `DeviceView`，不包含平台 API、
  `cl_mem` 转换、设备指针算术或后端选择分支。`DeviceView` 保存分配句柄、字节偏移、
  可访问长度和 backend kind；嵌套 slice 检查边界，`Buffer` slice 保留源存储的生命周期。
  不能对 OpenCL buffer 使用 `device_address()`，必须传递 view。
- `OperatorRuntime<T, Operation>` 是统一编译和执行入口。各算子目录中的
  `CudaOperator` / `OpenCLOperator` 特化负责验证、launch geometry、源码和算法选择。
  原来 `backend_cuda.cc` 中的 19 个 `LaunchCompute` 已全部移入对应 Runtime。
  CUDA Linear 使用后端持有的 BLAS 服务，不将服务句柄暴露给模型。
- Backend 仅管理上下文、内存、stream/queue、event、编译执行服务和执行计划。
  OpenCL executor 使用 backend 创建的同一 context/device/queue；
  参数绑定显式区分 buffer、标量和 local memory，buffer 传入基址及 64-bit 字节偏移，
  不再假设所有参数都是原生指针。
- OpenCL program 按源码、dtype 和编译选项缓存。prepared sequence 为每条命令创建独立
  kernel 实例并冻结参数，避免共享 mutable kernel 参数。准备阶段不执行计算或修改状态；
  运行时复用设备 token/position/KV state，异步提交命令，回传结果后只等待一次。
  所有绑定存储必须比计划活得更久，同一计划的 `Run` 由调用方串行执行。
- GB10 未提供 `cl_khr_command_buffer`，因此 OpenCL prepared sequence 仍逐命令入队，
  **不是硬件 graph replay**。CUDA 继续使用后端内部的 CUDA Graph，不修改业务链路。

公共计算描述覆盖 28 个操作：embedding、各类 norm、MRoPE、KV cache、attention、
linear、elementwise、采样、图像处理、vision patch/position/merge，以及 AdaRmsNorm、
FlowStep、GatedAdd、SplitGatedQuery、FusedFfn、MultiHeadAttention。
每种语义均提供 CUDA/OpenCL 实现和 FP32/FP16/BF16 对照测试。
FA V1/V2/V4 是 CUDA attention 算法版本，不是三个不同的业务操作；
OpenCL 通过同一 Attention 语义选择 portable、tensor prefill 或专用 decode，
不复制并伪装成 CUDA FA4。

新增后端只需实现 backend 资源/执行服务、注册工厂，并提供其算子 Runtime；
新增操作则扩展计算描述、对应 Runtime 和精度/边界测试。两者都不需要修改模型数学链路。
目前构建同时包含 CUDA 与 OpenCL；这里不宣称已提供不依赖 CUDA toolchain 的独立 OpenCL 构建。

#### 数值与设备优化

GB10 的 OpenCL 设备没有声明 native half、BF16 或 matrix 扩展，因此 portable 路径采用
16-bit 存储、FP32 累加和显式 round-to-even 转换，保留 FP16/BF16 的运算舍入边界。
默认不启用 fast-relaxed-math；设备支持时启用 correctly-rounded divide/sqrt。
没有将 OpenCL 请求转交 CUDA/cuBLAS 的隐藏回退。

NVIDIA capability 检测启用 `.cl` 内的 inline PTX 转换、shuffle、`mma.sync` 优化，
实际仍由 `clBuildProgram` / `clEnqueueNDRangeKernel` 编译和执行。它不是 tcgen05，
也不要求 `cl_khr_fp16`。其他厂商使用 portable 实现，性能不能从 GB10 的结果外推。

保留的优化包括：

- Tensor Core tiled GEMM、paired vision patch embedding 和 tiled prefill attention。
  大型对齐 GEMM 使用实测更优的 128×64 tile；小矩阵/尾部保留原通用 tile。
- Batch-one GEMV 使用 packed storage 与 warp 归约；仍主要受 FP16 权重带宽限制。
- 单 token attention 使用 128-token split；D128、GQA ratio=4 的低精度路径采用 grouped
  QK tensor 运算，但 softmax/PV 保持 FP32，merge 只读取有效分区。
  固定 grid 从设备读取真实长度，支持跨分区增长、取消和状态复位。
- 图像 resize 的系数精度扫描在 host 端只计算一次，CUDA/OpenCL 共享相同算法，
  不再由每个 OpenCL workgroup 重复扫描全图。
- 算子计算默认异步，不在每个算子末尾执行 `clFinish`；
  `OMNI_OPENCL_PROFILE=1` 可启用逐入口 event 汇总，默认关闭以免影响测量。

#### 选择后端与验证

Qwen3-VL 通过 `Qwen3VLPipeline::Init(conf_path)` 加载运行参数与其引用的算子图：

- `Omni-Runtime/inference/conf/qwen3_vl_8B_f16.conf`：唯一入口，包含权重、精度、
  context、生成参数和 `graph_path=qwen3_vl.yaml`。
- `Omni-Runtime/inference/conf/qwen3_vl.yaml`：完整算子图及 `backend: cuda`。

`graph_path` 优先相对 `.conf` 所在目录解析，也支持绝对路径及已有的仓库相对路径
解析规则；缺少该字段、文件不存在或 YAML 无效都会在加载权重前显式报错。
示例不再单独接收图路径或读取独立的图路径环境变量。

后端由所引用的 YAML 唯一决定。运行 CUDA 时设为 `backend: cuda`，运行 OpenCL 时
设为 `backend: opencl`，两者使用相同的 `.conf` 入口和运行命令。
`OMNI_BACKEND` 与脚本的 `--backend cuda|opencl|auto` 不覆盖 Qwen3-VL 的 YAML；
这些脚本选项仅对其他测试保留原有行为。未知或不可用的 YAML 后端显式报错，不静默切换平台。

算子图语法、层重复、条件/并行节点和后端边界见
[当前 YAML 架构](architecture_and_parallel_execution.md#当前架构yaml-算子图)。

```bash
bazel clean
./script/build.sh
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
# 修改 YAML 的 backend 后，使用同一条命令验证另一后端
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
./script/run_test.sh --jobs 1 --test-output errors
```

注意清理命令是 `bazel clean`，不是 `bazel --clean`。
若要彻底清理 Bazel 缓存并重新编译全部目标，可用 `./script/build.sh --clean`。

算子测试覆盖两后端的三种 dtype、非零/嵌套字节偏移、输出 canary、矩阵尾 tile、
GQA 分区边界、prepared sequence 不提前执行，以及动态长度/状态重置。
模型测试不改变 token/text 断言阈值：64-token 与完整 352-token ID、文本、流式、
取消/恢复及重入/异常行为均须通过。352 个输出中首个来自 prefill，decode 计时为 351 步。
本轮重新运行 `tools/qwen3_vl_infer.py` 并逐字节比对，其完整输出仍与原参考文件相同；
没有为适配新后端而改写 reference。

#### 本机整模型实测

GB10 / SM121，同一模型、图片和 prompt，串行运行且关闭逐算子 profiling；
下表取同一进程第二轮、完整 352-token 输出的 backend event 时间，不包含加载/JIT：

| 阶段 | CUDA | OpenCL |
|---|---:|---:|
| ViT | 583.72 ms | 1307.80 ms |
| Prefill，2766 tokens | 752.32 ms | 1253.80 ms |
| Prefill throughput | 3676.64 tokens/s | 2206.10 tokens/s |
| Decode，351 步 | 22175.08 ms | 23030.41 ms |
| Decode throughput | 15.83 tokens/s | 15.24 tokens/s |

OpenCL decode 达到本轮 CUDA 吞吐的约 96.3%，但 ViT/prefill 仍分别慢约 2.24×/1.67×，
不能宣称所有阶段性能相同或达到硬件峰值。单 token 的主要权重流量没有减少；
后续多 query 阶段仍可继续调优 tensor attention 和矩阵 staging/流水化。

独立 profiling 定位并验证了两个实际热点：图像预处理由每张约 645 ms 降至约 6.85 ms；
同一组 430 个 decode 步中，attention 从约 9103 ms 降至 grouped+merge 合计约 934 ms。
大型 GEMM 的 128×64 tile 在五个实际形状上相对 64×64 实测提升约 1.44–1.98×，
并检查 FP16/BF16 尾部与偏移。上述局部收益不应相乘后当作整模型收益。

最终 clean build、再次 build、CUDA/OpenCL 示例及全量脚本均通过：
全量为 14 个标签、13 个实际测试程序、195 个启用用例，其中 kernel 用例 129 个；
另有 3 个默认禁用 benchmark。CUDA 定向 memcheck 9 个 kernel 用例及 1 个执行计划用例
均为 0 errors，racecheck 5 个用例为 0 errors / 0 warnings。
这些 sanitizer 结果仅针对 CUDA；OpenCL 使用独立参考、两后端对照、偏移/canary 和
resident-state 回归，不将 CUDA sanitizer 结果当作 OpenCL 的内存检查。
