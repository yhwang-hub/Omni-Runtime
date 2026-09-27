# Omni-Runtime Bazel 构建与测试指南

本文说明如何直接使用 Bazel 完成 Omni-Runtime 的编译和测试，以及如何使用 `script/build.sh` 与 `script/run_test.sh`。

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

该输出与 `tools/qwen3_vl_infer.py` 的参考推理结果保持一致。
