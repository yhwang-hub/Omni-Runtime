# 架构梳理与 Qwen3-VL 并行执行方案

## 当前架构：组合式模型执行

模型不再继承 `VLMModelBase`，也不保留 `ModelPipeline<T>` 或量化目录下的类型别名。
Engine、已加载模型、设备执行器与 Qwen 组件分别承担生命周期、只读资源、运行时和输入语义，
底层 YAML、ComputeNode、CUDA/OpenCL 算子、精度与驻留 decode 算法保持不变。

```text
InferenceEngine → ModelRegistry / builtin_models
                          |
        Model::Load / Model::CreateExecutor
                /                     \
       Qwen3VLModel                Qwen35Model
                \                     /
      配置 / 已验证权重 / tokenizer / 图模板
                          |
       Qwen3VLExecutor / Qwen35Executor
         状态 + workspace + 执行计划
                          |
      QwenProcessor + AutoregressiveRunner
                          |
            OperatorGraph / ExecutionPlan
                          |
                 CUDA / OpenCL
```

- `inference/engine` 是统一调用入口，拥有模型及其设备执行器，通过注册表按配置装配。
  重载时先销毁旧执行器，析构时先释放执行器、再释放模型；原子作用域守卫拒绝并发/回调重入，
  失败和异常退出均释放占用。模型在执行器存活期间不可重新 Load，也不可绕过 Engine 并发执行。
  访问错误和后端信息的 getter 不应与执行操作并发。
- `models/model.h` 是加载与创建执行器的最小接口；具体模型只拥有已验证配置、权重、
  tokenizer 和图模板，不再保存当前请求 KV/recurrent 状态、workspace 或执行计划。
- `inference/executor` 定义设备执行器协议与任务输入输出，不依赖注册表或任何具体模型。
  具体执行器负责数据传输、vision/prefill/decode、状态复位及图绑定。
- `inference/generation` 仅管理 token 循环、EOS、UTF-8 输出和成功取消。
  首 token 仍来自 prefill，decode 计时与计数不包含该 token。
- `models/components/qwen_multimodal` 负责现有单图 ChatML、三轴 mRoPE；
  `models/components/qwen_vision` 负责视觉元数据和图维度。
  这些是 Qwen 组件而非所有 VLM/VLA 的公共契约；两个模型保持原有图像解码和融合路径。
- `models/qwen3_5` 按 GGUF `qwen35` 架构组织。逐张量权重编码与模型身份分离；
  所需权重形状在上传前校验。旧 `models/qwen3_8_nvfp4` 目录及两个 Pipeline 兼容头已删除；
  部署名称只保留为注册表中的配置别名，两个示例 target 名称不变。
- `inference/model_loader` 统一 GGUF 原编码上传与浮点转换，不含 Qwen 名称和 shape 规则。
  各模型的 `weight_schema` 仅校验元数据、名称、形状、编码及 scale；独立 Bazel target 不执行 GPU 上传。
- `ModelRegistry` 显式注册 factory，不依赖静态初始化副作用，不自动初始化 GPU。
  `qwen3vl`/`qwen3_vl` 和 `qwen35`/`qwen3_5`/`qwen3_8_nvfp4` 为相应架构别名。
- 两模型明确区分请求状态与执行 context。Qwen3-VL 的 serializer 只在加载时存活，
  Executor 只读引用已加载的模型数据，不再拥有另一份权重。context 中的执行计划先于
  其绑定的状态、workspace 和权重销毁。当前仍为单实例单请求，不宣称支持 continuous batching。
- 公共任务契约还支持 tensor 观测、状态及动作块。非 token 动作模型无需 tokenizer、
  EOS 或 decode hook；host fake 测试验证这条扩展边界，不代表已部署真实 VLA checkpoint。

配置保留现有两个 `.conf` 与 backend 选择规则。未知字段和非贪心采样设置明确报错，
旧 `gpu_layer_count`、`image_min_tokens`、`kv_page_tokens` 仅为兼容提示并输出警告。
没有引入随机采样、分页 KV、新的 workspace 分块策略或 kernel。

```bash
./script/run_test.sh --target //Omni-Runtime/inference:inference_engine_test
./script/run_test.sh --target //Omni-Runtime/inference:weight_loader_test
./script/run_test.sh --target //Omni-Runtime/inference:operator_graph_test
./script/run_test.sh --target //Omni-Runtime/models/components/qwen_multimodal:processor_test
./script/run_test.sh --target //Omni-Runtime/models:registry_test
bazel clean
./script/build.sh
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
./script/run_test.sh --target //example:qwen3.8_27B_nvfp4_inference_test --test-output all
./script/run_test.sh
```

扩展模型时实现 Model 并注册 factory；根据任务组合自己的 processor/runner。
只读权重可作为后续跨会话共享的边界，但当前绑定具体 Buffer 地址的 ExecutionPlan 不可
直接跨请求共享；分页 KV、多媒体输入和真正的多会话调度需要另外实现和验证。

### 与原目录方案的对齐

```text
models/
  model.h / model_define.h
  registry.h / registry.cc / builtin_models.cc
  components/
    qwen_multimodal/  processor.* / processor_define.h
    qwen_vision/      config.* / config_define.h / graph_binding.*
  qwen3_vl/
    config_define.h / model_define.h
    model.* / weight_schema.* / executor.*
    graph_binding.cc / runtime.cc
    graphs/model.yaml / model_test.cc
  qwen3_5/
    config_define.h
    model.* / weight_schema.* / executor.*
    graph_binding.cc / runtime.cc
    graphs/model.yaml / model_test.cc
inference/
  engine/inference_engine.*
  executor/model_executor.h / executor_define.h
  model_loader/weight_loader.* / weight_define.h
  generation/autoregressive_runner.*
  inference_graph/operator_graph* / operator_graph_test.cc
```

两模型使用相同的职责布局，而不是仅重命名文件：`model` 加载并装配只读资源，
`weight_schema` 校验，`graph_binding` 只构造绑定/节点，`runtime` 分配与复位存储，
`executor` 执行设备计算与请求流程。两个 YAML 已原样移入相应 `graphs/`，
原 `.conf` 入口通过相对路径引用，后端选择规则不变。

原草案中的 `config.*` 按本仓库规范落为 `config_define.h`；
两模型相同的 processor 放在共享组件，而不创建重复的转发类。
两模型的 State/Context 和 workspace 均只由执行器使用，统一作为 Executor 私有嵌套结构体；
分配、复位、dtype 与权重查找使用私有成员函数，`runtime.cc` 仅按职责拆分成员函数实现。
Qwen3-VL 已移除独立的 `runtime.h`、`runtime_define.h` 与 Runtime 聚合包装；
真正被模型加载与执行器共享的只读数据仍放入 `model_define.h`。
计时器和算子图不可移动，因此 Qwen3-VL 用独占 context 指针管理其整体生命周期，
重新初始化时先释放旧 context/执行计划，再替换请求状态，避免旧计划引用已释放的 KV。
没有为目录对称而创建空文件，也没有为了单测保留公开的运行时状态接口。
通用算子图测试已从模型目录迁出；新增 loader 精确转换、混合权重 schema 和 Engine 析构顺序测试。

公共 Executor/Loader/Registry 的 Bazel 依赖不包含具体模型；只有 `builtin_models` 负责装配。
当前不创建没有实现的通用分页内存规划、快照、动作 runner 或多会话 scheduler 占位层。

### 本轮运行时封装验收

- Qwen3-VL 的语言/视觉 workspace、请求状态和执行上下文全部收拢为 Executor
  私有嵌套结构体，删除 `runtime.h`、`runtime_define.h` 与独立 Runtime 聚合包装。
  初始化直接装配私有资源，分配/复位/dtype/权重查找为私有成员函数，
  不再传递可以被外部修改的运行时聚合对象，也不复制已加载模型引用。
- 原直接访问运行时结构体的单测改为公开 Executor 的未初始化和重复初始化失败契约；
  真实推理保留精确输出、重复请求内存稳定、取消后状态恢复断言，
  并补充取消/恢复前后的 prompt token 与取消结果内存一致性断言。
  没有增加 friend 或测试专用状态访问接口。
- 按指定顺序完成 `bazel clean`、默认 `build.sh`、两个指定模型命令及默认 `run_test.sh`：
  63 个目标构建成功，模型分别 5/5、3/3 通过，全量实际执行 22/22 个目标并全部通过。
  3 个默认禁用 benchmark 另行显式运行，全部通过。相关 C++ 格式和补丁检查通过。
- Qwen3-VL 的 64/352-token fixtures、Qwen3.8 的 399-token fixture 和全文断言均通过。
  未改变权重、图 YAML、kernel、运算顺序、精度或 Qwen3.8 的 512-token prefill 分块。

本轮改前/改后均使用同一 GB10、输入、权重与配置，两个模型独立串行运行。
Qwen3-VL 使用同进程复用请求的稳态数据：

| 模型/阶段 | 本轮改前 | 本轮改后 |
|---|---:|---:|
| Qwen3-VL 复用 vision | 593.94 ms | 592.70 ms |
| Qwen3-VL 复用 prefill | 3666.42 tokens/s | 3645.27 tokens/s |
| Qwen3-VL 复用 decode | 15.79 tokens/s | 15.80 tokens/s |
| Qwen3.8 vision | 2091.30 ms | 2083.46 ms |
| Qwen3.8 prefill | 581.60 tokens/s | 580.16 tokens/s |
| Qwen3.8 decode | 12.06 tokens/s | 12.03 tokens/s |

prefill 差异小于 0.6%，decode 差异小于 0.3%，未观察到实质性性能回退。
这些是指定输入下的本机回归数据，不是任意输入逐 token 一致性或跨硬件性能保证。

### 目录与生命周期对齐验收（上一阶段记录）

- 按上方命令顺序执行 `bazel clean`、默认构建、两个指定模型命令及默认全量测试。
  全部 63 个目标构建成功；两个模型分别通过 5/5、3/3 个用例；
  全量实际执行 22/22 个测试目标并全部通过，没有复用测试结果缓存。
- 默认禁用的 3 个 Blackwell benchmark 另行显式运行，3/3 通过，未修改启用状态。
- 保留 Qwen3-VL 的 64/352-token fixtures、Qwen3.8 的 399-token 精确 fixture
  和全文断言；流式、取消、异常日志、重复请求与失败恢复断言全部通过，未放宽输出要求。
- 新增并通过执行器先于模型释放/重载、executor 创建与初始化失败、CUDA/OpenCL
  权重逐位转换及原编码上传、混合权重 schema 无设备分配等契约测试。
- Bazel 实际依赖查询确认：Executor、Loader 不依赖模型层，通用 Registry
  不依赖具体模型；图 YAML 与原版本逐字节一致。变更 C++ 通过 clang-format，
  补丁通过 `git diff --check`。

该阶段性能先与改前即时基线比较，再在独立原版工作树运行 `8da5ff2`，随后运行对齐后版本。
两版使用同一 GB10、权重、图像、prompt 和配置，模型命令串行执行，不与其它本轮 GPU 测试并行。
以下为相邻原版/当前版本的实测；Qwen3-VL 采用同一进程内复用请求：

| 模型/阶段 | 原始版本 `8da5ff2` | 对齐后版本 |
|---|---:|---:|
| Qwen3-VL 复用 vision | 594.59 ms | 595.56 ms |
| Qwen3-VL 复用 prefill | 3624.69 tokens/s | 3618.70 tokens/s |
| Qwen3-VL 复用 decode | 15.80 tokens/s | 15.79 tokens/s |
| Qwen3.8 vision | 2090.24 ms | 2091.73 ms |
| Qwen3.8 prefill | 580.48 tokens/s | 578.56 tokens/s |
| Qwen3.8 decode | 12.01 tokens/s | 12.06 tokens/s |

prefill 差异小于 0.4%，decode 分别约 -0.06% 和 +0.42%，未观察到实质性回退。
该阶段较早即时基线的 Qwen3.8 decode 为 12.18 tokens/s，对齐后独立重复测量为
12.01–12.06 tokens/s；相邻原版也为 12.01 tokens/s，不能把跨时段差异直接归因于重构。
这是指定输入与本机条件下的回归结论，不是跨硬件或任意输入的性能/输出保证，
也不代表已达到历史 llama.cpp 绝对性能目标。

### 上一轮模型接口重构验收（历史记录）

- `bazel clean` 后默认 `./script/build.sh` 构建全部 51 个目标成功。
- 两个上述指定推理命令分别通过 5/5 和 3/3 个用例。
- 默认 `./script/run_test.sh` 实际执行并通过全部 19 个测试目标，无测试结果缓存复用。
  覆盖 backend、CUDA/OpenCL JIT、kernel、graph、内存、tokenizer、processor、executor、
  registry 及两个完整模型推理。
- 默认禁用的 3 个 Blackwell benchmark 用例也单独显式运行并全部通过，未修改其启用状态：
  `bazel test -c opt //Omni-Runtime/kernels:kernels_tests --test_filter='*DISABLED_*' --test_arg=--gtest_also_run_disabled_tests --test_output=all`。
- 没有放宽原有模型文本、token、流式、取消、异常日志和状态恢复断言。
  Qwen3-VL 保持原来的 64/352-token fixtures；27B 额外逐项比对改前版本的全部
  399 个 token（含 EOS），同时保留 llama.cpp 全文对照。
- 所有变更 C++ 文件通过仓库 clang-format 检查，`git diff --check` 通过。

GB10 同机对照（原始版本 `8da5ff2`，同一图像、prompt 和配置）：

| 模型/阶段 | 原始版本 | 重构后 clean build |
|---|---:|---:|
| Qwen3-VL 复用 vision | 597.53 ms | 594.77 ms |
| Qwen3-VL 复用 prefill | 3720.95 tokens/s | 3762.20 tokens/s |
| Qwen3-VL 复用 decode | 15.74 tokens/s | 15.72 tokens/s |
| Qwen3.8 vision | 2100.90 ms | 2085.07 ms |
| Qwen3.8 prefill | 584.95 tokens/s | 582.61 tokens/s |
| Qwen3.8 decode | 12.13 tokens/s | 12.10 tokens/s |

decode 差异约 0.1%–0.3%，小于本轮原始版本重复测量的波动；未观察到实质性性能回退。
这些是本机实测，不是跨硬件性能承诺。首次 vision 会受到 JIT/cache 状态影响，
不以单次冷启动与已预热运行比较得出性能结论。

## 历史架构：YAML 算子图与 VLM 基类

以下保留算子图引入过程的历史记录；其中的 VLM 基类和旧目标已由上述实现取代。

### 分层与运行路径

```text
qwen3_vl_8B_f16.conf
权重路径、dtype、context 容量、graph_path
                         |
                  qwen3_vl_8B_f16.yaml
        backend、算子、buffer 连线、repeat/parallel/when
                         |
                 VLMModelBase
       生命周期 / tokenizer / prompt / 流式生成
                  /                 \
         Qwen3VLPipeline         Qwen38Pipeline
       FP16 / KV / DeepStack     NVFP4/Q8 / recurrent state
                  \                 /
            模型权重映射 / shape 绑定
                         |
             OperatorGraphConfig::Append
          每条 op -> 一个 ComputeNode，带显式依赖
                         |
           backend::CompileComputeGraph
       校验依赖与内存冲突；lower 并行 Linear 分支
                  /                 \
      OperatorGraph / graph::Node    ExecutionPlan
        vision / embedding / prefill   常驻 decode
                  \                 /
                 Backend::Compute
                CUDA / OpenCL kernels
```

- `inference/inference_graph/operator_graph.*` 是模型无关的 YAML 展开、强类型字段绑定
  和算子节点执行层；复用已有 `ComputeOperation`，不重复定义一套算子接口。
- `models/qwen3_vl/qwen3_vl_graph.cc` 只负责 GGUF shape、workspace、KV slice、
  DeepStack 权重和张量绑定，不再手写 attention/FFN 的调用顺序。
- `models/vlm_model_base` 提供两个模型共同继承的 `VLMModelBase`，统一初始化状态、
  分词器加载、ChatML prompt 展开、三轴 mRoPE 位置、token 校验、流式输出、EOS、
  取消与重入保护。作用域守卫保证退出生成时恢复状态，不吞掉异常。
  `VLMConfig` 收拢公共语言模型维度；共享 helper 负责视觉元数据、图维度绑定、
  YAML section 展开与设备 buffer 分配计数。
- `qwen3_vl_generation.cc` 只保留 vision embedding 拷贝、prefill 传输与计时、
  FP16 KV/位置推进；流式 decode 循环由基类统一实现。CPU/设备传输边界不冒充数学算子。
- `qwen3_vl_pipeline.cc` 和 `qwen3_8_nvfp4_pipeline.cc` 提供模型专用初始化与生成 hook。
  FP16 类型转换/DeepStack 与 NVFP4/Q8 权重存储/混合 recurrent 状态不强行合并；
  保留各自的 YAML、prefill 分段、算子次序、精度与常驻 decode plan。
  FP16 权重注册表采用有序扁平清单，删除重复的逐层字段与加载分支；
  权重上传的实际顺序不变，共享 embedding/output 权重只保留一份。
  旧的 vision/prefill/language/decode GPU executor 已删除，没有保留第二条硬编码执行路径。
- decode 在初始化时由 YAML 构建一次常驻执行计划。vision/prefill 根据请求的实际 shape
  绑定算子节点，复用原有 GPU workspace、KV 和权重；不会每次生成重新分配这些存储。
- YAML 中每个 projection 都是独立 `Linear` 节点。已有 backend compiler 可以将合法的
  两路/三路 fork lower 成 `ParallelLinear`，这是后端执行优化，而非业务模块节点。
  原有 fused kernel（如 `ResidualRmsNorm`、`NormalizeRopeCache`）本身仍是单个公共算子。

业务代码不包含 CUDA/OpenCL 头文件、native handle 解包或平台分支。更换已有算子的
后端实现无需修改业务；新增一种数学算子仍须扩展公共计算契约、YAML 字段绑定和后端实现。
新模型还需提供自身的权重/shape/生成语义绑定，不承诺只替换 YAML 即可运行任意模型。
当前共享 prompt 实现针对这两个 Qwen 模型的既有 ChatML 语义；其他模型需要提供自身的
prompt 处理。GPU 数学算子与 launch 参数不因模型基类抽取而改变。

公共层回归测试可单独运行：

```bash
./script/run_test.sh --target //Omni-Runtime/models/vlm_model_base:vlm_model_base_test
```

### YAML 约定

唯一入口是 `Omni-Runtime/inference/conf/qwen3_vl_8B_f16.conf`，其中指定：

```ini
graph_path=qwen3_vl_8B_f16.yaml
```

相对路径优先相对 `.conf` 所在目录解析，也支持绝对路径及已有的仓库相对路径解析规则。
`.conf` 管理权重、精度和容量，引用的 YAML 管理算子图及后端；主 YAML 默认
`backend: cuda`，切换 OpenCL 只需修改为 `backend: opencl`，无需修改业务代码或运行命令。
解析器仍支持通过 YAML 的可选 `graph` 字段引用共享图（相对当前 YAML 解析，禁止链式导入），
但默认配置不需要额外的后端选择文件。

```yaml
backend: cuda
language:
  - repeat: language_layers
    as: layer
    body:
      - {op: RmsNorm, input: hidden, weight: "blk.{layer}.attn_norm.weight",
         rows: tokens, hidden_size: hidden_size, epsilon: epsilon, output: normalized}
      - parallel:
          - {op: Linear, input: normalized, weight: "blk.{layer}.attn_q.weight",
             rows: tokens, inner: hidden_size, columns: query_size, output: query}
          - {op: Linear, input: normalized, weight: "blk.{layer}.attn_k.weight",
             rows: tokens, inner: hidden_size, columns: kv_size, output: key}
```

- 普通列表按顺序连接；`parallel` 分支共享前驱，后续节点等待所有分支完成。
- `repeat/as/body` 展开模型层数；`{layer}` 只做绑定索引替换，不引入表达式语言。
- `when` 使用布尔绑定选择 prefill/decode 或 DeepStack 路径，条件块可以直接使用 `body`。
- YAML anchor/alias 复用视觉 projector 的同一组算子；列表 alias 直接展开。
- 字段名与公共算子结构一致；整数、uint64 元素数、float、bool、DeviceView 分别绑定，
  缺失/错误类型、未知算子/字段、重复字段/段、无效后端、递归 alias 和越界展开均报错。
  深度上限 16、单层 repeat 上限 4096、总节点上限 16384。
- 构图失败不向调用方发布半张图。随后由 backend 校验实际设备、依赖和无序内存冲突。

### 接入与验证

```cpp
Qwen3VLPipeline pipeline;
if (!pipeline.Init(conf_path)) {
  // Report pipeline.error_message().
}
```

示例只传入 `.conf`。初始化依次加载运行配置、解析 `graph_path`、加载 YAML、
选择后端、加载权重并构图；缺失路径、无效 YAML 或后端均显式报错。
Qwen3-VL 不再读取独立的图路径环境变量，也不使用 `OMNI_BACKEND` 或脚本
`--backend` 覆盖 YAML。脚本保留 `--backend cuda|opencl|auto`，仅供其他测试使用。
无论选择哪种 YAML 后端，模型测试命令均为：

```bash
./script/run_test.sh --target //example:qwen3_vl_inference_test --test-output all
```

设计参考：[ONNX IR](https://onnx.ai/onnx/repo-docs/IR.html) 的算子图与执行表示分离、
[ncnn 参数格式](https://github.com/Tencent/ncnn/blob/master/docs/developer-guide/param-and-model-file-structure.md)
的显式算子/张量连线，以及
[nndeploy](https://github.com/nndeploy/nndeploy) 的配置式 workflow。未引入这些框架，
仅通过 Bazel 引入 `yaml-cpp 0.8.0` 解析标准 YAML，执行仍完全复用 Omni-Runtime。

### 单配置入口回归验证

- 配置解析 7 个用例、模型与算子图 18 个用例通过，覆盖相对/绝对路径、
  缺失路径、无效 YAML、配置重载及 YAML 后端优先级。
- `build.sh --clean`、再次 `build.sh` 和全量串行测试通过，15/15 测试目标成功。
- CUDA 使用无 `--backend` 的示例命令连续运行两次，每次 5/5 用例通过；
  仅将 YAML 后端临时切为 OpenCL 后，同一命令同样 5/5 通过，再恢复默认 CUDA。
- 两后端均保持原有 64-token 和完整 352-token 的 ID、文本精确匹配断言，
  同时保留取消恢复、重入、异常和驻留状态回归。

### 首次 YAML 算子图重构验收（历史记录）

- 最终代码执行 `build.sh --clean` 和 `build.sh`，全工程编译通过。
- `run_test.sh --jobs 1 --test-output errors`：15/15 测试目标通过（含兼容别名）。
- YAML/模型回归：13 个用例通过，包含两后端的实际算子节点执行与复用、
  未准备/准备失败后的拒绝执行、非法配置和请求边界。
- 按原命令分别运行 CUDA/OpenCL 模型测试，每个后端 5/5 用例通过；
  64-token 与完整 352-token 输出的 token IDs 和文本精确匹配原参考，
  保留流式、取消/恢复、重入、回调异常及驻留存储复用断言，未修改参考数据。
- 额外设置 `OMNI_BACKEND=cuda`、选择 OpenCL YAML 且不传 `--backend`，
  取消/恢复与非法请求回归通过，并断言结果后端仍为 OpenCL。
- 修改的 C++ 格式、脚本语法、空白检查及生产业务代码的 native API/平台分支隔离检查通过。

## 1. 模块边界与本次修改

| 模块 | 职责与边界 | 本次处理 |
| --- | --- | --- |
| `graph` | 通用 CPU DAG、节点生命周期、状态传播和线程池；不负责 GPU stream 调度 | 保留现有实现，作为 C++ 风格依据 |
| `inference` | 配置、tokenizer、采样、流式文本与模型无关的公共处理 | 新增 `common/image_preprocessor`，统一 RGB 解码及 patch/merge 网格计算 |
| `models` | 模型权重映射、workspace、KV 状态和业务算子编排 | 将 `LanguageLauncher` 声明移入头文件；公共结构收拢到 `qwen3_vl_define.h`；视觉执行器不再实现图像解码和网格计算 |
| `memory` | Buffer/Tensor 所有权、切片、数据传输、stream/event 的 RAII 封装 | 保留；业务层通过 Buffer/DeviceView 访问设备数据，不解释 native handle |
| `backend` | 通用计算契约、设备资源、线性代数服务、执行计划及具体后端实现 | 将原先位于 kernels 的六种扩展算子定义合并到 `backend_compute_define.h`，删除反向依赖和多余定义目标 |
| `kernel` | JIT 编译、源码依赖解析及内存/磁盘缓存 | 保留；不把 JIT 或 native API 暴露到模型逻辑 |
| `kernels` | 每个算子的 CUDA/OpenCL 实现、参数适配与数值回归 | 按根目录 `.clang-format` 统一源文件、runtime 头文件和测试的排版；将图像 CUDA 入口从 `inference::example` 移到 `kernels::image_kernel` |
| `gguf_serializer` | GGUF 元数据和 tensor 存储读取，不承担模型执行 | 保留 |
| `utils` | 日志、状态宏、线程等基础能力 | 复用，不新增平行的错误处理框架 |

业务依赖的是 `Backend`、`ComputeOperation`、`DeviceView`、`Buffer` 等中立契约。
CUDA/OpenCL 的编译、调用、队列及句柄转换留在 backend 和底层算子实现中。
修订已有算子的某个后端实现，不需要改模型业务代码；新增业务算子则仍需要扩展公共契约及相应实现。

### 公共图像处理

- `ImagePreprocessor::LoadImage` 输出 RGB uint8，尺寸单位为像素；JPEG 继续使用
  libjpeg，其他格式继续使用 stb，避免更换 decoder 引起模型输入变化。
- 文件只打开一次；文件、stb 内存和 JPEG 状态由 RAII 管理。JPEG 的错误状态、
  decoder 和像素容器放在堆上，避免 `longjmp` 后读取被修改的栈变量。
- `MakeGrid` 只接收源尺寸、patch size 和 merge size，不依赖 Qwen 配置或 backend。
  保留向上对齐策略，不引入新的缩放上限或图像裁剪策略。
- 尺寸计算使用 64 位中间值，拒绝非法 alignment、溢出的对齐尺寸及 token 数；
  失败时不发布部分输出。共享输出结构位于 `image_preprocessor_define.h`。
- GPU bicubic resize、归一化和 patch 排列仍通过现有 `ImagePatchPreprocess`
  算子执行，不复制一份 CPU 数学实现。
- 模型 Bazel 目标不再直接依赖 stb 或配置 JPEG 链接；这些依赖归公共图像处理目标所有。

语言入口增加了 token 范围、三轴 position 长度/数值、KV 容量和 vision 区间检查，
在设备访问前拒绝非法输入；不改变合法输入的算子顺序和数学运算。

## 2. Language LaunchLayer 的依赖

下图表示依赖关系。本轮已实现 Q/K/V 与 gate/up 的 fork/join；Q/K RMSNorm
仍在 join 后执行，避免同时扩大融合算子的改动范围：

```text
hidden -> attn RMSNorm -> normalized
                            |-> Q Linear -> Q RMSNorm --+
                            |-> K Linear -> K RMSNorm --+-> MultiAxisRope --+
                            |-> V Linear ----------------------------------+-> StoreKvCache
                                                        Q after RoPE --------+------+
                                                                                    |
KV cache ready ----------------------------------------------------------------> Attention
  -> output Linear -> ResidualRmsNorm -> normalized
                              |-> gate Linear --+
                              |-> up Linear ----+-> SwiGLU -> down Linear -> residual Add
```

确切的并行机会：

1. Q/K/V projection 只读相同的 normalized，写入不同的 buffer，可以并行。
2. eager/prefill 路径中，Q RMSNorm 只等待 Q，K RMSNorm 只等待 K，可以各自提前启动；
   当前 `MultiAxisRope` 同时消费 Q/K，仍须等待两路。
3. 当前 `StoreKvCache` 同时消费旋转后的 K 和 V；须等待 RoPE 和 V projection。
   Attention 必须等待旋转后的 Q 及本层 KV 写入全部完成。
4. gate/up projection 相互独立；`SwiGLU` 必须等待两路。
5. Attention 输出、残差归一化、down projection 和最终残差有真实数据依赖，不能
   按“相邻函数调用”任意拆成并行任务。下一层必须等待本层最终 hidden。

prepared decode 使用融合的 `NormalizeRopeCache`，而不是上述三个独立阶段：
它同时依赖 Q/K/V，还读写设备上的 position/cache 状态。第一版只并行三路 projection，
随后 join 到现有融合算子，不拆融合 kernel。`AdvanceSequence` 必须等待所有层和
输出计算完成，不能提前更新设备位置。

## 3. 其他 Qwen3-VL 文件中的机会与限制

| 位置 | 可并行部分 | 必须处理的限制 |
| --- | --- | --- |
| `qwen3_vl_vision_encoder_executor_gpu.cc` / transformer 主干 | 暂无与语言 Q/K/V 同样的独立 projection 分支；已有 packed QKV Linear | 保留 packed GEMM；LN、QKV、bias、RoPE、attention、output、残差及 FFN 是依赖链，不能按 `&&` 链拆开 |
| 同文件 / DeepStack | 当前层的 DeepStack norm/projection 可与下一层主干部分重叠 | 当前复用 `normalized`、`qkv`，且后续层重写 `hidden`。必须增加独立的旁路 norm/projection scratch，并保证旁路读取完成后才允许覆盖 hidden；必要时保留快照 |
| 同文件 / patch embedding | CPU decode/grid 可以在 GPU 工作提交前完成 | resize -> paired embedding -> position embedding 存在真实依赖；不能并行这三个步骤 |
| `qwen3_vl_pipeline.cc` / BuildGraph | tokenizer 与 vision 已无相互依赖 | 当前 graph 只有一个 worker，实际串行。未来可用两个 worker，但先审计共享 context/error 字段和生命周期；prefill 依然等待二者完成 |
| `qwen3_vl_prefill_executor_gpu.cc` | 多模态 prompt 展开可与尚未完成的 GPU vision 主干重叠 | 必须先取得有效图像网格，拆分 CPU 元信息与设备 embedding 的就绪状态；当前完整 vision node 依赖先保留 |
| `qwen3_vl_decode_executor_gpu.cc` | 单步内部使用上述 QKV、gate/up 并行 | token t+1 依赖 token t 的采样结果，不能同时执行相邻 decode step；流式回调、取消和停止 token 的顺序必须保持 |
| `qwen3_vl_runtime.cc` / 权重加载 | 不同权重的 CPU 类型转换具有独立性 | 当前 weights map、内存计数和上传服务共享状态，不能直接并行写；若以后优化，先并行准备独立 host 结果，再受控发布/上传 |
| `qwen3_vl_weights.cc` | 主要是一次性的元数据校验 | 不应为低收益初始化代码引入设备调度框架 |

DeepStack 的输出仍需按原来的层顺序进入 `deepstack_embeddings`；
语言侧对应层的注入必须等到该 embedding ready，不能以任务完成顺序决定索引。

## 4. 原始 backend 中立架构路线

本节保留完整设计的约束和后续扩展方向；本轮已落地的范围及实际 lowering 见第 7 节。

### 4.1 复用 ComputeOperation，增加依赖，不新增业务 stream

复用现有 `ExecutionPlan` 和 `ComputeOperation`，增加一个依赖节点定义及一个
`CreateExecutionPlan` 重载。示意接口：

```cpp
struct ComputeNode final {
  ComputeOperation operation;
  std::vector<uint32_t> dependencies;
};

bool CreateExecutionPlan(const ScalarType type,
                         const std::vector<ComputeNode> &nodes,
                         const Allocation &readback,
                         std::unique_ptr<ExecutionPlan> *const plan);
```

这些共享定义放在 backend 的 `xxx_define.h` 中。保留原来的线性 operation 列表入口：
依赖图校验后 lowering 为现有的 operation 列表，兼容串行计划而不重复执行器实现。
模型只描述 Q/K/V 分支、join 和数据，不选择 CUDA stream 或 OpenCL queue。

计划创建时：

1. 校验节点编号、依赖是否存在、是否有环，以及 readback 的生产节点。
2. 对每种 operation 维护唯一的读/写区域描述，按底层 allocation 和字节区间
   检查 RAW/WAR/WAW；DeviceView 切片和原地操作必须参与别名检查。
3. 对缺失的冲突依赖明确报错；不能把“内存地址不同”误认为没有别名。
4. 先做 workspace 生命周期规划，再为可重叠节点分配 scratch；串行节点继续复用，
   不给每个节点永久分配完整 workspace。
5. 预编译 kernel、准备参数、选择/冻结 GEMM 算法，之后才构建可重复执行计划。
   不在 graph capture 或每 token 路径中调优/分配资源。
6. 由 backend 根据设备能力选择有界执行 lane；保留合法的串行执行策略，并可诊断
   实际采用的策略。第一版最多三路，不依赖“无限 stream”。

### 4.2 CUDA 实现

- eager 路径目前 `Backend::Compute` 没有 stream 参数；多行 Linear 使用共享的
  BLAS handle、32 MiB workspace 和 `compute_mutex_`，实际调用使用默认 stream。
  仅加 CPU future 或现有内存 stream 封装，既无法实现正确的算子路由，也无法保证重叠。
- 为每个执行 lane 提供 backend 私有的 stream、BLAS 执行状态和不重叠 workspace。
  复用现有算法缓存；编译/调优缓存的锁不应覆盖正常设备执行。
- 将当前 JIT `LaunchScope` 的路由与 GEMM 的执行 stream 打通；它目前只涵盖 JIT
  kernel，不能假定它同时控制所有 BLAS 调用。
- eager 路径以 backend 私有 event 建立 fork/join，不在每个节点后做设备全局同步。
  默认流中的 upload/前驱也须显式接入 fork 的依赖，不能依赖 legacy default-stream
  对 nonblocking stream 的隐式顺序。
- prepared decode 优先将同一 DAG 构建为 CUDA Graph 的显式节点依赖，
  或正确 join 的多流 capture；不让模型代码调用任何 CUDA API。
- 当前 CUDA prepared Linear 只接受单行。扩展到 prefill 前，必须补齐多行 GEMM
  的预热、stream 路由和 capture 支持，不能直接复用这个限制下的实现。

### 4.3 OpenCL 实现

- 当前 executor 固定一个 in-order queue，prepared sequence 也顺序 enqueue。
  仅增加 host 线程仍然进入同一队列，不能获得所需并行。
- 使用多条 backend 私有 in-order queue 和跨队列 event wait list；不强制要求
  out-of-order queue 或 `cl_khr_command_buffer`，保持 OpenCL 1.2 执行能力兼容。
- 每次提交使用独立或安全预绑定的 kernel 参数状态，避免共享 `cl_kernel` 上并行
  `clSetKernelArg`。复用 program 缓存，不复制编译后的 program。
- producer queue 必须正确 flush；event 生命周期持续到消费者已入队/完成相应使用。
  readback 必须等待全部必需节点，不能只等默认队列的末尾 marker。
- profiler 的总耗时度量从共同起点到最终 join 的 makespan，不能相加重叠节点耗时。

### 4.4 生命周期、失败和回退约束

- 同一 runtime 的 generation 仍串行，不能并发使用同一 hidden、KV cache 或 plan。
  并行仅发生在单次计划内部；业务层不新增 backend kind 分支。
- 计划持有必要资源，绑定的 allocation 必须活得比计划长，保持现有契约。
- 中途提交失败需等待/收拢已提交任务后再释放资源，并返回明确失败；不能发布部分
  token、在部分 KV 写入后伪装成功，或静默改走另一种数学路径。
- decode 的状态推进与 host readback 成功语义保持一致；失败后的 KV/position 状态
  不承诺事务回滚，必须标记不可继续执行并要求重置，不能直接重试同一个 step。
- 多路 GEMV/GEMM 可能争用同一显存带宽/SM，不保证加速。分别测量 prefill 与
  decode，只有实际降低端到端时延才启用相应并行策略。

## 5. 分期范围与验收

前一轮完成第 1 节重构。本轮完成 DAG 校验、串行兼容及两后端 QKV/gate-up
fork/join；vision DeepStack 与 tokenize/vision 重叠仍为后续独立扩展，不额外分配
hidden 快照或改变 CPU graph worker 数。
本轮 grouped GEMV/GEMM 是同一 kernel launch 内的独立矩阵调度，不打包权重、
不把矩阵拼接成一次更大的点积，也不融合 SwiGLU，因此沿用原来的累加及舍入顺序。

验收必须包括：

- 两后端基础算子、FP32/FP16/BF16、切片及原地 buffer 的原有回归；
- 无效 DAG、缺失依赖、别名冲突、提交失败及资源释放；
- eager 和 prepared 两种路径与串行基线对比；
- CUDA/OpenCL 的 Qwen3-VL 64-token 和长输出 token IDs、文本完全一致；
- 取消、恢复、KV 容量边界、重复 Generate 无额外持久显存增长；
- 分别统计 VIT/prefill/decode 的时间及新增持久/峰值 workspace 字节数；
  基线和候选使用相同输入、算法、预热次数及设备负载，不能用算子耗时之和代替总时延。

构建及运行命令见 [构建测试指南](build_and_test_guide.md)。

## 6. 前一轮重构验证结果

以下为前一轮重构的验证记录；本轮并行验收另见第 7 节：

```bash
./script/build.sh --clean
./script/build.sh
./script/run_test.sh --test-output errors
./script/run_test.sh --target //example:qwen3_vl_inference_test --backend cuda --test-output all
./script/run_test.sh --target //example:qwen3_vl_inference_test --backend opencl --test-output all
```

- 全量测试报告 15 个测试目标通过，其中包含 OpenCL JIT 的兼容别名目标。
- 算子测试实际运行 129 个用例，全部通过；原有 3 个禁用的性能用例保持禁用。
- 新公共图像处理运行 8 个用例，覆盖 RGB、灰度、JPEG、坏文件、空指针、
  对齐策略、网格溢出和失败时保持输出。
- 模型层运行 4 个用例，覆盖中立 operation 录制、无效 launcher、
  非法语言输入及跨次生成保留 workspace。
- 两个显式 backend 的端到端推理各通过 4 个用例，包含 64-token 和长输出的
  token IDs/文本精确匹配、流式输出、取消/恢复、重入拒绝、回调异常及状态复用。
- 算子和新公共模块的 clang-format 检查、`git diff --check`、
  业务模块 native CUDA/OpenCL 引用检查及旧定义/旧图像入口残留检查均通过。

## 7. 本轮并行执行实现

### 公共契约和依赖验证

- `ComputeNode`、`ParallelLinear` 均位于 `backend_compute_define.h`。模型以相同代码
  记录 QKV 的三个分支、FFN 的两个分支及 join；没有 backend kind 分支或 native API。
- `CompileComputeGraph` 支持前向编号的边，检查缺失边、自依赖、环、设备种类及
  未排序的 RAW/WAR/WAW 冲突；按 DeviceView 的字节区间检查切片和原地操作。
  CUDA 也识别不同 base pointer 指向同一地址区间的别名。OpenCL view 必须遵循
  Buffer 的同一 allocation handle 加 offset 契约，不将外部 sub-buffer 当成独立 allocation。
- 编译器只把已经 ready、无依赖且输入/rows/inner 相同的两至三个 Linear 合为
  `ParallelLinear`，其他节点保持拓扑顺序执行。它不是任意算子的多流调度器，
  不推测 DeepStack scratch 是否可以重用，也不额外分配模型 workspace。
- projection 的类型、容量、对齐及输出与各路输入/权重/输出的重叠在提交前统一验证；
  非法 group 不会先执行其中的合法分支。只读权重/输入可共享。
- 两种 `CreateExecutionPlan` 入口最终使用同一组后端执行器；旧的串行调用兼容。
  计划创建不执行 decode，也不推进 KV/position；prepared CUDA 多行 Linear
  的原有限制保留，多行并行当前在 eager/prefill 路径执行。

### CUDA

- 单行 projection lowering 为 grouped GEMV：多个矩阵的独立 block/warp 在同一个
  grid 中调度，不需要额外 event、stream、权重打包或中间 buffer。
  继续调用原有 `GemvDot`，保留 FP32/FP16/BF16 的累加顺序，以及适用形状的
  Blackwell streaming load 和编译期 inner 特化。
  若各路权重的对齐导致原来的向量化策略不同，则保留逐路 kernel，避免为了合并
  改变某一路的累加顺序。
- 多行 projection 支持三路：默认 stream 加两条 backend 私有 nonblocking stream，
  用 ready/done event fork/join。每条附加 lane 有自己的 BLAS handle 和 32 MiB
  workspace，不并发使用默认 BLAS scratch。总显式 BLAS workspace 从 32 MiB
  增至 96 MiB，初始化时一次分配，计入 `compute_workspace_bytes()`，重复生成不增长。
- 各路复用串行 GEMM 的算法描述和精度模式；先在默认 stream 完成算法调优，
  再向附加 lane 提交，不在 capture 中调优。
- 第一种多行 shape 会交替测量串行/并行各三次，比较包含 join 的 event makespan
  中位数，要求并行策略至少留出约 5% 的收益余量，再缓存所选策略，避免以微小波动
  启用多流。日志输出选择和时间；竞争带宽导致更慢的多流策略不会
  被强制使用。调优期间的耗时计入首次 prefill，不伪装成稳态耗时。
- prepared decode 将 grouped GEMV 捕获为单个 graph 节点；其后的融合
  `NormalizeRopeCache` 和最终 `AdvanceSequence` 保持依赖链。
- SM120 的单 token packed Q8 激活还支持独立 `QuantizedLinear` 的 capture
  依赖调度，适用于 Q8/NVFP4 权重。仅对相邻、共享输入且没有交叉读写重叠
  的投影建立 sibling 节点，再将全部分支的末端作为后继 frontier。
  不改变各投影的 kernel、整数点积、浮点累加及归约顺序，也不增加 stream、
  event 节点或 workspace。原逻辑图的共享状态、重叠 view 和后继 join
  仍保持正确顺序；不对多行、非 packed 输入及其他算子做此调度。
  复用编译器的字节区间别名检查，捕获依赖数组在修改 frontier 前复制；
  `OMNI_PARALLEL_LINEAR=0` 可恢复物理串行。CUDA 12/13 的 capture API
  签名差异在 backend 内处理，当前完整验证使用 CUDA 13。
- 小批次 Q8 激活量化按独立 K32 block 分组，在 `rows <= 8` 时沿列维度使用
  最多 64 个 CTA，避免单 token 的所有量化工作只使用一个 SM。组内最大值、
  舍入、scale 及输出布局不变；需要全行最大值的 NVFP4 prefill 保持原调度。
- Qwen3.8 的 KV cache 使用物理 FP16 存储，query、output 和 recurrent state
  仍为 FP32。`StoreKvCache.is_fp16_cache` 保留原有“FP32 存储半精度舍入值”
  的兼容语义；`is_fp16_key_value` 才指定两字节缓存，并与 Attention 使用
  同一布局。已有 FP16 attention 的乘加和 softmax 顺序不变。非对齐 view
  保留标量读取；launch 前按实际存储宽度校验容量、偏移和维度乘积。
  原 Qwen3-VL 配置不启用此布局，原浮点路径保持不变。

### OpenCL

- 采用单次 3D NDRange 的 branch 维度调度独立 projection，各 group 只访问自己的
  权重和输出，复用既有 Linear/LinearVector/LinearTensor 数学实现。
- 保留单个 in-order queue，不需要多 queue、跨 queue event 或 command-buffer 扩展。
  kernel 完成即是 join；后继操作和 readback 自然等待所有分支。
- 多行只合并采用相同 entry、tile 配置和 local size 的分支；策略不同的合法形状
  保持各自原来的串行 kernel，不改变所选算法和计算顺序。
- prepared sequence 为各次调用保留已绑定参数的独立 kernel，重放时不修改共享参数。
- Q8 小批次量化同样按独立 K32 block 使用二维 NDRange；混合 attention 支持
  与 CUDA 相同的物理 FP16 KV 布局、原精度开关和缓存尺寸校验。

### 失败、回归和 A/B 开关

- CUDA 部分提交失败会 drain 已提交 lane 和默认 stream，再返回失败。
  prepared CUDA/OpenCL 真正执行失败后会 drain 并失效该 plan；必须重新初始化，
  不允许在部分推进的 KV/position 上继续重试。仅传入无效 host 输出参数不会失效计划。
- 新增 DAG 合法性、别名冲突、空输出、policy 参数、eager 逐分支精确比较、
  prepared join/重复执行及无准备阶段副作用测试。
  覆盖两路/三路、FP32/FP16/BF16、尾部维度、切片和重复调用。
- `OMNI_PARALLEL_LINEAR=0` 用于同一版本下的串行 A/B 对照；未设置或为 `1`
  时启用 backend 的并行策略。其他值显式报错。该开关只在 backend 解释，
  `run_test.sh` 会在 Bazel test 模式传递它，Bazel run 模式直接继承进程环境。

```bash
OMNI_PARALLEL_LINEAR=0 ./script/run_test.sh \
  --target //example:qwen3_vl_inference_test --backend cuda --test-output all
OMNI_PARALLEL_LINEAR=1 ./script/run_test.sh \
  --target //example:qwen3_vl_inference_test --backend cuda --test-output all
```

将 `cuda` 换成 `opencl` 可执行另一后端的相同对照。

### 本轮实际验收与性能

已重新执行干净构建、增量构建和全量测试：15/15 测试目标通过（含一个兼容别名）。
backend 运行 23 个用例，模型层运行 5 个用例，算子运行 141 个用例，均通过；
原有 3 个禁用的算子性能用例保持禁用。格式、脚本语法、空白及业务层 native API
隔离检查通过。

在 NVIDIA GB10 上，同一份最终代码交替执行串行开关与默认策略，各三轮：
轮次顺序为 serial/parallel、parallel/serial、serial/parallel。每次都执行完整的
4 个 Qwen3-VL 用例，两后端共 12 次推理测试进程全部通过，包含 64-token 和
长输出 token IDs/文本精确一致、流式输出、取消/恢复、重入和回调异常。

以下取第二次复用生成（2766 个 prefill tokens、352 个输出 token IDs，
其中 351 个 decode steps）的阶段耗时中位数，不包含模型加载，也不是进程墙钟时间：

| Backend / 阶段 | 串行（ms） | 默认策略（ms） | 耗时降低 |
| --- | ---: | ---: | ---: |
| CUDA prefill | 768.79 | 769.16 | -0.05%，基本持平 |
| CUDA decode | 22982.92 | 22709.56 | 1.19% |
| OpenCL prefill | 1259.64 | 1222.58 | 2.94% |
| OpenCL decode | 23303.61 | 22987.47 | 1.36% |

必须同时注意成本和范围：

- CUDA 首次 prefill 包含策略测量，中位数从 898.96 ms 增至 984.51 ms，
  即增加 85.55 ms；不能将首次延迟宣称为加速。OpenCL 不做该多流调优，
  首次 prefill 从 1387.57 ms 降至 1359.50 ms。
- 本机大 prefill 的 CUDA 多流收益低于选择余量，因此实际保留串行 GEMM；
  12-token 短 prompt 的 QKV 分支实测选择多流。单 token decode 使用 grouped GEMV。
- CUDA 额外的显式常驻 BLAS workspace 为 64 MiB；模型权重、KV 和模型 scratch
  不复制。重复生成内存计数不增长的端到端断言通过。
- 这些是本机三轮交替测量，不承诺任意硬件、任意输入长度都有相同比例的提升。
  DeepStack/CPU graph 并行和 prepared 多行 CUDA graph 并未在本轮启用。
