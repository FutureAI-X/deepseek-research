# R1 复现
## 一 数据集准备
[Jiayi-Pan/Countdown-Tasks-3to4](https://huggingface.co/datasets/Jiayi-Pan/Countdown-Tasks-3to4)

**下载方式**

安装依赖
```
pip install huggingface_hub
```

在虚拟环境中执行以下命令

```
# 更换为国内镜像源，这个只用执行一次，每次重新打开终端就要重新执行
export HF_ENDPOINT=https://hf-mirror.com

# 从 Hugging Face 下载数据集
huggingface-cli download --repo-type dataset --resume-download Jiayi-Pan/Countdown-Tasks-3to4 --local-dir <你想要存放的路径>

# 数据集下载示例
huggingface-cli download --repo-type dataset --resume-download Jiayi-Pan/Countdown-Tasks-3to4 --local-dir /home/futureai/datasets/Jiayi-Pan/Countdown-Tasks-3to4
```

## 二 模型准备
### 2.1 模型选择
选择的模型为 Qwen/Qwen2.5-3B-Instruct，不建议使用小于3B的模型（小于3B的模型无法学会推理）
- [Hugging Face](https://huggingface.co/Qwen/Qwen2.5-3B-Instruct)
- [ModelScope](https://www.modelscope.cn/models/Qwen/Qwen2.5-3B-Instruct)
### 2.2 模型下载
此处选择从 ModelScope，首先安装依赖
```
pip install --upgrade setuptools
pip install modelscope
```
执行下载
```
modelscope download --model Qwen/Qwen2.5-3B-Instruct --local_dir /home/futureai/models/Qwen/Qwen2.5-3B-Instruct
```

## 三 配置文件
### 3.1 Accelerate 配置文件
accelerate 用于分布式训练（单机多卡、多机多卡）

新建 `accelerate_config.yaml` 文件，一般来说，这个文件内容不需要该，如果有定制需求，可以运行 `accelerate config` 自行设定
```
compute_environment: LOCAL_MACHINE
debug: false
deepspeed_config:
  deepspeed_multinode_launcher: standard
  offload_optimizer_device: none
  offload_param_device: none
  zero3_init_flag: true
  zero3_save_16bit_model: true
  zero_stage: 3
distributed_type: DEEPSPEED
downcast_bf16: 'no'
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```
**参数详细说明**
1. `compute_environment: LOCAL_MACHINE`
- 表示计算环境是本地机器（而不是远程集群或云环境）。
- 如果是在多台机器上进行分布式训练，则需要将此值更改为其他选项，如 `MULTI_NODE`
2. `debug: false`
- 是否启用调试模式。
- 如果设置为 `true`，会输出更多的调试信息，便于排查问题。
- 在生产环境中通常设置为 `false`。
3. `deepspeed_config`
DeepSpeed 的配置部分，用于优化大规模模型的分布式训练。
    - `deepspeed_multinode_launcher: standard`
        - 指定 DeepSpeed 的多节点启动器类型。
        - `standard` 是默认值，表示使用标准的启动方式（例如通过 `torch.distributed.launch` 或 `deepspeed` 命令）。
    - `offload_optimizer_device: none`
        - 是否将优化器状态卸载到 CPU 或 NVMe 存储设备。
        - `none` 表示不卸载优化器状态，所有状态都保留在 GPU 内存中
    - `offload_param_device: none`
        - 是否将模型参数卸载到 CPU 或 NVMe 存储设备。
        - `none` 表示不卸载模型参数，所有参数都保留在 GPU 内存中。
    - `zero3_init_flag: true`
        - 是否启用 DeepSpeed 的 Zero Redundancy Optimizer (ZeRO) 第 3 阶段初始化
        - `true` 表示启用 ZeRO-3 初始化，这可以减少内存使用并支持更大规模的模型
    - `zero3_save_16bit_model: true`
        - 在保存模型时是否保存为 16 位精度（FP16/BF16）
        - `true` 表示保存为 16 位模型，以节省磁盘空间
    - `zero_stage: 3`
        - 指定 DeepSpeed 的 ZeRO 阶段
        - `0`: 不启用 ZeRO
        - `1`: 仅分片优化器状态
        - `2`: 分片优化器状态和梯度
        - `3`: 分片优化器状态、梯度和模型参数（最节省内存的阶段）
4. `distributed_type: DEEPSPEED`
- 指定分布式训练的类型
- DEEPSPEED 表示使用 DeepSpeed 进行分布式训练
- 其他可能的值包括
    - `MULTI_GPU`: 使用 PyTorch 的原生多 GPU 支持
    - `TPU`: 使用 TPU 进行训练
    - `NO`: 不使用分布式训练
5. `downcast_bf16: 'no'`
- 是否将模型权重转换为 `bfloat16`（BF16）格式
- `no` 表示不进行转换，保持原始精度（通常是 FP32 或 FP16）
6. `machine_rank: 0`
- 指定当前机器在分布式训练中的排名（rank）
- 在单机训练中，通常设置为 `0`
- 在多机训练中，每台机器的 machine_rank 必须唯一
7. `main_training_function: main`
- 指定主训练函数的名称
- 在脚本中，`main` 函数通常是训练的入口点
- accelerate 会在运行时调用这个函数
8. `mixed_precision: bf16`
- 指定混合精度训练的类型
- `bf16` 表示使用 `bfloat16` 混合精度训练
- 其他可能的值包括
    - `fp16`: 使用 FP16 混合精度训练
    - `no`: 不使用混合精度训练
9. `num_machines: 1`
- 指定参与分布式训练的机器数量
- `1` 表示单机训练
- 如果设置为大于 `1`，则需要配置多机通信（如 RDZV 后端）
10. `num_processes: 8`
- 指定每个机器上的进程数（通常等于 GPU 数量）
- `8` 表示每个机器上使用 8 个进程（对应 8 个 GPU）
11. `rdzv_backend: static`
- 指定分布式训练的 Rendezvous（RDZV）后端。
- `static` 表示静态配置，适用于单机或多机环境。
- 其他可能的值包括 `c10d`（PyTorch 的分布式通信库）。
12. `same_network: true`
- 指定所有机器是否在同一网络中。
- `true` 表示所有机器在同一个局域网中，可以直接通信。
- 如果机器分布在不同的网络中，则需要额外配置网络连接。
13. `tpu_env: []`
- 指定 TPU 环境变量。
- [] 表示未使用 TPU。
- 如果使用 TPU，则需要提供相关的环境变量。
14. `tpu_use_cluster: false`
- 是否使用 TPU 集群。
- `false` 表示不使用 TPU 集群。
15. `tpu_use_sudo: false`
- 是否在 TPU 上使用 `sudo` 权限。
- `false` 表示不需要 `sudo` 权限
16. `use_cpu: false`
- 是否使用 CPU 进行训练。
- `false` 表示不使用 CPU，而是使用 GPU 或其他加速设备（如 TPU）。
- 如果设置为 `true`，则训练将在 CPU 上进行
### 3.2 TRL 配置文件
新建 `trl_config.yaml`，用于设定训练的超参数
```
# 模型参数
model_name_or_path: <你的模型存放的路径，比如：models/Qwen/Qwen2.5-3B-Instruct>
model_revision: main
torch_dtype: bfloat16
attn_implementation: flash_attention_2
bf16: true
tf32: true
output_dir: <你想要模型输出的路径，比如 output/Datawhale-R1>

# 数据集参数
dataset_id_or_path: <你的数据集存放的路径，比如：dataset>

# Swanlab 训练流程记录参数
swanlab: true # 是否开启 Swanlab 
workspace: <用户名>
project: <项目名，整个复现项目的名称，例如：Datawhale-R1-by_xxx>
experiment_name: <实验名，某次超参数运行的自定义名称，例如：qwen2.5-3B-lr:5e-7_beta:0.001>

# 训练参数
max_steps: 450 # 最大训练步长
per_device_train_batch_size: 1
gradient_accumulation_steps: 8
gradient_checkpointing: true
gradient_checkpointing_kwargs:
  use_reentrant: false
learning_rate: 5.0e-7 # 学习率，调整过，参见下文介绍
lr_scheduler_type: cosine # 学习率衰减方案
warmup_ratio: 0.03 # 学习率预热比率（对于整个步长），好用！
seed: 2025 # 随机种子，方便实验复现

# GRPO 算法参数
beta: 0.001 # KL 惩罚因子，调整过，参见下文介绍
max_prompt_length: 256 # 输入 prompt 最大长度，本实验基本不会有太大变化
max_completion_length: 4096 # 输出回答长度，包含推理思维链，设为 4K 比较合适
num_generations: 8
use_vllm: true # 启用 vllm 来加速推理
vllm_device: <计算卡编号，例如：cuda:2> # 留出一张卡来启用 vllm 推理，参见下文介绍
vllm_gpu_memory_utilization: 0.5

# Logging arguments
logging_strategy: steps
logging_steps: 1
save_strategy: "steps"
save_steps: 50 # 每隔多少步保存一次
```
**参数详细说明**
1. `learning_rate` 与 `beta`
在GRPO的原始论文[《DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models》](https://arxiv.org/abs/2402.03300)里分别为`1e-6`和`0.04`。这里根据[《Unraveling RLHF and Its Variants: Progress and Practical Engineering Insights》](https://hijkzzz.notion.site/unraveling-rlhf-and-its-variants-engineering-insights)调整为`5e-7`和`0.001`
2. `vllm_device`
留出一张卡作为 vllm 的推理卡，假设我们手上有 3 张卡（编号cuda: 0, cuda: 1, cuda: 2)，我们需要指定其中一张卡为 vllm 推理卡，例如我们指定最后一张 cuda:2。另外，如果你使用了CUDA_VISIBLE_DEVICES 情况会有些不一样，比如我们有 8 张卡（编号 cuda:0-7），指定编号为 1、2、3 的卡可见（CUDA_VISIBLE_DEVICES=1,2,3)，这时我们想指定最后一张卡为 vllm 推理卡，则是需要设置为 cuda:2，因为设置完可见性后，cuda:1 -> cuda:0，cuda:2 -> cuda:1，cuda:3 -> cuda:2，所以原先的 3 号卡变为了新编号的 2 号卡。
3. `save_steps`
在 [mini-r1](https://www.philschmid.de/mini-deepseek-r1) 中是被设为 25，但是跑完整个训练后，保存的文件大小达到了 700+ GB！因为不仅包含了模型，还包含了其他卡的优化器状态和其他检查点信息，我们在这里改为 50，但仍然要提醒同学们设置成合适自己的大小（训练代码中已经包含结束后保存模型的代码）
## 四 启动训练
```
bash train_r1.sh
```
注意：--num_processes 是由你希望使用的计算卡数量决定，我们之前在配置文件那里说过，要留一张卡作为 vllm 的推理卡，那么 --num_processes 的数值应该是你要使用的计算卡数量 n-1，例如我有 3 张卡，我的 --num_processes 应该为 2。这里的 --num_processes 的数值也会把 deepspeed_zero3.yaml 的num_processes 设置的 8 给覆盖掉。　