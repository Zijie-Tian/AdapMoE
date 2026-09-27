# AdapMoE 开发与测试约束

本文件适用于仓库根目录及其所有子目录。本文所说的“康大环境”即本项目专用的 **Conda 环境 `adapmoe`**。

## 强制规则

1. **构建、安装依赖、运行 Python 代码、测试、推理和评测，必须使用 `adapmoe` Conda 环境。** 不得使用 `base`、系统 Python、用户级 `~/.local` 包，或机器上已有的其他环境（例如 `bitmoe`）。创建环境本身可以在环境外执行 `conda create`；此后的 `pip`、构建和测试命令必须在 `adapmoe` 中执行。
2. 每次工作前先运行 `conda activate adapmoe`（若名称不可解析，使用环境绝对路径激活）；非交互命令可用 `conda run -n adapmoe <命令>`。安装包只用 `python -m pip`，不要使用可能指向其他环境的裸 `pip`。在本机，`CONDA_PREFIX` 应为 `/mnt/raid/tzj/anaconda3/envs/adapmoe`，`which python` 应指向该目录下的 `bin/python`。若不匹配，**停止构建或测试，先修正环境**。
3. 不得随意升级下面已验证的关键版本，尤其是 PyTorch、Transformers、HQQ 和 NumPy。确需变更时，先确认兼容性，再更新本文件，并重跑验证命令。
4. 保持 `PYTHONNOUSERSITE=1`，防止 `~/.local/lib/python3.10/site-packages` 混入环境。直接调用环境内 Python 可绕过 Conda 激活变量，因此优先激活环境或使用 `conda run`。

## 已验证的环境与依赖

安装前需要 Conda、Git（用于获取固定提交的 HQQ）及可用的 NVIDIA GPU/驱动，实际推理依赖 CUDA。本机为 Linux x86-64、NVIDIA A100（SM 80），驱动 590.44.01；已验证 CUDA 12.1 版 PyTorch。仓库 `README.md` 要求 Python 3.10；`requirements.txt` 固定 Transformers、HQQ 提交、NumPy 和 tqdm。该 HQQ 提交实际要求 `torch>=2.1.1`，所以使用下列经过验证的组合，而不是让 `torch>=2.1.0` 自动解析到最新版。

| 用途 | 包及版本 |
| --- | --- |
| Python / CUDA | Python `3.10.21`；PyTorch wheel 的 CUDA runtime `12.1` |
| 核心 GPU 栈 | `torch==2.1.2+cu121`、`torchvision==0.16.2+cu121`、`triton==2.1.0` |
| 仓库核心依赖 | `transformers==4.36.1`、`numpy==1.24.4`、`tqdm==4.66.1`、`hqq==0.1.1`（Git 提交 `37502bea31f2969c6680c0c4a88ca74b3bb234a5`） |
| 核心依赖的已验证版本 | `accelerate==0.25.0`、`timm==0.9.12`、`huggingface-hub==0.23.5`、`tokenizers==0.15.2`、`safetensors==0.8.0` |
| 仓库文档所述评测 | `lm-eval==0.4.2`、`bitsandbytes==0.43.1`、`pandas==2.0.3`、`datasets==2.19.2`、`evaluate==0.4.2`、`peft==0.7.1`、`pyarrow==15.0.2`、`fsspec==2024.3.1` |

`requirements.txt` **没有列出** `lm-eval`、`bitsandbytes`、`pandas` 等评测依赖；仅安装该文件不足以运行两个文档中的评测入口。`benchmarks/chain-of-thought-hub/MMLU/` 下的旧版第三方/API 客户端脚本还可能需要额外依赖和凭证，不在上述主入口的验证范围内。

## 在本机新建或重建环境

本机根分区空间很少；环境、Conda 包缓存、pip 临时文件、Hugging Face 与 Triton 缓存必须放在 `/mnt/raid/tzj`，不可写入默认的 `/tmp` 或 `~/.cache`。其他机器可将下方 `STORAGE` 改为容量充足的可写目录。以下命令从仓库根目录执行；若 `conda` 尚未在 `PATH` 中，先将 Conda 的 `condabin` 目录加入 `PATH`（本机为 `/mnt/data/tzj/anaconda3/condabin`）：

```bash
source "$(conda info --base)/etc/profile.d/conda.sh"
STORAGE=/mnt/raid/tzj
mkdir -p "$STORAGE/tmp/adapmoe" "$STORAGE/.cache/pip" \
  "$STORAGE/.cache/huggingface" "$STORAGE/.cache/triton" \
  "$STORAGE/.cache/torch_extensions" "$STORAGE/anaconda3/pkgs" \
  "$STORAGE/anaconda3/envs"
export TMPDIR="$STORAGE/tmp/adapmoe"
export CONDA_PKGS_DIRS="$STORAGE/anaconda3/pkgs"
export CONDA_ENVS_PATH="$STORAGE/anaconda3/envs"
conda create -y -n adapmoe python=3.10.21 pip
conda env config vars set -n adapmoe \
  TMPDIR="$STORAGE/tmp/adapmoe" \
  PIP_CACHE_DIR="$STORAGE/.cache/pip" \
  HF_HOME="$STORAGE/.cache/huggingface" \
  TRITON_CACHE_DIR="$STORAGE/.cache/triton" \
  TORCH_EXTENSIONS_DIR="$STORAGE/.cache/torch_extensions" \
  XDG_CACHE_HOME="$STORAGE/.cache" \
  PYTHONNOUSERSITE=1
conda activate adapmoe
test "$CONDA_DEFAULT_ENV" = adapmoe
test "$CONDA_PREFIX" = "$STORAGE/anaconda3/envs/adapmoe"
test "$(command -v python)" = "$CONDA_PREFIX/bin/python"
python -m pip install --index-url https://download.pytorch.org/whl/cu121 \
  'torch==2.1.2' 'torchvision==0.16.2'
```

为防止 HQQ 的传递依赖或评测依赖升级 PyTorch/Transformers，继续使用约束文件安装：

```bash
cat > "$TMPDIR/adapmoe-constraints.txt" <<'EOF'
torch==2.1.2
torchvision==0.16.2
triton==2.1.0
transformers==4.36.1
tokenizers==0.15.2
safetensors==0.8.0
numpy==1.24.4
tqdm==4.66.1
accelerate==0.25.0
timm==0.9.12
huggingface-hub==0.23.5
peft==0.7.1
datasets==2.19.2
pyarrow==15.0.2
fsspec==2024.3.1
evaluate==0.4.2
pandas==2.0.3
bitsandbytes==0.43.1
EOF
python -m pip install -c "$TMPDIR/adapmoe-constraints.txt" -r requirements.txt
python -m pip install -c "$TMPDIR/adapmoe-constraints.txt" \
  'lm-eval==0.4.2' 'bitsandbytes==0.43.1' 'pandas==2.0.3'
python -m pip check
```

HQQ 必须来自 `requirements.txt` 中指定的 Git 提交，不要用同名的最新版 PyPI 包替换。若环境已存在，不要直接执行 `conda create` 覆盖它；先激活并检查版本。

## 最小验证与运行限制

以下命令须在激活的 `adapmoe` 环境中运行；本机选择空闲 GPU 时，可用 `CUDA_VISIBLE_DEVICES=3` 将物理 GPU 3 映射为代码中的 `cuda:0`。

```bash
python -m pip check
python -m compileall -q run.py src benchmarks/run_acc_benchmark.py \
  benchmarks/modified_mixtral.py benchmarks/chain-of-thought-hub/run_mmlu.py
python run.py --help
(cd benchmarks && python run_acc_benchmark.py --help)
(cd benchmarks/chain-of-thought-hub && python run_mmlu.py --help)
CUDA_VISIBLE_DEVICES=3 python -c \
  "import torch; assert torch.cuda.is_available(); print(torch.__version__, torch.version.cuda, torch.cuda.get_device_name(0))"
CUDA_VISIBLE_DEVICES=3 python - <<'PY'
import torch
import bitsandbytes as bnb
from src.triton_kernels import triton_matmul2_transpose, triton_matmul4_transpose

a = torch.ones((1, 32), device="cuda", dtype=torch.float16)
scales = torch.ones((1, 32), device="cuda", dtype=torch.float16)
zeros = torch.zeros((1, 32), device="cuda", dtype=torch.float16)
for func, rows in ((triton_matmul4_transpose, 16), (triton_matmul2_transpose, 8)):
    qweight = torch.zeros((rows, 32), device="cuda", dtype=torch.uint8)
    assert torch.count_nonzero(func(32, a, qweight, scales, zeros)).item() == 0
linear = bnb.nn.Linear4bit(32, 16, bias=False, compute_dtype=torch.float16, quant_type="nf4").to("cuda")
assert torch.isfinite(linear(a)).all()
print("Triton 2/4-bit 与 bitsandbytes 4-bit CUDA 烟测通过")
PY
```

完整模型推理**尚未验证**，不可把导入、`--help` 或小张量 CUDA 烟测表述为端到端成功：

- `run.py` 把 Hugging Face ID `lavawolfiee/Mixtral-8x7B-Instruct-v0.1-offloading-demo` 当作 `state_path`，但 `src/build_model.py` 会直接从本地路径打开 `model.safetensors.index.json`；必须先准备模型快照，并让 `state_path` 指向本地权重目录。
- 两套主评测脚本把模型路径写死为 `/opt/pretrained_models/Mixtral-8x{size}B-Instruct-v0.1`；运行前须提供相应权重或调整路径。CoT 评测应从 `benchmarks/chain-of-thought-hub` 目录运行，因其导入和 `MMLU/data` 路径均依赖工作目录。
- `run.py` 固定使用 `cuda:0`；在共享主机上先检查 GPU 可用显存，再通过 `CUDA_VISIBLE_DEVICES` 选择物理 GPU。HQQ 提示缺少可选的 `hqq_aten` 后端，不代表本仓库使用的 Triton 路径失败。
