# If WORLD_SIZE is not provided, default to single node (1)
WORLD_SIZE=${WORLD_SIZE:-1}
# For single node default, derive global rank information
GLOBAL_RANK=${GLOBAL_RANK:-0}
#!/bin/bash
# Helper script to launch UniAD stage1 track_map training with local mini dataset.

set -e

# Resolve repository root and ensure PYTHONPATH includes it.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
case ":${PYTHONPATH:-}:" in
  *":${REPO_ROOT}:"*) ;;
  *) export PYTHONPATH="${REPO_ROOT}${PYTHONPATH:+:${PYTHONPATH}}";;
esac
cd "${REPO_ROOT}"

# Default environment overrides (can still be overridden by caller).
PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Resolve an actual python executable for helper snippets.
if command -v "${PYTHON_BIN}" >/dev/null 2>&1; then
  PYTHON_EXEC="${PYTHON_BIN}"
elif command -v python3 >/dev/null 2>&1; then
  PYTHON_EXEC="python3"
elif command -v python >/dev/null 2>&1; then
  PYTHON_EXEC="python"
else
  echo "[ERROR] 无法找到可用的 Python 解释器，请设置 PYTHON_BIN。" 1>&2
  exit 1
fi

# Values injected by the platform (fallbacks provided for local runs).
NUM_GPUS=${KUBERNETES_POD_GPU_COUNT:-${NUM_GPUS:-2}}
NUM_NODES=${WORLD_SIZE:-${NUM_NODES:-1}}
NODE_RANK=${RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

CONFIG="projects/configs/stage1_track_map/base_track_map.py"

# Default volume root (can be overridden)
VOLUME_ROOT=${VOLUME_ROOT:-/workspace/dataset}

# Allow overriding via env vars, otherwise fall back to mounted volume layout.
DATA_ROOT=${DATA_ROOT:-${ODVLA_DATA_ROOT:-${VOLUME_ROOT}/vla-11-data/v1.0.0/dataset/data}}
INFO_ROOT=${INFO_ROOT:-${ODVLA_INFO_ROOT:-${DATA_ROOT}/infos_mini}}
NUSCENES_ROOT="${DATA_ROOT}/nuscenes"

# Attempt to auto-resolve common dataset locations if defaults are missing.
if [ ! -d "${NUSCENES_ROOT}" ]; then
  for candidate in \
    "/dataset/vla-11-data/v1.0.0/dataset/data" \
    "/dataset/vla-1.0.0/dataset/data" \
    "/workspace/dataset/data"; do
    if [ -d "${candidate}/nuscenes" ]; then
      DATA_ROOT="${candidate}"
      NUSCENES_ROOT="${DATA_ROOT}/nuscenes"
      break
    fi
  done
fi

if [ ! -d "${INFO_ROOT}" ] || [ ! -f "${INFO_ROOT}/nuscenes_infos_temporal_train.pkl" ]; then
  for candidate in \
    "${DATA_ROOT}/infos_mini" \
    "/dataset/vla-11-data/v1.0.0/dataset/data/infos_mini" \
    "/dataset/vla-1.0.0/dataset/data/infos_mini"; do
    if [ -f "${candidate}/nuscenes_infos_temporal_train.pkl" ]; then
      INFO_ROOT="${candidate}"
      break
    fi
  done
fi

if [ ! -d "${NUSCENES_ROOT}" ]; then
  echo "[ERROR] 找不到 nuScenes 数据目录: ${NUSCENES_ROOT}" 1>&2
  echo "请设置 DATA_ROOT 或 ODVLA_DATA_ROOT 环境变量指向包含 nuscenes/ 的目录。" 1>&2
  exit 1
fi

if [ ! -f "${INFO_ROOT}/nuscenes_infos_temporal_train.pkl" ]; then
  echo "[ERROR] 找不到 infos_mini 数据: ${INFO_ROOT}/nuscenes_infos_temporal_train.pkl" 1>&2
  echo "请设置 INFO_ROOT 或 ODVLA_INFO_ROOT，或检查 infos_mini 目录是否正确。" 1>&2
  exit 1
fi

CKPT_PATH=${CKPT_PATH:-${REPO_ROOT}/ckpts/bevformer_r101_dcn_24ep.pth}
if [ ! -f "${CKPT_PATH}" ]; then
  for candidate in \
    "${VOLUME_ROOT}/ckpts/bevformer_r101_dcn_24ep.pth" \
    "${REPO_ROOT}/../OpenDriveVLA-main/ckpts/bevformer_r101_dcn_24ep.pth" \
    "${REPO_ROOT}/../ckpts/bevformer_r101_dcn_24ep.pth"; do
    if [ -f "${candidate}" ]; then
      CKPT_PATH="${candidate}"
      break
    fi
  done
fi

if [ ! -f "${CKPT_PATH}" ]; then
  echo "[ERROR] 找不到预训练权重 ckpts/bevformer_r101_dcn_24ep.pth，请设置 CKPT_PATH 指向实际文件。" 1>&2
  exit 1
fi

WORK_DIR=${WORK_DIR:-/workspace/OpenDriveVLA/outputs/track_map_mini}

# Determine torchrun command (fallback to python -m torch.distributed.run)
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD=(torchrun)
else
  TORCHRUN_CMD=(${PYTHON_BIN} -m torch.distributed.run)
fi

# torchrun will launch NUM_GPUS processes on each node.
# Total world size = NUM_NODES * NUM_GPUS.
AVAILABLE_GPUS=$("${PYTHON_EXEC}" - <<'PY'
import subprocess
try:
    import torch
    print(torch.cuda.device_count())
except Exception:
    try:
        out = subprocess.check_output(["nvidia-smi", "-L"], stderr=subprocess.DEVNULL).decode().strip()
        print(len([line for line in out.splitlines() if line.strip()]))
    except Exception:
        print(0)
PY
)

if [ "${AVAILABLE_GPUS}" -lt 1 ]; then
  echo "[ERROR] 未检测到可用 GPU。当前训练脚本要求至少一张 GPU。" 1>&2
  exit 1
fi

if [ "${AVAILABLE_GPUS}" -lt "${NUM_GPUS}" ]; then
  echo "[WARN] 检测到可用 GPU 数 (${AVAILABLE_GPUS}) 少于 NUM_GPUS=${NUM_GPUS}，自动降为单卡训练。" 1>&2
  NUM_GPUS=1
fi

if [ "${NUM_GPUS}" -eq 1 ]; then
  # 保留一个 GPU，避免 torch.distributed 额外开销。
  if [[ "${CUDA_VISIBLE_DEVICES}" == *,* ]]; then
    CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES%%,*}
  fi
  export CUDA_VISIBLE_DEVICES
  DEVICE_IDS="[0]"
  LAUNCHER_OPT="--launcher none"
else
  DEVICE_IDS="[0,$(seq -s, 1 $((NUM_GPUS-1)))]"
  LAUNCHER_OPT="--launcher pytorch"
fi

COMMON_CFG_OPTS=(
  data.samples_per_gpu=1
  data.workers_per_gpu=0
  data_root=${NUSCENES_ROOT}
  info_root=${INFO_ROOT}
  train_pipeline.0.img_root=${NUSCENES_ROOT}
  test_pipeline.0.img_root=${NUSCENES_ROOT}
  data.train.pipeline.0.img_root=${NUSCENES_ROOT}
  data.val.pipeline.0.img_root=${NUSCENES_ROOT}
  data.test.pipeline.0.img_root=${NUSCENES_ROOT}
  data.train.ann_file=${INFO_ROOT}/nuscenes_infos_temporal_train.pkl
  data.val.ann_file=${INFO_ROOT}/nuscenes_infos_temporal_val.pkl
  data.test.ann_file=${INFO_ROOT}/nuscenes_infos_temporal_val.pkl
  data.train.data_root=${NUSCENES_ROOT}
  data.val.data_root=${NUSCENES_ROOT}
  data.test.data_root=${NUSCENES_ROOT}
  load_from=${CKPT_PATH}
  work_dir=${WORK_DIR}
)

if [ "${NUM_GPUS}" -gt 1 ]; then
  "${TORCHRUN_CMD[@]}" \
    --nproc_per_node=${NUM_GPUS} \
    --nnodes=${NUM_NODES} \
    --node_rank=${NODE_RANK} \
    --master_addr=${MASTER_ADDR} \
    --master_port=${MASTER_PORT} \
    third_party/mmdetection3d_1_0_0rc6/tools/train.py \
    ${CONFIG} \
    --launcher pytorch \
    --cfg-options "${COMMON_CFG_OPTS[@]}"
else
  "${PYTHON_EXEC}" \
    third_party/mmdetection3d_1_0_0rc6/tools/train.py \
    ${CONFIG} \
    --cfg-options "${COMMON_CFG_OPTS[@]}" launcher="none"
fi

