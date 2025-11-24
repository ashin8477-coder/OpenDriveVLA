# If WORLD_SIZE is not provided, default to single node (1)
WORLD_SIZE=${WORLD_SIZE:-1}
# For single node default, derive global rank information
GLOBAL_RANK=${GLOBAL_RANK:-0}
#!/bin/bash
# Helper script to launch UniAD stage1 track_map training with local mini dataset.

set -e

# Default environment overrides (can still be overridden by caller).
PYTHON_BIN=${PYTHON_BIN:-python3}
CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

# Values injected by the platform (fallbacks provided for local runs).
NUM_GPUS=${KUBERNETES_POD_GPU_COUNT:-${NUM_GPUS:-2}}
NUM_NODES=${WORLD_SIZE:-${NUM_NODES:-1}}
NODE_RANK=${RANK:-${NODE_RANK:-0}}
MASTER_ADDR=${MASTER_ADDR:-127.0.0.1}
MASTER_PORT=${MASTER_PORT:-29500}

CONFIG="projects/configs/stage1_track_map/base_track_map.py"

DATA_ROOT=${DATA_ROOT:-/workspace/OpenDriveVLA/volume/data}
NUSCENES_ROOT="${DATA_ROOT}/nuscenes"
INFO_ROOT=${INFO_ROOT:-${DATA_ROOT}/infos_mini}

WORK_DIR=${WORK_DIR:-/workspace/OpenDriveVLA/outputs/track_map_mini}

# Determine torchrun command (fallback to python -m torch.distributed.run)
if command -v torchrun >/dev/null 2>&1; then
  TORCHRUN_CMD=(torchrun)
else
  TORCHRUN_CMD=(${PYTHON_BIN} -m torch.distributed.run)
fi

# torchrun will launch NUM_GPUS processes on each node.
# Total world size = NUM_NODES * NUM_GPUS.
"${TORCHRUN_CMD[@]}" \
  --nproc_per_node=${NUM_GPUS} \
  --nnodes=${NUM_NODES} \
  --node_rank=${NODE_RANK} \
  --master_addr=${MASTER_ADDR} \
  --master_port=${MASTER_PORT} \
  third_party/mmdetection3d_1_0_0rc6/tools/train.py \
  ${CONFIG} \
  --launcher pytorch \
  --cfg-options \
    data.samples_per_gpu=1 \
    data.workers_per_gpu=0 \
    data_root=${NUSCENES_ROOT} \
    info_root=${INFO_ROOT} \
    train_pipeline.0.img_root=${NUSCENES_ROOT} \
    test_pipeline.0.img_root=${NUSCENES_ROOT} \
    data.train.pipeline.0.img_root=${NUSCENES_ROOT} \
    data.val.pipeline.0.img_root=${NUSCENES_ROOT} \
    data.test.pipeline.0.img_root=${NUSCENES_ROOT} \
    data.train.ann_file=${INFO_ROOT}/nuscenes_infos_temporal_train.pkl \
    data.val.ann_file=${INFO_ROOT}/nuscenes_infos_temporal_val.pkl \
    data.test.ann_file=${INFO_ROOT}/nuscenes_infos_temporal_val.pkl \
    data.train.data_root=${NUSCENES_ROOT} \
    data.val.data_root=${NUSCENES_ROOT} \
    data.test.data_root=${NUSCENES_ROOT} \
    work_dir=${WORK_DIR}

