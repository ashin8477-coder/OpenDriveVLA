   #!/bin/bash
   set -e

   if [ $# -lt 2 ]; then
     echo "Usage: $0 <config> <num_gpus> [additional args...]"
     exit 1
   fi

   CONFIG=$1
   GPUS=$2
   shift 2

   PORT=${PORT:-29500}
   PYTHON_BIN=${PYTHON_BIN:-python3}
   CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-0,1}

   ${PYTHON_BIN} -m torch.distributed.launch \
     --nproc_per_node=${GPUS} \
     --master_port=${PORT} \
     --use_env \
     third_party/mmdetection3d_1_0_0rc6/tools/train.py \
     ${CONFIG} \
     --launcher pytorch \
     "$@"
   EOF
