#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -m torch.distributed.launch \
        --nproc_per_node=${GPUS_PER_NODE} \
        --use_env test_ddp.py
        # --use_env train.py \
        # --gpu 0,1,2,3 \
        # --lr 1e-4 \
        # --exp_name output/${WORK_DIR} \
        # --end_epoch 14 \
        # --train_batch_size 1 \
        # torchrun --nproc_per_node=${GPUS_PER_NODE} --master_port=3003 test_ddp.py
    # python -u train.py --gpu 0,1,2,3 --lr 1e-4 --exp_name output/${WORK_DIR} --end_epoch 14 --train_batch_size 1

        # torchrun --nnodes=1 \
        # --nproc_per_node=${GPUS_PER_NODE} \
        # --rdzv_id=100 \
        # --rdzv_backend=c10d \
        # --rdzv_endpoint=$MASTER_ADDR:29400 train.py --gpu 0,1,2,3 --lr 1e-4 --exp_name output/${WORK_DIR} --end_epoch 14 --train_batch_size 1