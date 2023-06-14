#!/usr/bin/env bash

set -x

PARTITION=Zoetrope
JOB_NAME=$1
GPUS=$2
CONFIG=$3

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python train.py \
        --num_gpus ${GPUS} \
        --exp_name output/train_${JOB_NAME} \
        --master_port $(($RANDOM % 50 + 45600)) \
        --config ${CONFIG}

