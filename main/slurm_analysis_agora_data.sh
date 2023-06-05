#!/usr/bin/env bash

set -x

PARTITION=Zoetrope
JOB_NAME=${1:-analysis}
GPUS=${2:-1}
CONFIG=${3:-config_data_analysis.py}

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=4 # ${CPUS_PER_TASK:-2}
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
    python analysis_agora_data.py \
        --num_gpus ${GPUS} \
        --exp_name output/train_${JOB_NAME} \
        --master_port 46669 \
        --config ${CONFIG}

