#!/usr/bin/env bash

set -x

PARTITION=Zoetrope
JOB_NAME=$1
GPUS=$2
RES_PATH=$3
CKPT=$4

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
    python demo.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/demo_${JOB_NAME} \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --img_path ../demo/demo_video/frames_00001.jpg \
        --output_folder ../demo/demo_video_out

