#!/usr/bin/env bash

set -x

PARTITION=Zoetrope
JOB_NAME=$1
GPUS=$2
RES_PATH=$3
CKPT=$4
GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
python demo.py \
    --num_gpus ${GPUS_PER_NODE} \
    --exp_name output/demo_${JOB_NAME} \
    --result_path ${RES_PATH} \
    --ckpt_idx ${CKPT} \
    --img_path ../demo/demo_video/frames \
    --start 1 \
    --end  477 \
    --output_folder ../demo/demo_video_out

