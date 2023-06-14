#!/usr/bin/env bash
#!/usr/bin/env bash

set -x

PARTITION=Zoetrope
JOB_NAME=$1
GPUS=$2
RES_PATH=$3
CKPT=$4

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_PW3D \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset PW3D \
        --agora_benchmark agora_model_val

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_EgoBody_Egocentric \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset EgoBody_Egocentric \
        --agora_benchmark agora_model_val

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_UBody \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset UBody \
        --agora_benchmark agora_model_val
        
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_EHF \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset EHF \
        --agora_benchmark agora_model_val

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_AGORA_val \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset AGORA \
        --agora_benchmark agora_model_val
    
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_ARCTIC \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset ARCTIC \
        --agora_benchmark agora_model_val
w
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_RenBody \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset RenBody_HiRes \
        --agora_benchmark agora_model_val


srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_AGORA_test \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset AGORA \
        --agora_benchmark agora_model_test