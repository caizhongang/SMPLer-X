#!/usr/bin/env bash
#!/usr/bin/env bash

set -x

PARTITION=Zoetrope
JOB_NAME=h4w_vis
GPUS=1
RES_PATH=hand4whole_official
CKPT=7

# model_path=../output/train_gta_synbody_ft_20230410_132110/model_dump/snapshot_2.pth.tar

GPUS_PER_NODE=$((${GPUS}<8?${GPUS}:8))
CPUS_PER_TASK=${CPUS_PER_TASK:-2}
SRUN_ARGS=${SRUN_ARGS:-""}


# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --quotatype=auto \
#     --kill-on-bad-exit=1 \
#     ${SRUN_ARGS} \
#     python test.py \
#         --num_gpus ${GPUS_PER_NODE} \
#         --exp_name output/test_${JOB_NAME}_ep${CKPT}_EgoBody_Egocentric \
#         --result_path ${RES_PATH} \
#         --ckpt_idx ${CKPT} \
#         --testset EgoBody_Egocentric \
#         --vis \
#         --h4w_original \
#         --agora_benchmark agora_model_val

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --quotatype=auto \
#     --kill-on-bad-exit=1 \
#     ${SRUN_ARGS} \
#     python test.py \
#         --num_gpus ${GPUS_PER_NODE} \
#         --exp_name output/test_${JOB_NAME}_ep${CKPT}_UBody \
#         --result_path ${RES_PATH} \
#         --ckpt_idx ${CKPT} \
#         --testset UBody \
#         --vis \
#         --h4w_original \
#         --agora_benchmark agora_model_val
        

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --quotatype=auto \
#     --kill-on-bad-exit=1 \
#     ${SRUN_ARGS} \
#     python test.py \
#         --num_gpus ${GPUS_PER_NODE} \
#         --exp_name output/test_${JOB_NAME}_ep${CKPT}_AGORA_val \
#         --result_path ${RES_PATH} \
#         --ckpt_idx ${CKPT} \
#         --testset AGORA \
#         --vis \
#         --h4w_original \
#         --agora_benchmark agora_model_val
    
# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --quotatype=auto \
#     --kill-on-bad-exit=1 \
#     ${SRUN_ARGS} \
#     python test.py \
#         --num_gpus ${GPUS_PER_NODE} \
#         --exp_name output/test_${JOB_NAME}_ep${CKPT}_ARCTIC \
#         --result_path ${RES_PATH} \
#         --ckpt_idx ${CKPT} \
#         --testset ARCTIC \
#         --vis \
#         --h4w_original \
#         --agora_benchmark agora_model_val

PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks-per-node=1 \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=auto \
    --exclusive \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python test.py \
        --num_gpus ${GPUS_PER_NODE} \
        --exp_name output/test_${JOB_NAME}_ep${CKPT}_RenBody \
        --result_path ${RES_PATH} \
        --ckpt_idx ${CKPT} \
        --testset RenBody_HiRes \
        --vis \
        --h4w_original \
        --agora_benchmark agora_model_val

# PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# srun -p ${PARTITION} \
#     --job-name=${JOB_NAME} \
#     --gres=gpu:${GPUS_PER_NODE} \
#     --ntasks-per-node=1 \
#     --cpus-per-task=${CPUS_PER_TASK} \
#     --quotatype=auto \
#     --kill-on-bad-exit=1 \
#     ${SRUN_ARGS} \
#     python test.py \
#         --num_gpus ${GPUS_PER_NODE} \
#         --exp_name output/test_${JOB_NAME}_ep${CKPT}_EHF \
#         --result_path ${RES_PATH} \
#         --ckpt_idx ${CKPT} \
#         --testset EHF \
#         --vis \
#         --osx_original \
#         --agora_benchmark agora_model_val
