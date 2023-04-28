#!/usr/bin/env bash
source pavi_env
set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "logs" ]]; then
  mkdir logs
fi

################
gpus=${1-4}
job_name=${2-h36m_FT_vitbase_pos_embed_b4_lr5e4x08_backboneclip_wd01_cosineLR_ld75_dpr03_sz480_10ep___SmallSetting___LSA_10p_small_setting6_add_posetrack_DGMarket_deepfashion}
CONFIG=${3-h36m_FT_vitbase_pos_embed_b4_lr5e4x08_backboneclip_wd01_cosineLR_ld75_dpr03_sz480_10ep.yaml}
################

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG}

AutoResume=checkpoints/${job_name}/ckpt_task_iter_newest.pth.tar

LOG_FILE=logs/${job_name}.log
now=$(date +"%Y%m%d_%H%M%S")
if [[ -f ${LOG_FILE} ]]; then
    echo 'log_file exists. mv: ' ${LOG_FILE} ' =>' ${LOG_FILE}_${now}.log
    mv ${LOG_FILE} ${LOG_FILE}_${now}
fi
echo 'log file: ' ${LOG_FILE}

LINKLINK_FUSED_BUFFER_LOG2_MB=-1 GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n${gpus} -p stc1_1080ti --gres=gpu:${g} --ntasks-per-node=${g} --mpi=pmi2 --quotatype=reserved \
    --job-name=${job_name} --cpus-per-task=5 --phx-priority P0 --comment wbsR-SC220052.001 -x SH-IDC1-10-5-40-[166,238,107,163,168,171,162,165,134,136,248,239,114,136,161,164,245,186,64,177,178],SH-IDC1-10-5-41-[3,4] \
python -W ignore -u ${ROOT}/multitask.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --auto-resume=checkpoints/${job_name}/ckpt_task_iter_newest.pth.tar \
    2>&1 | tee ${LOG_FILE}
