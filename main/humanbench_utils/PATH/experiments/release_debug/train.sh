#!/usr/bin/env bash
source /mnt/lustre/share/platform/env/pavi_env
# set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "train_logs" ]]; then
  mkdir train_logs
fi

################
gpus=${1-8}
job_name=${2-pedattr_debug}
CONFIG=${3-pedattr_debug.yaml}
################

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG}


LOG_FILE=train_logs/${job_name}.log
now=$(date +"%Y%m%d_%H%M%S")
if [[ -f ${LOG_FILE} ]]; then
    echo 'log_file exists. mv: ' ${LOG_FILE} ' =>' ${LOG_FILE}_${now}.log
    mv ${LOG_FILE} ${LOG_FILE}_${now}
fi
echo 'log file: ' ${LOG_FILE}

# resume_config='v100_32g_vitbase_size224_lr1e3_stepLRx3_bmp1_adafactor_wd01_layerdecay075_lpe_peddet_citypersons_LSA_reduct8_tbn1_heads2_gate1_peddetShareDecoder_exp3_setting_SharePosEmbed'
# resume_iter=70000

resume_config=''
resume_iter=0


LINKLINK_FUSED_BUFFER_LOG2_MB=-1 GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n${gpus} -p rdbp1_1080ti --gres=gpu:${g} --ntasks-per-node=${g} --mpi=pmi2 --quotatype=reserved \
    --job-name=${job_name} --cpus-per-task=5 --preempt --comment wbsR-SC220052.001 \
python -W ignore -u ${ROOT}/multitask.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --auto-resume=checkpoints/${resume_config}/ckpt_task_iter_${resume_iter}.pth.tar \
    2>&1 | tee ${LOG_FILE}
