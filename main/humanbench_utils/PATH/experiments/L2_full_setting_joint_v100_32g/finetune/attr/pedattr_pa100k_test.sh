#!/usr/bin/env bash
set -x
source pavi_env

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "eval_logs" ]]; then
  mkdir eval_logs
fi

################
gpus=${1-1}
job_name=${2-pa100k_vitbase_SGD_lr1e2x1_stepLRx2_wd5e4_backboneclip_dpr03_30ep___SmallSetting___LSA_10p_small_setting6_add_posetrack_DGMarket_deepfashion}
CONFIG=${3-pa100k_vitbase_SGD_lr1e2x1_stepLRx2_wd5e4_backboneclip_dpr03_30ep.yaml}
TEST_CONFIG=${4-pedattr_pa100k_test.yaml}
GINFO_INDEX=${5-0}   # task index config cherrypick (if necessary)
TEST_MODEL=${6-'/mnt/lustrenew/chencheng1/expr_files/vitruvian/devL2/L2_samll_setting_attr_FT/checkpoints/pa100k_vitbase_SGD_lr1e2x1_stepLRx2_wd5e4_backboneclip_dpr03_30ep___SmallSetting___LSA_10p_small_setting6_add_posetrack_DGMarket_deepfashion/ckpt_task0_iter_newest.pth.tar'}
################

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}

LOG_FILE=eval_logs/${job_name}.log
now=$(date +"%Y%m%d_%H%M%S")
if [[ -f ${LOG_FILE} ]]; then
    echo 'log_file exists. mv: ' ${LOG_FILE} ' =>' ${LOG_FILE}_${now}.log
    mv ${LOG_FILE} ${LOG_FILE}_${now}
fi
echo 'log file: ' ${LOG_FILE}

LINKLINK_FUSED_BUFFER_LOG2_MB=-1 GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n${gpus} -p stc1_1080ti --gres=gpu:${g} --ntasks-per-node=${g} --mpi=pmi2 --quotatype=auto \
    --job-name=${job_name} --comment wbsR-SC220052.001 --cpus-per-task=5 \
python -W ignore -u ${ROOT}/test.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --test_config ${TEST_CONFIG} \
    --spec_ginfo_index ${GINFO_INDEX} \
    --load-path=${TEST_MODEL} \
    2>&1 | tee ${LOG_FILE}

