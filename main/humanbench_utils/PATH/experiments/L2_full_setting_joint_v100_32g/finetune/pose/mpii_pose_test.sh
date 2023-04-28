#!/usr/bin/env bash

source pavi_env
set -x

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "mpii_pose_eval_logs" ]]; then
  mkdir mpii_pose_eval_logs
fi

################
gpus=${1-1}
job_name=${2-mpii_pose_lr5e4x08_wd01_backboneclip_stepLR_classichead_dpr3e1_LN_udp_50ep___SmallSetting___LSA_10p_small_setting6_add_posetrack_DGMarket_deepfashion}
CONFIG=${3-mpii_pose_lr5e4x08_wd01_backboneclip_stepLR_classichead_dpr3e1_LN_udp_50ep.yaml}
TEST_CONFIG=${4-mpii_pose_test.yaml}
GINFO_INDEX=${5-0}   # task index config cherrypick (if necessary)
TEST_MODEL=${6-'/mnt/lustrenew/chencheng1/expr_files/vitruvian/devL2/L2_samll_setting_pose_FT/checkpoints/mpii_pose_lr5e4x08_wd01_backboneclip_stepLR_classichead_dpr3e1_LN_udp_50ep___SmallSetting___LSA_10p_small_setting6_add_posetrack_DGMarket_deepfashion/ckpt_task0_iter_newest.pth.tar'}
################

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}


LOG_FILE=mpii_pose_eval_logs/test_${job_name}.log
now=$(date +"%Y%m%d_%H%M%S")
if [[ -f ${LOG_FILE} ]]; then
    echo 'log_file exists. mv: ' ${LOG_FILE} ' =>' ${LOG_FILE}_${now}.log
    mv ${LOG_FILE} ${LOG_FILE}_${now}
fi
echo 'log file: ' ${LOG_FILE}

LINKLINK_FUSED_BUFFER_LOG2_MB=-1 GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n${gpus} -p stc1_1080ti --gres=gpu:${g} --ntasks-per-node=${g} --mpi=pmi2 --quotatype=auto \
    --job-name=${job_name} --cpus-per-task=5 --preempt --comment wbsR-SC220052.001 -x SH-IDC1-10-5-40-[64,238,107,163,168,171,162,165,134,136,248,239,114,136,161,164,245,186,64],SH-IDC1-10-5-41-[3,4] \
python -W ignore -u ${ROOT}/test.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --test_config ${TEST_CONFIG} \
    --spec_ginfo_index ${GINFO_INDEX} \
    --load-path=${TEST_MODEL} \
    2>&1 | tee ${LOG_FILE}
