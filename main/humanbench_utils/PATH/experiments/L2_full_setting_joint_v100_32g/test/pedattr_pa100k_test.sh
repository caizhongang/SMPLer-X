#!/usr/bin/env bash
set -x
source /mnt/lustre/share/platform/env/pavi_env

# source /mnt/lustre/share/spring/r0.3.0
ROOT=../../
export PYTHONPATH=$ROOT:$PYTHONPATH

if [[ ! -d "pedattr_pa100k_eval_logs" ]]; then
  mkdir pedattr_pa100k_eval_logs
fi

################
gpus=${1-1}
job_name=${2-v100_32g_vitbase_size224_lr1e3_stepLRx3_bmp1_adafactor_wd01_clip05_layerdecay075_lpe_peddet_citypersons_LSA_reduct8_tbn1_heads2_gate1_peddetShareDecoder_exp3_setting_SharePosEmbed}
CONFIG=${3-v100_32g_vitbase_size224_lr1e3_stepLRx3_bmp1_adafactor_wd01_clip05_layerdecay075_lpe_peddet_citypersons_LSA_reduct8_tbn1_heads2_gate1_peddetShareDecoder_exp3_setting_SharePosEmbed.yaml}
TEST_CONFIG=${4-pedattr_pa100k_test.yaml}
GINFO_INDEX=${5-8}   # task index config cherrypick (if necessary)
TEST_MODEL=${6-/mnt/lustre/chencheng1/expr_files/vitruvian/L2_full_setting_joint/checkpoints/v100_32g_vitbase_size224_lr1e3_stepLRx3_bmp1_adafactor_wd01_clip05_layerdecay075_lpe_peddet_citypersons_LSA_reduct8_tbn1_heads2_gate1_peddetShareDecoder_exp3_setting_SharePosEmbed/ckpt_task${GINFO_INDEX}_iter_newest.pth.tar}
################

g=$((${gpus}<8?${gpus}:8))
echo 'start job:' ${job_name} ' config:' ${CONFIG} ' test_config:' ${TEST_CONFIG}


LOG_FILE=pedattr_pa100k_eval_logs/${job_name}.log
now=$(date +"%Y%m%d_%H%M%S")
if [[ -f ${LOG_FILE} ]]; then
    echo 'log_file exists. mv: ' ${LOG_FILE} ' =>' ${LOG_FILE}_${now}.log
    mv ${LOG_FILE} ${LOG_FILE}_${now}
fi
echo 'log file: ' ${LOG_FILE}

LINKLINK_FUSED_BUFFER_LOG2_MB=-1 GLOG_vmodule=MemcachedClient=-1 MKL_SERVICE_FORCE_INTEL=1 \
srun -n${gpus} -p rdbp1_v100_32g --gres=gpu:${g} --ntasks-per-node=${g} --mpi=pmi2 --quotatype=spot \
    --job-name=${job_name} --comment wbsR-SC220052.001 --cpus-per-task=5 \
python -W ignore -u ${ROOT}/test.py \
    --expname ${job_name} \
    --config ${CONFIG} \
    --test_config ${TEST_CONFIG} \
    --spec_ginfo_index ${GINFO_INDEX} \
    --load-path=${TEST_MODEL} \
    2>&1 | tee ${LOG_FILE}
