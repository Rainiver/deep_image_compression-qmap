#!/bin/sh

currenttime=`date "+%Y%m%d_%H%M%S"`
work_path=$(dirname $0)
jobname=$(pwd | awk -F "/" '{print $NF}')
partition=$1
gpu_num=8

# .. under spring_helper / .. under DIC / .. under root_dir / SpringDispatcher under SpringDispatcherMainDir
ROOT=../../../SpringDispatcher/
export PYTHONPATH=$ROOT:$PYTHONPATH

# under springnas_lite_MainDir
export PYTHONPATH=../../../springnas-lite/:$PYTHONPATH

# .. under spring_helper / .. under DIC /
export PYTHONPATH=../../:$PYTHONPATH

# .. under root_dir
export PYTHONPATH=../../../:$PYTHONPATH

GLOG_vmodule=MemcachedClient=-1 srun --mpi=pmi2 \
    --job-name=${jobname} \
    -p ${partition} \
    -n ${gpu_num} \
    --gres=gpu:8 --ntasks-per-node=8 --kill-on-bad-exit=1\
    python -u -m SpringDispatcher \
    --pipeline_config_path ${work_path}/pipeline.yaml \
    --root_folder ${work_path}/results \
    2>&1 | tee ${work_path}/${currenttime}.log

# --resume_path ${work_path}/xxx.pth.tar
