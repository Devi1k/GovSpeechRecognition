#!/bin/bash

if [ $# != 6 ];then
    echo "usage: ${0} config_path ckpt_path_prefix host_ip host_port speech_save_dir warmup_manifest"
    exit -1
fi


# start server
#ngpu=$(echo $CUDA_VISIBLE_DEVICES | awk -F "," '{print NF}')
#echo "using $ngpu gpus..."


config_path=$1
ckpt_prefix=$2
host_ip=$3
host_port=$4
speech_save_dir=$5
warmup_manifest=$6

python3 -u ${BIN_DIR}/deploy/server.py \
--host_ip ${host_ip} \
--config ${config_path} \
--checkpoint_path ${ckpt_prefix} \
--host_port ${host_port} \
--speech_save_dir ${speech_save_dir} \
--warmup_manifest ${warmup_manifest} \

if [ $? -ne 0 ]; then
    echo "Failed in start server!"
    exit 1
fi


exit 0