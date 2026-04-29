#!/bin/bash
# set -x 
if [[ $# -lt 4 ]]; then
    echo "Usage: $0 <op [read|write]> <xfer [gpu|cpu|gds]> <nvme mount> <output log dir>"
    exit 1 
fi

IO_OP=$1
XFER=$2
NVME_DIR=$3
LOG_DIR=$4


if [[ ${IO_OP} == "read" ]]; then 
    io_op_opt="--read"
elif [[ ${IO_OP} == "write" ]]; then 
    io_op_opt=""
else 
    echo "Error: ${IO_OP} is an invalid op. Valid ops are [read, write]"
    exit 1
fi 

if [[ ${XFER} == "cpu" ]]; then
    xfer_opt=""
elif [[ ${XFER} == "gpu" ]]; then
    xfer_opt="--gpu --use_accelerator_pin_memory"
elif [[ ${XFER} == "gds" ]]; then 
    xfer_opt="--gpu --use_gds"
else
    echo "Error: ${XFER} is an invalid op. Valid xfers are [cpu, gpu, gds]"
    exit 1
fi 

NUM_DRIVES=`ls -d ${NVME_DIR}* | wc -l`
if [[ $NUM_DRIVES -lt 1 ]]; then
    echo "Error: Found less than 1 folder in ${NVME_DIR}"
    exit 1
fi 



mkdir -p ${LOG_DIR}
IO_SIZE=1G

for numjobs in 1 4 8; do 
    if ((numjobs < NUM_DRIVES)); then
        continue
    fi 
    FTD_OPT="--folder_to_device_mapping "
    drive_num=0
    jobs_per_drive=$((numjobs/NUM_DRIVES))
     if ((jobs_per_drive == 0 )); then
        jobs_per_drive=1
    fi 
    for (( i=0; i<${numjobs}; i++ )); do
        FTD_OPT="${FTD_OPT} ${NVME_DIR}${drive_num}:${i}"
        if (( (i+1) % jobs_per_drive == 0)); then
            drive_num=$((drive_num+1))
        fi
    done 
    # echo ${FTD_OPT} 
    COMMON_OPTS="--io_size ${IO_SIZE} ${io_op_opt} ${xfer_opt} ${FTD_OPT}" 
    for ov in overlap sequential; do 
        if [[ ${ov} == "sequential" ]]; then
            ov_opt="--sequential_requests"
        else
            ov_opt=""
        fi
        for sub in single block; do 
            if [[ ${sub} == "single" ]]; then
                sub_opt="--single_submit"
            else
                sub_opt=""
            fi
            for io_para in 1 2 4 8; do
                io_para_opt="--io_parallel ${io_para}"
                for bs in 1M 2M; do 
                    bs_opt="--block_size ${bs}"
                    for qd in 128; do 
                        qd_opt="--queue_depth ${qd}"
                        RUN_OPTS="${ov_opt} ${sub_opt} ${io_para_opt} ${bs_opt} ${qd_opt}"
                        LOG="${LOG_DIR}/$IO_OPT_${sub}_${ov}_t${io_para}_p${numjobs}_d${qd}_bs${bs}.txt"
                        cmd="ds_io ${COMMON_OPTS} ${RUN_OPTS} &> ${LOG}"
                        echo ${cmd}
                        eval ${cmd}
                    done 
                done 
            done
        done
    done  
done 