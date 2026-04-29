#!/bin/bash

if [[ $# -lt 3 ]]; then
    echo "Usage: $0 <xfer [cpu|gpu|gds]> <nvme mount> <output log dir>"
    exit 1 
fi

XFER=$1
NVME_DIR=$2
LOG_DIR=$3

./ds_io_sweep.sh "read" ${XFER} ${NVME_DIR} ${LOG_DIR}
