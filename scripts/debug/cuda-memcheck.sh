#!/bin/bash

# CUDA-MEMCHECK parameters
tool=memcheck # Options: memcheck, racecheck, initcheck, synccheck
leak_check=no # Options: no, full (only active for memcheck tool)

# Get process ID
proc=
proc=${proc:-${SLURM_PROCID}}
proc=${proc:-${MV2_COMM_WORLD_RANK}}
proc=${proc:-${OMPI_COMM_WORLD_RANK}}

# Choose file basename
basename=$(pwd)/cuda-memcheck-${proc:+proc${proc}-}$(hostname)

# Launch CUDA-MEMCHECK
memcheck_opts=
if [[ "$tool" == "memcheck" ]]; then
    memcheck_opts="--leak-check ${leak_check}"
fi
cuda-memcheck --tool ${tool} ${memcheck_opts} --log-file ${basename}.log --save ${basename}.memcheck --print-limit 0 $*
