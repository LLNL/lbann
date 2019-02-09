#!/bin/bash

datadir="`dirname $0`/data"

lbplot --ind_var=time \
       ${datadir}/fmow_res_sgd_1e-3_1gpu.2.out \
       ${datadir}/pascal_res_sgd_1e-3_1gpu.json \
       ${datadir}/fmow_final_sgd_1e-3_1node.out \
       ${datadir}/pascal_res_sgd_1e-3_orig_1node.json \
       ${datadir}/fmow_res_sgd_1e-3_proc_1gpu.out \
       ${datadir}/pascal_res_sgd_1e-3_proc_1gpu.json \
       ${datadir}/fmow_res_sgd_1e-3_proc_1node.out \
       ${datadir}/pascal_res_sgd_1e-3_proc_1node.json \
       -n \
       'LBANN, raw data, GPUs=1' \
       'PyTorch, raw data, GPUs=1' \
       'LBANN, raw data, GPUs=2' \
       'PyTorch, raw data, GPUs=2' \
       'LBANN, pre-processed data, GPUs=1' \
       'PyTorch, pre-processed data, GPUs=1' \
       'LBANN, pre-processed data, GPUs=2' \
       'PyTorch, pre-processed data, GPUs=2' \
       $@
