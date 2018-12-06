#!/bin/bash

cd data && lbplot --ind_var=time \
    -n  'LBANN, raw data, GPUs=1' 'PyTorch, raw data, GPUs=1' 'LBANN, raw data, GPUs=2' 'PyTorch, raw data, GPUs=2' \
        'LBANN, pre-processed data, GPUs=1' 'PyTorch, pre-processed data, GPUs=1' 'LBANN, pre-processed data, GPUs=2' 'PyTorch, pre-processed data, GPUs=2' \
    -p  fmow_res_sgd_1e-3_1gpu.2.out pascal_res_sgd_1e-3_1gpu.json fmow_final_sgd_1e-3_1node.out ./pascal_res_sgd_1e-3_orig_1node.json \
        fmow_res_sgd_1e-3_proc_1gpu.out pascal_res_sgd_1e-3_proc_1gpu.json fmow_res_sgd_1e-3_proc_1node.out pascal_res_sgd_1e-3_proc_1node.json && cd ..
