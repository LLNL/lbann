#!/bin/zsh
#BSUB -nnodes 1
#BSUB -W 120
#BSUB -G exalearn
#BSUB -e err.log
#BSUB -o out.log
#BSUB -J cosmoflow
#BSUB -q pbatch
#BSUB -alloc_flags ipisolate

firsthost=`jsrun --nrs 1 -r 1 /bin/hostname`
export MASTER_ADDR=$firsthost
export MASTER_PORT=23456

echo "Started at $(date)"
jsrun -r 4 --bind none -c 10 --smpiargs="off" python main.py --preload --enable-amp --use-batchnorm
echo "Finished at $(date)"