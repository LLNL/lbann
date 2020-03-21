#!/usr/bin/bash

#SBATCH --account=baasic
#SBATCH --partition=pbatch
#SBATCH --time=1-00:00:00
#SBATCH --job-name=enamine_train

cd /g/g13/jones289/workspace/lbann/applications/ATOM

spack env activate -p lbann

# fyi, sequence length is set to max_sequence_len-1 because the merge_samples data reader that is used when chunked-data is specified expects this
# batch

batch_size=$1

echo "global batch size is ${batch_size}"

time python train_atom_char_rnn.py --account $SLURM_JOB_ACCOUNT  --partition ${SLURM_JOB_PARTITION} --job-name ${SLURM_JOB_NAME} --embedding-dim 38 --num-embeddings 38 --num-epochs 1 --data-reader-prototext charvae_data/enamine/enamine.prototext --pad-index 36 --sequence-length 117 --chunked-data --nodes ${SLURM_JOB_NUM_NODES} --procs-per-trainer ${SLURM_JOB_NUM_NODES} --batch-size=$batch_size

