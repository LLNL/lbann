#!/bin/sh

cat << EOF | bsub -nnodes 4 -W 720
#!/bin/bash 
#BSUB -G guests -J "16gpusdplassen" 
#BSUB -e lassen_16gpus_jag_1M.err.%J
#BSUB -o lassen_16gpus_jag_1M.out.%J
cd lbann_dir

echo hostname 
echo date

jsrun -E AL_PROGRESS_RANKS_PER_NUMA_NODE=2 -E OMP_NUM_THREADS=4 -n 4 -r 1 -a 4 -b packed:11 -c 44 -g 4 build/gnu.Release.lassen.llnl.gov/lbann/build/model_zoo/lbann_cycgan --model={model_zoo/models/jag/ae_cycle_gan/cycgan_m1.prototext,model_zoo/models/jag/ae_cycle_gan/cycgan_m2.prototext,model_zoo/models/jag/ae_cycle_gan/cycgan_m3.prototext,model_zoo/models/jag/ae_cycle_gan/vae1.prototext,model_zoo/models/jag/ae_cycle_gan/vae_cyc.prototext} --disable_cuda=0 --optimizer=model_zoo/optimizers/opt_adam.prototext --reader=model_zoo/models/jag/ae_cycle_gan/data_reader_jag_conduit_lassen.prototext --st_on


echo date

EOF
