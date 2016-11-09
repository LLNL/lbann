#!/bin/sh

# You can submit this with something like:
# sbatch -Abrain -N16 --enable-hyperthreads -t 1440 --clear-ssd --msr-safe --output="slurm-lbann-<scriptname>-test-%j.out" tests/<scriptname>.sh
# and can include the working directory with:
# --workdir=/g/g19/vanessen/DeepLearning/lbann.git/lbann_v0.2/examples 

#SBATCH --time=1440

TESTDIR=`dirname $0`
DIRNAME=`dirname $TESTDIR`

FULLSCRIPT=.
SCRIPT=run_lbann_dnn_multi_mnist.sh

if [ -e "${DIRNAME}/${SCRIPT}" ] ; then
    FULLSCRIPT="${DIRNAME}/${SCRIPT}"
fi

if [ ! -z "$SLURM_SUBMIT_DIR" ] ; then
  if [ -e "${SLURM_SUBMIT_DIR}/${SCRIPT}" ] ; then
      FULLSCRIPT="${SLURM_SUBMIT_DIR}/${SCRIPT}"
  fi
fi

EPOCHS=1
#NET="5000,4000,3000,1000"
NET="25000,1000"

echo "Executing script $0 -> ${SLURM_JOB_NAME}"
echo "Clearing /l/ssd for batch execution"
#srun -N${SLURM_NNODES} hostname
srun -N${SLURM_NNODES} --clear-ssd hostname

MAX_MB=300
STD_OPTS="-u -e ${EPOCHS}"
#STD_OPTS="-t ${TRAIN} -v ${VAL} -e ${EPOCHS} -n ${NET} -r 0.001"
echo "################################################################################"
for b in 300 150 100 75 60 50; do
  for k in 1 2 3 4 5 6; do
    CMD="${FULLSCRIPT} ${STD_OPTS} -b ${b} -k ${k} -z $((${k}*${MAX_MB}/${b}))"
    echo ${CMD}
    ${CMD}
    echo "################################################################################"
  done
done
