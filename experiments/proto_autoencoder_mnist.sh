#!/bin/sh

DIRNAME=`dirname $0`
#Set Script Name variable
SCRIPT=`basename ${0}`

# Figure out which cluster we are on
CLUSTER=`hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g'`

RUN="srun"

#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`


TASKS_PER_NODE=12
export SLURM_NNODES=$SLURM_JOB_NUM_NODES

# Look for the binary in the cluster specific build directory
BINDIR="${DIRNAME}/../build/${CLUSTER}.llnl.gov${DEBUGDIR}/model_zoo"

#add whatever is on the command line to options
OPTS=""
for v in "$@"; do
  OPTS="$OPTS $v"
done

TASKS=$((${SLURM_JOB_NUM_NODES} * ${SLURM_CPUS_ON_NODE}))
if [ ${TASKS} -gt 384 ]; then
TASKS=384
fi
LBANN_TASKS=$((${SLURM_NNODES} * ${TASKS_PER_NODE}))

CMD="${RUN} -n${LBANN_TASKS}  \
  --ntasks-per-node=${TASKS_PER_NODE} \
  ${BINDIR}/lbann \
  --model=../model_zoo/models/autoencoder_mnist/model_autoencoder_mnist.prototext \
  --reader=../model_zoo/data_readers/data_reader_mnist.prototext \
  --optimizer=../model_zoo/optimizers/opt_adam.prototext \
  $OPTS"

echo ${CMD}
${CMD}
