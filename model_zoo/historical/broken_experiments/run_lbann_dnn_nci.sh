#!/bin/sh

DIRNAME=`dirname $0`
#Set Script Name variable
SCRIPT=`basename ${0}`

# Figure out which cluster we are on
CLUSTER=`hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g'`

#Initialize variables to default values.
TRAINING_SAMPLES=1
VALIDATION_SAMPLES=1
EPOCHS=10


NETWORK="1000"

PARIO=0
BLOCK_SIZE=256
MODE="false"
MB_SIZE=128
LR=0.0001
ACT=1
LRM=1
TEST_W_TRAIN_DATA=0
LR_DECAY=0.5
#TRAIN_FILE="shuffle.trn.1"
#TEST_FILE="shuffle.tst.1"
TRAIN_FILE="cl.LE.SR.dsc.all.norm" #small unbalanced
TEST_FILE="cl.LE.SR.dsc.all.norm"

RUN="srun"

ROOT_DATASET_DIR="/l/ssd"
# Originated from /usr/mic/post1/metagenomics/cancer/anl_datasets/tmp_norm/
DATASET_DIR="datasets/cancer/anl_datasets/tmp_norm"
OUTPUT_DIR="/l/ssd/lbann/outputs"
PARAM_DIR="/l/ssd/lbann/models"
SAVE_MODEL=false
LOAD_MODEL=false
TASKS_PER_NODE=8

if [ "${CLUSTER}" = "catalyst" ]; then
LUSTRE_FILEPATH="/p/lscratchf/brainusr"
#ENABLE_HT=--enable-hyperthread
else
LUSTRE_FILEPATH="/p/lscratche/brainusr"
ENABLE_HT=
fi

USE_LUSTRE_DIRECT=0

#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT -t <training set size> -e <epochs> -v <validation set size>${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-a${NORM} <val> --Sets the ${BOLD}activation type${NORM}. Default is ${BOLD}${ACT}${NORM}."
  echo "${REV}-b${NORM} <val> --Sets the ${BOLD}mini-batch size${NORM}. Default is ${BOLD}${MB_SIZE}${NORM}."
  echo "${REV}-c${NORM}       --(CHEAT) Test / validate with the ${BOLD}training data${NORM}. Default is ${BOLD}${TEST_W_TRAIN_DATA}${NORM}."
  echo "${REV}-d${NORM}       --Sets the ${BOLD}debug mode${NORM}."
  echo "${REV}-e${NORM} <val> --Sets the ${BOLD}number of epochs${NORM}. Default is ${BOLD}${EPOCHS}${NORM}."
  echo "${REV}-f${NORM} <val> --Path to the ${BOLD}datasets${NORM}. Default is ${BOLD}${ROOT_DATASET_DIR}${NORM}."
  echo "${REV}-i${NORM} <val> --Sets the ${BOLD}parallel I/O limit${NORM}. Default is ${BOLD}${PARIO}${NORM}."
  echo "${REV}-j${NORM} <val> --Sets the ${BOLD}learning rate decay${NORM}. Default is ${BOLD}${LR_DECAY}${NORM}."
  echo "${REV}-l${NORM} <val> --Determines if the model is ${BOLD}loaded${NORM}. Default is ${BOLD}${LOAD_MODEL}${NORM}."
  echo "${REV}-m${NORM} <val> --Sets the ${BOLD}mode${NORM}. Default is ${BOLD}${MODE}${NORM}."
  echo "${REV}-n${NORM} <val> --Sets the ${BOLD}network topology${NORM}. Default is ${BOLD}${NETWORK}${NORM}."
  echo "${REV}-o${NORM} <val> --Sets the ${BOLD}output directory${NORM}. Default is ${BOLD}${OUTPUT_DIR}${NORM}."
  echo "${REV}-p${NORM} <val> --Sets the ${BOLD}input parameter directory${NORM}. Default is ${BOLD}${PARAM_DIR}${NORM}."
  echo "${REV}-q${NORM} <val> --Sets the ${BOLD}learning rate method${NORM}. Default is ${BOLD}${LRM}${NORM}."
  echo "${REV}-r${NORM} <val> --Sets the ${BOLD}inital learning rate${NORM}. Default is ${BOLD}${LR}${NORM}."
  echo "${REV}-s${NORM} <val> --Determines if the model is ${BOLD}saved${NORM}. Default is ${BOLD}${SAVE_MODEL}${NORM}."
  echo "${REV}-t${NORM} <val> --Sets the number of ${BOLD}training samples${NORM}. Default is ${BOLD}${TRAINING_SAMPLES}${NORM}."
  echo "${REV}-u${NORM}       --Use the ${BOLD}Lustre filesystem${NORM} directly. Default is ${BOLD}${USE_LUSTRE_DIRECT}${NORM}."
  echo "${REV}-v${NORM} <val> --Sets the number of ${BOLD}validation samples${NORM}. Default is ${BOLD}${VALIDATION_SAMPLES}${NORM}."
  echo "${REV}-x${NORM} <val> --Set ${BOLD}train file name ${NORM}. Default is ${BOLD}${TRAIN_FILE}${NORM}."
  echo "${REV}-y${NORM} <val> --Set ${BOLD}test file name ${NORM}. Default is ${BOLD}${TEST_FILE}${NORM}."
  echo "${REV}-z${NORM} <val> --Sets the ${BOLD}tasks per node${NORM}. Default is ${BOLD}${TASKS_PER_NODE}${NORM}."
  echo -e "${REV}-h${NORM}    --Displays this help message. No further functions are performed."\\n
  exit 1
}

while getopts ":a:b:cde:f:hi:j:l:m:n:o:p:q:r:s:t:uv:x:y:z:" opt; do
  case $opt in
    a)
      ACT=$OPTARG
      ;;
    b)
      MB_SIZE=$OPTARG
      ;;
    c)
      TEST_W_TRAIN_DATA=1
      ;;
    d)
      RUN="totalview srun -a"
      DEBUGDIR="-debug"
      ;;
    e)
      EPOCHS=$OPTARG
      ;;
    f)
      ROOT_DATASET_DIR=$OPTARG
      ;;
    h)
      HELP
      exit 1
      ;;
    i)
      PARIO=$OPTARG
      ;;
    j)
      LR_DECAY=$OPTARG
      ;;
    l)
      LOAD_MODEL=$OPTARG
      ;;
    m)
      MODE=$OPTARG
      ;;
    n)
      NETWORK=$OPTARG
      ;;
    o)
      OUTPUT_DIR=$OPTARG
      ;;
    p)
      PARAM_DIR=$OPTARG
      ;;
    q)
      LRM=$OPTARG
      ;;
    r)
      LR=$OPTARG
      ;;
    s)
      SAVE_MODEL=$OPTARG
      ;;
    t)
      TRAINING_SAMPLES=$OPTARG
      ;;
    u)
      USE_LUSTRE_DIRECT=1
      ;;
    v)
      VALIDATION_SAMPLES=$OPTARG
      ;;
    x)
      TRAIN_FILE=$OPTARG
      ;;
    y)
      TEST_FILE=$OPTARG
      ;;
    z)
      TASKS_PER_NODE=$OPTARG
      ;;
    \?)
      echo "Invalid option: -$OPTARG" >&2
      exit 1
      ;;
    :)
      echo "Option -$OPTARG requires an argument." >&2
      exit 1
      ;;
  esac
done

shift $((OPTIND-1))
# now do something with $@

# Look for the binary in the cluster specific build directory
BINDIR="${DIRNAME}/../build/${CLUSTER}.llnl.gov${DEBUGDIR}/model_zoo"

# Once all of the options are parsed, you can setup the environment
#source ${DIRNAME}/setup_brain_lbann_env.sh -m debug_mvapich2 -v 0.86
#source ${DIRNAME}/setup_brain_lbann_env.sh -m openmpi -v 0.86
#source ${DIRNAME}/setup_brain_lbann_env.sh -m debug_openmpi -v 0.86
source ${DIRNAME}/setup_brain_lbann_env.sh -m mvapich2 -v El_0.86/v86-6ec56a

TASKS=$((${SLURM_NNODES} * ${SLURM_CPUS_ON_NODE}))
if [ ${TASKS} -gt 384 ]; then
TASKS=384
fi
LBANN_TASKS=$((${SLURM_NNODES} * ${TASKS_PER_NODE}))

export PATH=/collab/usr/global/tools/stat/file_bcast/${SYS_TYPE}/fbcast:${PATH}

if [ ${USE_LUSTRE_DIRECT} -eq 1 ]; then

ROOT_DATASET_DIR=${LUSTRE_FILEPATH}

else

if [ ! -d ${ROOT_DATASET_DIR}/${DATASET_DIR} ]; then
    CMD="pdsh mkdir -p ${ROOT_DATASET_DIR}/${DATASET_DIR}"
    echo "${CMD}"
    ${CMD}
fi

FILES=(${TEST_FILE} ${TRAIN_FILE})
for f in "${FILES[@]}"
do
    FILE=`basename $f`
    if [ ! -e ${ROOT_DATASET_DIR}/${DATASET_DIR}/${FILE} ]; then
        CMD="srun -n${TASKS} -N${SLURM_NNODES} file_bcast_par13 1MB ${LUSTRE_FILEPATH}/${DATASET_DIR}/${f} ${ROOT_DATASET_DIR}/${DATASET_DIR}/${FILE}"
        echo "${CMD}"
        ${CMD}
    fi
done

fi

CMD="${RUN} -N${SLURM_NNODES} -n${LBANN_TASKS} ${ENABLE_HT} --ntasks-per-node=${TASKS_PER_NODE} ${BINDIR}/dnn_nci  --learning-rate ${LR} --activation-type ${ACT} --learning-rate-method ${LRM} --lr-decay-rate ${LR_DECAY} --lambda 0.1 --dataset ${ROOT_DATASET_DIR}/${DATASET_DIR}/ --train-file ${TRAIN_FILE} --test-file ${TEST_FILE}"
echo ${CMD}
${CMD}
