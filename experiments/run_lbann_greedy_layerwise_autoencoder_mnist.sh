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
MB_SIZE=192
LR=0.01
ACT=1
LRM=1
TEST_W_TRAIN_DATA=0
LR_DECAY=0.5
DROPOUT=-1.0

RUN="srun"

ROOT_DATASET_DIR="/l/ssd"
OUTPUT_DIR="/l/ssd/lbann/outputs"
PARAM_DIR="/l/ssd/lbann/models"
SAVE_MODEL=false
LOAD_MODEL=false
CKPT_EPOCHS=0
CKPT_STEPS=0
USE_LUSTRE_DIRECT=0

TASKS_PER_NODE=1

if [ "${CLUSTER}" = "catalyst" ]; then
LUSTRE_FILEPATH="/p/lscratchf/brainusr"
DATASET_DIR="datasets/MNIST"
TRAIN_LABEL_FILE="train-labels-idx1-ubyte"
TRAIN_IMAGE_FILE="train-images-idx3-ubyte"
TEST_LABEL_FILE="t10k-labels-idx1-ubyte"
TEST_IMAGE_FILE="t10k-images-idx3-ubyte"
#ENABLE_HT=--enable-hyperthread
ENABLE_HT=
else
DATASET_DIR="datasets/mnist-bin"
LUSTRE_FILEPATH="/p/lscratche/brainusr"
TRAIN_LABEL_FILE="train-labels-idx1-ubyte"
TRAIN_IMAGE_FILE="train-images-idx3-ubyte"
TEST_LABEL_FILE="t10k-labels-idx1-ubyte"
TEST_IMAGE_FILE="t10k-images-idx3-ubyte"
ENABLE_HT=
USE_LUSTRE_DIRECT=1
fi


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
  echo "${REV}-k${NORM} <val> --Checkpoint after every ${BOLD}N${NORM} steps Default is ${BOLD}${CKPT_STEPS}${NORM}."
  echo "${REV}-K${NORM} <val> --Checkpoint after every ${BOLD}N${NORM} epochs. Default is ${BOLD}${CKPT_EPOCHS}${NORM}."
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
  echo "${REV}-z${NORM} <val> --Sets the ${BOLD}tasks per node${NORM}. Default is ${BOLD}${TASKS_PER_NODE}${NORM}."
  echo -e "${REV}-h${NORM}    --Displays this help message. No further functions are performed."\\n
  exit 1
}

while getopts ":a:b:cde:f:hi:j:k:K:l:m:n:o:p:q:r:s:t:uv:z:" opt; do
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
      #DEBUGDIR="-debug"
      DEBUGDIR=""
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
    k)
      CKPT_STEPS=$OPTARG
      ;;
    K)
      CKPT_EPOCHS=$OPTARG
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

TASKS=$((${SLURM_JOB_NUM_NODES} * ${SLURM_CPUS_ON_NODE}))
if [ ${TASKS} -gt 384 ]; then
TASKS=384
fi
LBANN_TASKS=$((${SLURM_JOB_NUM_NODES} * ${TASKS_PER_NODE}))

export PATH=/collab/usr/global/tools/stat/file_bcast/${SYS_TYPE}/fbcast:${PATH}

if [ ${USE_LUSTRE_DIRECT} -eq 1 ]; then

ROOT_DATASET_DIR=${LUSTRE_FILEPATH}

else

FILES=(labels.tar resized_256x256/train.tar resized_256x256/val.tar resized_256x256/test.tar)
for tarball in "${FILES[@]}"
do
    FILE=`basename $tarball`
    if [ ! -e ${ROOT_DATASET_DIR}/${FILE} ]; then
#        CMD="pdcp /p/lscratchf/brainusr/datasets/ILSVRC2012/${tarball} /l/ssd/"
        CMD="srun -n${TASKS} -N${SLURM_NNODES} file_bcast_par13 1MB ${LUSTRE_FILEPATH}/${DATASET_DIR}/${tarball} ${ROOT_DATASET_DIR}/${FILE}"
        echo "${CMD}"
        ${CMD}
    fi
done

if [ ! -d ${ROOT_DATASET_DIR}/${DATASET_DIR}/resized_256x256 ]; then
    CMD="pdsh mkdir -p ${ROOT_DATASET_DIR}/${DATASET_DIR}/resized_256x256"
    echo "${CMD}"
    ${CMD}
fi

FILES=(labels)
for tarball in "${FILES[@]}"
do
    if [ ! -e ${ROOT_DATASET_DIR}/${DATASET_DIR}/${tarball} ]; then
        CMD="pdsh tar xf ${ROOT_DATASET_DIR}/${tarball}.tar -C ${ROOT_DATASET_DIR}/${DATASET_DIR}/"
        echo "${CMD}"
        ${CMD}
    fi
done

FILES=(train val test)
for tarball in "${FILES[@]}"
do
    if [ ! -e ${ROOT_DATASET_DIR}/${DATASET_DIR}/resized_256x256/${tarball} ]; then
        CMD="pdsh tar xf ${ROOT_DATASET_DIR}/${tarball}.tar -C ${ROOT_DATASET_DIR}/${DATASET_DIR}/resized_256x256/"
        echo "${CMD}"
        ${CMD}
    fi
done

if [ ! -d ${PARAM_DIR} ]; then
    CMD="mkdir -p ${PARAM_DIR}"
    echo ${CMD}
    ${CMD}
fi

if [ ! -d ${OUTPUT_DIR} ]; then
    CMD="mkdir -p ${OUTPUT_DIR}"
    echo ${CMD}
    ${CMD}
fi

fi

CMD="${RUN} -n${LBANN_TASKS} ${ENABLE_HT} --ntasks-per-node=${TASKS_PER_NODE}  ${BINDIR}/greedy_layerwise_autoencoder_mnist  --learning-rate ${LR} --activation-type ${ACT} --network ${NETWORK} --learning-rate-method ${LRM} --test-with-train-data ${TEST_W_TRAIN_DATA} --lr-decay-rate ${LR_DECAY} --lambda 0.1 --dataset ${ROOT_DATASET_DIR}/${DATASET_DIR} --train-label-file ${TRAIN_LABEL_FILE} --train-image-file ${TRAIN_IMAGE_FILE} --test-label-file ${TEST_LABEL_FILE} --test-image-file ${TEST_IMAGE_FILE} --num-epochs ${EPOCHS} --mb-size ${MB_SIZE} --drop-out ${DROPOUT} --ckpt-epochs ${CKPT_EPOCHS} --ckpt-steps ${CKPT_STEPS}"
#CMD="${RUN} -N1 -n${LBANN_TASKS} ${ENABLE_HT} --ntasks-per-node=${TASKS_PER_NODE} --distribution=block --drop-caches=pagecache ${DIRNAME}/lbann_dnn_mnist --par-IO ${PARIO} --dataset ${ROOT_DATASET_DIR}/${DATASET_DIR}/  --max-validation-samples ${VALIDATION_SAMPLES} --profiling true --max-training-samples ${TRAINING_SAMPLES} --block-size ${BLOCK_SIZE} --output ${OUTPUT_DIR} --mode ${MODE} --num-epochs ${EPOCHS} --params ${PARAM_DIR} --save-model ${SAVE_MODEL} --load-model ${LOAD_MODEL} --mb-size ${MB_SIZE} --learning-rate ${LR} --activation-type ${ACT} --network ${NETWORK} --learning-rate-method ${LRM} --test-with-train-data ${TEST_W_TRAIN_DATA} --lr-decay-rate ${LR_DECAY}"
echo ${CMD}
${CMD}
