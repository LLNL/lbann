#!/bin/sh

SPACK_RECIPES=`dirname ${0}`
#Set Script Name variable
SCRIPT=`basename ${0}`

SCRIPTS_DIR=`dirname ${SPACK_RECIPES}`

if [[ "$SCRIPTS_DIR" = /* ]]; then
  ROOT_DIR=`dirname ${SCRIPTS_DIR}`
else
  LVL_TO_ROOT_DIR=`dirname ${SCRIPTS_DIR}`
fi

BLAS=openblas
BUILD_TYPE=Release
COMPILER=gcc@4.9.3
DTYPE=4
EL_VER=master
MPI=mvapich2
VARIANTS=

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT -c gcc@7.1.0${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-b${NORM} <val> --Select ${BOLD}BLAS library${NORM}. Default is ${BOLD}${BLAS}${NORM}."
  echo "${REV}-c${NORM} <val> --Select ${BOLD}compiler${NORM}. Default is ${BOLD}${COMPILER}${NORM}."
  echo "${REV}-d${NORM}       --Build with ${BOLD}Debug mode${NORM} enabled. Default is ${BOLD}${BUILD_TYPE}${NORM}."
  echo "${REV}-e${NORM} <val> --Select ${BOLD}Elemental version${NORM}. Default is ${BOLD}${EL_VER}${NORM}."
  echo "${REV}-m${NORM} <val> --Select ${BOLD}MPI library${NORM}. Default is ${BOLD}${MPI}${NORM}."
  echo "${REV}-s${NORM}       --Build with ${BOLD}sequential initialization mode${NORM} enabled."
  echo "${REV}-t${NORM} <val> --Select ${BOLD}datatype${NORM}. Default is ${BOLD}${DTYPE}${NORM}."
  echo -e "${REV}-h${NORM}       --Displays this help message. No further functions are performed."\\n
  exit 1
}

while getopts "b:c:de:hm:t:" opt; do
  case $opt in
    b)
      BLAS=$OPTARG
      ;;
    c)
      COMPILER=$OPTARG
      ;;
    d)
      BUILD_TYPE=Debug
      ;;
    e)
      EL_VER=$OPTARG
      ;;
    h)
      HELP
      exit 1
      ;;
    m)
      MPI=$OPTARG
      ;;
    s)
      VARIANTS="${VARIANTS} +seq_init"
      ;;
    t)
      DTYPE=$OPTARG
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

# Figure out which cluster we are on
CLUSTER=`hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g'`
ARCH=`uname -m`

PLATFORM=
if [ "${CLUSTER}" == "surface" -o "${CLUSTER}" == "ray" ]; then
  PLATFORM="+gpu"
  EL_VER="${EL_VER}+cublas"
fi

SPACK_OPTIONS="lbann@local build_type=${BUILD_TYPE} dtype=${DTYPE} ${PLATFORM} ${VARIANTS} %${COMPILER} ^elemental@${EL_VER} blas=${BLAS} ^${MPI}"

SPEC="spack spec ${SPACK_OPTIONS}"
CMD="spack setup ${SPACK_OPTIONS}"

DIST=
case ${BUILD_TYPE} in
  Release)
    DIST=rel
    ;;
  Debug)
    DIST=debug
    ;;
  :)
    DIST=unkwn
    ;;
esac

# Create a directory for the build
DIR="${COMPILER}_${ARCH}_${MPI}_${BLAS}_${DIST}"
DIR=${DIR//@/-}

echo "Creating directory ${DIR}"
mkdir -p ${DIR}/build
cd ${DIR}

echo $SPEC
echo $SPEC > spack_build_lbann.sh
$SPEC
err=$?
if [ $err -eq 1 ]; then
  echo "Spack spec command returned error: $err"
  exit -1
fi

echo $CMD
echo $CMD >> spack_build_lbann.sh
chmod +x spack_build_lbann.sh
$CMD
err=$?
if [ $err -eq 1 ]; then
  echo "Spack setup command returned error: $err"
  exit -1
fi

# Find the root of the git repo
cd build
PATH_TO_SRC=
if [ ! -z ${LVL_TO_ROOT_DIR} ]; then
  PATH_TO_SRC="${LVL_TO_ROOT_DIR}/../.."
elif [ ! -z ${ROOT_DIR} ]; then
  PATH_TO_SRC="${ROOT_DIR}"
fi

if [ ! -z ${PATH_TO_SRC} -a -d ${PATH_TO_SRC}/src ]; then
  CMD="../spconfig.py ${PATH_TO_SRC}"
  echo $CMD
  $CMD
fi

# Deal with the fact that spack should not install a package when doing setup"
FIX="spack uninstall -y lbann"
echo $FIX
$FIX
