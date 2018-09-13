#!/bin/bash

set -e

if [ ! -z "$bamboo_SPACK_ROOT" ]; then
    . $bamboo_SPACK_ROOT/share/spack/setup-env.sh
fi

CLUSTER=`hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g'`

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
DTYPE=float
EL_VER=develop
if [ "${CLUSTER}" == "ray" -o "${CLUSTER}" == "sierra" ]; then
  MPI=spectrum-mpi
elif [ "${CLUSTER}" == "pascal" -o "${CLUSTER}" == "surface" ]; then
  MPI='mvapich2 +cuda'
else
  MPI=mvapich2
fi
VARIANTS=
GPU=0 # usually ignored

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT -c gcc@7.1.0${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-b${NORM} <val> --Select ${BOLD}BLAS library${NORM}. Default is ${BOLD}${BLAS}${NORM}."
  echo "${REV}-c${NORM} <val> --Select ${BOLD}compiler${NORM}. Default is ${BOLD}${COMPILER}${NORM}."
  echo "${REV}-d${NORM}       --Build with ${BOLD}Debug mode${NORM} enabled. Default is ${BOLD}${BUILD_TYPE}${NORM}."
  echo "${REV}-e${NORM} <val> --Select ${BOLD}Elemental version${NORM}. Default is ${BOLD}${EL_VER}${NORM}."
  echo "${REV}-g${NORM}       --Build with ${BOLD}GPU support${NORM} enabled."
  echo "${REV}-m${NORM} <val> --Select ${BOLD}MPI library${NORM}. Default is ${BOLD}${MPI}${NORM}."
  echo "${REV}-s${NORM}       --Build with ${BOLD}sequential initialization mode${NORM} enabled."
  echo "${REV}-t${NORM} <val> --Select ${BOLD}datatype${NORM}. Default is ${BOLD}${DTYPE}${NORM}."
  echo -e "${REV}-h${NORM}       --Displays this help message. No further functions are performed."\\n
  exit 1
}

while getopts "b:c:de:ghm:st:z" opt; do
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
    g)
      GPU=1
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
    z)
      SPACK_DIRTY=1
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
ARCH=`uname -m`

PLATFORM=
FEATURE=
if [ "${GPU}" == "1" -o "${CLUSTER}" == "surface" -o "${CLUSTER}" == "ray" -o "${CLUSTER}" == "sierra" -o "${CLUSTER}" == "pascal" ]; then
  if [ "${CLUSTER}" == "flash" ]; then
    PLATFORM="+gpu ^cuda@7.5 ^cudnn@5.1"
    FEATURE="_gpu_cuda-7.5_cudnn-5.1"
  elif [ "${CLUSTER}" == "sierra" -o "${CLUSTER}" == "ray" ]; then
    PLATFORM="+gpu ^cuda@9.2.64 ^cudnn@7.0"
    FEATURE="_gpu_cuda-9.2.64_cudnn-7.0"
  elif [ "${CLUSTER}" == "pascal" ]; then
    PLATFORM="+gpu ^cuda@9.1.85 ^cudnn@7.1"
    FEATURE="_gpu_cuda-9.1.85_cudnn-7.1"
  else
    PLATFORM="+gpu"
    FEATURE="_gpu"
  fi
  EL_VER="${EL_VER}+cuda"
else
  PLATFORM="~gpu"
fi

C_FLAGS=
CXX_FLAGS=
Fortran_FLAGS=

DIST=
case ${BUILD_TYPE} in
  Release)
    DIST=rel
    # Don't use the march=native flag for gcc and intel compilers since that
    # wouldn't allow spack to differentiate between optimization sets
    # C_FLAGS="${C_FLAGS} -march=native"
    # CXX_FLAGS="${CXX_FLAGS} -march=native"
    # Fortran_FLAGS="${Fortran_FLAGS} -march=native"
    if [[ (${COMPILER} == gcc@*) ]]; then
        if [ "${CLUSTER}" == "catalyst" ]; then
            ARCH_FLAGS="-march=ivybridge -mtune=ivybridge"
        elif [ "${CLUSTER}" == "quartz"  -o "${CLUSTER}" == "pascal" ]; then
            ARCH_FLAGS="-march=broadwell -mtune=broadwell"
        elif [ "${CLUSTER}" == "surface" ]; then
            ARCH_FLAGS="-march=sandybridge -mtune=sandybridge"
        elif [ "${CLUSTER}" == "flash" ]; then
            ARCH_FLAGS="-march=haswell -mtune=haswell"
        fi
    elif [[ (${COMPILER} == intel@*) ]]; then
        if [ "${CLUSTER}" == "catalyst" ]; then
            ARCH_FLAGS="-march=corei7-avx -mtune=ivybridge"
        elif [ "${CLUSTER}" == "quartz"  -o "${CLUSTER}" == "pascal" ]; then
            ARCH_FLAGS="-march=core-avx2 -mtune=broadwell"
        elif [ "${CLUSTER}" == "surface" ]; then
            ARCH_FLAGS="-march=corei7-avx -mtune=sandybridge"
        elif [ "${CLUSTER}" == "flash" ]; then
            ARCH_FLAGS="-march=core-avx2 -mtune=haswell"
        fi
    elif [[ ${COMPILER} == clang@* ]]; then
        if [ "${CLUSTER}" == "catalyst" -o "${CLUSTER}" == "surface" ]; then
            ARCH_FLAGS="-mavx -march=native"
        elif [ "${CLUSTER}" == "quartz" -o "${CLUSTER}" == "flash"  -o "${CLUSTER}" == "pascal" ]; then
            ARCH_FLAGS="-mavx2 -march=native"
        fi
    fi
    C_FLAGS="-O3 -g ${ARCH_FLAGS}"
    CXX_FLAGS="-O3 -g ${ARCH_FLAGS}"
    Fortran_FLAGS="-O3 -g ${ARCH_FLAGS}"
    ;;
  Debug)
    DIST=debug
    C_FLAGS="-g"
    CXX_FLAGS="-g"
    Fortran_FLAGS="-g"
    ;;
  :)
    DIST=unkwn
    ;;
esac

SPACK_CFLAGS=
if [ ! -z "${C_FLAGS}" ]; then
    SPACK_CFLAGS="cflags=\"${C_FLAGS}\""
fi
SPACK_CXXFLAGS=
if [ ! -z "${CXX_FLAGS}" ]; then
    SPACK_CXXFLAGS="cxxflags=\"${CXX_FLAGS}\""
fi
SPACK_FFLAGS=
if [ ! -z "${Fortran_FLAGS}" ]; then
    SPACK_FFLAGS="fflags=\"${Fortran_FLAGS}\""
fi

SPACK_SETUP_FLAGS=
if [ "${SPACK_DIRTY}" == "1" ]; then
  SPACK_SETUP_FLAGS="--dirty"
fi

if [ "${CLUSTER}" == "ray" ]; then
  MPI="spectrum-mpi@2018.04.27"
fi

SPACK_OPTIONS="lbann@local %${COMPILER} build_type=${BUILD_TYPE} dtype=${DTYPE} ${PLATFORM} ${VARIANTS} ^hydrogen@${EL_VER} build_type=${BUILD_TYPE} blas=${BLAS} ^${MPI}"
# Disable the extra compiler flags until spack supports propagating flags properly
#SPACK_OPTIONS="lbann@local build_type=${BUILD_TYPE} dtype=${DTYPE} ${PLATFORM} ${VARIANTS} %${COMPILER} ${SPACK_CFLAGS} ${SPACK_CXXFLAGS} ${SPACK_FFLAGS} ^elemental@${EL_VER} blas=${BLAS} ^${MPI}"

# Use older cmake to avoid passing -pthread to nvcc
if [ "${CLUSTER}" == "ray" -o "${CLUSTER}" == "sierra" ]; then
  SPACK_OPTIONS="$SPACK_OPTIONS ^cmake@3.9.0"
fi

SPEC="spack spec ${SPACK_OPTIONS}"
CMD="spack setup ${SPACK_SETUP_FLAGS} ${SPACK_OPTIONS}"

# Create a directory for the build
if [ ! -z "$bamboo_SPACK_ROOT" ]; then
  DIR="${CLUSTER}_${COMPILER}_${DIST}"
  DIR=${DIR//@/-}
  DIR=${DIR// /-}
  DIR=${DIR//+/-}
else
  DIR="${CLUSTER}_${COMPILER}_${ARCH}${FEATURE}_${MPI}_${BLAS}_${DIST}"
  DIR=${DIR//@/-}
  DIR=${DIR// /-}
  DIR=${DIR//+/-}
fi
echo "Creating directory ${DIR}"
mkdir -p ${DIR}/build
cd ${DIR}

echo $SPEC
echo $SPEC > spack_build_lbann.sh
eval $SPEC
err=$?
if [ $err -eq 1 ]; then
  echo "Spack spec command returned error: $err"
  exit -1
fi

echo $CMD
echo $CMD >> spack_build_lbann.sh
chmod +x spack_build_lbann.sh
eval $CMD
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
  eval $CMD
fi

# Deal with the fact that spack should not install a package when doing setup"
FIX="spack uninstall --all -y lbann %${COMPILER} build_type=${BUILD_TYPE}"
echo $FIX
if [ ! -z "$bamboo_SPACK_ROOT" ]; then
    eval $FIX &> /dev/null
    exit 0
else
    eval $FIX
fi


