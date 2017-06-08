#!/bin/bash

# Detect OS version
TOSS=$(uname -r | sed 's/\([0-9][0-9]*\.*\)\-.*/\1/g')
HasGPU=`hostname | grep -e surface -e ray`
ARCH=$(uname -m)

if [ "${TOSS}" == "3.10.0" ]; then
  if [ "${ARCH}" == "x86_64" ]; then
    module load cmake/3.5.2
  elif [ "${ARCH}" == "ppc64le" ]; then
    module load cmake/3.7.2
  fi
  BY_MODULE=1
elif [ "${TOSS}" == "2.6.32" ]; then
  # initialize dotkit
  . /usr/local/tools/dotkit/init.sh
  use cmake-3.4.1
  BY_MODULE=0
else
  # need to initialize modules on earlier versions of TOSS
  . /usr/share/[mM]odules/init/bash
  BY_MODULE=1
fi

# Detect the cuda toolkit version loaded or use default
if [ "${HasGPU}" == "" ] ; then
  echo "This platform has no GPU"
elif [ "$CUDA_PATH" == "" ] || [ `basename "$CUDA_PATH"` == "" ] ; then
  # use default
  CUDATOOLKIT_VERSION=8.0
  module load cudatoolkit/$CUDATOOLKIT_VERSION
  if [ -d /opt/cudatoolkit/$CUDATOOLKIT_VERSION ] ; then
      CUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit/$CUDATOOLKIT_VERSION
  elif [ -d /opt/cudatoolkit-$CUDATOOLKIT_VERSION ] ; then
      CUDA_TOOLKIT_ROOT_DIR=/opt/cudatoolkit-$CUDATOOLKIT_VERSION
  fi
elif [ "${ARCH}" == "ppc64le" ]; then
  CUDA_TOOLKIT_ROOT_DIR=/usr/local/cuda
  CUDATOOLKIT_VERSION=`ls -l ${CUDA_TOOLKIT_ROOT_DIR} | awk '{print $NF}' | cut -d '-' -f 2`
else
  CUDATOOLKIT_VERSION=`basename "$CUDA_PATH" | sed 's/cudatoolkit-//'`
  CUDA_TOOLKIT_ROOT_DIR=$CUDA_PATH
fi

################################################################
# Default options
################################################################

COMPILER=gnu
if [ "${ARCH}" == "x86_64" ]; then
  MPI=mvapich2
elif [ "${ARCH}" == "ppc64le" ]; then
  MPI=spectrum
fi
COMPILER_CC_NAME_VERSION=
BUILD_TYPE=Release
Elemental_DIR=
if [ "${TOSS}" == "3.10.0" ]; then
  OpenCV_DIR=""
  if [ "${ARCH}" == "x86_64" ]; then
    VTUNE_DIR=/usr/tce/packages/vtune/default
  elif [ "${ARCH}" == "ppc64le" ]; then
    VTUNE_DIR=""
  fi
else
  OpenCV_DIR=/usr/gapps/brain/tools/OpenCV/2.4.13
  VTUNE_DIR=/usr/local/tools/vtune
fi
if [ "${ARCH}" == "x86_64" ]; then
  cuDNN_DIR=/usr/gapps/brain/installs/cudnn/v5
elif [ "${ARCH}" == "ppc64le" ]; then
  cuDNN_DIR=""
fi
ELEMENTAL_MATH_LIBS=
PATCH_OPENBLAS=ON
CMAKE_C_FLAGS=
CMAKE_CXX_FLAGS=-DLBANN_SET_EL_RNG
CMAKE_Fortran_FLAGS=
CLEAN_BUILD=0
VERBOSE=0
CMAKE_INSTALL_MESSAGE=LAZY
MAKE_NUM_PROCESSES=$(($(nproc) + 1))
GEN_DOC=0
INSTALL_LBANN=0
ALTERNATE_BUILD_DIR=none
BUILD_SUFFIX=
SEQ_INIT=OFF

# In case that autoconf fails during on-demand buid on surface, try the newer
# version of autoconf installed under '/p/lscratche/brainusr/autoconf/bin'
# by putting it at the beginning of the PATH or use the preinstalled library
# by enabling LIBJPEG_TURBO_DIR
WITH_LIBJPEG_TURBO=ON
#LIBJPEG_TURBO_DIR="/p/lscratche/brainusr/libjpeg-turbo"
#LIBJPEG_TURBO_DIR="/p/lscratchf/brainusr/libjpeg-turbo"

function version_gt() { test "$(printf '%s\n' "$@" | sort -V | head -n 1)" != "$1"; }

if [ "${CLUSTER}" == "surface" ]; then
  AUTOCONF_CUSTOM_DIR=/p/lscratche/brainusr/autoconf/bin
  AUTOCONF_VER_DEFAULT=`autoconf --version | awk '(FNR==1){print $NF}'`
  AUTOCONF_VER_CUSTOM=`${AUTOCONF_CUSTOM_DIR}/autoconf --version | awk '(FNR==1){print $NF}'`

  if version_gt ${AUTOCONF_VER_CUSTOM} ${AUTOCONF_VER_DEFAULT}; then
    export PATH=${AUTOCONF_CUSTOM_DIR}:${PATH}
  fi
fi

################################################################
# Help message
################################################################

function help_message {
  local SCRIPT=$(basename ${0})
  local N=$(tput sgr0)    # Normal text
  local C=$(tput setf 4)  # Colored text
cat << EOF
Build LBANN on an LLNL LC system.
Can be called anywhere in the LBANN project tree.
Usage: ${SCRIPT} [options]
Options:
  ${C}--help${N}                  Display this help message and exit.
  ${C}--compiler${N} <val>        Specify compiler ('gnu' or 'intel' or 'clang').
  ${C}--mpi${N} <val>             Specify MPI library ('mvapich2' or 'openmpi' or 'spectrum').
  ${C}--verbose${N}               Verbose output.
  ${C}--debug${N}                 Build with debug flag.
  ${C}--tbinf${N}                 Build with Tensorboard interface.
  ${C}--vtune${N}                 Build with VTune profiling libraries.
  ${C}--clean-build${N}           Clean build directory before building.
  ${C}--make-processes${N} <val>  Number of parallel processes for make.
  ${C}--doc${N}                   Generate documentation.
  ${C}--install-lbann${N}         Install LBANN headers and dynamic library into the build directory.
  ${C}--build${N}                 Specify alternative build directory; default is <lbann_home>/build.
  ${C}--suffix${N}                Specify suffix for build directory. If you are, e.g, building on surface, your build will be <someplace>/surface.llnl.gov, regardless of your choice of compiler or other flags. This option enables you to specify, e.g: --suffix gnu_debug, in which case your build will be in the directory <someplace>/surface.llnl.gov.gnu_debug
EOF
}

################################################################
# Parse command-line arguments
################################################################

while :; do
  case ${1} in
    -h|--help)
      # Help message
      help_message
      exit 0
      ;;
    --build)
      # Change default build directory
      if [ -n "${2}" ]; then
        ALTERNATE_BUILD_DIR=${2}
        shift
      else
        echo "\"${1}\" option requires a non-empty option argument" >&2
        exit 1
      fi  
      ;;
    --suffix)
      # Specify suffix for build directory
       if [ -n "${2}" ]; then
         BUILD_SUFFIX=${2}
         shift
       else
         echo "\"${1}\" option requires a non-empty option argument" >&2
         exit 1
       fi
       ;;
    --compiler)
      # Choose compiler
      if [ -n "${2}" ]; then
        COMPILER=${2}
        shift
      else
        echo "\"${1}\" option requires a non-empty option argument" >&2
        exit 1
      fi
      ;;
    --mpi)
      # Choose mpi library
      if [ -n "${2}" ]; then
        MPI=${2}
        shift
      else
        echo "\"${1}\" option requires a non-empty option argument" >&2
        exit 1
      fi
      ;;
    -v|--verbose)
      # Verbose output
      VERBOSE=1
      CMAKE_INSTALL_MESSAGE=ALWAYS
      ;;
    -d|--debug)
      # Debug mode
      BUILD_TYPE=Debug
      SEQ_INIT=ON
      ;;
    --tbinf)
      # Tensorboard interface
      WITH_TBINF=ON
      ;;
    --vtune)
      # VTune libraries
      WITH_VTUNE=ON
      ;;
    --clean-build|--build-clean)
      # Clean build directory
      CLEAN_BUILD=1
      ;;
    -j|--make-processes)
      if [ -n "${2}" ]; then
        MAKE_NUM_PROCESSES=${2}
        shift
      else
        echo "\"${1}\" option requires a non-empty option argument" >&2
        exit 1
      fi
      ;;
    --doc)
      # Generate documentation
      GEN_DOC=1
      ;;
    -i|--install-lbann)
      INSTALL_LBANN=1
      ;;
    -?*)
      # Unknown option
      echo "Unknown option (${1})" >&2
      exit 1
      ;;
    *)
      # Break loop if there are no more options
      break
  esac
  shift
done

################################################################
# Load modules
################################################################

if [ "${BY_MODULE}" == "1" ]; then
  module load git
else
  use git
fi

################################################################
# Initialize variables
################################################################

# Detect computing system
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Build and install directories
ROOT_DIR=$(git rev-parse --show-toplevel)
if [ "${ALTERNATE_BUILD_DIR}" !=  "none" ]; then
  BUILD_DIR=${ALTERNATE_BUILD_DIR}/${CLUSTER}.llnl.gov
else
  BUILD_DIR=${ROOT_DIR}/build/${CLUSTER}.llnl.gov
fi
INSTALL_DIR=${BUILD_DIR}

if [ -n "${BUILD_SUFFIX}" ]; then
  BUILD_DIR=${BUILD_DIR}.${BUILD_SUFFIX} 
  INSTALL_DIR=${BUILD_DIR}
fi

mkdir -p ${BUILD_DIR}
mkdir -p ${INSTALL_DIR}

########################
#         Compilers
########################
# Get C/C++/Fortran compilers 

COMPILER_=${COMPILER}
if [ ${COMPILER} == "gnu" ] ; then
  COMPILER_=gcc
fi

if [ "${BY_MODULE}" == "1" ]; then
  if [ "`module list 2> /dev/stdout | grep ${COMPILER_}`" == "" ] ; then
    module load ${COMPILER_}
  fi
  if [ "`module list 2> /dev/stdout | grep ${COMPILER_}`" == "" ] ; then
    echo "Try load ${COMPILER} module first"
    echo The MPI version might not have been compiled with the compiler ${COMPILER} version. Try other versions.
    exit 1
  fi
  COMPILER_BASE="`module show ${COMPILER_} 2> /dev/stdout | grep '\"PATH\"' | cut -d ',' -f 2 | cut -d ')' -f 1 | sed 's/\/bin//' | sed 's/\"//g'`"
else
  if [ ${COMPILER} == "intel" ] ; then
    COMPILER_=ic
  fi
  COMPILER_="`use | sed 's/ //g' | egrep -e "^${COMPILER_}"`"
  if [ "${COMPILER_}" == "" ] ; then
    echo load "\`"${COMPILER}"\` first"
    exit 1
  fi
  COMPILER_BASE="`use -hv ${COMPILER_} | grep 'dk_alter PATH' | awk '{print $3}' | sed 's/\/bin//'`"
fi


if [ "${COMPILER}" == "gnu" ]; then
  CMAKE_C_COMPILER=${COMPILER_BASE}/bin/gcc
  CMAKE_CXX_COMPILER=${COMPILER_BASE}/bin/g++
  CMAKE_Fortran_COMPILER=${COMPILER_BASE}/bin/gfortran
  COMPILER_VERSION=`${CMAKE_C_COMPILER} --version | head -n 1 | awk '(NF==3){print $3}'`
  FORTRAN_LIB=${COMPILER_BASE}/lib64/libgfortran.so

elif [ "${COMPILER}" == "intel" ]; then
  CMAKE_C_COMPILER=${COMPILER_BASE}/bin/icc
  CMAKE_CXX_COMPILER=${COMPILER_BASE}/bin/icpc
  CMAKE_Fortran_COMPILER=${COMPILER_BASE}/bin/ifort
  COMPILER_VERSION=`${CMAKE_C_COMPILER} --version | head -n 1 | awk '(NF==4){print $3}'`
  FORTRAN_LIB=${COMPILER_BASE}/lib/intel64/libifcore.so

elif [ "${COMPILER}" == "clang" ]; then
  # clang depends on gnu fortran library. so, find the dependency
  if [ "${CLUSTER}" == "ray" ] ; then
    #gccdep=`ldd ${COMPILER_BASE}/lib/*.so 2> /dev/null | grep gcc | awk '(NF>2){print $3}' | sort | uniq | head -n 1`
    #GCC_VERSION=`ls -l $gccdep | awk '{print $NF}' | cut -d '-' -f 2 | cut -d '/' -f 1`
    # Forcing to gcc 4.9.3 because of the current way of ray's gcc and various clang installation
    GCC_VERSION=4.9.3
    GNU_DIR=/usr/tcetmp/packages/gcc/gcc-${GCC_VERSION}
  elif [ "${BY_MODULE}" == "1" ]; then
    GCC_VERSION=`ldd ${COMPILER_BASE}/lib/libclang.so | grep gcc- | awk 'BEGIN{FS="/"} {for (i=1;i<=NF; i++) if ($i~/^gcc-/) print $i}' | tail -n 1`
    GNU_DIR=/usr/tce/packages/gcc/${GCC_VERSION}
  else
    GCC_VERSION=`ldd ${COMPILER_BASE}/lib/libclang.so | grep gnu | awk 'BEGIN{FS="/"} {for (i=1;i<=NF; i++) if ($i~/^gnu$/) print $(i+1)}' | tail -n 1`
    GNU_DIR=/usr/apps/gnu/${GCC_VERSION}
  fi

  CMAKE_C_COMPILER=${COMPILER_BASE}/bin/clang
  CMAKE_CXX_COMPILER=${COMPILER_BASE}/bin/clang++
  CMAKE_Fortran_COMPILER=${GNU_DIR}/bin/gfortran
  FORTRAN_LIB=${GNU_DIR}/lib64/libgfortran.so
  COMPILER_VERSION=`${CMAKE_C_COMPILER} --version | awk '(($1=="clang")&&($2=="version")){print $3}'`
  export MPICH_FC=${GNU_DIR}/bin/gfortran
  #MPI_Fortran_COMPILER="${MPI_DIR}/bin/mpifort -fc=${CMAKE_Fortran_COMPILER}" $ done by exporting MPICH_FC

else
  # Unrecognized compiler
  echo "Unrecognized compiler (${COMPILER})"
  exit 1
fi

export CC=${CMAKE_C_COMPILER}
export CXX=${CMAKE_CXX_COMPILER}
echo ${CMAKE_C_COMPILER}        > COMPILERS_USED.txt
echo ${CMAKE_CXX_COMPILER}     >> COMPILERS_USED.txt
echo ${CMAKE_Fortran_COMPILER} >> COMPILERS_USED.txt


########################
#         MPI          
########################
# Get top-level MPI directory and corresponding MPI wrappers of the compilers  

if [ "${ARCH}" == "x86_64" ] && [ "${MPI}" != "mvapich2" ] && [ "${MPI}" != "openmpi" ] ; then
  echo "Invalid MPI library selected ${MPI}"
  exit 1
fi

if [ "${ARCH}" == "ppc64le" ] && [ "${MPI}" != "spectrum" ] ; then
  echo "Invalid MPI library selected ${MPI}"
  exit 1
elif [ "${MPI}" == "spectrum" ] ; then
  MPI=spectrum-mpi
fi

if [ "${BY_MODULE}" == "1" ]; then
  if [ "`module list 2> /dev/stdout | grep ${MPI}`" == "" ] ; then
    module load ${MPI}
  fi
  if [ "`module list 2> /dev/stdout | grep ${MPI}`" == "" ] ; then
    echo "Try load ${MPI} module first"
    echo The MPI version might not have been compiled with the compiler ${COMPILER} version. Try other versions.
    exit 1
  fi
  MPI_DIR="`module show ${MPI} 2> /dev/stdout | grep '\"PATH\"' | cut -d ',' -f 2 | cut -d ')' -f 1 | sed 's/\/bin//' | sed 's/\"//g'`"
else
  if [ "`use | grep ${MPI}`" == "" ] ; then
    echo "do \'use ${MPI}\'"
    exit 1
  fi
  MPI_VER="`use | grep ${MPI}`"
  if [ "${COMPILER}" == "gnu" ] || [ "${COMPILER}" == "intel" ] || [ "${COMPILER}" == "pgi" ] ; then
    if [ "`echo ${MPI_VER} | grep ${COMPILER}`" == "" ] ; then
      echo "switch to an MPI version that is consistent with ${COMPILER} compilers"
      exit 1
    fi
  fi
  MPI_DIR="`use -hv ${MPI_VER} | grep 'dk_alter PATH' | awk '{print $3}' | sed 's/\/bin//'`"
fi

# Get MPI compilers
MPI_C_COMPILER=${MPI_DIR}/bin/mpicc
MPI_CXX_COMPILER=${MPI_DIR}/bin/mpicxx
if [ "${MPI_Fortran_COMPILER}" == "" ]; then
  MPI_Fortran_COMPILER=${MPI_DIR}/bin/mpifort
fi

# Get CUDA and cuDNN
if [ "${HasGPU}" != "" ] ; then
  WITH_CUDA=ON
  WITH_CUDNN=ON
  ELEMENTAL_USE_CUBLAS=1
fi


echo ${MPI_C_COMPILER}       >> COMPILERS_USED.txt
echo ${MPI_CXX_COMPILER}     >> COMPILERS_USED.txt
echo ${MPI_Fortran_COMPILER} >> COMPILERS_USED.txt
${MPI_Fortran_COMPILER} --version >> COMPILERS_USED.txt
echo ${FORTRAN_LIB} >> COMPILERS_USED.txt
autoconf --version >> COMPILERS_USED.txt


################################################################
# Build LBANN
################################################################

# Work in build directory
pushd ${BUILD_DIR}

  # Clean up build directory
  if [ ${CLEAN_BUILD} -ne 0 ]; then
    CLEAN_COMMAND="rm -rf ${BUILD_DIR}/*"
    if [ ${VERBOSE} -ne 0 ]; then
      echo "${CLEAN_COMMAND}"
    fi
    ${CLEAN_COMMAND}
  fi

# ATM: goes after Elemental_DIR
#-D OpenCV_DIR=${OpenCV_DIR} \

  # Configure build with CMake
  CONFIGURE_COMMAND=$(cat << EOF
cmake \
-D CMAKE_BUILD_TYPE=${BUILD_TYPE} \
-D CMAKE_INSTALL_MESSAGE=${CMAKE_INSTALL_MESSAGE} \
-D CMAKE_INSTALL_PREFIX=${INSTALL_DIR} \
-D CMAKE_CXX_COMPILER=${CMAKE_CXX_COMPILER} \
-D CMAKE_C_COMPILER=${CMAKE_C_COMPILER} \
-D CMAKE_Fortran_COMPILER=${CMAKE_Fortran_COMPILER} \
-D GFORTRAN_LIB=${FORTRAN_LIB} \
-D MPI_CXX_COMPILER=${MPI_CXX_COMPILER} \
-D MPI_C_COMPILER=${MPI_C_COMPILER} \
-D MPI_Fortran_COMPILER=${MPI_Fortran_COMPILER} \
-D CMAKE_CXX_FLAGS=${CMAKE_CXX_FLAGS} \
-D CMAKE_C_FLAGS=${CMAKE_C_FLAGS} \
-D CMAKE_Fortran_FLAGS=${CMAKE_Fortran_FLAGS} \
-D WITH_CUDA=${WITH_CUDA} \
-D WITH_CUDNN=${WITH_CUDNN} \
-D WITH_TBINF=${WITH_TBINF} \
-D WITH_VTUNE=${WITH_VTUNE} \
-D Elemental_DIR=${Elemental_DIR} \
-D CUDA_TOOLKIT_ROOT_DIR=${CUDA_TOOLKIT_ROOT_DIR} \
-D cuDNN_DIR=${cuDNN_DIR} \
-D VTUNE_DIR=${VTUNE_DIR} \
-D ELEMENTAL_MATH_LIBS=${ELEMENTAL_MATH_LIBS} \
-D VERBOSE=${VERBOSE} \
-D MAKE_NUM_PROCESSES=${MAKE_NUM_PROCESSES} \
-D LBANN_HOME=${ROOT_DIR} \
-D SEQ_INIT=${SEQ_INIT} \
-D COMPILER_VERSION=${COMPILER_VERSION} \
-D COMPILER_BASE=${COMPILER_BASE} \
-D WITH_LIBJPEG_TURBO=${WITH_LIBJPEG_TURBO} \
-D LIBJPEG_TURBO_DIR=${LIBJPEG_TURBO_DIR} \
-D PATCH_OPENBLAS=${PATCH_OPENBLAS} \
-D ELEMENTAL_USE_CUBLAS=${ELEMENTAL_USE_CUBLAS}
${ROOT_DIR}
EOF
)

echo "${CONFIGURE_COMMAND}" > build_command_used.txt

  if [ ${VERBOSE} -ne 0 ]; then
    echo "${CONFIGURE_COMMAND}"
  fi
  ${CONFIGURE_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "CONFIGURE FAILED"
    echo "--------------------"
    exit 1
  fi

  # Build LBANN with make
  BUILD_COMMAND="make -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
  if [ ${VERBOSE} -ne 0 ]; then
    echo "${BUILD_COMMAND}"
  fi
  ${BUILD_COMMAND}
  if [ $? -ne 0 ] ; then
    echo "--------------------"
    echo "MAKE FAILED"
    echo "--------------------"
    exit 1
  fi

  # Install LBANN with make
  if [ ${INSTALL_LBANN} -ne 0 ]; then
      INSTALL_COMMAND="make install -j${MAKE_NUM_PROCESSES} VERBOSE=${VERBOSE}"
      if [ ${VERBOSE} -ne 0 ]; then
          echo "${INSTALL_COMMAND}"
      fi
      ${INSTALL_COMMAND}
      if [ $? -ne 0 ] ; then
          echo "--------------------"
          echo "MAKE INSTALL FAILED"
          echo "--------------------"
          exit 1
      fi
  fi

  # Generate documentation with make
  if [ ${GEN_DOC} -ne 0 ]; then
    DOC_COMMAND="make doc"
    if [ ${VERBOSE} -ne 0 ]; then
      echo "${DOC_COMMAND}"
    fi
    ${DOC_COMMAND}
    if [ $? -ne 0 ] ; then
      echo "--------------------"
      echo "MAKE DOC FAILED"
      echo "--------------------"
      exit 1
    fi
  fi
  
popd
