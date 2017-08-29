#Disable unbound variable error
set +u

CLUSTER=`hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g'`

#Here is a way to check if you are running in sourced environment
#[[ $BASH_SOURCE != $0 ]] && printf '%s running sourced ...\n' "$BASH_SOURCE"

#Set Script Name variable
SCRIPT=`basename ${BASH_SOURCE}`
EL_VER=El_0.86/v86-6ec56a
COMPILER=intel_cc
#MPI=openmpi
MPI=mvapich2

#Set fonts for Help.
NORM=`tput sgr0`
BOLD=`tput bold`
REV=`tput smso`

#Help function
function HELP {
  echo -e \\n"Help documentation for ${BOLD}${SCRIPT}.${NORM}"\\n
  echo -e "${REV}Basic usage:${NORM} ${BOLD}$SCRIPT${NORM}"\\n
  echo "Command line switches are optional. The following switches are recognized."
  echo "${REV}-c${NORM} <val> --Sets the ${BOLD}compiler name${NORM}. Default is ${BOLD}${COMPILER}${NORM}."
  echo "${REV}-m${NORM} <val> --Sets the ${BOLD}mpi library${NORM}. Default is ${BOLD}${MPI}${NORM}."
  echo "${REV}-v${NORM} <val> --Sets the number of ${BOLD}version of libElemental${NORM}. Default is ${BOLD}${EL_VER}${NORM}."
  echo -e "${REV}-h${NORM}    --Displays this help message. No further functions are performed."\\n
  exit 1
}

while getopts ":c:hm:v:" opt; do
  case $opt in
    c)
      COMPILER=$OPTARG
      ;;
    h)
      HELP
      exit 1
      ;;
    m)
      MPI=$OPTARG
      ;;
    v)
      EL_VER=$OPTARG
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

# If the script is sourced setup the compiler modules
#if  [[ $BASH_SOURCE != $0 ]] ; then
#    module load intel/14.0
#    module load mkl/14.0
#fi

#source /usr/gapps/brain/tools/Elemental/${EL_VER}/${COMPILER}/setup_libElemental_env.sh

BOOST_LIB_PATH=/usr/gapps/brain/installs/boost_stages/intel
BOOST_VER=boost_1_58_0
BOOST_LIB=${BOOST_LIB_PATH}/${BOOST_VER}/lib

export LD_LIBRARY_PATH=${BOOST_LIB}:$LD_LIBRARY_PATH
