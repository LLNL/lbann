set -e
CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
LBANN_DIR=$(git rev-parse --show-toplevel)
DEBUG=''
source /usr/share/lmod/lmod/init/bash
source /etc/profile.d/00-modulepath.sh

while :; do
    case ${1} in
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

        -d|--debug)
            # Debug mode
            DEBUG='--debug'
            ;;
        *)
            # Break loop if there are no more options
            break

    esac
    shift
done

if [ "${COMPILER}" == 'clang6' ]; then
    module load clang/6.0.0
    ${LBANN_DIR}/scripts/build_lbann_lc.sh --compiler clang ${DEBUG} --reconfigure --with-conduit
fi


if [ "${COMPILER}" == 'gcc7' ]; then
    module load gcc/7.1.0
    ${LBANN_DIR}/scripts/build_lbann_lc.sh --compiler gnu ${DEBUG} --reconfigure --with-conduit
fi

if [ "${COMPILER}" == 'intel19' ]; then
    module load intel/19.0.0
    ${LBANN_DIR}/scripts/build_lbann_lc.sh --compiler intel ${DEBUG} --reconfigure --with-conduit
fi
