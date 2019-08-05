BUILD_DIR=
COMPILER=
while :; do
    case ${1} in
        -b|--build-dir)
            # Set build directory
            # -n: check if string has non-zero length.
            if [ -n "${2}" ]; then
                BUILD_DIR=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                help_message
                exit 1
            fi
            ;;
        -c|--compiler)
            # Set compiler name
            # -n: check if string has non-zero length.
            if [ -n "${2}" ]; then
                COMPILER=${2}
                shift
            else
                echo "\"${1}\" option requires a non-empty option argument" >&2
                help_message
                exit 1
            fi
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

cd ${BUILD_DIR}
pwd
LBANN_DIR=$(git rev-parse --show-toplevel)
echo $LBANN_DIR
OUTPUT_DIR=${LBANN_DIR}/bamboo/sequential_catch_tests/${COMPILER}
echo $OUTPUT_DIR
ctest --no-compress-output -T Test
mkdir ${OUTPUT_DIR}/Testing
rsync -rautivh Testing/ $OUTPUT_DIR/Testing
cd -
pwd
