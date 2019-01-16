GIT_TOPLEVEL=`git rev-parse --show-toplevel`
LBANN_ROOT=${LBANN_ROOT:-${GIT_TOPLEVEL}}
LBANN_PROTO_DIR="${LBANN_ROOT}/src/proto"
LBANN_PROTO="${LBANN_PROTO_DIR}/lbann.proto"
PROTOC=`which protoc`

${PROTOC} -I=${LBANN_PROTO_DIR} ${LBANN_PROTO} --python_out=.
