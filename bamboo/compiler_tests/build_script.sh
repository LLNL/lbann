CLUSTER=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')
if [ "${CLUSTER}" != 'surface' ]; then
    source /usr/share/lmod/lmod/init/bash
    source /etc/profile.d/00-modulepath.sh
fi
LBANN_DIR=$(git rev-parse --show-toplevel)
${LBANN_DIR}/scripts/build_lbann_lc.sh --with-conduit
