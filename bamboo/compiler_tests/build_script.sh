source /usr/share/lmod/lmod/init/bash
source /etc/profile.d/00-modulepath.sh
LBANN_DIR=$(git rev-parse --show-toplevel)
${LBANN_DIR}/scripts/build_lbann_lc.sh --with-conduit
