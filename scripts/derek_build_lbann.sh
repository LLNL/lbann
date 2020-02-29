#!/usr/bin/bash
sys_arch=$(uname -m)

export LBANN_HOME=/g/g13/jones289/workspace/lbann
export LBANN_BUILD_DIR=$LBANN_HOME/build_${sys_arch}
export LBANN_INSTALL_DIR=$LBANN_BUILD_DIR/install

if [ -d "${LBANN_BUILD_DIR}" ]
then
    echo "Directory ${LBANN_BUILD_DIR} exists."
else
    echo "Directory ${LBANN_BUILD_DIR} does not yet exist. Creating it."
    mkdir ${LBANN_BUILD_DIR}
fi

cd ${LBANN_BUILD_DIR}

if [ -d "${LBANN_INSTALL_DIR}" ]
then
    echo "Directory ${LBANN_INSTALL_DIR} exists."
else
    echo "Directory ${LBANN_INSTALL_DIR} does not yet exist. Creating it."
    mkdir ${LBANN_INSTALL_DIR}
fi
spack env create -d . ${LBANN_HOME}/spack_environments/developer_release_cuda_spack.yaml
cp ${LBANN_HOME}/spack_environments/std_versions_and_variants_llnl_lc_cz.yaml .
echo "detected system architecture to be ${sys_arch}"
if [ $sys_arch = "x86_64" ]
then  
    cp ${LBANN_HOME}/spack_environments/externals_x86_64_broadwell_llnl_lc_cz.yaml externals_llnl_lc_cz.yaml # where <arch> = x86_64_broadwell | power9le
elif [ $sys_arch = "ppc64le" ]
then
    cp ${LBANN_HOME}/spack_environments/externals_power9le_llnl_lc_cz.yaml externals_llnl_lc_cz.yaml # where <arch> = x86_64_broadwell | power9le
else
    echo "${sys_arch} not supported"
fi
spack install
spack env loads # Spack creates a file named loads that has all of the correct modules
source ${SPACK_ROOT}/share/spack/setup-env.sh # Rerun setup since spack doesn't modify MODULEPATH unless there are module files defined
source loads
