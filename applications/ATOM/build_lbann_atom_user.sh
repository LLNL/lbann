#LBANN_REPO_PATH="https://github.com/LLNL/lbann/raw/develop"
LBANN_REPO_PATH="https://github.com/bvanessen/lbann/raw/extensible_build_script"
wget --backups=1 ${LBANN_REPO_PATH}/scripts/build_lbann.sh \
    ${LBANN_REPO_PATH}/scripts/customize_build_env.sh \
    ${LBANN_REPO_PATH}/scripts/utilities.sh \
    ${LBANN_REPO_PATH}/applications/ATOM/external_packages_atom.sh

chmod +x build_lbann.sh customize_build_env.sh utilities.sh external_packages_atom.sh

# Identify the center that we are running at
CENTER=
# Customize the build based on the center
source customize_build_env.sh
#source $(dirname ${BASH_SOURCE})/customize_build_env.sh
set_center_specific_fields

if [[ -z "${CENTER}" ]]; then
    echo "Building at unknown HPC Center -- please setup appropriate externals and system installed packages"
    echo "https://lbann.readthedocs.io/en/latest/building_lbann.html#building-installing-lbann-as-a-user"
else
    echo "Building using preconfigured defaults for ${CENTER}"
fi

CENTER_USER_VARIANTS=
SPACK_ARCH_TARGET=$(spack arch -t)
set_center_specific_variants ${CENTER} ${SPACK_ARCH_TARGET}

CMD="./build_lbann.sh -j $(($(nproc)+2)) -d -s -u develop -e external_packages_atom.sh -- ${CENTER_USER_VARIANTS}"
echo ${CMD}
${CMD}
