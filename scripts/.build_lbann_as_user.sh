LBANN_REPO_PATH="https://github.com/LLNL/lbann/raw/develop"
curl -fsSL -O ${LBANN_REPO_PATH}/scripts/build_lbann.sh
curl -fsSL -O ${LBANN_REPO_PATH}/scripts/customize_build_env.sh
curl -fsSL -O ${LBANN_REPO_PATH}/scripts/utilities.sh
curl -fsSL -O ${LBANN_REPO_PATH}/scripts/find_externals_and_lbann_top_level_dependencies.py
curl -fsSL -O ${LBANN_REPO_PATH}/ci_test/requirements.txt

chmod +x build_lbann.sh customize_build_env.sh utilities.sh find_externals_and_lbann_top_level_dependencies.py

# Identify the center that we are running at
CENTER=
# Customize the build based on the center
source customize_build_env.sh
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

if [[ -n "${LBANN_VARIANTS}" ]]; then
    echo "Overriding ${CENTER} variants: ${CENTER_USER_VARIANTS}"
    VARIANTS="${LBANN_VARIANTS}"
else
    if [[ -n "${CENTER_USER_VARIANTS}" ]]; then
        echo "Configuring LBANN with the following standard variants: ${CENTER_USER_VARIANTS}"
    fi
    VARIANTS="${CENTER_USER_VARIANTS}"
fi

if [[ -n "${LBANN_VERSION}" ]]; then
    echo "Overriding default version: develop"
    VERSION="${LBANN_VERSION}"
else
    VERSION="develop"
fi

if [[ -n "${LBANN_EXTRAS}" ]]; then
    echo "Overriding default extras: ${EXTRAS}"
    EXTRAS="${LBANN_EXTRAS}"
fi

CMD="./build_lbann.sh -j $(($(nproc)+2)) -d -s -u ${VERSION} --pip ./requirements.txt ${EXTRAS} -- ${VARIANTS}"
echo ${CMD}
${CMD}
