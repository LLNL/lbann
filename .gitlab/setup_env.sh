# This is a collection of common variables and whatnot that may change
# based on the value of "${cluster}" or other variables.

# To make things work with modules, the user can set "COMPILER_FAMILY"
# to "gnu", "clang", "amdclang", or "cray" and the suitable compiler
# paths will be deduced from the current PATH. Alternatively, users
# can set "CC"/"CXX" directly, in which case the
# "COMPILER_FAMILY" variable will be ignored.

# Prefer RPATH to RUNPATH (stability over flexibility)
common_linker_flags="-Wl,--disable-new-dtags"
CFLAGS=${CFLAGS:-""}
CXXFLAGS=${CXXFLAGS:-""}
LDFLAGS=${LDFLAGS:-""}
LDFLAGS="${common_linker_flags} ${LDFLAGS}"

compiler_family=${COMPILER_FAMILY:-gnu}
case "${compiler_family,,}" in
    gnu|gcc)
        CC=${CC:-$(command -v gcc)}
        CXX=${CXX:-$(command -v g++)}
        EXTRA_LINK_FLAGS="-fuse-ld=gold ${common_linker_flags}"
        ;;
    clang)
        CC=${CC:-$(command -v clang)}
        CXX=${CXX:-$(command -v clang++)}
        EXTRA_LINK_FLAGS="-fuse-ld=lld ${common_linker_flags}"
        ;;
    amdclang)
        CC=${CC:-$(command -v amdclang)}
        CXX=${CXX:-$(command -v amdclang++)}
        ROCM_VER=$(basename ${ROCM_PATH})
        ROCM_VER_NUM=$(echo "${ROCM_VER}" | tr -d '[a-z]')
        COMPILER_VER="amdclang${ROCM_VER_NUM}"
        EXTRA_LINK_FLAGS="-fuse-ld=lld ${common_linker_flags}"
        ;;
    cray)
        CC=${CC:-$(command -v cc)}
        CXX=${CXX:-$(command -v CC)}
        EXTRA_LINK_FLAGS="-fuse-ld=lld ${common_linker_flags}"
        ;;
    craycc)
        CC=${CC:-$(command -v craycc)}
        CXX=${CXX:-$(command -v craycxx)}
        EXTRA_LINK_FLAGS="-fuse-ld=lld ${common_linker_flags}"
        ;;
    *)
        echo "Unknown compiler family: ${compiler_family}. Using gnu."
        CC=${CC:-$(command -v gcc)}
        CXX=${CXX:-$(command -v g++)}
        EXTRA_LINK_FLAGS="-fuse-ld=gold ${common_linker_flags}"
        ;;
esac

# Set the compiler version based on the path of the compiler
COMPILER_VER=${COMPILER_VER:-"$(basename $(dirname $(dirname $(which ${CC}))))"}

# HIP/CUDA configuration and launcher are platform-specific
CUDACXX=${CUDACXX:=""}
CUDAHOSTCXX=${CUDAHOSTCXX:=${CXX}}

cuda_platform=OFF
rocm_platform=OFF

launcher=mpiexec

extra_rpaths=${extra_rpaths:-""}

# Set to the preferred install directory for the external dependencies
CI_STABLE_DEPENDENCIES_ROOT=/usr/workspace/lbann/ci_stable_dependencies
INSTALL_EXTERNALS_ROOT=${CI_STABLE_DEPENDENCIES_ROOT}/${cluster}

case "${cluster}" in
    pascal)
        CUDACXX=${CUDACXX:-$(command -v nvcc)}
        CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
        cuda_platform=ON
        gpu_arch=60
        launcher=slurm
        CUDA_VER=$(basename ${CUDA_HOME})
        SYSTEM_INSTALL_PREFIX_EXTERNALS=${CUDA_VER}/${COMPILER_VER}/openmpi-4.1.2
        ;;
    lassen)
        CUDACXX=${CUDACXX:-$(command -v nvcc)}
        CUDAHOSTCXX=${CUDAHOSTCXX:-${CXX}}
        cuda_platform=ON
        gpu_arch=70
        launcher=lsf
        CUDA_VER=$(basename ${CUDA_HOME})
        SYSTEM_INSTALL_PREFIX_EXTERNALS=${CUDA_VER}/${COMPILER_VER}/spectrum-mpi-rolling-release
        export CMAKE_PREFIX_PATH="${CI_STABLE_DEPENDENCIES_ROOT}/${cluster}/${CUDA_VER}/nccl_2.20.3-1+cuda12.2_ppc64le:${CI_STABLE_DEPENDENCIES_ROOT}/${cluster}/${CUDA_VER}/cudnn-linux-ppc64le-8.9.7.29_cuda12-archive:${CMAKE_PREFIX_PATH:-""}"
        ;;
    tioga)
        cray_libs_dir=${CRAYLIBS_X86_64:-""}
        if [[ -n "${cray_libs_dir}" ]]
        then
            extra_rpaths="${cray_libs_dir}:${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${extra_rpaths}"
            export LD_LIBRARY_PATH=${CRAY_LD_LIBRARY_PATH}:${LD_LIBRARY_PATH}
        else
            extra_rpaths="${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${extra_rpaths}"
        fi
        rocm_platform=ON
#        gpu_arch=gfx90a,gfx942
        gpu_arch=gfx90a
        launcher=flux
        ROCM_VER=$(basename ${ROCM_PATH})
        PE_ENV_lc=$(echo "${PE_ENV}" | tr '[:upper:]' '[:lower:]')
        # case "${compiler_family,,}" in
        #     craycc)
        #         PE_ENV_lc=${PE_ENV_lc}cc
        #         ;;
        #     *)
        #         ;;
        # esac
        SYSTEM_INSTALL_PREFIX_EXTERNALS=${ROCM_VER}/${PE_ENV_lc}/cray-mpich-${CRAY_MPICH_VERSION}
        ;;
    corona)
        extra_rpaths="${ROCM_PATH}/lib:${ROCM_PATH}/llvm/lib:${extra_rpaths}"
        rocm_platform=ON
        gpu_arch=gfx906
        launcher=flux
        ROCM_VER=$(basename ${ROCM_PATH})
        SYSTEM_INSTALL_PREFIX_EXTERNALS=${ROCM_VER}/${COMPILER_VER}/openmpi-4.1.2
        ;;
    *)
        ;;
esac

export CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH:-""}
ci_core_cmake_prefix_path="${INSTALL_EXTERNALS_ROOT}/${SYSTEM_INSTALL_PREFIX_EXTERNALS}/logs/lbann_sb_suggested_cmake_prefix_path.sh"
if [[ -e ${ci_core_cmake_prefix_path} ]]; then
    source ${ci_core_cmake_prefix_path}
fi
if [[ "${build_half:-""}" = "ON" ]]; then
    export CMAKE_PREFIX_PATH=${CI_STABLE_DEPENDENCIES_ROOT}/half-2.1.0:${CMAKE_PREFIX_PATH}
fi
case "${cluster}" in
    tioga)
        ROCM_VER=$(basename ${ROCM_PATH})
        if [[ "${ROCM_VER}" = "6.2.0" ]]; then
            CMAKE_PREFIX_PATH=/p/vast1/lbann/stable_dependencies/${cluster}/rocm-6.2.0/miopen:${CMAKE_PREFIX_PATH}
        fi
        ;;
    *)
        ;;
esac
CMAKE_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/;}

# Improve debugging info and remove some misguided warnings. These are
# passed only to the LBANN stack.  Add -v for debugging
EXTRA_CXX_FLAGS="-g3 -Wno-deprecated-declarations"
EXTRA_HIP_FLAGS="-g3 -Wno-deprecated-declarations"

# Update the location of external packages
FWD_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/|}

# Make sure the compilers and flags are exported
export CC CXX CUDACXX CUDAHOSTCXX CFLAGS CXXFLAGS LDFLAGS
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Environment Info"
echo "~~~~~"
echo "~~~~~  Cluster: ${cluster}"
echo "~~~~~  CUDA? ${cuda_platform}"
echo "~~~~~  ROCm? ${rocm_platform}"
echo "~~~~~  GPU arch: ${gpu_arch}"
echo "~~~~~  Launcher: ${launcher}"
echo "~~~~~"
echo "~~~~~  Compiler family: ${compiler_family}"
echo "~~~~~  Compiler version: ${COMPILER_VER}"
echo "~~~~~  CC: ${CC}"
echo "~~~~~  CXX: ${CXX}"
echo "~~~~~  CUDACXX: ${CUDACXX}"
echo "~~~~~  CUDAHOSTCXX: ${CUDAHOSTCXX}"
echo "~~~~~"
echo "~~~~~  CFLAGS: ${CFLAGS}"
echo "~~~~~  CXXFLAGS: ${CXXFLAGS}"
echo "~~~~~  LDFLAGS: ${LDFLAGS}"
echo "~~~~~  Extra rpaths: ${extra_rpaths}"
echo "~~~~~  CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
echo "-----"
echo "-----  Dependency Flags:"
echo "-----  HALF: \"${build_half:-""}\""
echo "-----  DISTCONV: \"${build_distconv:-""}\""
echo "-----  FFT: \"${build_fft:-""}\""
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

# Get Breathe, gcovr, and Ninja. Putting this off to the side because
# I don't want to tweak "the real" python environment, but it's just
# these one or two things so it's not worth a venv.
if [[ -n "${run_coverage:-""}" ]]
then
    python_pkgs="ninja gcovr"
else
    python_pkgs="ninja"
fi

VENV_DIR="${TMPDIR}/${USER}/lbann_venv"
CMD="python3 -m venv ${VENV_DIR}"
echo "${CMD}"
${CMD}
CMD="source ${VENV_DIR}/bin/activate"
echo "${CMD}"
${CMD}

export PYTHONUSERBASE=${TMPDIR}/${USER}/python/${cluster}
export PATH=${PYTHONUSERBASE}/bin:${PATH}
CMD="python3 -m pip install --prefix ${PYTHONUSERBASE} ${python_pkgs}"
echo "${CMD}"
${CMD}

# Make sure the PYTHONPATH is all good.
export PYTHONPATH=$(ls --color=no -1 -d ${PYTHONUSERBASE}/lib/python*/site-packages | paste -sd ":" - ):${PYTHONPATH:-""}
