#!/bin/sh

# Grab some helper functions
source $(dirname ${BASH_SOURCE})/superbuild_externals.sh

# Function to setup the following fields based on the center that you are running at:
# CENTER
# BUILD_SUFFIX
# COMPILER
# NINJA_NUM_PROCESSES
set_center_specific_fields()
{
    SYS=$(uname -s)

    if [[ ${SYS} = "Darwin" ]]; then
        CENTER="osx"
        COMPILER="clang"
        BUILD_SUFFIX=llnl.gov
    else
        CORI=$([[ $(hostname) =~ (cori|cgpu) ]] && echo 1 || echo 0)
        PERLMUTTER=$([[ $(printenv LMOD_SYSTEM_NAME) =~ (perlmutter) ]] && echo 1 || echo 0)
#	LMOD_SITE_NAME
	DOMAINNAME=$(python3 -c 'import socket; domain = socket.getfqdn().split("."); print(domain[-2] + "." + domain[-1]) if len(domain) > 1 else print(domain)')
#	domainname -A | egrep -o '[a-z]+\.[a-z]+( |$)' | sort -u
        if [[ ${CORI} -eq 1 || ${PERLMUTTER} -eq 1 ]]; then
            CENTER="nersc"
        elif [[ ${DOMAINNAME} = "ornl.gov" ]]; then
            CENTER="olcf"
            NINJA_NUM_PROCESSES=16 # Don't let OLCF kill build jobs
        elif [[ ${DOMAINNAME} = "llnl.gov" ]]; then
            CENTER="llnl_lc"
        elif [[ ${DOMAINNAME} = "riken.jp" ]]; then
            CENTER="riken"
        else
            CENTER="unknown"
        fi
        COMPILER="gnu"
    fi
}

# Function to setup the following fields based on the center that you are running at:
# GPU_ARCH_VARIANTS
# CMAKE_GPU_ARCH
set_center_specific_gpu_arch()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le") # Lassen
                GPU_ARCH_VARIANTS="cuda_arch=70"
                CMAKE_GPU_ARCH="70"
                ;;
            "broadwell") # Pascal
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="60"
                ;;
            "haswell") # RZHasGPU
                GPU_ARCH_VARIANTS="cuda_arch=37"
                CMAKE_GPU_ARCH="37"
                ;;
            "ivybridge") # Catalyst
                ;;
            "sandybridge") # Surface
                GPU_ARCH_VARIANTS="cuda_arch=35"
                CMAKE_GPU_ARCH="35"
                ;;
            "zen" | "zen2") # Corona
                # Use a HIP Clang variant
                GPU_ARCH_VARIANTS="amdgpu_target=gfx906"
                ;;
            "zen3") # Tioga, RZVernal
                # Use a HIP Clang variant
                GPU_ARCH_VARIANTS="amdgpu_target=gfx90a"
                ;;
            *)
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "zen3") # Crusher, Frontier
                GPU_ARCH_VARIANTS="amdgpu_target=gfx90a"
                ;;
            *)
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "zen3") # Perlmutter
                GPU_ARCH_VARIANTS="cuda_arch=80"
                CMAKE_GPU_ARCH="80"
                ;;
            *)
                ;;
        esac
    fi
}


set_center_specific_modules()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        # Disable the StdEnv for systems in LC
        case ${spack_arch_target} in
            "power9le") # Lassen
                MODULE_CMD_GCC="module load StdEnv gcc/8.3.1 cuda/11.8.0 spectrum-mpi/rolling-release cmake/3.23.1 essl/6.3.0.2 python/3.8.2"
                MODULE_CMD_CLANG="module load clang/10.0.1-gcc-8.3.1 cuda/11.6.1 spectrum-mpi/rolling-release python/3.7.2"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface

                MODULE_CMD_GCC="module load jobutils/1.0 StdEnv gcc/10.3.1-magic ninja/1.11.1 openmpi/4.1.2 cuda/11.8.0 python/3.9.12"
                # Note that clang is installed in /usr/workspace/brain/tom/pascal/llvm/latest/ and it is version 17.0.0
                MODULE_CMD_CLANG="module load gcc/10.3.1 cuda/11.8.0 mvapich2/2.3.7 python/3.9.12"
                ;;
            "ivybridge" | "cascadelake") # Catalyst, Ruby
                MODULE_CMD="module load gcc/10.2.1 mvapich2/2.3.6 python/3.7.2"
                ;;
            "zen" | "zen2") # Corona
                MODULE_CMD="module load StdEnv gcc/10.3.1-magic openmpi/4.1.2 git/2.36.1 cmake/3.26.3 emacs/28.2 rocm/5.7.0"
                # ; ml use /opt/toss/modules/modulefiles && ml openmpi-gnu/4.1
                ;;
            "zen3") # Tioga, RZVernal
                MODULE_CMD="module load craype-x86-trento craype-network-ofi libfabric/2.1 perftools-base/23.09.0 cce/17.0.0 craype/2.7.23 cray-mpich/8.1.28 cray-libsci/23.09.1.1 PrgEnv-cray StdEnv rocm/5.7.1 cmake/3.24.2"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                MODULE_CMD="module load gcc/9.3.0 cuda/11.1.1 spectrum-mpi/10.3.1.2-20200121"
                ;;
            "zen3") # Frontier, Crusher
                MODULE_CMD="module load ums/default ums002/default; module unload rocm; module load craype-x86-trento craype-network-ofi libfabric/1.15.2.0 perftools-base/22.12.0 amd/5.5.1 craype/2.7.21 cray-mpich/8.1.26 openblas/0.3.17-omp PrgEnv-amd/8.4.0 cmake/3.23.2 cray-python/3.9.13.1 hdf5/1.14.0"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                MODULE_CMD="module purge; module load cgpu modules/3.2.11.4 gcc/8.3.0 cuda/11.1.1 openmpi/4.0.3 cmake/3.18.2"
                ;;
            "zen3") # Perlmutter
		MODULE_CMD="module load PrgEnv-cray/8.4.0 craype-x86-milan libfabric/1.11.0.4.116 craype-network-ofi cmake/3.22.0 cce/13.0.2 craype/2.7.15 cray-mpich/8.1.15 cray-libsci/21.08.1.2 nccl/2.11.4 cudnn/8.3.2 cray-python/3.9.7.1 craype-accel-host cudatoolkit/11.5"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                MODULE_CMD="module load system/fx700 gcc/10.1 gcc/openmpi/4.0.3"
                ;;
            *)
                echo "No pre-specified modules found for this system. Make sure to setup your own"
                ;;
        esac
    else
        echo "No pre-specified modules found for this system. Make sure to setup your own"
    fi
}

set_center_specific_spack_dependencies()
{
    local center="$1"
    local spack_arch_target="$2"

    if [[ ${center} = "llnl_lc" ]]; then
        if [[ -n "${USE_CENTER_MIRRORS:-}" ]]; then
            POSSIBLE_MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
            for m in ${POSSIBLE_MIRRORS}
            do
                if [[ -r "${m}" ]]; then
                    MIRRORS="${m} $MIRRORS"
                fi
            done
        fi
        # CENTER_UPSTREAM_PATH="/p/vast1/lbann/spack_installed_packages/opt/spack"
        # MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
        case ${spack_arch_target} in
            "power9le") # Lassen
                CENTER_COMPILER_PATHS="/usr/tce/packages/gcc/gcc-8.3.1 /usr/tce/packages/clang/clang-10.0.1-gcc-8.3.1/"
                CENTER_COMPILER="%gcc@8.3.1"
                DEPENDENTS_CENTER_COMPILER="%gcc@8.3.1"
                CENTER_DEPENDENCIES="^spectrum-mpi ^cuda@11.8.89 ^libtool@2.4.2 ^python@3.9.10: ^protobuf@3.21.5 ^py-protobuf@4.21.5 ^nccl@2.19.4 ^hwloc@1.11.8"
                CENTER_BLAS_LIBRARY="blas=openblas"
                CENTER_PYTHON_ROOT_PACKAGES="py-numpy@1.16.0:1.24.3^openblas@0.3.6 py-pip@22.2.2:"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER_PATHS="/usr/tce/packages/gcc/gcc-10.3.1-magic /usr/workspace/brain/tom/pascal/llvm/latest/"
                CENTER_COMPILER="%gcc"
#                CENTER_COMPILER="%clang"
#                DEPENDENTS_CENTER_COMPILER="%gcc@10.3.1"
                # There is something weird about the python@3.9.13 on Pascal right now 5/31/2023
                CENTER_DEPENDENCIES="^openmpi@4.1.2"
                CENTER_PIP_PACKAGES="${LBANN_HOME}/scripts/common_python_packages/requirements.txt ${LBANN_HOME}/ci_test/requirements.txt"
                ;;
            "ivybridge" | "cascadelake") # Catalyst, Ruby
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%gcc"
                CENTER_DEPENDENCIES="^mvapich2@2.3.6 ^hwloc@1.11.13 ^libtool@2.4.2 ^python@3.9.10 ^protobuf@3.10.0 ^py-protobuf@3.10.0"
                ;;
            "zen" | "zen2") # Corona
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%rocmcc@5.7.0"
                CENTER_DEPENDENCIES="^openmpi@4.1.2 ^hip@5.7.0 ^python@3.9.12 ^py-protobuf@4.21.5"
                CENTER_PIP_PACKAGES="${LBANN_HOME}/scripts/common_python_packages/requirements.txt ${LBANN_HOME}/ci_test/requirements.txt"
                ;;
            "zen3") # Tioga, RZVernal
                CENTER_COMPILER="%rocmcc@5.7.1"
                CENTER_DEPENDENCIES="^cray-mpich@8.1.27 ^hip@5.7.1 ^python@3.9.12"
                CENTER_BLAS_LIBRARY="blas=libsci"
                # Override the conduit variants for the cray compilers
                CONDUIT_VARIANTS="~hdf5_compat~fortran~parmetis"
                CENTER_PIP_PACKAGES="${LBANN_HOME}/scripts/common_python_packages/requirements.txt ${LBANN_HOME}/ci_test/requirements.txt"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12"
                ;;
            "zen3") # Frontier, Crusher
                CENTER_COMPILER="%rocmcc@5.5.1"
                CENTER_DEPENDENCIES="^cray-mpich@8.1.26 ^hip@5.5.1 ^python@3.9.13 ^protobuf@3.21.12"
                CENTER_BLAS_LIBRARY="blas=openblas"
                # Override the conduit variants for the cray compilers
                CONDUIT_VARIANTS="~hdf5_compat~fortran~parmetis+blt_find_mpi~test"
                CENTER_PIP_PACKAGES="${LBANN_HOME}/scripts/common_python_packages/requirements.txt ${LBANN_HOME}/ci_test/requirements.txt"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                CENTER_DEPENDENCIES="^openmpi"
                ;;
            "zen3") # Perlmutter
                CENTER_COMPILER="%cce@13.0.2"
                CENTER_DEPENDENCIES="^cray-mpich@8.1.15~wrappers ^python@3.9.7 ^cuda+allow-unsupported-compilers"
                CENTER_BLAS_LIBRARY="blas=libsci"
                # Override the conduit variants for the cray compilers
                CONDUIT_VARIANTS="~hdf5_compat~fortran~parmetis~blt_find_mpi"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                CENTER_DEPENDENCIES="^openmpi"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    elif [[ ${center} = "osx" ]]; then
        case ${spack_arch_target} in
            "skylake")
                CENTER_DEPENDENCIES="^hdf5+hl"
                CENTER_BLAS_LIBRARY="blas=accelerate"
                ;;
            "m1")
                CENTER_DEPENDENCIES="^hdf5+hl ^python@3.10 ^protobuf@3.21.5 ^py-protobuf@4.21.5"
                CENTER_BLAS_LIBRARY="blas=accelerate"
                CENTER_COMPILER="%apple-clang"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
                ;;
        esac
    else
        echo "No center found and no center-specified CENTER_DEPENDENCIES for ${spack_arch_target} at ${center}."
    fi
}

set_center_specific_externals()
{
    local center="$1"
    local spack_arch_target="$2"
    local spack_arch="$3"
    local yaml="$4"
    local module_dir="$5"

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "broadwell" | "haswell" | "sandybridge" | "ivybridge")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: false
      version:
      - '20'
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
    mvapich2:
      buildable: false
      version:
      - '2.3.7'
      externals:
      - spec: mvapich2@2.3.7 arch=${spack_arch}
        modules:
        - mvapich2/2.3.7
EOF
        set_superbuild_externals "pascal" "cuda-11.8.0" "openmpi-4.1.2" "$yaml" "${LOG}"
        set_superbuild_DHA_externals "pascal" "cuda-11.8.0" "openmpi-4.1.2" "$yaml" "${LOG}"
                ;;
            "power9le" | "power8le")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: false
      version:
      - '20'
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
        set_superbuild_externals "lassen" "cuda-11.8.0" "spectrum-mpi-rolling-release" "$yaml" "${LOG}"
        set_superbuild_DHA_externals "lassen" "cuda-11.8.0" "spectrum-mpi-rolling-release" "$yaml" "${LOG}"
        set_superbuild_power_externals "lassen" "cuda-11.8.0" "spectrum-mpi-rolling-release" "$yaml" "${LOG}"

                ;;
            "zen" | "zen2")
cat <<EOF  >> ${yaml}
  packages:
    hipcub:
      buildable: false
      version:
      - '5.7.0'
      externals:
      - spec: hipcub@5.7.0 arch=${spack_arch}
        prefix: /opt/rocm-5.7.0/hipcub
        extra_attributes:
          compilers:
            c: /opt/rocm-5.7.0/llvm/bin/clang
            c++: /opt/rocm-5.7.0/llvm/bin/clang++
    llvm-amdgpu:
      buildable: false
      version:
      - '5.7.0'
      externals:
      - spec: llvm-amdgpu@5.7.0 arch=${spack_arch}
        prefix: /opt/rocm-5.7.0/llvm
        extra_attributes:
          compilers:
            c: /opt/rocm-5.7.0/llvm/bin/clang
            c++: /opt/rocm-5.7.0/llvm/bin/clang++
    openmpi:
      buildable: false
      version:
      - '4.1.2'
      externals:
      - spec: openmpi@4.1.2 arch=${spack_arch}
        modules:
        - openmpi/4.1.2
EOF

        set_superbuild_externals "corona" "rocm-5.7.0" "openmpi-4.1.2" "$yaml" "${LOG}"
        set_superbuild_DHA_externals "corona" "rocm-5.7.0" "openmpi-4.1.2" "$yaml" "${LOG}"

                ;;
            "zen3")
cat <<EOF  >> ${yaml}
  compilers:
  - compiler:
      spec: rocmcc@5.7.1
      paths:
        cc: amdclang
        cxx: amdclang++
        # cc: craycc
        # cxx: crayCC
        f77: amdflang
        fc: amdflang
      flags: {}
      operating_system: rhel8
      target: any
      modules:
      - PrgEnv-cray/8.4.0
      - cce/17.0.0
      - rocm/5.7.1
      environment: {}
      extra_rpaths:
      - /opt/cray/pe/cce/17.0.0/cce/x86_64/lib
      - /opt/cray/pe/cce/17.0.0/cce-clang/x86_64/lib/x86_64-unknown-linux-gnu
  - compiler:
      spec: cce@17.0.0
      paths:
        cc: craycc
        cxx: crayCC
        f77: crayftn
        fc: crayftn
      flags: {}
      operating_system: rhel8
      target: any
      modules:
      - PrgEnv-cray
      - cce/17.0.0
      - rocm/5.7.1
      environment: {}
      extra_rpaths:
      - /opt/cray/pe/cce/17.0.0/cce/x86_64/lib
      - /opt/cray/pe/cce/17.0.0/cce-clang/x86_64/lib/x86_64-unknown-linux-gnu
  packages:
    all:
      require:
        target=zen3
      providers:
        mpi: [cray-mpich]
    hipcub:
      buildable: false
      version:
      - '5.7.1'
      externals:
      - spec: hipcub@5.7.1 arch=${spack_arch}
        prefix: /opt/rocm-5.7.1/hipcub
    rocthrust:
      buildable: false
      version:
      - '5.7.1'
      externals:
      - spec: rocthrust@5.7.1 arch=${spack_arch}
        prefix: /opt/rocm-5.7.1
    llvm-amdgpu:
      buildable: false
      version:
      - '5.7.1'
      externals:
      - spec: llvm-amdgpu@5.7.1 arch=${spack_arch}
        prefix: /opt/rocm-5.7.1/llvm
    cray-libsci:
      buildable: false
      version:
      - '23.09.1.1'
      externals:
      - spec: cray-libsci@23.09.1.1 %rocmcc arch=${spack_arch}
        modules:
        - cce/17.0.0 PrgEnv-cray cray-libsci/23.09.1.1
    cray-mpich:
      buildable: false
      version:
      - '8.1.28'
      externals:
      - spec: cray-mpich@8.1.27 %rocmcc arch=${spack_arch}
        modules:
        - cce/17.0.0 PrgEnv-cray cray-mpich/8.1.28
EOF
        set_superbuild_externals "tioga" "rocm-5.7.1" "cray-mpich-8.1.28" "$yaml" "${LOG}"
        set_superbuild_DHA_externals "tioga" "rocm-5.7.1" "cray-mpich-8.1.28" "$yaml" "${LOG}"

                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le" | "power8le")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: false
      version:
      - '20'
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "zen3")
cat <<EOF  >> ${yaml}
  compilers:
  - compiler:
      spec: rocmcc@5.5.1
      paths:
        cc: cc
        cxx: CC
        f77: ftn
        fc: ftn
      flags: {}
      operating_system: rhel8
      target: any
      modules:
      - PrgEnv-amd
      - amd/5.5.1
      environment: {}
      extra_rpaths: []
  packages:
    all:
      providers:
        mpi: [cray-mpich]
    hipcub:
      buildable: False
      version:
      - '5.5.1'
      externals:
      - spec: hipcub@5.5.1 arch=${spack_arch}
        prefix: /opt/rocm-5.5.1/hipcub
    llvm-amdgpu:
      buildable: False
      version:
      - '5.5.1'
      externals:
      - spec: llvm-amdgpu@5.5.1 arch=${spack_arch}
        prefix: /opt/rocm-5.5.1/llvm
    cray-libsci:
      buildable: False
      version:
      - '23.05.1.4'
      externals:
      - spec: cray-libsci@23.05.1.4 arch=${spack_arch}
        modules:
        - amd/5.5.1 PrgEnv-amd/8.4.0 cray-libsci/23.05.1.4
    cray-mpich:
      buildable: False
      version:
      - '8.1.26'
      externals:
      - spec: cray-mpich@8.1.26 arch=${spack_arch}
        modules:
        - amd/5.6.0 PrgEnv-amd/8.4.0 cray-mpich/8.1.26
    # libfabric:
    #   buildable: false
    #   version:
    #   - 1.15.0
    #   externals:
    #   - spec: libfabric@1.15.0 arch=${spack_arch}
    #     modules:
    #     - libfabric/1.15.0
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: false
      version:
      - '20'
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "zen3") #perlmutter
cat <<EOF  >> ${yaml}
  packages:
    all:
      providers:
        mpi: [cray-mpich]
    nvhpc:
      buildable: false
      version:
      - '21.11'
      externals:
      - spec: nvhpc@21.11 arch=${spack_arch}
        modules:
        - cudatoolkit/11.5
    cudnn:
      buildable: false
      version:
      - '8.3.2'
      externals:
      - spec: cudnn@8.3.2 arch=${spack_arch}
        modules:
        - cudnn/8.3.2
    cray-libsci:
      buildable: false
      version:
      - '21.08.1.2'
      externals:
      - spec: cray-libsci@21.08.1.2 arch=${spack_arch}
        modules:
        - cray-libsci/21.08.1.2
    cray-mpich:
      buildable: false
      version:
      - '8.1.15'
      externals:
      - spec: "cray-mpich@8.1.15~wrappers arch=${spack_arch}"
        modules:
        - cray-mpich/8.1.15
    nccl:
      buildable: false
      version:
      - '2.11.4'
      externals:
      - spec: nccl@2.11.4 arch=${spack_arch}
        modules:
        - nccl/2.11.4
EOF
                ;;
            *)
                echo "No center-specified externals."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            *)
        echo "No center-specified externals."
                ;;
        esac
    else
        echo "No center-specified externals."
    fi

    echo "Setting up a sane definition of how to represent modules."
cat <<EOF >> ${yaml}
  modules:
    default:
      enable::
        - lmod
      lmod:
        all:
          autoload: direct
        core_compilers:
        - 'cce@13.0.0'
    lbann_lmod_modules:
      roots:
        lmod: ${module_dir}
      arch_folder: false
      lmod:
        all:
          autoload: direct
EOF
    if [[ ${CENTER_COMPILER} ]]; then
        CORE_COMPILER=$(echo "${CENTER_COMPILER}" | tr -d '%')
cat <<EOF >> ${yaml}
        core_compilers:
        - '${CORE_COMPILER}'
EOF
        if [[ ${DEPENDENTS_CENTER_COMPILER} ]]; then
            DEPENDENTS_CORE_COMPILER=$(echo "${DEPENDENTS_CENTER_COMPILER}" | tr -d '%')
cat <<EOF >> ${yaml}
        - '${DEPENDENTS_CORE_COMPILER}'
EOF


        fi
    fi

}

cleanup_clang_compilers()
{
    local center="$1"
    local spack_arch_os="$2"
    local yaml="$3"

    if [[ ${center} = "llnl_lc" ]]; then
        if [[ ${spack_arch_os} = "rhel7" ]]; then
            # Point compilers that don't have a fortran compiler a default one
            sed -i.sed_bak -e 's/\([[:space:]]*f[c7]7*:[[:space:]]*\)null$/\1\/usr\/tce\/packages\/gcc\/gcc-8.3.1\/bin\/gfortran/g' ${yaml}
            echo "Updating Clang compiler's to see the gfortran compiler."

            # LC uses a old default gcc and clang needs a newer default gcc toolchain
            # Also set LC clang compilers to use lld for faster linking ldflags: -fuse-ld=lld
            perl -i.perl_bak -0pe 's/(- compiler:.*?spec: clang.*?flags:) (\{\})/$1 \{cflags: --gcc-toolchain=\/usr\/tce\/packages\/gcc\/gcc-8.3.1, cxxflags: --gcc-toolchain=\/usr\/tce\/packages\/gcc\/gcc-8.3.1\}/smg' ${yaml}
        else
            # Point compilers that don't have a fortran compiler a default one
            sed -i.sed_bak -e 's/\([[:space:]]*f[c7]7*:[[:space:]]*\)null$/\1\/usr\/tce\/bin\/gfortran/g' ${yaml}
            echo "Updating Clang compiler's to see the gfortran compiler."
        fi
    elif [[ ${center} = "osx" ]]; then
        # Point compilers that don't have a fortran compiler a default one
        sed -i.sed_bak -e 's/\([[:space:]]*f[c7]7*:[[:space:]]*\)null$/\1\/opt\/homebrew\/bin\/gfortran/g' ${yaml}
        echo "Updating Clang compiler's to see the homebrew gfortran compiler."
    else
        # Point compilers that don't have a fortran compiler a default one
        sed -i.sed_bak -e 's/\([[:space:]]*f[c7]7*:[[:space:]]*\)null$/\1\/usr\/bin\/gfortran/g' ${yaml}
        echo "Updating Clang compiler's to see the gfortran compiler."
    fi
}

set_center_specific_variants()
{
    local center="$1"
    local spack_arch_target="$2"

    STD_USER_VARIANTS="+vision +numpy"
    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le" | "broadwell" | "haswell" | "sandybridge") # Lassen, Pascal, RZHasGPU, Surface
                CENTER_USER_VARIANTS="+cuda"
                ;;
            "ivybridge") # Catalyst
                CENTER_USER_VARIANTS="+onednn"
                ;;
            "zen" | "zen2") # Corona
                CENTER_USER_VARIANTS="+rocm"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le") # Summit
                CENTER_USER_VARIANTS="+cuda"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512") # CoriGPU
                CENTER_USER_VARIANTS="+cuda"
                ;;
            "zen3") # Perlmutter
                CENTER_USER_VARIANTS="+cuda"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                CENTER_USER_VARIANTS="+onednn"
                ;;
            *)
                echo "No center-specified CENTER_USER_VARIANTS."
                ;;
        esac
    else
        echo "No center found and no center-specified CENTER_USER_VARIANTS."
    fi
    CENTER_USER_VARIANTS+=" ${STD_USER_VARIANTS}"
}
