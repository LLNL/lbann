#!/bin/sh

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
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.6.1 spectrum-mpi/rolling-release python/3.7.2"
                # MODULE_CMD="module --force unload StdEnv; module load clang/12.0.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2 essl/6.2.1"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface
                MODULE_CMD_GCC="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.6.1 mvapich2/2.3 python/3.7.2"
                MODULE_CMD_CLANG="module --force unload StdEnv; module load clang/12.0.1 cuda/11.6.1 mvapich2/2.3.6 python/3.7.2"
                ;;
            "ivybridge" | "cascadelake") # Catalyst, Ruby
                MODULE_CMD="module --force unload StdEnv; module load gcc/10.2.1 mvapich2/2.3 python/3.7.2"
                ;;
            "zen" | "zen2") # Corona
                MODULE_CMD="module load gcc-tce/10.3.1 rocm/5.2.0 mvapich2-tce/2.3.6"
#                MODULE_CMD="module load gcc-tce/10.3.1 rocm/5.2.0 openmpi-tce/4.1.2"
                # ; ml use /opt/toss/modules/modulefiles && ml openmpi-gnu/4.1
                ;;
            "zen3") # Tioga, RZVernal
                MODULE_CMD="module load craype-x86-trento craype-network-ofi libfabric/1.7.2-llnl perftools-base/22.09.0 amd/5.3.0 craype/2.7.17 cray-mpich/8.1.19 cray-libsci/22.08.1.1 PrgEnv-amd/8.3.3 StdEnv cmake/3.23.1"
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
		MODULE_CMD="module load PrgEnv-cray/8.3.3 craype-x86-milan libfabric/1.11.0.4.116 craype-network-ofi cmake/3.22.0 cce/13.0.2 craype/2.7.15 cray-mpich/8.1.15 cray-libsci/21.08.1.2 nccl/2.11.4 cudnn/8.3.2 cray-python/3.9.7.1 craype-accel-host cudatoolkit/11.5"
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
        if [[ -z "${SKIP_MIRRORS:-}" ]]; then
            POSSIBLE_MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
            for m in ${POSSIBLE_MIRRORS}
            do
                if [[ -r "${m}" ]]; then
                    MIRRORS="${m} $MIRRORS"
                fi
            done
        fi
        # MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
        case ${spack_arch_target} in
            "power9le") # Lassen
                CENTER_COMPILER="%gcc"
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12 threads=openmp ^cuda@11.6.112 ^libtool@2.4.2 ^python@3.9.10 ^protobuf@3.10.0 ^py-protobuf@3.10.0"
                CENTER_BLAS_LIBRARY="blas=openblas"
                # CENTER_COMPILER="%clang"
                # CENTER_DEPENDENCIES="^spectrum-mpi ^cuda@11.1.105 ^libtool@2.4.2 ^python@3.9.10"
                # CENTER_BLAS_LIBRARY="blas=essl"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%clang"
                CENTER_DEPENDENCIES="^mvapich2 ^hwloc@1.11.13 ^libtool@2.4.2 ^python@3.9.10 ^protobuf@3.10.0 ^py-protobuf@3.10.0"
                ;;
            "ivybridge" | "cascadelake") # Catalyst, Ruby
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%gcc"
                CENTER_DEPENDENCIES="^mvapich2 ^hwloc@1.11.13 ^libtool@2.4.2 ^python@3.9.10 ^protobuf@3.10.0 ^py-protobuf@3.10.0"
                ;;
            "zen" | "zen2") # Corona
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_COMPILER="%rocmcc@5.2.0"
                CENTER_DEPENDENCIES="^mvapich2@2.3.6 ^hip@5.2.0 ^python@3.9.12"
#                CENTER_DEPENDENCIES="^openmpi@4.1.2 ^hip@5.2.0 ^python@3.9.12"
#                CENTER_DEPENDENCIES="^openmpi@4.1.2 ^hip@5.2.0 ^python@3.9.10 ^protobuf@3.10.0 ^py-protobuf@3.10.0"
                CENTER_PIP_PACKAGES="${LBANN_HOME}/scripts/common_python_packages/requirements.txt ${LBANN_HOME}/ci_test/requirements.txt"
                ;;
            "zen3") # Tioga, RZVernal
                CENTER_COMPILER="%rocmcc@5.3.0"
                CENTER_DEPENDENCIES="^cray-mpich@8.1.19 ^hip@5.3.0 ^python@3.9.12 ^aluminum@1.0.0-lbann +ofi_libfabric_plugin"
                CENTER_BLAS_LIBRARY="blas=libsci"
                # Override the conduit variants for the cray compilers
                CONDUIT_VARIANTS="~hdf5_compat~fortran~parmetis~blt_find_mpi"
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
                CENTER_DEPENDENCIES="^hdf5+hl"
                CENTER_BLAS_LIBRARY="blas=accelerate"
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

    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "broadwell" | "haswell" | "sandybridge" | "ivybridge")
cat <<EOF  >> ${yaml}
  packages:
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "power9le" | "power8le")
cat <<EOF  >> ${yaml}
  packages:
    essl:
      buildable: False
      version:
      - 6.2.1
      externals:
      - spec: essl@6.2.1 arch=${spack_arch}
        modules:
        - essl/6.2.1
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
EOF
                ;;
            "zen" | "zen2")
cat <<EOF  >> ${yaml}
  packages:
    hipcub:
      buildable: False
      version:
      - 5.2.0
      externals:
      - spec: hipcub@5.2.0 arch=${spack_arch}
        prefix: /opt/rocm-5.2.0/hipcub
        extra_attributes:
          compilers:
            c: /opt/rocm-5.2.0/llvm/bin/clang
            c++: /opt/rocm-5.2.0/llvm/bin/clang++
    llvm-amdgpu:
      buildable: False
      version:
      - 5.2.0
      externals:
      - spec: llvm-amdgpu@5.2.0 arch=${spack_arch}
        prefix: /opt/rocm-5.2.0/llvm
        extra_attributes:
          compilers:
            c: /opt/rocm-5.2.0/llvm/bin/clang
            c++: /opt/rocm-5.2.0/llvm/bin/clang++
    rdma-core:
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
    openmpi:
      buildable: False
      version:
      - 4.1.2
      externals:
      - spec: openmpi@4.1.2 arch=${spack_arch}
        modules:
        - openmpi-tce/4.1.2
    mvapich2:
      buildable: False
      version:
      - 2.3.6
      externals:
      - spec: mvapich2@2.3.6 arch=${spack_arch}
        modules:
        - mvapich2-tce/2.3.6
EOF
                ;;
            "zen3")
cat <<EOF  >> ${yaml}
  compilers:
  - compiler:
      spec: rocmcc@5.3.0
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
      - amd/5.3.0
      environment: {}
      extra_rpaths: []
  packages:
    all:
      providers:
        mpi: [cray-mpich]
    hipcub:
      buildable: False
      version:
      - 5.3.0
      externals:
      - spec: hipcub@5.3.0 arch=${spack_arch}
        prefix: /opt/rocm-5.3.0/hipcub
    llvm-amdgpu:
      buildable: False
      version:
      - 5.3.0
      externals:
      - spec: llvm-amdgpu@5.3.0 arch=${spack_arch}
        prefix: /opt/rocm-5.3.0/llvm
    cray-libsci:
      buildable: False
      version:
      - 22.08.1.1
      externals:
      - spec: cray-libsci@22.08.1.1 arch=${spack_arch}
        modules:
        - cray-libsci/22.08.1.1
    cray-mpich:
      buildable: False
      version:
      - 8.1.19
      externals:
      - spec: cray-mpich@8.1.19 arch=${spack_arch}
        modules:
        - cray-mpich/8.1.19
    libfabric:
      buildable: false
      version:
      - 1.7.2-llnl
      externals:
      - spec: libfabric@1.7.2-llnl arch=${spack_arch}
        modules:
        - libfabric/1.7.2-llnl
EOF
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
      buildable: False
      version:
      - 20
      externals:
      - spec: rdma-core@20 arch=${spack_arch}
        prefix: /usr
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
      buildable: False
      version:
      - 20
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
      buildable: False
      version:
      - 21.11
      externals:
      - spec: nvhpc@21.11 arch=${spack_arch}
        modules:
        - cudatoolkit/11.5
    cudnn:
      buildable: False
      version:
      - 8.3.2
      externals:
      - spec: cudnn@8.3.2 arch=${spack_arch}
        modules:
        - cudnn/8.3.2
    cray-libsci:
      buildable: False
      version:
      - 21.08.1.2
      externals:
      - spec: cray-libsci@21.08.1.2 arch=${spack_arch}
        modules:
        - cray-libsci/21.08.1.2
    cray-mpich:
      buildable: False
      version:
      - 8.1.15
      externals:
      - spec: "cray-mpich@8.1.15~wrappers arch=${spack_arch}"
        modules:
        - cray-mpich/8.1.15
    nccl:
      buildable: False
      version:
      - 2.11.4
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
      lmod:
        core_compilers:
          - 'cce@13.0.0'
EOF
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
