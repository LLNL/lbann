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
        DOMAINNAME=$(python3 -c 'import socket; domain = socket.getfqdn().split("."); print(domain[-2] + "." + domain[-1])')
        if [[ ${CORI} -eq 1 ]]; then
            CENTER="nersc"
        elif [[ ${DOMAINNAME} = "ornl.gov" ]]; then
            CENTER="olcf"
            NINJA_NUM_PROCESSES=16 # Don't let OLCF kill build jobs
        elif [[ ${DOMAINNAME} = "llnl.gov" ]]; then
            CENTER="llnl_lc"
        elif [[ ${DOMAINNAME} = "riken.jp" ]]; then
            CENTER="riken"
        else
            CENTER="llnl_lc"
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
            "power8le") # Ray
                GPU_ARCH_VARIANTS="cuda_arch=60"
                CMAKE_GPU_ARCH="60"
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
                GPU_ARCH_VARIANTS="amdgpu_target=gfx906 %clang@amd"
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
            "power9le" | "power8le") # Lassen, Ray
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.1 spectrum-mpi/rolling-release python/3.7.2"
                ;;
            "broadwell" | "haswell" | "sandybridge") # Pascal, RZHasGPU, Surface
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 cuda/11.1.0 mvapich2/2.3 python/3.7.2"
                ;;
            "ivybridge") # Catalyst
                MODULE_CMD="module --force unload StdEnv; module load gcc/8.3.1 mvapich2/2.3 python/3.7.2"
                ;;
            "zen" | "zen2") # Corona
                MODULE_CMD="module --force unload StdEnv; module load clang/11.0.0 python/3.7.2 opt rocm/4.1.0 openmpi-gnu/4.0"
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
        MIRRORS="/p/vast1/lbann/spack/mirror /p/vast1/atom/spack/mirror"
        case ${spack_arch_target} in
            "power9le" | "power8le") # Lassen, Ray
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12 threads=openmp ^cuda@11.1.105"
                CENTER_FLAGS="+gold"
                ;;
            "broadwell" | "haswell" | "sandybridge" | "ivybridge") # Pascal, RZHasGPU, Surface, Catalyst
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_DEPENDENCIES="^mvapich2 ^hwloc@1.11.13"
                CENTER_FLAGS="+gold"
                ;;
            "zen" | "zen2") # Corona
                # On LC the mvapich2 being used is built against HWLOC v1
                CENTER_DEPENDENCIES="^openmpi ^hwloc@2.3.0"
                CENTER_FLAGS="+lld"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES."
                ;;
        esac
    elif [[ ${center} = "olcf" ]]; then
        case ${spack_arch_target} in
            "power9le")
                CENTER_DEPENDENCIES="^spectrum-mpi ^openblas@0.3.12"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES."
                ;;
        esac
    elif [[ ${center} = "nersc" ]]; then
        case ${spack_arch_target} in
            "skylake_avx512")
                CENTER_DEPENDENCIES="^openmpi"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES."
                ;;
        esac
    elif [[ ${center} = "riken" ]]; then
        case ${spack_arch_target} in
            "a64fx")
                CENTER_DEPENDENCIES="^openmpi"
                ;;
            *)
                echo "No center-specified CENTER_DEPENDENCIES."
                ;;
        esac
    else
        echo "No center found and no center-specified CENTER_DEPENDENCIES."
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
            "broadwell" | "haswell" | "sandybridge" | "power9le" | "power8le")
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
            "zen" | "zen2")
cat <<EOF  >> ${yaml}
  compilers:
    - compiler:
        spec: clang@amd
        paths:
          cc: /opt/rocm-4.1.0/llvm/bin/clang
          cxx: /opt/rocm-4.1.0/llvm/bin/clang++
          f77: /usr/bin/gfortran
          fc: /usr/bin/gfortran
        flags: {}
        operating_system: rhel7
        target: x86_64
        modules: []
        environment: {}
        extra_rpaths: []
  packages:
    hip:
      buildable: False
      version:
      - 4.1.0
      externals:
      - spec: hip@4.1.0 arch=${spack_arch}
        prefix: /opt/rocm-4.1.0/hip
        extra_attributes:
          compilers:
            c: /opt/rocm-4.1.0/llvm/bin/clang
            c++: /opt/rocm-4.1.0/llvm/bin/clang++
            hip: /opt/rocm-4.1.0/hip/bin/hipcc
    hipcub:
      buildable: False
      version:
      - 4.1.0
      externals:
      - spec: hipcub@4.1.0 arch=${spack_arch}
        prefix: /opt/rocm-4.1.0/hipcub
        extra_attributes:
          compilers:
            c: /opt/rocm-4.1.0/llvm/bin/clang
            c++: /opt/rocm-4.1.0/llvm/bin/clang++
    hsa-rocr-dev:
      buildable: False
      version:
      - 4.1.0
      externals:
      - spec: hsa-rocr-dev@4.1.0 arch=${spack_arch}
        prefix: /opt/rocm-4.1.0
        extra_attributes:
          compilers:
            c: /opt/rocm-4.1.0/llvm/bin/clang
            c++: /opt/rocm-4.1.0/llvm/bin/clang++
    llvm-amdgpu:
      buildable: False
      version:
      - 4.1.0
      externals:
      - spec: llvm-amdgpu@4.1.0 arch=${spack_arch}
        prefix: /opt/rocm-4.1.0/llvm
        extra_attributes:
          compilers:
            c: /opt/rocm-4.1.0/llvm/bin/clang
            c++: /opt/rocm-4.1.0/llvm/bin/clang++
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
      - 4.0
      externals:
      - spec: openmpi@4.0.0 arch=${spack_arch}
        prefix: /opt/openmpi/4.0/gnu
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
    enable::
      - tcl
      - lmod
    lmod::
      hash_length: 7
      projections:
        all: '\${PACKAGE}/\${VERSION}'
      all:
        filter:
          # Exclude changes to any of these variables
          environment_blacklist:
          - 'CPATH'
          - 'LIBRARY_PATH'
      ^python:
        autoload:  'direct'
    tcl:
      hash_length: 7
      projections:
        all: '\${PACKAGE}/\${VERSION}'
      all:
        filter:
          # Exclude changes to any of these variables
          environment_blacklist:
          - 'CPATH'
          - 'LIBRARY_PATH'
      ^python:
        autoload:  'direct'
EOF
}

cleanup_clang_compilers()
{
    local center="$1"
    local yaml="$2"

    # Point compilers that don't have a fortran compiler a default one
    sed -i.sed_bak -e 's/\(f[c7]7*:\s\)null$/\1 \/usr\/bin\/gfortran/g' ${yaml}
    echo "Updating Clang compiler's to see the gfortran compiler."

    if [[ ${center} = "llnl_lc" ]]; then
        # LC uses a old default gcc and clang needs a newer default gcc toolchain
        # Also set LC clang compilers to use lld for faster linking ldflags: -fuse-ld=lld
        perl -i.perl_bak -0pe 's/(- compiler:.*?spec: clang.*?flags:) (\{\})/$1 \{cflags: --gcc-toolchain=\/usr\/tce\/packages\/gcc\/gcc-8.1.0, cxxflags: --gcc-toolchain=\/usr\/tce\/packages\/gcc\/gcc-8.1.0\}/smg' ${yaml}
    fi
}

set_center_specific_variants()
{
    local center="$1"
    local spack_arch_target="$2"

    STD_USER_VARIANTS="+vision +numpy"
    if [[ ${center} = "llnl_lc" ]]; then
        case ${spack_arch_target} in
            "power9le" | "power8le" | "broadwell" | "haswell" | "sandybridge") # Lassen, Ray, Pascal, RZHasGPU, Surface
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
