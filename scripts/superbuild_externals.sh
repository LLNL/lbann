set_superbuild_externals()
{
    local system="$1"
    local dnn_lib="$2"
    local compiler_ver="$3"
    local mpi="$4"
    local yaml="$5"
    local prefix="$6"
    local gpu_arch="$7"

    if [ -n "${gpu_arch}" ]; then
        dnn_lib="${dnn_lib}/${gpu_arch}"
    fi

    #/usr/workspace/lbann/stable_dependencies/rzvernal/rocm-5.7.1/mi300a/cray-mpich-8.1.27

    #/usr/workspace/lbann/lbann-superbuild/

    local sb_extra_prefix="${system}/${dnn_lib}/${compiler_ver}/${mpi}"
    CMD="source ${prefix}/${sb_extra_prefix}/logs/lbann_sb_suggested_cmake_prefix_path.sh"
#    CMD="source /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/logs/lbann_sb_suggested_cmake_prefix_path.sh"
    echo ${CMD} | tee -a ${LOG}
    ${CMD}

cat <<EOF  >> ${yaml}
    adiak:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: adiak@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/adiak
    caliper:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: caliper@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/caliper
    catch2:
      buildable: false
      version:
      - '2.9.2'
      externals:
      - spec: catch2@2.9.2 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/catch2
    half:
      buildable: false
      version:
      - '2.1.0'
      externals:
      - spec: half@2.1.0 arch=${spack_arch}
        prefix: ${prefix}/half-2.1.0
    hdf5:
      buildable: false
      version:
      - '1.10.9'
      externals:
      - spec: hdf5@1.10.9 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/hdf5
    jpeg-turbo:
      buildable: false
      version:
      - '2.0.3'
      externals:
      - spec: jpeg-turbo@2.0.3 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/jpeg-turbo
    spdlog:
      buildable: false
      version:
      - '1.12.0'
      externals:
      - spec: spdlog@1.12.0 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/spdlog
    cereal:
      buildable: false
      version:
      - '1.3.0'
      externals:
      - spec: cereal@1.3.0 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/cereal
    clara:
      buildable: false
      version:
      - '1.1.5'
      externals:
      - spec: clara@1.1.5 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/clara
    cnpy:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: cnpy@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/cnpy
    conduit:
      buildable: false
      version:
      - 'develop'
      externals:
      - spec: conduit@develop arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/conduit
    hiptt:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: hiptt@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/hiptt
    opencv:
      buildable: false
      version:
      - '4.1.0'
      externals:
      - spec: opencv@4.1.0 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/opencv
    protobuf:
      buildable: false
      version:
      - '3.21.5'
      externals:
      - spec: protobuf@3.21.5+shared arch=${spack_arch}
#      - spec: protobuf@3.21.5~shared arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/protobuf
    zstr:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: zstr@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/zstr
EOF

    if [[ ${dnn_lib} =~ "rocm" ]]; then
cat <<EOF  >> ${yaml}
    hwloc:
      buildable: false
      version:
      - '3.0.0'
      externals:
      - spec: hwloc@3.0.0 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/hwloc
    aws-ofi-rccl:
      buildable: false
      version:
      - 'cxi'
      externals:
      - spec: aws-ofi-rccl@cxi arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/aws_ofi_rccl
    hiptt:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: hiptt@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/hiptt
EOF
    fi

    if [[ ${dnn_lib} =~ "cuda" ]]; then
cat <<EOF  >> ${yaml}
    nccl:
      buildable: false
      version:
      - '2.19.4'
      externals:
      - spec: nccl@2.19.4 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/nccl
    cudnn:
      buildable: false
      version:
      - '8.9.4'
      externals:
      - spec: cudnn@8.9.4 arch=linux-rhel8-broadwell
        prefix: ${prefix}/cudnn-8.9.4/cuda_11_x86_64
      - spec: cudnn@8.9.4 arch=linux-rhel7-power9le
        prefix: ${prefix}/cudnn-8.9.4/cuda_11_ppc64le
    cutensor:
      buildable: false
      version:
      - '1.7.0.1'
      externals:
      - spec: cutensor@1.7.0.1 arch=linux-rhel8-broadwell
        prefix: ${prefix}/cutensor-1.7.0.1/libcutensor-linux-x86_64-1.7.0.1-archive
      - spec: cutensor@1.7.0.1 arch=linux-rhel7-power9le
        prefix: ${prefix}/cutensor-1.7.0.1/libcutensor-linux-ppc64le-1.7.0.1-archive

EOF
    fi
}

set_superbuild_DHA_externals()
{
    local system="$1"
    local dnn_lib="$2"
    local compiler_ver="$3"
    local mpi="$4"
    local yaml="$5"
    local prefix="$6"
    local dha_dir="$7"
    local gpu_arch="$8"

    if [ -n "${gpu_arch}" ]; then
        dnn_lib="${dnn_lib}/${gpu_arch}"
    fi

    local sb_extra_prefix="${system}/${dnn_lib}/${compiler_ver}/${mpi}"
#    source ${prefix}/${system}/${dnn_lib}/${mpi}/logs/lbann_sb_suggested_cmake_prefix_path.sh
    CMD="source ${prefix}/${sb_extra_prefix}/${dha_dir}/logs/lbann_sb_suggested_cmake_prefix_path.sh"
    echo ${CMD} | tee -a ${LOG}
    ${CMD}

cat <<EOF  >> ${yaml}
    aluminum:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: aluminum@master arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/${dha_dir}/aluminum
    hydrogen:
      buildable: false
      version:
      - 'develop'
      externals:
      - spec: hydrogen@develop arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/${dha_dir}/hydrogen
    dihydrogen:
      buildable: false
      version:
      - 'develop'
      externals:
      - spec: dihydrogen@develop arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/${dha_dir}/dihydrogen
EOF
}

set_superbuild_power_externals()
{
    local system="$1"
    local dnn_lib="$2"
    local compiler_ver="$3"
    local mpi="$4"
    local yaml="$5"
    local prefix="$6"
    local gpu_arch="$7"

    if [ -n "${gpu_arch}" ]; then
        dnn_lib="${dnn_lib}/${gpu_arch}"
    fi

    local sb_extra_prefix="${system}/${dnn_lib}/${compiler_ver}/${mpi}"
#    source ${prefix}/${sb_extra_prefix}/logs/lbann_sb_suggested_cmake_prefix_path.sh

cat <<EOF  >> ${yaml}
    openblas:
      buildable: false
      version:
      - '0.3.6'
      externals:
      - spec: openblas@0.3.6 arch=${spack_arch}
        prefix: ${prefix}/${sb_extra_prefix}/openblas
EOF
}
