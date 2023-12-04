set_superbuild_externals()
{
    local system="$1"
    local dnn_lib="$2"
    local mpi="$3"
    local yaml="$4"
    local LOG="$5"

    CMD="source /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/logs/lbann_sb_suggested_cmake_prefix_path.sh"
    echo ${CMD} | tee -a ${LOG}
    ${CMD}

cat <<EOF  >> ${yaml}
    adiak:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: adiak@master arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/adiak
    caliper:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: caliper@master arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/caliper
    catch2:
      buildable: false
      version:
      - '2.9.2'
      externals:
      - spec: catch2@2.9.2 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/catch2
    hdf5:
      buildable: false
      version:
      - '1.10.9'
      externals:
      - spec: hdf5@1.10.9 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/hdf5
    jpeg-turbo:
      buildable: false
      version:
      - '2.0.3'
      externals:
      - spec: jpeg-turbo@2.0.3 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/jpeg-turbo
    spdlog:
      buildable: false
      version:
      - '1.12.0'
      externals:
      - spec: spdlog@1.12.0 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/spdlog
    cereal:
      buildable: false
      version:
      - '1.3.0'
      externals:
      - spec: cereal@1.3.0 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/cereal
    clara:
      buildable: false
      version:
      - '1.1.5'
      externals:
      - spec: clara@1.1.5 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/clara
    cnpy:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: cnpy@master arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/cnpy
    conduit:
      buildable: false
      version:
      - 'develop'
      externals:
      - spec: conduit@develop arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/conduit
    hiptt:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: hiptt@master arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/hiptt
    opencv:
      buildable: false
      version:
      - '4.1.0'
      externals:
      - spec: opencv@4.1.0 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/opencv
    protobuf:
      buildable: false
      version:
      - '3.21.5'
      externals:
      - spec: protobuf@3.21.5 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/protobuf
    zstr:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: zstr@master arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/zstr
    nccl:
      buildable: false
      version:
      - '2.19.4'
      externals:
      - spec: nccl@2.19.4 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/nccl
    cudnn:
      buildable: false
      version:
      - '8.9.4'
      externals:
      - spec: cudnn@8.9.4 arch=linux-rhel8-broadwell
        prefix: /p/vast1/lbann/stable_dependencies/cudnn-8.9.4/cuda_11_x86_64
      - spec: cudnn@8.9.4 arch=linux-rhel7-power9le
        prefix: /p/vast1/lbann/stable_dependencies/cudnn-8.9.4/cuda_11_ppc64le
    cutensor:
      buildable: false
      version:
      - '1.7.0.1'
      externals:
      - spec: cutensor@1.7.0.1 arch=linux-rhel8-broadwell
        prefix: /p/vast1/lbann/stable_dependencies/cutensor-1.7.0.1/libcutensor-linux-x86_64-1.7.0.1-archive
      - spec: cutensor@1.7.0.1 arch=linux-rhel7-power9le
        prefix: /p/vast1/lbann/stable_dependencies/cutensor-1.7.0.1/libcutensor-linux-ppc64le-1.7.0.1-archive

EOF
}

set_superbuild_DHA_externals()
{
    local system="$1"
    local dnn_lib="$2"
    local mpi="$3"
    local yaml="$4"

    source /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/logs/lbann_sb_suggested_cmake_prefix_path.sh

cat <<EOF  >> ${yaml}
    aluminum:
      buildable: false
      version:
      - 'master'
      externals:
      - spec: aluminum@master arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/aluminum
    hydrogen:
      buildable: false
      version:
      - 'develop'
      externals:
      - spec: hydrogen@develop arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/hydrogen
    dihydrogen:
      buildable: false
      version:
      - 'develop'
      externals:
      - spec: dihydrogen@develop arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/dihydrogen
EOF
}

set_superbuild_power_externals()
{
    local system="$1"
    local dnn_lib="$2"
    local mpi="$3"
    local yaml="$4"

    source /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/logs/lbann_sb_suggested_cmake_prefix_path.sh

cat <<EOF  >> ${yaml}
    openblas:
      buildable: false
      version:
      - '0.3.6'
      externals:
      - spec: openblas@0.3.6 arch=${spack_arch}
        prefix: /p/vast1/lbann/stable_dependencies/${system}/${dnn_lib}/${mpi}/openblas
EOF
}
