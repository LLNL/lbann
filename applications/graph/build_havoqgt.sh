#!/bin/bash

# Options
system=$(hostname | sed 's/\([a-zA-Z][a-zA-Z]*\)[0-9]*/\1/g')

# Load modulefiles
module load git
module load cmake
module load gcc/8.3.1
module load boost
if [ "${system}" == "lassen" -o "${system}" == "sierra" ]; then
    module load spectrum-mpi
else
    module load mvapich2
fi

# Compilers
cxx_compiler=$(which g++)
cxx_flags="-std=c++17 -fopenmp -lrt -lstdc++fs -lpthread"
mpi_cxx_compiler=$(which mpicxx)

# Source directories
pushd $(dirname $(realpath $0)) > /dev/null
lbann_dir=$(git rev-parse --show-toplevel)
popd > /dev/null
app_dir=${lbann_dir}/applications/graph
node2vec_dir=${app_dir}/largescale_node2vec
havoqgt_dir=${app_dir}/havoqgt
boost_dir=/usr/tcetmp/packages/boost/boost-1.70.0

# Create clean build directory and move there
build_dir=${app_dir}/build
rm -rf ${build_dir}
mkdir -p ${build_dir}
pushd ${build_dir} > /dev/null

# Build HavoqGT
mkdir havoqgt
pushd havoqgt > /dev/null
configure_command=$(cat << EOF
cmake \
--verbose=1 \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_COMPILER=${cxx_compiler} \
-DCMAKE_CXX_FLAGS="${cxx_flags}" \
-DMPI_CXX_COMPILER=${mpi_cxx_compiler} \
-DBOOST_ROOT=${boost_dir} \
${havoqgt_dir}
EOF
)
echo "${configure_command}"
eval ${configure_command}
build_command=$(cat << EOF
make -j$(nproc) ingest_edge_list
EOF
)
echo "${build_command}"
eval ${build_command}
popd > /dev/null

# Build largescale_node2vec
mkdir largescale_node2vec
pushd largescale_node2vec > /dev/null
configure_command=$(cat << EOF
cmake \
--verbose=1 \
-DCMAKE_BUILD_TYPE=Release \
-DCMAKE_CXX_COMPILER=${cxx_compiler} \
-DCMAKE_CXX_FLAGS="${cxx_flags}" \
-DMPI_CXX_COMPILER=${mpi_cxx_compiler} \
-DBOOST_ROOT=${boost_dir} \
-DHAVOQGT_ROOT=${havoqgt_dir} \
${node2vec_dir}
EOF
)
echo "${configure_command}"
eval ${configure_command}
build_command=$(cat << EOF
make -j$(nproc) run_dist_node2vec_rw
EOF
)
echo "${build_command}"
eval ${build_command}
popd > /dev/null

# Return to original directory
dirs -c
