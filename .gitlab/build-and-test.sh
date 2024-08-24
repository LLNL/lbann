#!/usr/bin/env bash

################################################################################
## Copyright (c) 2014-2024, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################

# Initialize modules for users not using bash as a default shell
modules_home=${MODULESHOME:-"/usr/share/lmod/lmod"}
if [[ -e ${modules_home}/init/bash ]]
then
    source ${modules_home}/init/bash
fi

set -o errexit
set -o nounset

hostname="$(hostname)"
cluster=${hostname//[0-9]/}
project_dir="$(git rev-parse --show-toplevel)"
if [[ $? -eq 1 ]]
then
    project_dir="$(pwd)"
fi

# NOTE: No modules will be explicitly unloaded or purged. Obviously,
# loading a new compiler will trigger the auto-unload of the existing
# compiler module (and all the other side-effects wrt mpi, etc), but
# no explicit action is taken by this script.
modules=${MODULES:-""}
run_coverage=${WITH_COVERAGE:-""}
build_distconv=${WITH_DISTCONV:-""}
build_half=${WITH_HALF:-""}
build_fft=${WITH_FFT:-""}

TEST_FLAG=${WITH_DISTCONV:-""}
if [[ ${build_distconv} ]]; then
    TEST_FLAG="test_*_distconv.py"
fi

job_unique_id=${CI_JOB_ID:-""}
prefix=""

# Setup the module environment
if [[ -n "${modules}" ]]
then
    echo "Loading modules: \"${modules}\""
    module load ${modules}
fi

# Finish setting up the environment
source ${project_dir}/.gitlab/setup_env.sh

# Make sure our working directory is something sane.
cd ${project_dir}

# Create some temporary build space.
if [[ -z "${job_unique_id}" ]]; then
    job_unique_id=manual_job_$(date +%F_%0H%0M)
    while [[ -d ${prefix}-${job_unique_id} ]] ; do
        sleep 1
        job_unique_id=manual_job_$(date +%F_%0H%0M)
    done
fi
build_dir=${BUILD_DIR:-"${project_dir}/build-${job_unique_id}"}
mkdir -p ${build_dir}

# Dependencies
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test started"
echo "~~~~~         Start: $(date)"
echo "~~~~~          Host: ${hostname}"
echo "~~~~~   Project dir: ${project_dir}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

prefix="${project_dir}/install-deps-${CI_JOB_NAME_SLUG:-${job_unique_id}}"
dha_prefix=${prefix}

# Just for good measure...
export CMAKE_PREFIX_PATH=${dha_prefix}/aluminum:${dha_prefix}/hydrogen:${dha_prefix}/dihydrogen:${CMAKE_PREFIX_PATH}
CMAKE_CMAKE_PREFIX_PATH=${CMAKE_PREFIX_PATH//:/;}

# Allow a user to force this
rebuild_deps=${REBUILD_DEPS:-""}

# Rebuild if the prefix doesn't exist.
if [[ ! -d "${prefix}" ]]
then
    rebuild_deps=1
fi

# Rebuild if latest hashes don't match
if [[ -z "${rebuild_deps}" ]]
then
    function fetch-sha {
        # $1 is the LLNL package name (e.g., 'aluminum')
        # $2 is the branch name (e.g., 'master')
        curl -s -H "Accept: application/vnd.github.VERSION.sha" \
             "https://api.github.com/repos/llnl/$1/commits/$2"
    }

    al_head=$(fetch-sha aluminum master)
    al_prebuilt="<not found>"
    if [[ -f "${prefix}/al-prebuilt-hash.txt" ]]
    then
        al_prebuilt=$(cat ${prefix}/al-prebuilt-hash.txt)
    fi

    h_head=$(fetch-sha elemental hydrogen)
    h_prebuilt="<not found>"
    if [[ -f "${prefix}/h-prebuilt-hash.txt" ]]
    then
        h_prebuilt=$(cat ${prefix}/h-prebuilt-hash.txt)
    fi

    h2_head=$(fetch-sha dihydrogen develop)
    h2_prebuilt="<not found>"
    if [[ -f "${prefix}/h2-prebuilt-hash.txt" ]]
    then
        h2_prebuilt=$(cat ${prefix}/h2-prebuilt-hash.txt)
    fi

    if [[ "${al_head}" != "${al_prebuilt}" ]]
    then
        echo "Prebuilt Aluminum hash does not match latest head; rebuilding."
        echo "  (prebuilt: ${al_prebuilt}; head: ${al_head})"
        rebuild_deps=1
    fi
    if [[ "${h_head}" != "${h_prebuilt}" ]]
    then
        echo "Prebuilt Hydrogen hash does not match latest head; rebuilding."
        echo "  (prebuilt: ${h_prebuilt}; head: ${h_head})"
        rebuild_deps=1
    fi
    if [[ "${h2_head}" != "${h2_prebuilt}" ]]
    then
        echo "Prebuilt DiHydrogen hash does not match latest head; rebuilding."
        echo "  (prebuilt: ${h2_prebuilt}; head: ${h2_head})"
        rebuild_deps=1
    fi
fi

if [[ -n "${rebuild_deps}" ]]
then

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Building Dependencies"
    echo "~~~~~     Build dir: ${build_dir}"
    echo "~~~~~   Install dir: ${prefix}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    # Set the superbuild dir
    lbann_sb_dir=${project_dir}/scripts/superbuild

    cd ${build_dir}
    # Uses "${cluster}", "${prefix}", and "${lbann_sb_dir}"
    source ${project_dir}/.gitlab/configure_deps.sh
    cmake --build build-deps
    ninja -C build-deps gather-all

    # Stamp these commits
    cd ${build_dir}/build-deps/aluminum/src && git rev-parse HEAD > ${prefix}/al-prebuilt-hash.txt
    cd ${build_dir}/build-deps/hydrogen/src && git rev-parse HEAD > ${prefix}/h-prebuilt-hash.txt
    cd ${build_dir}/build-deps/dihydrogen/src && git rev-parse HEAD > ${prefix}/h2-prebuilt-hash.txt

    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Dependencies Built"
    echo "~~~~~ $(date)"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
else
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
    echo "~~~~~ Using Cached Dependencies"
    echo "~~~~~     Prefix: ${prefix}"
    echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

    for f in $(find ${prefix} -iname "*.pc");
    do
        pfx=$(realpath $(dirname $(dirname $(dirname $f))))
        echo " >> Changing prefix in $(realpath $f) to: ${pfx}"
        sed -i -e "s|^prefix=.*|prefix=${pfx}|g" $f
    done
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Building LBANN"
echo "~~~~~ $(date)"
echo "~~~~~ CMAKE_PREFIX_PATH: ${CMAKE_PREFIX_PATH}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

prefix=${build_dir}/install
cd ${build_dir}
source ${project_dir}/.gitlab/configure_lbann.sh
if ! cmake --build build-lbann ;
then
    echo "ERROR: compilation failed, building with verbose output..."
    cmake --build build-lbann --verbose -j 1
else
    ninja -C build-lbann install
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Installing Python Packages with PIP"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

#CMD="python3 -m pip install -i https://pypi.org/simple --prefix ${prefix}/lbann protobuf tqdm numpy scipy"
CMD="python3 -m pip install -i https://pypi.org/simple --prefix ${prefix}/lbann pytest protobuf tqdm numpy"
echo ${CMD}
${CMD}

LBANN_MODFILES_DIR=${build_dir}/install/lbann/etc/modulefiles
ml use ${LBANN_MODFILES_DIR}
ml load lbann

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Testing LBANN: $(which lbann)"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

failed_tests=0
source ${project_dir}/.gitlab/run_catch_tests.sh

source ${project_dir}/.gitlab/run_unit_and_integration_tests.sh

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ LBANN Tests Complete"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "~~~~~ Build and test completed"
echo "~~~~~ $(date)"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

[[ "${failed_tests}" -eq 0 ]] && exit 0 || exit 1
