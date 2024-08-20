if [[ "$cluster" == "lassen" ]]
then
    hydrogen_lapack_opt="-D LBANN_SB_FWD_Hydrogen_BLA_VENDOR=Generic"
    dihydrogen_lapack_opt="-D LBANN_SB_FWD_DiHydrogen_BLA_VENDOR=Generic"
else
    hydrogen_lapack_opt=""
    dihydrogen_lapack_opt=""
fi

echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"
echo "-----  BVE Dependency Flags:"
echo "-----  HALF: ${build_half}"
echo "-----  DISTCONV: ${build_distconv}"
echo "-----  FFT: ${build_fft}"
echo "~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~"

#    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_GPU_FP16=${build_half:-OFF} \
cmake \
    -G Ninja \
    -S ${lbann_sb_dir} \
    -B ${build_dir}/build-deps \
    \
    -D CMAKE_PREFIX_PATH=${CMAKE_CMAKE_PREFIX_PATH} \
    -D CMAKE_BUILD_TYPE=Release \
    -D CMAKE_INSTALL_PREFIX=${prefix} \
    \
    -D CMAKE_EXE_LINKER_FLAGS=${common_linker_flags} \
    -D CMAKE_SHARED_LINKER_FLAGS=${common_linker_flags} \
    \
    -D CMAKE_BUILD_RPATH="${extra_rpaths//:/|}" \
    -D CMAKE_INSTALL_RPATH="${extra_rpaths//:/|}" \
    \
    -D BUILD_SHARED_LIBS=ON \
    -D CMAKE_BUILD_RPATH_USE_ORIGIN=OFF \
    -D CMAKE_BUILD_WITH_INSTALL_RPATH=OFF \
    -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
    -D CMAKE_SKIP_BUILD_RPATH=OFF \
    -D CMAKE_SKIP_INSTALL_RPATH=OFF \
    -D CMAKE_SKIP_RPATH=OFF \
    \
    -D CMAKE_CXX_STANDARD=17 \
    -D CMAKE_CUDA_STANDARD=17 \
    -D CMAKE_HIP_STANDARD=17 \
    \
    -D CMAKE_CUDA_ARCHITECTURES=${gpu_arch} \
    -D CMAKE_HIP_ARCHITECTURES=${gpu_arch} \
    \
    -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
    \
    -D LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY="PKG_LC" \
    -D LBANN_SB_DEFAULT_CUDA_OPTS=${cuda_platform} \
    -D LBANN_SB_DEFAULT_ROCM_OPTS=${rocm_platform} \
    \
    -D LBANN_SB_BUILD_Aluminum=ON \
    -D LBANN_SB_Aluminum_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_Aluminum_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_CALIPER=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_NCCL=ON \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_HOST_TRANSFER=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_TESTS=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_BENCHMARKS=OFF \
    -D LBANN_SB_FWD_Aluminum_ALUMINUM_ENABLE_THREAD_MULTIPLE=OFF \
    -D LBANN_SB_FWD_Aluminum_CMAKE_PREFIX_PATH=${FWD_CMAKE_PREFIX_PATH} \
    \
    -D LBANN_SB_BUILD_Hydrogen=ON \
    ${hydrogen_lapack_opt} \
    -D LBANN_SB_Hydrogen_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_Hydrogen_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_HALF=${build_half:-OFF} \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_TESTING=ON \
    -D LBANN_SB_FWD_Hydrogen_Hydrogen_ENABLE_UNIT_TESTS=OFF \
    -D LBANN_SB_FWD_Hydrogen_CMAKE_PREFIX_PATH=${FWD_CMAKE_PREFIX_PATH} \
    \
    -D LBANN_SB_BUILD_DiHydrogen=ON \
    ${dihydrogen_lapack_opt} \
    -D LBANN_SB_DiHydrogen_TAG=fix-rocm-6-2-0-build \
    -D LBANN_SB_DiHydrogen_URL=https://github.com/benson31/dihydrogen \
    -D LBANN_SB_DiHydrogen_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
    -D LBANN_SB_DiHydrogen_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
    -D LBANN_SB_FWD_DiHydrogen_H2_ENABLE_DISTCONV_LEGACY=${build_distconv:-OFF} \
    -D LBANN_SB_FWD_DiHydrogen_CMAKE_PREFIX_PATH=${FWD_CMAKE_PREFIX_PATH}
