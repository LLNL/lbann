if [[ "$cluster" == "lassen" ]]
then
    lapack_opt="-D BLA_VENDOR=Generic"
else
    lapack_opt=""
fi

cmake -G Ninja \
      -S ${project_dir} \
      -B ${build_dir}/build-lbann \
      \
      -D CMAKE_PREFIX_PATH=${CMAKE_CMAKE_PREFIX_PATH} \
      -D CMAKE_BUILD_TYPE=Release \
      -D CMAKE_INSTALL_PREFIX=${prefix}/lbann \
      \
      -D CMAKE_BUILD_RPATH="${extra_rpaths//:/\;}" \
      -D CMAKE_INSTALL_RPATH="${extra_rpaths//:/\;}" \
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CUDA_STANDARD=17 \
      -D CMAKE_HIP_STANDARD=17 \
      \
      -D CMAKE_CUDA_ARCHITECTURES=${gpu_arch} \
      -D CMAKE_HIP_ARCHITECTURES=${gpu_arch} \
      \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
      -D CMAKE_EXE_LINKER_FLAGS="${EXTRA_LINK_FLAGS}" \
      -D CMAKE_SHARED_LINKER_FLAGS="${EXTRA_LINK_FLAGS}" \
      \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -D CMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
      -D CMAKE_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
      -D LBANN_DATATYPE=float \
      -D LBANN_WITH_CALIPER=OFF \
      -D LBANN_WITH_DISTCONV=${build_distconv:-OFF} \
      -D LBANN_WITH_TBINF=OFF \
      -D LBANN_WITH_UNIT_TESTING=ON \
      -D LBANN_WITH_CNPY=ON \
      -D LBANN_DETERMINISTIC=ON \
      -D LBANN_WITH_ADDRESS_SANITIZER=OFF \
      -D LBANN_WITH_FFT=OFF \
      -D LBANN_WITH_EMBEDDED_PYTHON=ON \
      -D LBANN_WITH_PYTHON_FRONTEND=ON \
      -D LBANN_WITH_VISION=ON


#      -D LBANN_WITH_CUTENSOR=OFF \

# \
#       -D CMAKE_PREFIX_PATH=${FWD_CMAKE_PREFIX_PATH}

#      -D LBANN_SB_LBANN_SOURCE_DIR=${LBANN_SRC_DIR} \

      # -D LBANN_SB_DEFAULT_CUDA_OPTS=${cuda_platform} \
      # -D LBANN_SB_DEFAULT_ROCM_OPTS=${rocm_platform} \
      # -D LBANN_WITH_NVSHMEM=OFF \

      # \
      # -D LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY="PKG_LC" \
