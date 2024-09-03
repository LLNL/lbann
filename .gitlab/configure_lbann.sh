if [[ "$cluster" == "lassen" ]]
then
    lbann_lapack_opt="-D BLA_VENDOR=Generic"
    build_fft=OFF
else
    lbann_lapack_opt=""
fi

if [[ "$cluster" == "tioga" ]]
then
    build_fft=OFF
fi

# Default RPATH rules will not include in-source libraries from the prefix path... add them here.
if [ -z "${extra_rpaths}" ]; then
    extra_rpaths=${dha_prefix}/aluminum/lib64:${dha_prefix}/hydrogen/lib:${dha_prefix}/dihydrogen/lib64
else
    extra_rpaths=${dha_prefix}/aluminum/lib64:${dha_prefix}/hydrogen/lib:${dha_prefix}/dihydrogen/lib64:${extra_rpaths:-""}
fi

cmake -G Ninja \
      -S ${project_dir} \
      -B ${build_dir}/build-lbann \
      \
      -D CMAKE_PREFIX_PATH=${CMAKE_CMAKE_PREFIX_PATH} \
      -D CMAKE_BUILD_TYPE=RelWithDebInfo \
      -D CMAKE_INSTALL_PREFIX=${prefix}/lbann \
      \
      -D CMAKE_BUILD_RPATH="${extra_rpaths//:/;}" \
      -D CMAKE_INSTALL_RPATH="${extra_rpaths//:/;}" \
      -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=ON \
      \
      -D CMAKE_CXX_STANDARD=17 \
      -D CMAKE_CUDA_STANDARD=17 \
      -D CMAKE_HIP_STANDARD=17 \
      \
      -D CMAKE_CUDA_ARCHITECTURES=${gpu_arch} \
      -D CMAKE_HIP_ARCHITECTURES=${gpu_arch} \
      -D AMDGPU_TARGETS=${gpu_arch} \
      -D GPU_TARGETS=${gpu_arch} \
      \
      -D BUILD_SHARED_LIBS=ON \
      -D CMAKE_POSITION_INDEPENDENT_CODE=ON \
      -D CMAKE_EXE_LINKER_FLAGS="${EXTRA_LINK_FLAGS}" \
      -D CMAKE_SHARED_LINKER_FLAGS="${EXTRA_LINK_FLAGS}" \
      \
      -D CMAKE_EXPORT_COMPILE_COMMANDS=ON \
      -D CMAKE_CXX_FLAGS="${EXTRA_CXX_FLAGS}" \
      -D CMAKE_HIP_FLAGS="${EXTRA_HIP_FLAGS}" \
      ${lbann_lapack_opt} \
      -D LBANN_DATATYPE=float \
      -D LBANN_WITH_CALIPER=OFF \
      -D LBANN_WITH_DISTCONV=${build_distconv:-OFF} \
      -D LBANN_WITH_TBINF=OFF \
      -D LBANN_WITH_UNIT_TESTING=ON \
      -D LBANN_WITH_CNPY=ON \
      -D LBANN_DETERMINISTIC=ON \
      -D LBANN_WITH_ADDRESS_SANITIZER=OFF \
      -D LBANN_WITH_FFT=${build_fft:-OFF} \
      -D LBANN_WITH_EMBEDDED_PYTHON=ON \
      -D LBANN_WITH_PYTHON_FRONTEND=ON \
      -D LBANN_WITH_VISION=ON
