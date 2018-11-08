# This handles the non-compiler aspect of the CUDA toolkit.
# NCCL and cuDNN are handled separately.

if (NOT CUDA_FOUND)
  find_package(CUDA REQUIRED)
endif ()

find_package(CUB REQUIRED)
find_package(NVTX REQUIRED)
find_package(cuDNN REQUIRED)

if (NOT TARGET cuda::toolkit)
  add_library(cuda::toolkit INTERFACE IMPORTED)
endif ()

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES cuda::cub cuda::cudnn cuda::nvtx
  "${CUDA_CUBLAS_LIBRARIES}" "${CUDA_CUDA_LIBRARY}")

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CUDA>:-arch=sm_30>)

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_INCLUDE_DIRECTORIES "${CUDA_INCLUDE_DIRS}")
