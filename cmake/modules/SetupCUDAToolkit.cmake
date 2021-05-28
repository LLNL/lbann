# This handles the non-compiler aspect of the CUDA toolkit.
# NCCL and cuDNN are handled separately.

if (NOT CUDA_FOUND)
  find_package(CUDA REQUIRED)
endif ()

find_package(NVTX REQUIRED)
find_package(cuDNN REQUIRED)
find_package(cuFFT REQUIRED)

if (NOT TARGET cuda::toolkit)
  add_library(cuda::toolkit INTERFACE IMPORTED)
endif ()

set_property(TARGET cuda::toolkit APPEND PROPERTY
  INTERFACE_LINK_LIBRARIES cuda::cudnn cuda::cufft cuda::nvtx)
