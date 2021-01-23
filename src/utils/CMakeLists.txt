# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  argument_parser.cpp
  cnpy_utils.cpp
  commify.cpp
  cudnn.cpp
  description.cpp
  environment_variable.cpp
  exception.cpp
  file_utils.cpp
  graph.cpp
  im2col.cpp
  jag_common.cpp
  lbann_library.cpp
  miopen.cpp
  number_theory.cpp
  omp_diagnostics.cpp
  options.cpp
  profiling.cpp
  protobuf_utils.cpp
  python.cpp
  random.cpp
  random_number_generators.cpp
  serialization.cpp
  stack_profiler.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
  system_info.cpp
  trainer_file_utils.cpp
  visitor_hooks.cpp
)

if (LBANN_HAS_CUDA)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cufft_wrapper.cpp")

  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    cuda.cu
    nvshmem.cu
    )
endif ()

if (LBANN_HAS_ROCM)
  # Add the ROCM source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    rocm.cpp
    )
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distconv.cpp")
endif ()

if (LBANN_HAS_OPENCV)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/image.cpp")
endif ()

add_subdirectory(serialize_matrices)
add_subdirectory(threads)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
