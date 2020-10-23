
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
  image.cpp
  jag_common.cpp
  lbann_library.cpp
  number_theory.cpp
  omp_diagnostics.cpp
  options.cpp
  profiling.cpp
  protobuf_utils.cpp
  python.cpp
  random.cpp
  random_number_generators.cpp
  stack_profiler.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
  system_info.cpp
  trainer_file_utils.cpp
)

if (LBANN_HAS_HALF)
  list(APPEND THIS_DIR_SOURCES
    ${CMAKE_CURRENT_SOURCE_DIR}/serialization.cpp)
endif (LBANN_HAS_HALF)

if (LBANN_HAS_GPU)
  list(APPEND THIS_DIR_SOURCES
    "${CMAKE_CURRENT_SOURCE_DIR}/cufft_wrapper.cpp")

  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    cuda.cu
    nvshmem.cu
    )
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distconv.cpp")
endif ()

add_subdirectory(threads)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
