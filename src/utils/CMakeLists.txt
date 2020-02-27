# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  argument_parser.cpp
  cnpy_utils.cpp
  cublas.cpp
  cudnn.cpp
  description.cpp
  environment_variable.cpp
  exception.cpp
  file_utils.cpp
  graph.cpp
  im2col.cpp
  image.cpp
  number_theory.cpp
  omp_diagnostics.cpp
  options.cpp
  profiling.cpp
  protobuf_utils.cpp
  python.cpp
  random.cpp
  stack_profiler.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
  lbann_library.cpp
  jag_common.cpp
)

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    cuda.cu
    )
endif ()

add_subdirectory(threads)

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/distconv.cpp")
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
