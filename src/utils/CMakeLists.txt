# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  cnpy_utils.cpp
  cublas.cpp
  cudnn.cpp
  exception.cpp
  file_utils.cpp
  graph.cpp
  im2col.cpp
  number_theory.cpp
  omp_diagnostics.cpp
  options.cpp
  profiling.cpp
  protobuf_utils.cpp
  quantizer.cpp
  random.cpp
  stack_profiler.cpp
  stack_trace.cpp
  statistics.cpp
  summary.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
