# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  any.hpp
  compiler_control.hpp
  cublas.hpp
  cuda.hpp
  cudnn.hpp
  dataset.hpp
  description.hpp
  entrywise_operator.hpp
  exception.hpp
  factory.hpp
  factory_error_policies.hpp
  file_utils.hpp
  glob.hpp
  im2col.hpp
  image.hpp
  jag_utils.hpp
  lbann_library.hpp
  mild_exception.hpp
  number_theory.hpp
  omp_diagnostics.hpp
  opencv.hpp
  options.hpp
  profiling.hpp
  prototext.hpp
  python.hpp
  random.hpp
  statistics.hpp
  summary.hpp
  timer.hpp
  type_erased_matrix.hpp
  )

# Add the subdirectories
add_subdirectory(threads)
add_subdirectory(impl)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
