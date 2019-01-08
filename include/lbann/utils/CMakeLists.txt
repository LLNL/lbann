# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  compiler_control.hpp
  cublas.hpp
  cuda.hpp
  cudnn.hpp
  dataset.hpp
  description.hpp
  entrywise_operator.hpp
  exception.hpp
  glob.hpp
  im2col.hpp
  mild_exception.hpp
  number_theory.hpp
  omp_diagnostics.hpp
  options.hpp
  profiling.hpp
  prototext.hpp
  quantizer.hpp
  quantizer_impl.hpp
  random.hpp
  statistics.hpp
  summary.hpp
  timer.hpp
  lbann_library.hpp
  jag_utils.hpp
  )

# Add the subdirectories
add_subdirectory(threads)
add_subdirectory(impl)

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
