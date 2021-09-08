# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  common.hpp

  abs.cpp
  binary.cpp
  binary_with_constant.cpp
  clamp.cpp
  math_builders.cpp
  unary.cpp
  )

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    common.cuh

    abs.cu
    binary.cu
    binary_with_constant.cu
    clamp.cu
    unary.cu
    )
endif ()

add_subdirectory(cereal_registration)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
