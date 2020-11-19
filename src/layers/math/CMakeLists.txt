# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  unary.cpp
  binary.cpp
  clamp.cpp
  math_builders.cpp
  matmul.cpp
  )

if (LBANN_HAS_GPU)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    unary.cu
    binary.cu
    clamp.cu
    )
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(GPU_SOURCES "${GPU_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
