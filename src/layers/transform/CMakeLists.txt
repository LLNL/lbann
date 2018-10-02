# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  crop.cpp
  evaluation.cpp
  in_top_k.cpp
  sort.cpp
  tessellate.cpp
  )

if (LBANN_HAS_CUDA)
  # Add the CUDA source files for this directory
  set_full_path(THIS_DIR_CU_SOURCES
    crop.cu
    in_top_k.cu
    sort.cu
    tessellate.cu
    )
endif ()

if (LBANN_HAS_DISTCONV)
  list(APPEND THIS_DIR_CU_SOURCES "${CMAKE_CURRENT_SOURCE_DIR}/split.cu")
endif ()

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
