# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  Elemental_extensions.cpp
  base.cpp
  comm.cpp
  )

# Add the subdirectories
add_subdirectory(callbacks)
add_subdirectory(data_readers)
add_subdirectory(data_store)
add_subdirectory(io)
add_subdirectory(layers)
add_subdirectory(metrics)
add_subdirectory(models)
add_subdirectory(objective_functions)
add_subdirectory(optimizers)
add_subdirectory(proto)
add_subdirectory(transforms)
add_subdirectory(utils)
add_subdirectory(weights)

if (TARGET LbannProto)
  set(LBANN_HAS_LBANN_PROTO TRUE PARENT_SCOPE)
endif ()

# Propagate the files up the tree
set(LBANN_HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
set(LBANN_SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
set(LBANN_CUDA_SOURCES "${CUDA_SOURCES}" "${THIS_DIR_CU_SOURCES}" PARENT_SCOPE)
