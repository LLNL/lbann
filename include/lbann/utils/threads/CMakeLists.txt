# Add the headers for this directory
set_full_path(THIS_DIR_HEADERS
  thread_pool.hpp
  thread_safe_queues.hpp
  thread_topology.hpp
  type_erased_function.hpp
  memory.hpp
  thread_utils.hpp
  )

# Propagate the files up the tree
set(HEADERS "${HEADERS}" "${THIS_DIR_HEADERS}" PARENT_SCOPE)
