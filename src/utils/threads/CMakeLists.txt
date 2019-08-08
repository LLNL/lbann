# Test for non-portable POSIX features
include(CheckCXXSourceCompiles)
set(CMAKE_REQUIRED_LIBRARIES Threads::Threads)
set(_PTHREAD_SETAFFINITY_NP_TEST_CODE
  "#include <pthread.h>
int main(int, char* argv[]) {
  pthread_t thd = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  return pthread_setaffinity_np(thd, sizeof(cpu_set_t), &cpuset);
}")
check_cxx_source_compiles(
  "${_PTHREAD_SETAFFINITY_NP_TEST_CODE}" LBANN_HAS_PTHREAD_SETAFFINITY_NP)
set(_PTHREAD_GETAFFINITY_NP_TEST_CODE
  "#include <pthread.h>
int main(int, char* argv[]) {
  pthread_t thd = pthread_self();
  cpu_set_t cpuset;
  CPU_ZERO(&cpuset);
  return pthread_getaffinity_np(thd, sizeof(cpu_set_t), &cpuset);
}")
check_cxx_source_compiles(
  "${_PTHREAD_GETAFFINITY_NP_TEST_CODE}" LBANN_HAS_PTHREAD_GETAFFINITY_NP)
set(CMAKE_REQUIRED_LIBRARIES)

if (LBANN_HAS_PTHREAD_SETAFFINITY_NP AND LBANN_HAS_PTHREAD_GETAFFINITY_NP)
  set(LBANN_HAS_PTHREAD_AFFINITY_SUPPORT TRUE CACHE INTERNAL
    "LBANN has pthread affinity support")
endif ()

# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  thread_pool.cpp
  thread_utils.cpp
)

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
