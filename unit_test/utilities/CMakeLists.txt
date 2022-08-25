# Add the library
add_library(unit_test_utilities
  # Catch2 version abstraction
  Catch2BasicSupport.hpp

  # Headers
  MPITestHelpers.hpp
  ReplaceEscapes.hpp

  # C++
  MPITestHelpers.cpp
  ReplaceEscapes.cpp
  ) # add_library unit_test_utilities

target_include_directories(unit_test_utilities
  PUBLIC ${CMAKE_CURRENT_SOURCE_DIR})

target_link_libraries(unit_test_utilities PUBLIC lbann)

# Add the unit tests for the library
add_subdirectory(unit_test)
set(LBANN_SEQ_CATCH2_TEST_FILES
  "${LBANN_SEQ_CATCH2_TEST_FILES}"
  PARENT_SCOPE)
