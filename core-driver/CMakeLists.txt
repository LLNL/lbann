cmake_minimum_required(VERSION 3.21.0)
project(lbann-test-driver CXX)
find_package(LBANN 0.102.0 REQUIRED)
add_executable(lbann-core main.cpp)
target_link_libraries(lbann-core PRIVATE LBANN::lbann)

#target_link_libraries(lbann-bin lbann)
set_target_properties(lbann-core
  PROPERTIES
  OUTPUT_NAME lbann-core-driver
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

#list(APPEND LBANN_EXE_TGTS lbann-core)

install(TARGETS lbann-core
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
