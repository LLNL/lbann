add_executable(generate_schema_and_sample_list
  generate_schema_and_sample_list.cpp
  helpers.cpp)
target_link_libraries(generate_schema_and_sample_list
  PRIVATE
  clara::clara
  conduit::conduit)
set_target_properties(generate_schema_and_sample_list
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if ("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  target_compile_features(generate_schema_and_sample_list
    PRIVATE
    cxx_std_17)
endif ()
install(TARGETS generate_schema_and_sample_list
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})

# Testing
add_executable(helpers_tests test_main.cpp test_helpers.cpp helpers.cpp)
set_target_properties(helpers_tests
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
target_link_libraries(helpers_tests
  PRIVATE
  conduit::conduit
  Catch2::Catch2)
target_compile_features(helpers_tests PRIVATE cxx_std_17)
