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
if (LBANN_WITH_UNIT_TESTING)
  add_executable(helpers_tests
    $<$<NOT:$<BOOL:${LBANN_USE_CATCH2_V3}>>:test_main.cpp>
    test_helpers.cpp
    helpers.cpp)
  set_target_properties(helpers_tests
    PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/unit_test)
  target_link_libraries(helpers_tests
    PRIVATE
    conduit::conduit
    $<IF:$<BOOL:${LBANN_USE_CATCH2_V3}>,Catch2::Catch2WithMain,Catch2::Catch2>)
  if (LBANN_USE_CATCH2_V3)
    target_compile_definitions(helpers_tests PRIVATE LBANN_USE_CATCH2_V3)
  endif ()
  if ("cxx_std_17" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
    target_compile_features(helpers_tests PRIVATE cxx_std_17)
  else ()
    lbann_check_and_append_flag(LBANN_CXX17_FLAG "-std=c++17")
    target_compile_options(helpers_tests PRIVATE ${LBANN_CXX17_FLAG})
  endif ()
endif ()
