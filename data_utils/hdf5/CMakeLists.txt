add_executable(generate_HDF5_schema_and_sample_list
  generate_HDF5_schema_and_sample_list.cpp)
target_link_libraries(generate_HDF5_schema_and_sample_list
  PRIVATE
  conduit::conduit)
set_target_properties(generate_HDF5_schema_and_sample_list
  PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)
if ("cxx_std_11" IN_LIST CMAKE_CXX_COMPILE_FEATURES)
  target_compile_features(generate_HDF5_schema_and_sample_list
    PRIVATE
    cxx_std_11)
endif ()
install(TARGETS generate_HDF5_schema_and_sample_list
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR})
