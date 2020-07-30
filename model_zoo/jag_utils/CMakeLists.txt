# Add a target to control building all the utilities
add_custom_target(jag-utils)

add_executable(build_index
  EXCLUDE_FROM_ALL build_index.cpp)
target_link_libraries(build_index lbann)
add_dependencies(jag-utils build_index)

add_executable(extract_random_samples
  EXCLUDE_FROM_ALL extract_random_samples.cpp)
target_link_libraries(extract_random_samples lbann)
add_dependencies(jag-utils extract_random_samples)

add_executable(dump_bundle
  EXCLUDE_FROM_ALL dump_bundle.cpp)
target_link_libraries(dump_bundle lbann)
add_dependencies(jag-utils dump_bundle)

add_executable(check_images
  EXCLUDE_FROM_ALL check_images.cpp)
target_link_libraries(check_images lbann)
add_dependencies(jag-utils check_images)

add_executable(detect_corruption
  EXCLUDE_FROM_ALL detect_corruption.cpp)
target_link_libraries(detect_corruption lbann)
add_dependencies(jag-utils detect_corruption)

add_executable(load_bundle2raw
  EXCLUDE_FROM_ALL load_bundle2raw.cpp)
target_link_libraries(load_bundle2raw lbann)
add_dependencies(jag-utils load_bundle2raw)

add_executable(compute_min_max_images
  EXCLUDE_FROM_ALL compute_min_max_images.cpp)
target_link_libraries(compute_min_max_images lbann)
add_dependencies(jag-utils compute_min_max_images)

add_executable(compute_per_channel_image_avg_min_max
  EXCLUDE_FROM_ALL compute_per_channel_image_avg_min_max.cpp)
target_link_libraries(compute_per_channel_image_avg_min_max lbann)
add_dependencies(jag-utils compute_per_channel_image_avg_min_max)

add_executable(load_balance
  EXCLUDE_FROM_ALL load_balance.cpp)
target_link_libraries(load_balance lbann)
add_dependencies(jag-utils load_balance)

add_executable(check_for_duplicate_samples
  EXCLUDE_FROM_ALL check_for_duplicate_samples.cpp)
target_link_libraries(check_for_duplicate_samples lbann)
add_dependencies(jag-utils extract_random_samples)

add_executable(test_conduit_hdf5
  EXCLUDE_FROM_ALL test_conduit_hdf5.cpp)
target_link_libraries(test_conduit_hdf5 lbann)
add_dependencies(jag-utils test_conduit_hdf5)

add_executable(select_samples
  EXCLUDE_FROM_ALL select_samples.cpp)
target_link_libraries(select_samples lbann)
add_dependencies(jag-utils select_samples)

add_executable(build_sample_id_mapping
  EXCLUDE_FROM_ALL build_sample_id_mapping.cpp)
target_link_libraries(build_sample_id_mapping lbann)
add_dependencies(jag-utils build_sample_id_mapping)

add_executable(generate_corrupt_samples
  EXCLUDE_FROM_ALL generate_corrupt_samples.cpp)
target_link_libraries(generate_corrupt_samples lbann)
add_dependencies(jag-utils generate_corrupt_samples)

add_executable(compute_hydra_normalization
  EXCLUDE_FROM_ALL compute_hydra_normalization.cpp)
target_link_libraries(compute_hydra_normalization lbann)
add_dependencies(jag-utils compute_hydra_normalization)

add_executable(test_reading_speed
  EXCLUDE_FROM_ALL test_reading_speed.cpp)
target_link_libraries(test_reading_speed lbann)
add_dependencies(jag-utils test_reading_speed)

add_executable(convert
  EXCLUDE_FROM_ALL convert.cpp)
target_link_libraries(convert lbann)
add_dependencies(jag-utils convert)

add_executable(convert_npz_to_conduit
  EXCLUDE_FROM_ALL convert_npz_to_conduit.cpp)
target_link_libraries(convert_npz_to_conduit lbann)
add_dependencies(jag-utils convert_npz_to_conduit)

# Install the binaries
install(
  TARGETS select_samples build_sample_id_mapping build_index convert_npz_to_conduit 
  OPTIONAL
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

# The use of `OPTIONAL` here will trigger CMake warnings. These can
# safely be ignored and tests confirm that. See these for more info:
#
# https://gitlab.kitware.com/cmake/cmake/issues/18258
# https://cmake.org/pipermail/cmake/2011-August/046014.html
