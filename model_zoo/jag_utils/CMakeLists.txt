add_executable(build_index build_index.cpp)
target_link_libraries(build_index lbann)

add_executable(extract_random_samples extract_random_samples.cpp)
target_link_libraries(extract_random_samples lbann)

add_executable(dump_bundle dump_bundle.cpp)
target_link_libraries(dump_bundle lbann)

add_executable(check_images check_images.cpp)
target_link_libraries(check_images lbann)

add_executable(detect_corruption detect_corruption.cpp)
target_link_libraries(detect_corruption lbann)

add_executable(load_bundle2raw load_bundle2raw.cpp)
target_link_libraries(load_bundle2raw lbann)

add_executable(compute_min_max_images compute_min_max_images.cpp)
target_link_libraries(compute_min_max_images lbann)

add_executable(compute_per_channel_image_avg_min_max compute_per_channel_image_avg_min_max.cpp)
target_link_libraries(compute_per_channel_image_avg_min_max lbann)

add_executable(load_balance load_balance.cpp)
target_link_libraries(load_balance lbann)

add_executable(check_for_duplicate_samples check_for_duplicate_samples.cpp)
target_link_libraries(check_for_duplicate_samples lbann)

add_executable(test_conduit_hdf5 test_conduit_hdf5.cpp)
target_link_libraries(test_conduit_hdf5 lbann)

add_executable(select_samples select_samples.cpp)
target_link_libraries(select_samples lbann)

add_executable(build_sample_id_mapping build_sample_id_mapping.cpp)
target_link_libraries(build_sample_id_mapping lbann)

add_executable(generate_corrupt_samples generate_corrupt_samples.cpp)
target_link_libraries(generate_corrupt_samples lbann)

add_executable(compute_hydra_normalization compute_hydra_normalization.cpp)
target_link_libraries(compute_hydra_normalization lbann)

add_executable(test_reading_speed test_reading_speed.cpp)
target_link_libraries(test_reading_speed lbann)

add_executable(convert convert.cpp)
target_link_libraries(convert lbann)

install(
  TARGETS select_samples
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
