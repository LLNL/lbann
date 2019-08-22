add_executable(build_index build_index.cpp)
target_link_libraries(build_index PRIVATE lbann)

add_executable(extract_random_samples extract_random_samples.cpp)
target_link_libraries(extract_random_samples PRIVATE lbann)

add_executable(dump_bundle dump_bundle.cpp)
target_link_libraries(dump_bundle PRIVATE lbann)

add_executable(check_images check_images.cpp)
target_link_libraries(check_images PRIVATE lbann)

add_executable(detect_corruption detect_corruption.cpp)
target_link_libraries(detect_corruption PRIVATE lbann)

add_executable(load_bundle2raw load_bundle2raw.cpp)
target_link_libraries(load_bundle2raw PRIVATE lbann)

add_executable(compute_min_max_images compute_min_max_images.cpp)
target_link_libraries(compute_min_max_images PRIVATE lbann)

add_executable(compute_per_channel_image_avg_min_max compute_per_channel_image_avg_min_max.cpp)
target_link_libraries(compute_per_channel_image_avg_min_max PRIVATE lbann)

add_executable(load_balance load_balance.cpp)
target_link_libraries(load_balance PRIVATE lbann)

add_executable(check_for_duplicate_samples check_for_duplicate_samples.cpp)
target_link_libraries(check_for_duplicate_samples PRIVATE lbann)

add_executable(test_conduit_hdf5 test_conduit_hdf5.cpp)
target_link_libraries(test_conduit_hdf5 PRIVATE lbann)

add_executable(select_samples select_samples.cpp)
target_link_libraries(select_samples PRIVATE lbann)

add_executable(build_sample_id_mapping build_sample_id_mapping.cpp)
target_link_libraries(build_sample_id_mapping PRIVATE lbann)

add_executable(generate_corrupt_samples generate_corrupt_samples.cpp)
target_link_libraries(generate_corrupt_samples PRIVATE lbann)

add_executable(compute_hydra_normalization compute_hydra_normalization.cpp)
target_link_libraries(compute_hydra_normalization PRIVATE lbann)

add_executable(test_reading_speed test_reading_speed.cpp)
target_link_libraries(test_reading_speed PRIVATE lbann)

add_executable(convert convert.cpp)
target_link_libraries(convert PRIVATE lbann)

install(
  TARGETS select_samples
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )
