if (LBANN_HAS_CONDUIT)

  add_executable( build_index-bin build_index.cpp )
  target_link_libraries(build_index-bin lbann )
  set_target_properties(build_index-bin PROPERTIES OUTPUT_NAME build_index)

  add_executable( extract_random_samples-bin extract_random_samples.cpp )
  target_link_libraries(extract_random_samples-bin lbann )
  set_target_properties(extract_random_samples-bin PROPERTIES OUTPUT_NAME extract_random_samples)

  add_executable( dump_bundle-bin dump_bundle.cpp )
  target_link_libraries(dump_bundle-bin lbann )
  set_target_properties(dump_bundle-bin PROPERTIES OUTPUT_NAME dump_bundle)

  add_executable( check_images-bin check_images.cpp )
  target_link_libraries(check_images-bin lbann )
  set_target_properties(check_images-bin PROPERTIES OUTPUT_NAME check_images)

  add_executable( detect_corruption-bin detect_corruption.cpp )
  target_link_libraries(detect_corruption-bin lbann )
  set_target_properties(detect_corruption-bin PROPERTIES OUTPUT_NAME detect_corruption)

  add_executable( load_bundle2raw-bin load_bundle2raw.cpp )
  target_link_libraries(load_bundle2raw-bin lbann )
  set_target_properties(load_bundle2raw-bin PROPERTIES OUTPUT_NAME load_bundle2raw)

  add_executable( compute_min_max_images-bin compute_min_max_images.cpp )
  target_link_libraries(compute_min_max_images-bin lbann )
  set_target_properties(compute_min_max_images-bin PROPERTIES OUTPUT_NAME compute_min_max_images)

  add_executable( load_balance-bin load_balance.cpp )
  target_link_libraries(load_balance-bin lbann )
  set_target_properties(load_balance-bin PROPERTIES OUTPUT_NAME load_balance)

  add_executable( check_for_duplicate_samples-bin check_for_duplicate_samples.cpp )
  target_link_libraries(check_for_duplicate_samples-bin lbann )
  set_target_properties(check_for_duplicate_samples-bin PROPERTIES OUTPUT_NAME check_for_duplicate_samples)

  add_executable( test_conduit_hdf5-bin test_conduit_hdf5.cpp )
  target_link_libraries(test_conduit_hdf5-bin lbann )
  set_target_properties(test_conduit_hdf5-bin PROPERTIES OUTPUT_NAME test_conduit_hdf5)

  add_executable( select_samples select_samples.cpp )
endif ()
