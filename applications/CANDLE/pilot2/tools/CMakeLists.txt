# Add a target to control building all the utilities
add_custom_target(pilot2-utils)

add_executable(test_ras_lipid_data_files_for_errors
  EXCLUDE_FROM_ALL test_ras_lipid_data_files_for_errors.cpp)
  target_link_libraries(test_ras_lipid_data_files_for_errors lbann)
  add_dependencies(pilot2-utils test_ras_lipid_data_files_for_errors)

add_executable(compute_ras_lipid_sig1_normalization
  EXCLUDE_FROM_ALL compute_ras_lipid_sig1_normalization.cpp)
  target_link_libraries(compute_ras_lipid_sig1_normalization lbann)
  add_dependencies(pilot2-utils compute_ras_lipid_sig1_normalization)

add_executable(compute_ras_lipid_bbs_euclid_distances
  EXCLUDE_FROM_ALL compute_ras_lipid_bbs_euclid_distances.cpp)
  target_link_libraries(compute_ras_lipid_bbs_euclid_distances lbann)
  add_dependencies(pilot2-utils compute_ras_lipid_bbs_euclid_distances)

add_executable(compute_ras_lipid_bbs_max_min
  EXCLUDE_FROM_ALL compute_ras_lipid_bbs_max_min.cpp)
  target_link_libraries(compute_ras_lipid_bbs_max_min lbann)
  add_dependencies(pilot2-utils compute_ras_lipid_bbs_max_min)

add_executable(compute_ras_lipid_bbs_euclid_normalization
  EXCLUDE_FROM_ALL compute_ras_lipid_bbs_euclid_normalization.cpp)
  target_link_libraries(compute_ras_lipid_bbs_euclid_normalization lbann)
  add_dependencies(pilot2-utils compute_ras_lipid_bbs_euclid_normalization)

# Install the binaries
install( TARGETS 
  test_ras_lipid_data_files_for_errors 
  compute_ras_lipid_sig1_normalization 
  compute_ras_lipid_bbs_euclid_distances
  compute_ras_lipid_bbs_euclid_normalization

  OPTIONAL
  EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  INCLUDES DESTINATION ${CMAKE_INSTALL_INCLUDEDIR}
  )

#(from Tom)
# The use of `OPTIONAL` here will trigger CMake warnings. These can
# safely be ignored and tests confirm that. See these for more info:
#
# https://gitlab.kitware.com/cmake/cmake/issues/18258
# https://cmake.org/pipermail/cmake/2011-August/046014.html

