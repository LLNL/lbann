add_executable( build_index-bin build_index.cpp )
target_link_libraries(build_index-bin lbann )
set_target_properties(build_index-bin PROPERTIES OUTPUT_NAME build_index)

add_executable( extract_random_samples-bin extract_random_samples.cpp )
target_link_libraries(extract_random_samples-bin lbann )
set_target_properties(extract_random_samples-bin PROPERTIES OUTPUT_NAME extract_random_samples)
