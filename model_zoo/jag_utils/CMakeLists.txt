add_executable( build_index-bin build_index.cpp )
target_link_libraries(build_index-bin lbann )
set_target_properties(build_index-bin PROPERTIES OUTPUT_NAME build_index)

