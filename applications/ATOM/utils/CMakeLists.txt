# Add a target to control building all the utilities
add_custom_target(atom-utils)

add_executable(compute_vocab
  EXCLUDE_FROM_ALL compute_vocab.cpp)
  target_link_libraries(compute_vocab lbann)
  add_dependencies(atom-utils compute_vocab)

# Install the binaries
install( 
  TARGETS compute_vocab
  OPTIONAL
  #EXPORT LBANNTargets
  RUNTIME DESTINATION ${CMAKE_INSTALL_BINDIR}
  ARCHIVE DESTINATION ${CMAKE_INSTALL_LIBDIR}
  LIBRARY DESTINATION ${CMAKE_INSTALL_LIBDIR}
  )

#(from Tom)
# The use of `OPTIONAL` here will trigger CMake warnings. These can
# safely be ignored and tests confirm that. See these for more info:
#
# https://gitlab.kitware.com/cmake/cmake/issues/18258
# https://cmake.org/pipermail/cmake/2011-August/046014.html

