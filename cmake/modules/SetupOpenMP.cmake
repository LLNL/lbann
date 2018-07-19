# NOTE: If using AppleClang on OSX, you'll probably want to set
#
#   OpenMP_CXX_FLAGS
#
# on the command line. An example, using the latest Homebrew
# installation, might be:
#
#   -DOpenMP_CXX_FLAGS="-fopenmp -I/usr/local/include/libiomp/ -L/usr/local/lib/"

find_package(OpenMP REQUIRED CXX)

if (NOT TARGET OpenMP::OpenMP_CXX)
  add_library(OpenMP::OpenMP_CXX INTERFACE IMPORTED)

  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${OpenMP_CXX_FLAGS}>)

  # Propagate to the link flags
  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_LINK_LIBRARIES ${OpenMP_CXX_FLAGS})
  # The imported target will be defined in the same version as CMake
  # introduced the "OpenMP_<lang>_LIBRARIES" variable. Thus we don't
  # provide a contingency for them here.
else ()

  get_target_property(_OMP_FLAGS OpenMP::OpenMP_CXX INTERFACE_COMPILE_OPTIONS)

  set_property(TARGET OpenMP::OpenMP_CXX PROPERTY
    INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${_OMP_FLAGS}>)

endif ()
