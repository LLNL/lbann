find_package(MPI REQUIRED)

if (MPI_CXX_FOUND)
  if (NOT TARGET MPI::MPI_CXX)
    add_library(MPI::MPI_CXX INTERFACE IMPORTED)
    if (MPI_CXX_COMPILE_FLAGS)
      separate_arguments(_MPI_CXX_COMPILE_OPTIONS UNIX_COMMAND
        "${MPI_CXX_COMPILE_FLAGS}")
      set_property(TARGET MPI::MPI_CXX PROPERTY
        INTERFACE_COMPILE_OPTIONS "${_MPI_CXX_COMPILE_OPTIONS}")
    endif()

    if (MPI_CXX_LINK_FLAGS)
      separate_arguments(_MPI_CXX_LINK_LINE UNIX_COMMAND
        "${MPI_CXX_LINK_FLAGS}")
    endif()
    list(APPEND _MPI_CXX_LINK_LINE "${MPI_CXX_LIBRARIES}")

    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_LINK_LIBRARIES "${_MPI_CXX_LINK_LINE}")

    set_property(TARGET MPI::MPI_CXX PROPERTY
      INTERFACE_INCLUDE_DIRECTORIES "${MPI_CXX_INCLUDE_PATH}")

  endif (NOT TARGET MPI::MPI_CXX)
else ()
  message(FATAL_ERROR "MPI CXX compiler was not found and is required")
endif (MPI_CXX_FOUND)
