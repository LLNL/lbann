# First, this checks for Hydrogen. If it fails to find Hydrogen, it
# searches for Elemental. If it finds Elemental, it sets
# HYDROGEN_LIBRARIES to point to Elemental. If neither is found, an
# error is thrown.
#
# FIXME: This file exists to hide the implementation detail of
# Hydrogen vs. Elemental. Once we decide to move fully over to
# Hydrogen, this file is no longer necessary as it's just one
# find_package() line.

find_package(Hydrogen NO_MODULE
  PATH_SUFFIXES lib/cmake/hydrogen)

if (Hydrogen_FOUND)
  message(STATUS "Found Hydrogen: ${Hydrogen_DIR}")
else ()
  find_package(Elemental NO_MODULE
    PATH_SUFFIXES lib/cmake/elemental)

  if (Elemental_FOUND)
    set(HYDROGEN_LIBRARIES "${Elemental_LIBRARIES}")
    message(STATUS "Found Elemental: {Elemental_DIR}")

    if (TARGET El)
      set_property(TARGET El PROPERTY
        INTERFACE_INCLUDE_DIRECTORIES ${Elemental_INCLUDE_DIRS})
    endif ()
  else ()
    message(FATAL_ERROR "Neither Hydrogen nor Elemental was found! "
      "Try setting Hydrogen_DIR or Elemental_DIR and try again!")
  endif (Elemental_FOUND)
endif (Hydrogen_FOUND)
