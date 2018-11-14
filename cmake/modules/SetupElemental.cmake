# First, this checks for Hydrogen. If it fails to find Hydrogen, it
# searches for Elemental. If it finds Elemental, it sets
# HYDROGEN_LIBRARIES to point to Elemental. If neither is found, an
# error is thrown.
#
# FIXME: This file exists to hide the implementation detail of
# Hydrogen vs. Elemental. Once we decide to move fully over to
# Hydrogen, this file is no longer necessary as it's just one
# find_package() line.

set(_MIN_H_VERSION 1.0.0)
find_package(Hydrogen ${_MIN_H_VERSION} NO_MODULE
  HINTS ${Hydrogen_DIR} ${HYDROGEN_DIR} $ENV{Hydrogen_DIR} $ENV{HYDROGEN_DIR}
  PATH_SUFFIXES lib/cmake/hydrogen
  NO_DEFAULT_PATH)
if (NOT Hydrogen_FOUND)
  find_package(Hydrogen ${_MIN_H_VERSION} NO_MODULE REQUIRED)
endif ()

set(LBANN_HAS_HYDROGEN TRUE)
