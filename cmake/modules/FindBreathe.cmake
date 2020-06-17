# Exports the following variables:
#
#   BREATHE_EXECUTABLE
#   BREATHE_FOUND
#

find_program(BREATHE_EXECUTABLE breathe-apidoc
  HINTS ${BREATHE_DIR} $ENV{BREATHE_DIR}
  ${SPHINX_DIR} $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
  DOC "The breathe documentation tool."
  NO_DEFAULT_PATH)
find_program(BREATHE_EXECUTABLE breathe-apidoc)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(BREATHE
  DEFAULT_MSG BREATHE_EXECUTABLE)
