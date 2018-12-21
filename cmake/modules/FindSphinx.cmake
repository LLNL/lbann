# Exports the following variables:
#
#   SPHINX_EXECUTABLE
#   SPHINX_FOUND
#

find_program(SPHINX_EXECUTABLE sphinx-build
  HINTS ${SPHINX_DIR} $ENV{SPHINX_DIR}
  PATH_SUFFIXES bin
  DOC "The sphinx-build documentation tool."
  NO_DEFAULT_PATH)
find_program(SPHINX_EXECUTABLE sphinx-build)

# Standard handling of the package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(SPHINX
  DEFAULT_MSG SPHINX_EXECUTABLE)
