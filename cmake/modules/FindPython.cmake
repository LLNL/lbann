# Detect Python interpreter and Python C API
#
# This makes several improvements over the FindPython.cmake module
# that comes included with CMake:
#   - The stock version ignores user-provded hints if it thinks it has
#     found a newer Python version. This is a problem if a virtual
#     environment doesn't override the 'python<major>.<minor>'
#     executable since that executable will take
#     precedence. User-provided hints now take precedence.
#   - Python C API objects are deduced by querying the Python
#     interpreter rather than directly looking for files. This is
#     helpful if a virtual environment doesn't create all the
#     necessary copies or symlinks.
#
# Hint variables
#
#   Python_EXECUTABLE
#   Python_ROOT_DIR
#
# Exports the following variables
#
#   Python_FOUND
#   Python_EXECUTABLE
#   Python_VERSION
#   Python_VERSION_MAJOR
#   Python_VERSION_MINOR
#   Python_VERSION_PATCH
#   Python_INCLUDE_DIRS
#   Python_LIBRARIES
#

set(Python_FOUND FALSE)

# Find executable
if (NOT Python_EXECUTABLE)
  if (Python_ROOT_DIR)
    set(_HINT "${Python_ROOT_DIR}/bin")
  endif (Python_ROOT_DIR)
  find_program(Python_EXECUTABLE
    NAMES python3 python
    HINTS "${_HINT}")
endif (NOT Python_EXECUTABLE)
if (NOT Python_EXECUTABLE)
  message(WARNING "Could not find Python executable")
  return()
endif (NOT Python_EXECUTABLE)

# Get version
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
  "import sys; sys.stdout.write('.'.join([str(x) for x in sys.version_info[:3]]))"
  OUTPUT_VARIABLE Python_VERSION)
string(REGEX MATCHALL "[0-9]+" _VERSION_PARSED "${Python_VERSION}")
list(GET _VERSION_PARSED 0 Python_VERSION_MAJOR)
list(GET _VERSION_PARSED 1 Python_VERSION_MINOR)
list(GET _VERSION_PARSED 2 Python_VERSION_PATCH)

# Find Python C API
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
  "import sys; from distutils.sysconfig import get_python_inc; sys.stdout.write(get_python_inc())"
  OUTPUT_VARIABLE Python_INCLUDE_DIRS)
execute_process(
  COMMAND "${Python_EXECUTABLE}" "-c"
  "import sys; from distutils.sysconfig import get_config_var; sys.stdout.write(get_config_var('LIBDIR'))"
  OUTPUT_VARIABLE _LIB_DIR)
if (BUILD_SHARED_LIBS)
  set(_GLOB_EXPR "${_LIB_DIR}/libpython*${CMAKE_SHARED_LIBRARY_SUFFIX}")
ELSE (BUILD_SHARED_LIBS)
  set(_GLOB_EXPR "${_LIB_DIR}/libpython*${CMAKE_STATIC_LIBRARY_SUFFIX}")
endif (BUILD_SHARED_LIBS)
FILE(GLOB _GLOB_RESULT "${_GLOB_EXPR}")
get_filename_component(Python_LIBRARIES "${_GLOB_RESULT}" ABSOLUTE)

# Handle the find_package arguments
include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(
  Python
  REQUIRED_VARS Python_EXECUTABLE Python_INCLUDE_DIRS Python_LIBRARIES
  Python_VERSION_MAJOR Python_VERSION_MINOR Python_VERSION_PATCH
  VERSION_VAR Python_VERSION)

# Build the imported target
if (NOT TARGET Python::Python)
  add_library(Python::Python INTERFACE IMPORTED)
  set_property(TARGET Python::Python
    PROPERTY INTERFACE_INCLUDE_DIRECTORIES "${Python_INCLUDE_DIRS}")
  set_property(TARGET Python::Python
    PROPERTY INTERFACE_LINK_LIBRARIES "${Python_LIBRARIES}")
endif (NOT TARGET Python::Python)
