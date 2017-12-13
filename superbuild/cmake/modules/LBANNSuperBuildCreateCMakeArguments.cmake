# This function takes a list of variables and writes a CMake argument string of the format:
#   -DARGNAME:ARGTYPE=ARGVALUE ...
#
function (create_cmake_arguments)
  set(_OPTIONS REMOVE_PKG_NAME)
  set(_ONE_VALUE_PARAMS PACKAGE_NAME OUTPUT_VARIABLE)
  set(_MULTI_VALUE_PARAMS VARIABLES)

  cmake_parse_arguments(_CREATEARGS
    "${_OPTIONS}" "${_ONE_VALUE_PARAMS}" "${_MULTI_VALUE_PARAMS}" ${ARGN})

  # Short-circuit. IDK if is the best decision...
  if (${_CREATEARGS_OUTPUT_VARIABLE})
    return()
  endif ()

  set(_DO_NOT_PRINT_TYPES INTERNAL STATIC UNINITIALIZED)

  foreach(_variable ${_CREATEARGS_VARIABLES})

    # Cleanup the variable name
    if (_CREATEARGS_REMOVE_PKG_NAME)
      # We must be careful to only remove the first instance of the
      # package name.
      string(REGEX REPLACE "^${_CREATEARGS_PACKAGE_NAME}_\(.+\)" "\\1"
        _CMAKE_ARG_NAME ${_variable})
    else ()
      set(_CMAKE_ARG_NAME ${_variable})
    endif ()

    # Check the variable's type, if possible.
    get_property(_CMAKE_ARG_TYPE CACHE ${_variable} PROPERTY TYPE)

    # Check if it's a type we want to propagate
    if (_CMAKE_ARG_TYPE)
      list(FIND _DO_NOT_PRINT_TYPES ${_CMAKE_ARG_TYPE} _DO_NOT_PRINT_VAR)
    else ()
      set(_DO_NOT_PRINT_VAR 0)
    endif ()

    # Add the variable to the CMake line
    if (${_DO_NOT_PRINT_VAR} STREQUAL "-1")
      list(APPEND _output_string
        "-D${_CMAKE_ARG_NAME}:${_CMAKE_ARG_TYPE}=${${_variable}}")
    else ()
      list(APPEND _output_string "-D${_CMAKE_ARG_NAME}=${${_variable}}")
    endif ()

  endforeach ()

  # Return
  set(${_CREATEARGS_OUTPUT_VARIABLE} ${_output_string} PARENT_SCOPE)

endfunction(create_cmake_arguments)
