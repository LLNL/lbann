# Helper function that uses "dumb" logic to try to figure out if a
# library file is a shared or static library. This won't work on
# Windows; it will just return "unknown" for everything.
#
# FIXME: move to "LBANNCMakeUtilities.cmake" or the like
function(lbann_determine_library_type lib_name output_var)

  # Test if ends in ".a"
  string(REGEX MATCH "\.a$" _static_match ${lib_name})
  if (_static_match)
    set(${output_var} STATIC PARENT_SCOPE)
    return()
  endif (_static_match)

  # Test if ends in ".so(.version.id.whatever)"
  string(REGEX MATCH "\.so($|\..*$)" _shared_match ${lib_name})
  if (_shared_match)
    set(${output_var} SHARED PARENT_SCOPE)
    return()
  endif (_shared_match)

  # Test if ends in ".dylib(.version.id.whatever)"
  string(REGEX MATCH "\.dylib($|\..*$)" _mac_shared_match ${lib_name})
  if (_mac_shared_match)
    set(${output_var} SHARED PARENT_SCOPE)
    return()
  endif (_mac_shared_match)

  set(${output_var} "UNKNOWN" PARENT_SCOPE)
endfunction(lbann_determine_library_type lib_name output)

# A handy macro to add the current source directory to a local
# filename. To be used for creating a list of sources.
macro(set_full_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    list(APPEND __tmp_names "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()
