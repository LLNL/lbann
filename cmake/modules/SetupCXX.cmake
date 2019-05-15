# This sets up all of the proper compiler information, including some
# custom flags. <Tom's skeptical face>

include(CheckCXXCompilerFlag)
include(CheckIncludeFileCXX)

# MACRO LBANN_CHECK_AND_APPEND_FLAG
#
# Purpose: checks that all flags are valid and appends them to the
#   given list. Valid means that the compiler does not throw an error
#   upon encountering the flag.
#
# Arguments:
#   VAR The list of current flags
#   ARGN The flags to check
#
# Note: If flag is not valid, it is not appended.
macro(lbann_check_and_append_flag MY_VAR)
  foreach(flag ${ARGN})
    string(REPLACE "-" "_" _CLEAN_FLAG "${flag}")

    set(CMAKE_REQUIRED_LIBRARIES "${flag}")
    check_cxx_compiler_flag("${flag}" FLAG_${_CLEAN_FLAG}_OK)
    unset(CMAKE_REQUIRED_LIBRARIES)

    if (FLAG_${_CLEAN_FLAG}_OK)
      set(${MY_VAR} "${${MY_VAR}} ${flag}")
    endif ()
  endforeach()
endmacro()

# Temporary workaround to force CMake to recognize the XL
# compiler. The latest beta XL compiler is recognized as Clang.
if (CMAKE_CXX_COMPILER MATCHES "xlc" AND CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  set(CMAKE_CXX_COMPILER_ID "XL")
endif ()

################################################################
# Check Compiler versions
################################################################

execute_process(COMMAND ${CMAKE_CXX_COMPILER} -dumpversion OUTPUT_VARIABLE CXX_VERSION)

if (CMAKE_CXX_COMPILER_ID MATCHES "GNU")
   if (NOT (CXX_VERSION VERSION_GREATER 4.9 OR CXX_VERSION VERSION_EQUAL 4.9))
     message(FATAL_ERROR "LBANN & Elemental requires G++ Version >= 4.9")
   endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
   if (NOT (CXX_VERSION VERSION_GREATER 16.0 OR CXX_VERSION VERSION_EQUAL 16.0))
     message(FATAL_ERROR "LBANN & Elemental requires icpc Version >= 16.0")
   endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   if (NOT (CXX_VERSION VERSION_GREATER 3.5 OR CXX_VERSION VERSION_EQUAL 3.5))
     message(FATAL_ERROR "LBANN & Elemental requires clang Version >= 3.5")
   endif ()
   if (CMAKE_BUILD_TYPE MATCHES Debug)
     lbann_check_and_append_flag(CMAKE_CXX_FLAGS
       -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address)
   else()
     lbann_check_and_append_flag(CMAKE_CXX_FLAGS -fno-omit-frame-pointer)
   endif ()
elseif (CMAKE_CXX_COMPILER_ID MATCHES "XL")
  # Version requirement for xlc++? The latest compiler on ray is 14.1.0
  if (CMAKE_CXX_COMPILER_VERSION VERSION_LESS 14.1.0)
    message(WARNING "LBANN & Elemental may not work with xlc++ Version < 14.1.0")
  endif ()
else()
   message(FATAL_ERROR "Unsupported compiler: Unknown compiler vendor (${CMAKE_CXX_COMPILER_ID})")
endif ()

################################################################
# Muck with flags
################################################################

# Initialize C++ flags
lbann_check_and_append_flag(CMAKE_CXX_FLAGS
  -fPIC -g -Wall -Wextra -Wno-unused-parameter -Wnon-virtual-dtor -Wshadow
  -Wno-deprecated-declarations)

# Disable all optimization in debug for better viewing under debuggers
# (cmake already adds -g)
lbann_check_and_append_flag(CMAKE_CXX_FLAGS_DEBUG -O0)

if (${UPPER_PROJECT_NAME}_WARNINGS_AS_ERRORS)
  lbann_check_and_append_flag(_WERROR_FLAGS -Werror)
  separate_arguments(_WERROR_FLAGS NATIVE_COMMAND "${_WERROR_FLAGS}")
  if (NOT TARGET LBANN_CXX_FLAGS_werror)
    add_library(LBANN_CXX_FLAGS_werror INTERFACE)
    set_property(TARGET LBANN_CXX_FLAGS_werror PROPERTY
      INTERFACE_COMPILE_OPTIONS $<$<COMPILE_LANGUAGE:CXX>:${_WERROR_FLAGS}>)

    # Add the "library" to the export
    install(TARGETS LBANN_CXX_FLAGS_werror EXPORT LBANNTargets)
  endif ()
endif ()

# Some behavior is dependent on the compiler version.
if (NOT CMAKE_CXX_COMPILER_VERSION)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
    OUTPUT_VARIABLE CXX_VERSION)
else ()
  set(CXX_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
endif ()

# Special handling if we're compiling with Clang's address sanitizer
if (CMAKE_CXX_COMPILER_ID MATCHES "Clang")
   if (NOT (CXX_VERSION VERSION_GREATER 3.5 OR CXX_VERSION VERSION_EQUAL 3.5))
     message(FATAL_ERROR "LBANN & Elemental requires clang Version >= 3.5")
   endif ()
   if (CMAKE_BUILD_TYPE MATCHES Debug)
     lbann_check_and_append_flag(CMAKE_CXX_FLAGS
       -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address)
   else()
     lbann_check_and_append_flag(CMAKE_CXX_FLAGS -fno-omit-frame-pointer)
   endif ()
endif ()

# Turn off some annoying warnings
if (CMAKE_CXX_COMPILER_ID MATCHES "Intel")
  lbann_check_and_append_flag(CMAKE_CXX_FLAGS -diag-disable=2196)
endif ()

################################################################
# Initialize RPATH (always full RPATH)
# Note: see https://cmake.org/Wiki/CMake_RPATH_handling
################################################################

# Use RPATH on OS X
if (APPLE)
  set(CMAKE_MACOSX_RPATH ON)
endif ()

# Use (i.e. don't skip) RPATH for build
set(CMAKE_SKIP_BUILD_RPATH FALSE)

# Use same RPATH for build and install
set(CMAKE_BUILD_WITH_INSTALL_RPATH FALSE)

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

list(FIND CMAKE_PLATFORM_IMPLICIT_LINK_DIRECTORIES
  "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}" _IS_SYSTEM_DIR)
if (${_IS_SYSTEM_DIR} STREQUAL "-1")
  # Set the install RPATH correctly
  list(APPEND CMAKE_INSTALL_RPATH
    "${CMAKE_INSTALL_PREFIX}/${CMAKE_INSTALL_LIBDIR}")
endif ()

# Check if we can use Linux's sys/sendfile.h
check_include_file_cxx(sys/sendfile.h LBANN_SYS_SENDFILE_OK)

# Testing for std::any
include(CheckCXXSourceCompiles)
set(_ANY_TEST_CODE
  "#include <any>
int main(int, char* argv[]) { std::any x; }")
check_cxx_source_compiles("${_ANY_TEST_CODE}" LBANN_HAS_STD_ANY)

set(_MAKE_UNIQUE_TEST_CODE
  "#include <memory>
int main(int, char* argv[]) { auto x = std::make_unique<double>(); }")
check_cxx_source_compiles(
  "${_MAKE_UNIQUE_TEST_CODE}" LBANN_HAS_STD_MAKE_UNIQUE)
