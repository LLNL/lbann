################################################################################
## Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
## Produced at the Lawrence Livermore National Laboratory.
## Written by the LBANN Research Team (B. Van Essen, et al.) listed in
## the CONTRIBUTORS file. <lbann-dev@llnl.gov>
##
## LLNL-CODE-697807.
## All rights reserved.
##
## This file is part of LBANN: Livermore Big Artificial Neural Network
## Toolkit. For details, see http://software.llnl.gov/LBANN or
## https://github.com/LLNL/LBANN.
##
## Licensed under the Apache License, Version 2.0 (the "Licensee"); you
## may not use this file except in compliance with the License.  You may
## obtain a copy of the License at:
##
## http://www.apache.org/licenses/LICENSE-2.0
##
## Unless required by applicable law or agreed to in writing, software
## distributed under the License is distributed on an "AS IS" BASIS,
## WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
## implied. See the License for the specific language governing
## permissions and limitations under the license.
################################################################################

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

if (LBANN_WITH_ADDRESS_SANITIZER)
  lbann_check_and_append_flag(CMAKE_CXX_FLAGS
    -fsanitize=address -fno-omit-frame-pointer -fsanitize-recover=address)
endif ()

################################################################
# Muck with flags
################################################################

# Initialize C++ flags
set(CMAKE_POSITION_INDEPENDENT_CODE ON) # -fPIC
lbann_check_and_append_flag(CMAKE_CXX_FLAGS
  -g -Wall -Wextra -Wno-unused-parameter -Wnon-virtual-dtor -Wshadow
  -Wno-deprecated-declarations)

# Disable all optimization in debug for better viewing under debuggers
# (cmake already adds -g)
lbann_check_and_append_flag(CMAKE_CXX_FLAGS_DEBUG -O0)

if (LBANN_WARNINGS_AS_ERRORS)
  set(CMAKE_REQUIRED_LIBRARIES "-Werror")
  check_cxx_compiler_flag("${flag}" LBANN_FLAG_Werror_OK)
  unset(CMAKE_REQUIRED_LIBRARIES)
endif ()

# Some behavior is dependent on the compiler version.
if (NOT CMAKE_CXX_COMPILER_VERSION)
  execute_process(
    COMMAND ${CMAKE_CXX_COMPILER} -dumpversion
    OUTPUT_VARIABLE CXX_VERSION)
else ()
  set(CXX_VERSION "${CMAKE_CXX_COMPILER_VERSION}")
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
set(CMAKE_SKIP_BUILD_RPATH OFF)

# Use same RPATH for build and install
set(CMAKE_BUILD_WITH_INSTALL_RPATH OFF)

# Add the automatically determined parts of the RPATH
# which point to directories outside the build tree to the install RPATH
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH ON)

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
