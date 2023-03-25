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

################################################################################
## Copyright 2019-2020 Lawrence Livermore National Security, LLC and other
## DiHydrogen Project Developers. See the top-level LICENSE file for details.
##
## SPDX-License-Identifier: Apache-2.0
################################################################################

get_filename_component(COMPILER_BIN_DIR "${CMAKE_CXX_COMPILER}" DIRECTORY)
get_filename_component(COMPILER_PREFIX "${COMPILER_BIN_DIR}" DIRECTORY)

# Let the user override clang-format with its own variable. This would
# help if building with an older LLVM installation.
find_program(CLANG_FORMAT_PROGRAM clang-format
  HINTS ${CLANG_FORMAT_DIR} $ENV{CLANG_FORMAT_DIR}
  PATH_SUFFIXES bin
  DOC "The clang-format executable."
  NO_DEFAULT_PATH)

# Normal search inspired by the compiler choice. If the compiler
# happens to be, GCC, for example, users can also use LLVM_DIR. If all
# else fails, this falls back on default CMake searching.
find_program(CLANG_FORMAT_PROGRAM clang-format
  HINTS
  ${COMPILER_BIN_DIR}
  ${COMPILER_PREFIX}
  ${LLVM_DIR} $ENV{LLVM_DIR}
  PATH_SUFFIXES bin
  DOC "The clang-format executable."
  NO_DEFAULT_PATH)

# Default CMake searching.
find_program(CLANG_FORMAT_PROGRAM clang-format)

if (CLANG_FORMAT_PROGRAM)
  execute_process(COMMAND ${CLANG_FORMAT_PROGRAM} --version
    OUTPUT_VARIABLE CLANG_FORMAT_VERSION_STRING
    OUTPUT_STRIP_TRAILING_WHITESPACE)

  string(REGEX MATCH "[0-9]+\.[0-9]+\.[0-9]+"
    CLANG_FORMAT_VERSION
    "${CLANG_FORMAT_VERSION_STRING}")

  if (CLANG_FORMAT_VERSION VERSION_GREATER_EQUAL "10.0.0")
    set(CLANG_FORMAT_VERSION_OK TRUE)
  else ()
    set(CLANG_FORMAT_VERSION_OK FALSE)
  endif ()
endif ()

set(CLANG_FORMAT_AVAILABLE)
if (CLANG_FORMAT_PROGRAM AND CLANG_FORMAT_VERSION_OK)
  set(CLANG_FORMAT_AVAILABLE TRUE)
  add_custom_target(
    clang-format
    COMMAND ${CLANG_FORMAT_PROGRAM} -i
    $<TARGET_PROPERTY:clang-format,FORMAT_SOURCES>
    WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
    COMMENT "Applying clang-format."
    COMMAND_EXPAND_LISTS
    VERBATIM)
  define_property(TARGET PROPERTY FORMAT_SOURCES
    BRIEF_DOCS "Sources for clang-format."
    FULL_DOCS "Sources for clang-format.")

  # Add the sources from the given target to the "clang-format"
  # target.
  macro (add_clang_format IN_TARGET)

    get_target_property(TGT_TYPE ${IN_TARGET} TYPE)

    if ((TGT_TYPE MATCHES "(STATIC|SHARED|OBJECT|MODULE)_LIBRARY")
        OR (TGT_TYPE MATCHES "EXECUTABLE"))

      unset(TGT_SOURCES_FULL_PATH)
      get_target_property(TGT_SOURCES ${IN_TARGET} SOURCES)
      get_target_property(TGT_SRC_DIR ${IN_TARGET} SOURCE_DIR)

      foreach (src IN LISTS TGT_SOURCES)
        get_filename_component(SRC_NAME "${src}" NAME)
        if (src STREQUAL SRC_NAME)
          list(APPEND TGT_SOURCES_FULL_PATH "${TGT_SRC_DIR}/${src}")
        else ()
          list(APPEND TGT_SOURCES_FULL_PATH "${src}")
        endif ()
      endforeach ()

      set_property(TARGET clang-format APPEND
        PROPERTY FORMAT_SOURCES "${TGT_SOURCES_FULL_PATH}")
    elseif (TGT_TYPE MATCHES "INTERFACE_LIBRARY")
      get_target_property(TGT_SOURCES ${IN_TARGET} INTERFACE_SOURCES)

      if (TGT_SOURCES)
        # Sources might be in generator expressions! :/ We want to only
        # change the BUILD_INTERFACE objects with absolute paths.
        foreach (src IN LISTS TGT_SOURCES)
          # Skip install files
          if (src MATCHES ".*INSTALL_INTERFACE.*")
            continue()
          endif ()

          if (src MATCHES ".*BUILD_INTERFACE:(.*)>")
            set(my_src "${CMAKE_MATCH_1}")
          else ()
            set(my_src "${src}")
          endif ()
          get_filename_component(SRC_NAME "${my_src}" NAME)
          # Assume a relative path is
          if (my_src STREQUAL SRC_NAME)
            message(FATAL_ERROR "Not expecting relative path: ${my_src}")
            list(APPEND TGT_SOURCES_FULL_PATH "${TGT_SRC_DIR}/${my_src}")
          else ()
            list(APPEND TGT_SOURCES_FULL_PATH "${my_src}")
          endif ()
        endforeach ()

        set_property(TARGET clang-format APPEND
          PROPERTY FORMAT_SOURCES "${TGT_SOURCES_FULL_PATH}")
      endif (TGT_SOURCES)
    endif ()
  endmacro ()

  function (add_cf_to_tgts_in_dir IN_DIR)

    # Handle this directory
    get_property(_targets
      DIRECTORY "${IN_DIR}"
      PROPERTY BUILDSYSTEM_TARGETS)

    foreach (tgt IN LISTS _targets)
      add_clang_format(${tgt})
    endforeach ()

    # Recursive call.
    get_property(_subdirs
      DIRECTORY "${IN_DIR}"
      PROPERTY SUBDIRECTORIES)

    foreach (dir IN LISTS _subdirs)
      add_cf_to_tgts_in_dir("${dir}")
    endforeach ()
  endfunction ()

  function (add_clang_format_to_all_targets)
    add_cf_to_tgts_in_dir("${CMAKE_SOURCE_DIR}")
  endfunction ()

  message(STATUS "Found clang-format: ${CLANG_FORMAT_PROGRAM}")
  message(STATUS "Clang-format Version: ${CLANG_FORMAT_VERSION})")
  message(STATUS
    "Added target \"clang-format\" "
    "for applying clang-format to source.")
endif ()
