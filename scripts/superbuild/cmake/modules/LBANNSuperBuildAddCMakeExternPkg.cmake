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
function(lbann_sb_create_extern_pkg_cmake_args OUTPUT_VAR)
  set(_fn_opts)
  set(_fn_one_val PKG_NAME)
  set(_fn_multi_val VARIABLES)
  cmake_parse_arguments(lbann_sb_args
    "${_fn_opts}"
    "${_fn_one_val}"
    "${_fn_multi_val}" ${ARGN})

  set(lbann_sb_args_strip_regex "^LBANN_SB_FWD_${lbann_sb_args_PKG_NAME}_")

  set(lbann_sb_args_arg_list)
  foreach (var ${lbann_sb_args_VARIABLES})
    string(REGEX REPLACE "${lbann_sb_args_strip_regex}" ""
      lbann_sb_args_arg_name "${var}")
    list(APPEND lbann_sb_args_arg_list
      "-D${lbann_sb_args_arg_name}=${${var}}")
  endforeach ()

  # Return
  set(${OUTPUT_VAR} "${lbann_sb_args_arg_list}" PARENT_SCOPE)
endfunction ()

function(lbann_sb_create_pkg_lang_cmake_args OUTPUT_VAR)
  set(_fn_opts)
  set(_fn_one_val PKG_NAME)
  set(_fn_multi_val LANGUAGES)
  cmake_parse_arguments(lbann_sb_args
    "${_fn_opts}"
    "${_fn_one_val}"
    "${_fn_multi_val}" ${ARGN})

  set(NAME "${lbann_sb_args_PKG_NAME}")
  set(lbann_sb_args_arg_list)
  foreach (lang ${lbann_sb_args_LANGUAGES})
    if ("${lang}" STREQUAL "NONE")
      continue()
    endif ()
    if (LBANN_SB_${NAME}_${lang}_COMPILER)
      list(APPEND lbann_sb_args_arg_list
        "-DCMAKE_${lang}_COMPILER:STRING=${LBANN_SB_${NAME}_${lang}_COMPILER}")
    endif ()
    if (LBANN_SB_${NAME}_${lang}_STANDARD)
      list(APPEND lbann_sb_args_arg_list
        "-DCMAKE_${lang}_STANDARD:STRING=${LBANN_SB_${NAME}_${lang}_STANDARD}")
    endif ()
    if (LBANN_SB_${NAME}_${lang}_FLAGS)
      list(APPEND lbann_sb_args_arg_list
        "-DCMAKE_${lang}_FLAGS:STRING=${LBANN_SB_${NAME}_${lang}_FLAGS}")
    endif ()
    if (LBANN_SB_${NAME}_${lang}_HOST_COMPILER)
      list(APPEND lbann_sb_args_arg_list
        "-DCMAKE_${lang}_HOST_COMPILER:STRING=${LBANN_SB_${NAME}_${lang}_HOST_COMPILER}")
    endif ()
    if ("${lang}" STREQUAL "CUDA")
      list(APPEND lbann_sb_args_arg_list
        "-DCMAKE_${lang}_ARCHITECTURES:STRING=${LBANN_SB_${NAME}_${lang}_ARCHITECTURES}")
    endif ()
    if ("${lang}" STREQUAL "HIP")
      list(APPEND lbann_sb_args_arg_list
        "-DCMAKE_${lang}_ARCHITECTURES:STRING=${LBANN_SB_${NAME}_${lang}_ARCHITECTURES}"
        "-DAMDGPU_TARGETS:STRING=${LBANN_SB_${NAME}_${lang}_ARCHITECTURES}"
        "-DGPU_TARGETS:STRING=${LBANN_SB_${NAME}_${lang}_ARCHITECTURES}")
    endif ()

    if (LBANN_SB_${PKG_NAME}_BUILD_TYPE)
      string(TOUPPER "${LBANN_SB_${PKG_NAME}_BUILD_TYPE}" build_type)
      if (LBANN_SB_${NAME}_${lang}_FLAGS_${build_type})
        string(CONCAT _argstring
          "-DCMAKE_${lang}_FLAGS_${build_type}:STRING="
          "${LBANN_SB_${NAME}_${lang}_FLAGS_${build_type}}")
        list(APPEND lbann_sb_args_arg_list "${_argstring}")
      endif ()
    endif ()
  endforeach ()

  # Not language-specific, but this seems like the best place for now.
  if (LBANN_SB_${NAME}_SHARED_LINKER_FLAGS)
    string(CONCAT _argstring
      "-DCMAKE_SHARED_LINKER_FLAGS:STRING="
      "${LBANN_SB_${NAME}_SHARED_LINKER_FLAGS}")
    list(APPEND lbann_sb_args_arg_list "${_argstring}")
  endif ()
  if (LBANN_SB_${NAME}_STATIC_LINKER_FLAGS)
    string(CONCAT _argstring
      "-DCMAKE_STATIC_LINKER_FLAGS:STRING="
      "${LBANN_SB_${NAME}_STATIC_LINKER_FLAGS}")
    list(APPEND lbann_sb_args_arg_list "${_argstring}")
  endif ()
  if (LBANN_SB_${NAME}_EXE_LINKER_FLAGS)
    string(CONCAT _argstring
      "-DCMAKE_EXE_LINKER_FLAGS:STRING="
      "${LBANN_SB_${NAME}_EXE_LINKER_FLAGS}")
    list(APPEND lbann_sb_args_arg_list "${_argstring}")
  endif ()
  if (LBANN_SB_${PKG_NAME}_BUILD_TYPE)
    string(TOUPPER "${LBANN_SB_${PKG_NAME}_BUILD_TYPE}" build_type)
    if (LBANN_SB_${NAME}_SHARED_LINKER_FLAGS_${build_type})
      string(CONCAT _argstring
        "-DCMAKE_SHARED_LINKER_FLAGS_${build_type}:STRING="
        "${LBANN_SB_${NAME}_SHARED_LINKER_FLAGS_${build_type}}")
      list(APPEND lbann_sb_args_arg_list "${_argstring}")
    endif ()
    if (LBANN_SB_${NAME}_STATIC_LINKER_FLAGS_${build_type})
      string(CONCAT _argstring
        "-DCMAKE_STATIC_LINKER_FLAGS_${build_type}:STRING="
        "${LBANN_SB_${NAME}_STATIC_LINKER_FLAGS_${build_type}}")
      list(APPEND lbann_sb_args_arg_list "${_argstring}")
    endif ()
    if (LBANN_SB_${NAME}_EXE_LINKER_FLAGS_${build_type})
      string(CONCAT _argstring
        "-DCMAKE_EXE_LINKER_FLAGS_${build_type}:STRING="
        "${LBANN_SB_${NAME}_EXE_LINKER_FLAGS_${build_type}}")
      list(APPEND lbann_sb_args_arg_list "${_argstring}")
    endif ()
  endif ()

  # RPATHs
  if (LBANN_SB_${NAME}_BUILD_RPATH)
    list(APPEND lbann_sb_args_arg_list
      "-DCMAKE_BUILD_RPATH:STRING=${LBANN_SB_${NAME}_BUILD_RPATH}")
  endif ()
  if (LBANN_SB_${NAME}_INSTALL_RPATH)
    list(APPEND lbann_sb_args_arg_list
      "-DCMAKE_INSTALL_RPATH:STRING=${LBANN_SB_${NAME}_INSTALL_RPATH}")
  endif ()

  # Return
  set(${OUTPUT_VAR} "${lbann_sb_args_arg_list}" PARENT_SCOPE)
endfunction ()

# Keywords:
#   NAME -- the name of the package
#
#   LANGUAGES -- the languages for the package
#
#   DEPENDS_ON -- Any dependencies of the package. These are packages
#                 known to the superbuild and MUST be initialized
#                 prior to invoking this macro. NOTE: this is strictly
#                 an ordering mechanism required for correct build
#                 system generation! If detection is not automatic,
#                 the user is responsible for ensuring this package
#                 knows to find the dependencies at its build time
#                 (e.g., by setting CMAKE_PREFIX_PATH or similar).
#
#   EXTRA_CMAKE_ARGS -- strings to be forwarded as arguments to
#                       CMake for this package.
#
#   SOURCE_SUBDIR -- Path in the source directory to the toplevel
#                    CMakeLists.txt file.
#
# This macro also accepts the following keywords, which are forwarded
# directly to lbann_sb_init_extern_pkg:
#
#   GITHUB_URL -- the github URL for the package
#   GIT_TAG -- the git tag to checkout for this package
include(ExternalProject)
macro(lbann_sb_add_cmake_extern_pkg)
  set(_macro_opts)
  set(_macro_one_val
    NAME
    GITHUB_URL
    GIT_TAG
    SOURCE_SUBDIR)
  set(_macro_multi_val
    LANGUAGES
    EXTRA_CMAKE_ARGS
    DEPENDS_ON
    OPTIONAL_LANGUAGES)
  cmake_parse_arguments(lbann_sb_add
    "${_macro_opts}"
    "${_macro_one_val}"
    "${_macro_multi_val}" ${ARGN})

  # Perform the common initialization tasks. A non-CMake package (or
  # something with weird stuff going on such that they need to
  # customize the ExternalProject_Add call) would just call this
  # directly. The subset of arguments that it cares about are noted in
  # that macro's documentation.
  lbann_sb_init_extern_pkg(
    NAME ${lbann_sb_add_NAME}
    LANGUAGES ${lbann_sb_add_LANGUAGES}
    OPTIONAL_LANGUAGES ${lbann_sb_add_OPTIONAL_LANGUAGES}
    GITHUB_URL ${lbann_sb_add_GITHUB_URL}
    GIT_TAG ${lbann_sb_add_GIT_TAG})

  # Handle the DEPENDS_ON tag. This assumes that any package name
  # passed here has already been initialized.
  set(LBANN_SB_DEPENDS_TAG)
  set(LBANN_SB_${PKG_NAME}_DEPENDS)
  set(LBANN_SB_${PKG_NAME}_DEPENDS_PATHS)
  foreach (pkg ${lbann_sb_add_DEPENDS_ON})
    message(STATUS "${PKG_NAME}: Checking for ${pkg}")
    if (TARGET ${pkg})
      list(APPEND LBANN_SB_${PKG_NAME}_DEPENDS ${pkg})
      list(APPEND
        LBANN_SB_${PKG_NAME}_DEPENDS_PATHS
        "${LBANN_SB_${pkg}_PREFIX}")
      set(LBANN_SB_FWD_${PKG_NAME}_${pkg}_ROOT "${LBANN_SB_${pkg}_PREFIX}")
    endif ()
  endforeach ()
  if (LBANN_SB_${PKG_NAME}_DEPENDS)
    set(LBANN_SB_DEPENDS_TAG "DEPENDS")
    string(REPLACE ";" "|"
      LBANN_SB_FWD_${PKG_NAME}_CMAKE_PREFIX_PATH
      "${LBANN_SB_${PKG_NAME}_DEPENDS_PATHS}")
    message(STATUS "${PKG_NAME} depends on: ${LBANN_SB_${PKG_NAME}_DEPENDS}")
  endif ()

  # Get the variables to forward.
  get_property(PKG_VARIABLES DIRECTORY PROPERTY VARIABLES)
  set(lbann_sb_add_var_incl_regex "^LBANN_SB_FWD_${PKG_NAME}_.*")
  list(FILTER PKG_VARIABLES INCLUDE REGEX "${lbann_sb_add_var_incl_regex}")
  # Unlike previous versions of the code, we don't need the EXCLUDE
  # REGEX clause because we now prefix any and all SuperBuild
  # variables with "LBANN_SB". Of those, we only care about those
  # followed by "_FWD_${PKG_NAME}", which we collect via the INCLUDE
  # REGEX.

  # This function will take the list of variables and copy them into a
  # string of the form "-D${REAL_VAR_NAME}=${${VAR_NAME}}", where
  # "${REAL_VAR_NAME}" is the name of the variable with
  # "LBANN_SB_FWD_${PKG_NAME}_" stripped off.
  lbann_sb_create_extern_pkg_cmake_args(
    LBANN_SB_${PKG_NAME}_CMAKE_ARGS
    PKG_NAME ${PKG_NAME}
    VARIABLES ${PKG_VARIABLES})

  # This uses the output variables of "lbann_sb_init_extern_pkg" to
  # forward basic language compiler and flag information.
  lbann_sb_create_pkg_lang_cmake_args(
    LBANN_SB_${PKG_NAME}_CMAKE_LANG_ARGS
    PKG_NAME ${PKG_NAME}
    LANGUAGES ${lbann_sb_add_LANGUAGES} ${lbann_sb_add_OPTIONAL_LANGUAGES})

  # Handle the SOURCE_SUBDIR tag.
  set(LBANN_SB_SOURCE_SUBDIR_TAG)
  set(LBANN_SB_${PKG_NAME}_SOURCE_SUBDIR)
  if (lbann_sb_add_SOURCE_SUBDIR)
    set(LBANN_SB_SOURCE_SUBDIR_TAG "SOURCE_SUBDIR")
    set(LBANN_SB_${PKG_NAME}_SOURCE_SUBDIR "${lbann_sb_add_SOURCE_SUBDIR}")
  endif ()

  set(LBANN_SB_GIT_SUBMODULES_TAG)
  set(LBANN_SB_GIT_SUBMODULES_VAL)
  if ("${PKG_NAME}" STREQUAL "LBANN")
    set(LBANN_SB_GIT_SUBMODULES_TAG "GIT_SUBMODULES")
    set(LBANN_SB_GIT_SUBMODULES_VAL "")
    message("\nLBANN!\n")
  endif ()

  # Finally add the project.
  ExternalProject_Add(${PKG_NAME}
    # Standard stuff; users shouldn't care.
    PREFIX ${CMAKE_CURRENT_BINARY_DIR}
    TMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/tmp
    STAMP_DIR ${CMAKE_CURRENT_BINARY_DIR}/stamp
    BINARY_DIR ${CMAKE_CURRENT_BINARY_DIR}/build

    # User-configured paths.
    SOURCE_DIR ${LBANN_SB_${PKG_NAME}_SOURCE_DIR}
    INSTALL_DIR ${LBANN_SB_${PKG_NAME}_PREFIX}

    ${LBANN_SB_SOURCE_SUBDIR_TAG} ${LBANN_SB_${PKG_NAME}_SOURCE_SUBDIR}

    # Hack to handle building from an existing source directory.
    ${LBANN_SB_GIT_REPOSITORY_TAG} ${LBANN_SB_GIT_REPOSITORY}
    ${LBANN_SB_GIT_TAG_TAG} ${LBANN_SB_GIT_TAG}

    # Setup any dependencies
    ${LBANN_SB_DEPENDS_TAG} ${LBANN_SB_${PKG_NAME}_DEPENDS}

    # FIXME (trb 03/07/2023): This needs to be conditionally set!
    #GIT_SUBMODULES ""
    #${LBANN_SB_GIT_SUBMODULES_TAG} ${LBANN_SB_GIT_SUBMODULES_VAL}
    GIT_SHALLOW 1

    # Log everything.
    LOG_DOWNLOAD 1
    LOG_UPDATE 1
    LOG_CONFIGURE 1
    LOG_BUILD 1
    LOG_INSTALL 1
    LOG_TEST 1

    USES_TERMINAL_BUILD 1
    LIST_SEPARATOR |

    CMAKE_GENERATOR ${LBANN_SB_${PKG_NAME}_CMAKE_GENERATOR}
    CMAKE_ARGS
    # Compilers and flags
    ${LBANN_SB_${PKG_NAME}_CMAKE_LANG_ARGS}

    # Standard CMakery
    -D BUILD_SHARED_LIBS=${LBANN_SB_${PKG_NAME}_BUILD_SHARED_LIBS}
    -D CMAKE_INSTALL_PREFIX=${LBANN_SB_${PKG_NAME}_PREFIX}
    -D CMAKE_BUILD_TYPE=${LBANN_SB_${PKG_NAME}_BUILD_TYPE}
    -D CMAKE_POSITION_INDEPENDENT_CODE=${LBANN_SB_${PKG_NAME}_PIC}
    -D CMAKE_INTERPROCEDURAL_OPTIMIZATION=${LBANN_SB_${PKG_NAME}_IPO}

    -D CMAKE_INSTALL_RPATH_USE_LINK_PATH=${LBANN_SB_${PKG_NAME}_INSTALL_RPATH_USE_LINK_PATH}
    -D CMAKE_SKIP_RPATH=${LBANN_SB_${PKG_NAME}_SKIP_RPATH}
    -D CMAKE_BUILD_RPATH_USE_ORIGIN=${LBANN_SB_${PKG_NAME}_BUILD_RPATH_USE_ORIGIN}
    -D CMAKE_BUILD_WITH_INSTALL_RPATH=${LBANN_SB_${PKG_NAME}_BUILD_WITH_INSTALL_RPATH}
    -D CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH=${LBANN_SB_${PKG_NAME}_INSTALL_REMOVE_ENVIRONMENT_RPATH}
    -D CMAKE_SKIP_BUILD_RPATH=${LBANN_SB_${PKG_NAME}_SKIP_BUILD_RPATH}
    -D CMAKE_SKIP_INSTALL_RPATH=${LBANN_SB_${PKG_NAME}_SKIP_INSTALL_RPATH}

    # Extra arguments for CMake
    ${LBANN_SB_${PKG_NAME}_CMAKE_ARGS}
    ${lbann_sb_add_EXTRA_CMAKE_ARGS}
  )

endmacro ()
