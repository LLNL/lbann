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
# This should duplicate CMP0102 "NEW" behavior.
function(lbann_sb_mark_as_advanced)
  foreach (var ${ARGN})
    if (DEFINED CACHE{${var}})
      mark_as_advanced(${var})
    endif ()
  endforeach ()
endfunction ()

# This macro handles a lot of the boiler-plate associated with getting
# a package configured.
#
# usage: lbann_sb_init_extern_pkg(PKG_NAME
#                                 LANGUAGES [C,CXX,HIP,CUDA,etc]
#                                 OPTIONAL_LANGUAGES [HIP,CUDA,etc])
#
# PKG_NAME: The name of the package as you'd like it to appear in the
#           interface. E.g., "Protobuf" will setup variables like
#           "LBANN_SB_Protobuf_C_COMPILER".
#
# LANGUAGES: The programming languages that should be enabled. Each
#            language listed will be the argument to a call to
#            enable_language(). Thus, the arguments here must be valid
#            CMake languages. For each language, we setup the
#            following variables:
#
#               LBANN_SB_${PKG_NAME}_${LANGUAGE}_COMPILER
#                              (default: CMAKE_${LANG}_COMPILER)
#               LBANN_SB_${PKG_NAME}_${LANGUAGE}_STANDARD
#                              (default: CMAKE_${LANG}_STANDARD)
#               LBANN_SB_${PKG_NAME}_${LANGUAGE}_FLAGS
#                              (default: CMAKE_${LANG}_FLAGS)
#
#            Because CUDA is an epic pain, if CUDA is one of the
#            required languages, we also consider
#            LBANN_SB_${PKG_NAME}_CUDA_HOST_COMPILER. If this variable
#            isn't explicitly set by the user but
#            CMAKE_CUDA_HOST_COMPILER is set, we will set it to that
#            value. Otherwise, we leave it unset.
#
# OPTIONAL_LANGUAGES:
#                     The note about CUDA applies here as well.
#
# In addition to the language configuration described above, this
# handles the default build type, whether to build shared libraries,
# whether to use PIC, whether to use IPO, and installation prefix. The
# variables are:
#
#     LBANN_SB_${PKG_NAME}_BUILD_TYPE (default: CMAKE_BUILD_TYPE)
#     LBANN_SB_${PKG_NAME}_BUILD_SHARED_LIBS (default: BUILD_SHARED_LIBS)
#     LBANN_SB_${PKG_NAME}_PIC (default: CMAKE_POSITION_INDEPENDENT_CODE)
#     LBANN_SB_${PKG_NAME}_IPO (default: CMAKE_INTERPROCEDURAL_OPTIMIZATION)
#     LBANN_SB_${PKG_NAME}_PREFIX (default: CMAKE_INSTALL_PREFIX)
#
include(CheckLanguage)
macro(lbann_sb_init_extern_pkg)
  set(_macro_opts)
  set(_macro_one_val NAME GITHUB_URL GIT_TAG)
  set(_macro_multi_val LANGUAGES OPTIONAL_LANGUAGES)
  cmake_parse_arguments(sb_init
    "${_macro_opts}"
    "${_macro_one_val}"
    "${_macro_multi_val}" "${ARGN}")

  # These are just convenience names. HOWEVER, they are left in the
  # environment at the end of the macro. Thus, they are valid in
  # downstream code (at least at the calling scope).
  set(PKG_NAME ${sb_init_NAME})
  set(PKG_LANGS ${sb_init_LANGUAGES} ${sb_init_OPTIONAL_LANGUAGES})

  set(LBANN_SB_${PKG_NAME}_BUILD_TYPE
    ${CMAKE_BUILD_TYPE} CACHE
    STRING "The CMake build type for this build.")
  string(TOUPPER "${LBANN_SB_${PKG_NAME}_BUILD_TYPE}" build_type)

  # Setup the language-specific stuff (compiler/flags).
  foreach (lang ${PKG_LANGS})
    if ("${lang}" STREQUAL "NONE")
      continue()
    endif ()

    # I'm not sure if this is worthwhile or not. It might give a user
    # early feedback, but it doesn't take a custom compiler into
    # account.
    #
    # TODO: Write some custom code to check that the given compiler
    # exists. This shouldn't be TOO hard, especially if we assume
    # we're NOT cross-compiling.
    check_language(${lang})

    # NOTE: These must be type STRING! If the COMPILER has type
    # FILEPATH and is just an executable name (e.g., literally the
    # string "g++"), CMake will force the string to an absolute path
    # by prepending its working directory.
    set(LBANN_SB_${PKG_NAME}_${lang}_COMPILER
      ${CMAKE_${lang}_COMPILER} CACHE
      STRING "The ${lang} compiler to use for ${PKG_NAME}.")

    if (${lang} STREQUAL "CUDA")
      set(LBANN_SB_${PKG_NAME}_${lang}_HOST_COMPILER
        ${CMAKE_${lang}_HOST_COMPILER} CACHE
        STRING "The ${lang} HOST compiler to use for ${PKG_NAME}.")
      set(LBANN_SB_${PKG_NAME}_${lang}_ARCHITECTURES
        ${CMAKE_${lang}_ARCHITECTURES} CACHE
        STRING "The ${lang} architectures for ${PKG_NAME}.")
    endif ()

    if (${lang} STREQUAL "HIP")
      set(LBANN_SB_${PKG_NAME}_${lang}_ARCHITECTURES
        ${CMAKE_${lang}_ARCHITECTURES} CACHE
        STRING "The ${lang} architectures for ${PKG_NAME}.")
    endif ()

    set(LBANN_SB_${PKG_NAME}_${lang}_STANDARD
      ${CMAKE_${lang}_STANDARD} CACHE
      STRING "The ${lang} standard to use for ${PKG_NAME}.")

    set(LBANN_SB_${PKG_NAME}_${lang}_FLAGS
      ${CMAKE_${lang}_FLAGS} CACHE
      STRING "The ${lang} compiler flags to use for ${PKG_NAME}.")

    set(LBANN_SB_${PKG_NAME}_${lang}_FLAGS_${build_type}
      ${CMAKE_${lang}_FLAGS_${build_type}} CACHE
      STRING
      "The ${lang} compiler flags to use for ${PKG_NAME} in ${build_type} mode.")

    lbann_sb_mark_as_advanced(
      LBANN_SB_${PKG_NAME}_${lang}_COMPILER
      LBANN_SB_${PKG_NAME}_${lang}_STANDARD
      LBANN_SB_${PKG_NAME}_${lang}_FLAGS
      LBANN_SB_${PKG_NAME}_${lang}_FLAGS_${build_type}
      LBANN_SB_${PKG_NAME}_${lang}_HOST_COMPILER
    )
  endforeach()

  set(LBANN_SB_${PKG_NAME}_BUILD_RPATH
    "${CMAKE_BUILD_RPATH}"
    CACHE STRING
    "The build RPATHs to add to ${PKG_NAME}.")
  set(LBANN_SB_${PKG_NAME}_INSTALL_RPATH
    "${CMAKE_INSTALL_RPATH}"
    CACHE STRING
    "The install RPATHs to add to ${PKG_NAME}.")

  set(LBANN_SB_${PKG_NAME}_SHARED_LINKER_FLAGS
    "${CMAKE_SHARED_LINKER_FLAGS}"
    CACHE STRING
    "The shared linker flags for ${PKG_NAME}.")
  set(LBANN_SB_${PKG_NAME}_STATIC_LINKER_FLAGS
    "${CMAKE_STATIC_LINKER_FLAGS}"
    CACHE STRING
    "The static linker flags for ${PKG_NAME}.")
  set(LBANN_SB_${PKG_NAME}_EXE_LINKER_FLAGS
    "${CMAKE_EXE_LINKER_FLAGS}"
    CACHE STRING
    "The exe linker flags for ${PKG_NAME}.")

  if (build_type)
    set(LBANN_SB_${PKG_NAME}_SHARED_LINKER_FLAGS_${build_type}
      "${CMAKE_SHARED_LINKER_FLAGS_${build_type}}"
      CACHE STRING
      "The shared linker flags for ${PKG_NAME} in ${build_type} mode.")
    set(LBANN_SB_${PKG_NAME}_STATIC_LINKER_FLAGS_${build_type}
      "${CMAKE_STATIC_LINKER_FLAGS_${build_type}}"
      CACHE STRING
      "The static linker flags for ${PKG_NAME} in ${build_type} mode.")
    set(LBANN_SB_${PKG_NAME}_EXE_LINKER_FLAGS_${build_type}
      "${CMAKE_EXE_LINKER_FLAGS_${build_type}}"
      CACHE STRING
      "The exe linker flags for ${PKG_NAME} in ${build_type} mode.")
  endif ()


  if (LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY STREQUAL "PKG")
    set(LBANN_SB_${PKG_NAME}_PREFIX
      ${CMAKE_INSTALL_PREFIX}/${PKG_NAME} CACHE
      PATH "The CMake installation prefix for this build.")
  elseif (LBANN_SB_DEFAULT_INSTALL_PATH_STRATEGY STREQUAL "PKG_LC")
    string(TOLOWER "${PKG_NAME}" pkg_name_lower)
    set(LBANN_SB_${PKG_NAME}_PREFIX
      ${CMAKE_INSTALL_PREFIX}/${pkg_name_lower} CACHE
      PATH "The CMake installation prefix for this build.")
    set(pkg_name_lower)
  else ()
    set(LBANN_SB_${PKG_NAME}_PREFIX
      ${CMAKE_INSTALL_PREFIX} CACHE
      PATH "The CMake installation prefix for this build.")
  endif ()

  # TODO: Untangle the potential conflict between
  # "BUILD_SHARED_LIBS=ON" and "CMAKE_POSITION_INDEPENDENT_CODE={OFF or
  # unset}"
  option(LBANN_SB_${PKG_NAME}_BUILD_SHARED_LIBS
    "Build shared libraries for ${PKG_NAME}?" ${BUILD_SHARED_LIBS})
  option(LBANN_SB_${PKG_NAME}_PIC
    "Force PIC for ${PKG_NAME}?" ${CMAKE_POSITION_INDEPENDENT_CODE})
  option(LBANN_SB_${PKG_NAME}_IPO
    "Use interprocedural optimization for ${PKG_NAME}?"
    ${CMAKE_INTERPROCEDURAL_OPTIMIZATION})

  # RPATH stuff
  option(LBANN_SB_${PKG_NAME}_INSTALL_RPATH_USE_LINK_PATH
    "Add linker paths to install RPATH for ${PKG_NAME}"
    ${CMAKE_INSTALL_RPATH_USE_LINK_PATH})
  option(LBANN_SB_${PKG_NAME}_SKIP_RPATH
    "If true, skip adding rpath info to ${PKG_NAME}"
    ${CMAKE_SKIP_RPATH})
  option(LBANN_SB_${PKG_NAME}_BUILD_RPATH_USE_ORIGIN
    "Use ORIGIN in build RPATH for ${PKG_NAME}"
    ${CMAKE_BUILD_RPATH_USE_ORIGIN})
  option(LBANN_SB_${PKG_NAME}_BUILD_WITH_INSTALL_RPATH
    "Use install rpath as build rpath for ${PKG_NAME}"
    ${CMAKE_BUILD_WITH_INSTALL_RPATH})
  option(LBANN_SB_${PKG_NAME}_INSTALL_REMOVE_ENVIRONMENT_RPATH
    "Remove toolchain-specific paths from install RPATH for ${PKG_NAME}"
    ${CMAKE_INSTALL_REMOVE_ENVIRONMENT_RPATH})
  option(LBANN_SB_${PKG_NAME}_SKIP_BUILD_RPATH
    "If true, skip adding build rpath info to ${PKG_NAME}"
    ${CMAKE_SKIP_BUILD_RPATH})
  option(LBANN_SB_${PKG_NAME}_SKIP_INSTALL_RPATH
    "If true, skip adding install rpath info to ${PKG_NAME}"
    ${CMAKE_SKIP_INSTALL_RPATH})

  if (sb_init_GITHUB_URL)
    option(LBANN_SB_${PKG_NAME}_CLONE_VIA_SSH
      "Use the SSH protocol to clone ${PKG_NAME}?"
      ${LBANN_SB_CLONE_VIA_SSH})

    set(LBANN_SB_${PKG_NAME}_HTTPS_URL
      "https://github.com/${sb_init_GITHUB_URL}" CACHE
      STRING "The HTTPS URL CMake installation prefix for ${PKG_NAME}.")
    set(LBANN_SB_${PKG_NAME}_SSH_URL
      "git@github.com:${sb_init_GITHUB_URL}" CACHE
      STRING "The SSH URL CMake installation prefix for ${PKG_NAME}.")

    if (LBANN_SB_${PKG_NAME}_CLONE_VIA_SSH)
      set(LBANN_SB_${PKG_NAME}_URL
        ${LBANN_SB_${PKG_NAME}_SSH_URL} CACHE
        STRING "The URL to use to clone ${PKG_NAME}.")
    else ()
      set(LBANN_SB_${PKG_NAME}_URL
        ${LBANN_SB_${PKG_NAME}_HTTPS_URL} CACHE
        STRING "The URL to use to clone ${PKG_NAME}.")
    endif ()
  endif ()

  set(LBANN_SB_${PKG_NAME}_TAG
    ${sb_init_GIT_TAG} CACHE
    STRING "The git tag to checkout for ${PKG_NAME}.")

  set(LBANN_SB_${PKG_NAME}_CMAKE_GENERATOR
    ${CMAKE_GENERATOR} CACHE
    STRING "The CMake generator to use for ${PKG_NAME}.")

  set(LBANN_SB_${PKG_NAME}_SOURCE_DIR
    "${CMAKE_CURRENT_BINARY_DIR}/src" CACHE
    PATH "The source directory for ${PKG_NAME}.")

  if ("${LBANN_SB_${PKG_NAME}_SOURCE_DIR}"
      STREQUAL
      "${CMAKE_CURRENT_BINARY_DIR}/src")
    set(LBANN_SB_GIT_REPOSITORY_TAG "GIT_REPOSITORY")
    set(LBANN_SB_GIT_TAG_TAG "GIT_TAG")
    set(LBANN_SB_GIT_REPOSITORY "${LBANN_SB_${PKG_NAME}_URL}")
    set(LBANN_SB_GIT_TAG "${LBANN_SB_${PKG_NAME}_TAG}")
  else ()
    # If the user has provided some other dir, we need to disable the
    # download/update stages. This is done by clearing the repository
    # URL and tag.
    set(LBANN_SB_GIT_REPOSITORY_TAG)
    set(LBANN_SB_GIT_TAG_TAG)
    set(LBANN_SB_GIT_REPOSITORY)
    set(LBANN_SB_GIT_TAG)
    message(STATUS
      "Using ${PKG_NAME} source in: ${LBANN_SB_${PKG_NAME}_SOURCE_DIR}")
  endif ()

  lbann_sb_mark_as_advanced(
    LBANN_SB_${PKG_NAME}_BUILD_SHARED_LIBS
    LBANN_SB_${PKG_NAME}_BUILD_TYPE
    LBANN_SB_${PKG_NAME}_CLONE_VIA_SSH
    LBANN_SB_${PKG_NAME}_CMAKE_GENERATOR
    LBANN_SB_${PKG_NAME}_HTTPS_URL
    LBANN_SB_${PKG_NAME}_IPO
    LBANN_SB_${PKG_NAME}_PIC
    LBANN_SB_${PKG_NAME}_SOURCE_DIR
    LBANN_SB_${PKG_NAME}_SSH_URL
  )

endmacro()
