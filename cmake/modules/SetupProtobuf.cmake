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

set(PROTOBUF_MIN_VERSION "3.0.0")

# On cross-compilation machines, we want to use the module because we
# will use the host protoc and the target libprotobuf. In this case,
# users should set Protobuf_PROTOC_EXECUTABLE=/path/to/host/bin/protoc
# and set PROTOBUF_DIR=/path/to/target/protobuf/prefix.
option(${PROJECT_NAME}_USE_PROTOBUF_MODULE
  "Use the FindProtobuf module instead of Protobuf's config file." OFF)

if (${PROJECT_NAME}_USE_PROTOBUF_MODULE)
  if (PROTOBUF_DIR)
    set(__remove_protobuf_from_paths TRUE)
    list(APPEND CMAKE_LIBRARY_PATH ${PROTOBUF_DIR}/lib)
    list(APPEND CMAKE_INCLUDE_PATH ${PROTOBUF_DIR}/include)
    list(APPEND CMAKE_PREFIX_PATH ${PROTOBUF_DIR})
  endif ()

  # At this point, throw an error if Protobuf is not found.
  find_package(Protobuf "${PROTOBUF_MIN_VERSION}" MODULE)

  if (__remove_protobuf_from_paths)
    list(REMOVE_ITEM CMAKE_LIBRARY_PATH ${PROTOBUF_DIR}/lib)
    list(REMOVE_ITEM CMAKE_INCLUDE_PATH ${PROTOBUF_DIR}/include)
    list(REMOVE_ITEM CMAKE_PREFIX_PATH ${PROTOBUF_DIR})
    set(__remove_protobuf_from_paths)
  endif ()

else ()
  option(protobuf_MODULE_COMPATIBLE
    "Be compatible with FindProtobuf.cmake" ON)
  option(protobuf_VERBOSE
    "Enable verbose protobuf output" OFF)

  find_package(Protobuf "${PROTOBUF_MIN_VERSION}" CONFIG QUIET
    NAMES protobuf PROTOBUF
    HINTS
    "${Protobuf_DIR}" "${PROTOBUF_DIR}"
    "$ENV{Protobuf_DIR}" "$ENV{PROTOBUF_DIR}"
    PATH_SUFFIXES lib64/cmake/protobuf lib/cmake/protobuf
    NO_DEFAULT_PATH)
  if(NOT Protobuf_FOUND)
    find_package(Protobuf "${PROTOBUF_MIN_VERSION}" CONFIG QUIET REQUIRED)
  endif ()
  message(STATUS "Found Protobuf: ${Protobuf_DIR}")
endif ()

if (NOT Protobuf_FOUND)
  message(FATAL_ERROR
    "Protobuf still not found. This should never throw.")
endif ()

# Setup the imported target for old versions of CMake
if (NOT TARGET protobuf::libprotobuf)
  add_library(protobuf::libprotobuf INTERFACE IMPORTED)
  set_property(TARGET protobuf::libprotobuf PROPERTY
    INTERFACE_LINK_LIBRARIES "${PROTOBUF_LIBRARIES}")

  set_property(TARGET protobuf::libprotobuf PROPERTY
    INTERFACE_INCLUDE_DIRECTORIES "${PROTOBUF_INCLUDE_DIRS}")
endif ()

# This can just be "TRUE" since protobuf is REQUIRED above.
set(LBANN_HAS_PROTOBUF TRUE)
