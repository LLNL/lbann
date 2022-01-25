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

# A handy macro to add the current source directory to a local
# filename. To be used for creating a list of sources.
macro(set_full_path VAR)
  unset(__tmp_names)
  foreach(filename ${ARGN})
    list(APPEND __tmp_names "${CMAKE_CURRENT_SOURCE_DIR}/${filename}")
  endforeach()
  set(${VAR} "${__tmp_names}")
endmacro()

# A function to get a string of spaces. Useful for formatting output.
function(lbann_get_space_string OUTPUT_VAR LENGTH)
  set(_curr_length 0)
  set(_out_str "")
  while (${_curr_length} LESS ${LENGTH})
    string(APPEND _out_str " ")
    math(EXPR _curr_length "${_curr_length} + 1")
  endwhile ()

  set(${OUTPUT_VAR} "${_out_str}" PARENT_SCOPE)
endfunction ()

# This computes the maximum length of the things given in "ARGN"
# interpreted as simple strings.
macro(lbann_get_max_str_length OUTPUT_VAR)
  set(${OUTPUT_VAR} 0)
  foreach(var ${ARGN})
    string(LENGTH "${var}" _var_length)
    if (_var_length GREATER _max_length)
      set(${OUTPUT_VAR} ${_var_length})
    endif ()
  endforeach ()
endmacro ()
