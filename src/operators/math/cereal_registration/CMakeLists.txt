################################################################################
## Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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
# Add the source files for this directory
set_full_path(THIS_DIR_SOURCES
  abs.cpp
  acos.cpp
  acosh.cpp
  add.cpp
  add_constant.cpp
  asin.cpp
  asinh.cpp
  atan.cpp
  atanh.cpp
  ceil.cpp
  clamp.cpp
  constant_subtract.cpp
  cos.cpp
  cosh.cpp
  divide.cpp
  equal.cpp
  equal_constant.cpp
  erf.cpp
  erfinv.cpp
  exp.cpp
  expm1.cpp
  floor.cpp
  greater.cpp
  greater_equal.cpp
  less.cpp
  less_equal.cpp
  greater_constant.cpp
  greater_equal_constant.cpp
  less_constant.cpp
  less_equal_constant.cpp
  log.cpp
  log1p.cpp
  logical_and.cpp
  logical_not.cpp
  logical_or.cpp
  logical_xor.cpp
  max.cpp
  min.cpp
  max_constant.cpp
  min_constant.cpp
  mod.cpp
  multiply.cpp
  negative.cpp
  not_equal.cpp
  not_equal_constant.cpp
  pow.cpp
  reciprocal.cpp
  round.cpp
  rsqrt.cpp
  safe_divide.cpp
  safe_reciprocal.cpp
  scale.cpp
  select.cpp
  sign.cpp
  sin.cpp
  sinh.cpp
  sqrt.cpp
  square.cpp
  squared_difference.cpp
  subtract.cpp
  subtract_constant.cpp
  tan.cpp
  tanh.cpp
  )

# Propagate the files up the tree
set(SOURCES "${SOURCES}" "${THIS_DIR_SOURCES}" PARENT_SCOPE)
