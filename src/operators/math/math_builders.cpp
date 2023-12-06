////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
// Produced at the Lawrence Livermore National Laboratory.
// Written by the LBANN Research Team (B. Van Essen, et al.) listed in
// the CONTRIBUTORS file. <lbann-dev@llnl.gov>
//
// LLNL-CODE-697807.
// All rights reserved.
//
// This file is part of LBANN: Livermore Big Artificial Neural Network
// Toolkit. For details, see http://software.llnl.gov/LBANN or
// https://github.com/LLNL/LBANN.
//
// Licensed under the Apache License, Version 2.0 (the "Licensee"); you
// may not use this file except in compliance with the License.  You may
// obtain a copy of the License at:
//
// http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or
// implied. See the License for the specific language governing
// permissions and limitations under the license.
////////////////////////////////////////////////////////////////////////////////

#include <lbann/operators/math/math_builders_impl.hpp>

#define LBANN_ABS_OP_COMPLEX_ETI(T, D)                                         \
  template std::unique_ptr<lbann::Operator<El::Complex<T>, T, D>>              \
  lbann::build_abs_operator<El::Complex<T>, D>(lbann_data::Operator const&);
LBANN_ABS_OP_COMPLEX_ETI(float, El::Device::CPU);
#ifdef LBANN_HAS_DOUBLE
LBANN_ABS_OP_COMPLEX_ETI(double, El::Device::CPU);
#endif // LBANN_HAS_DOUBLE
#ifdef LBANN_HAS_GPU
LBANN_ABS_OP_COMPLEX_ETI(float, El::Device::GPU);
#ifdef LBANN_HAS_DOUBLE
LBANN_ABS_OP_COMPLEX_ETI(double, El::Device::GPU);
#endif // LBANN_HAS_DOUBLE
#endif
#undef LBANN_ABS_OP_COMPLEX_ETI

#define PROTO_DEVICE(T, D)                                                     \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(abs, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(acos, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(acosh, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(add, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(add_constant, T, D);                  \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(asin, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(asinh, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(atan, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(atanh, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(ceil, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(clamp, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(constant_subtract, T, D);             \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(cos, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(cosh, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(divide, T, D);                        \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(equal, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(equal_constant, T, D);                \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(erf, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(erfinv, T, D);                        \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(exp, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(expm1, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(floor, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(gelu, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(greater, T, D);                       \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(greater_constant, T, D);              \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(greater_equal, T, D);                 \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(greater_equal_constant, T, D);        \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(less, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(less_constant, T, D);                 \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(less_equal, T, D);                    \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(less_equal_constant, T, D);           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(log, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(log1p, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(logical_and, T, D);                   \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(logical_not, T, D);                   \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(logical_or, T, D);                    \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(logical_xor, T, D);                   \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(max, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(max_constant, T, D);                  \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(min, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(min_constant, T, D);                  \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(mod, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(multiply, T, D);                      \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(negative, T, D);                      \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(not_equal, T, D);                     \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(not_equal_constant, T, D);            \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(pow, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(reciprocal, T, D);                    \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(round, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(rsqrt, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(safe_divide, T, D);                   \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(safe_reciprocal, T, D);               \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(select, T, D);                        \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(scale, T, D);                         \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(sign, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(sin, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(sinh, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(sqrt, T, D);                          \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(square, T, D);                        \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(squared_difference, T, D);            \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(subtract, T, D);                      \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(subtract_constant, T, D);             \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(tan, T, D);                           \
  LBANN_SINGLE_TYPE_OPERATOR_BUILDER_ETI(tanh, T, D)

#include <lbann/macros/instantiate_device.hpp>
