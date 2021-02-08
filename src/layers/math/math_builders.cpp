////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#include <lbann/layers/math/binary.hpp>
#include <lbann/layers/math/clamp.hpp>
#include <lbann/layers/math/math_builders.hpp>
#include <lbann/layers/math/matmul.hpp>
#include <lbann/layers/math/unary.hpp>

#include <lbann/proto/proto_common.hpp> // IWYU pragma: export
#include <layers.pb.h> // IWYU pragma: export

namespace lbann
{

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_clamp_layer_from_pbuf(
  lbann_comm* /*comm*/, lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, clamp);
  using LayerType = clamp_layer<TensorDataType, Layout, Device>;
  auto const& params = proto_layer.clamp();
  return lbann::make_unique<LayerType>(
    El::To<TensorDataType>(params.min()),
    El::To<TensorDataType>(params.max()));
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
std::unique_ptr<Layer> build_matmul_layer_from_pbuf(
  lbann_comm* comm, lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, matmul);
  if constexpr (Layout == data_layout::DATA_PARALLEL) {
    using LayerType = matmul_layer<TensorDataType, Layout, Device>;
    const auto& params = proto_layer.matmul();
    return lbann::make_unique<LayerType>(
      comm,
      params.transpose_a(),
      params.transpose_b());
  }
  else {
    (void) comm;
    (void) proto_layer;
    LBANN_ERROR("matrix multiply layer is only supported with "
                "a data-parallel layout");
  }
}

LBANN_LAYER_DEFAULT_BUILDER(abs);
LBANN_LAYER_DEFAULT_BUILDER(acos);
LBANN_LAYER_DEFAULT_BUILDER(acosh);
LBANN_LAYER_DEFAULT_BUILDER(add);
LBANN_LAYER_DEFAULT_BUILDER(asin);
LBANN_LAYER_DEFAULT_BUILDER(asinh);
LBANN_LAYER_DEFAULT_BUILDER(atan);
LBANN_LAYER_DEFAULT_BUILDER(atanh);
LBANN_LAYER_DEFAULT_BUILDER(ceil);
LBANN_LAYER_DEFAULT_BUILDER(cos);
LBANN_LAYER_DEFAULT_BUILDER(cosh);
LBANN_LAYER_DEFAULT_BUILDER(divide);
LBANN_LAYER_DEFAULT_BUILDER(equal);
LBANN_LAYER_DEFAULT_BUILDER(exp);
LBANN_LAYER_DEFAULT_BUILDER(expm1);
LBANN_LAYER_DEFAULT_BUILDER(floor);
LBANN_LAYER_DEFAULT_BUILDER(greater);
LBANN_LAYER_DEFAULT_BUILDER(greater_equal);
LBANN_LAYER_DEFAULT_BUILDER(erf);
LBANN_LAYER_DEFAULT_BUILDER(erfinv);
LBANN_LAYER_DEFAULT_BUILDER(less);
LBANN_LAYER_DEFAULT_BUILDER(less_equal);
LBANN_LAYER_DEFAULT_BUILDER(log);
LBANN_LAYER_DEFAULT_BUILDER(log1p);
LBANN_LAYER_DEFAULT_BUILDER(logical_and);
LBANN_LAYER_DEFAULT_BUILDER(logical_not);
LBANN_LAYER_DEFAULT_BUILDER(logical_or);
LBANN_LAYER_DEFAULT_BUILDER(logical_xor);
LBANN_LAYER_DEFAULT_BUILDER(max);
LBANN_LAYER_DEFAULT_BUILDER(min);
LBANN_LAYER_DEFAULT_BUILDER(mod);
LBANN_LAYER_DEFAULT_BUILDER(multiply);
LBANN_LAYER_DEFAULT_BUILDER(negative);
LBANN_LAYER_DEFAULT_BUILDER(not_equal);
LBANN_LAYER_DEFAULT_BUILDER(pow);
LBANN_LAYER_DEFAULT_BUILDER(reciprocal);
LBANN_LAYER_DEFAULT_BUILDER(round);
LBANN_LAYER_DEFAULT_BUILDER(rsqrt);
LBANN_LAYER_DEFAULT_BUILDER(safe_divide);
LBANN_LAYER_DEFAULT_BUILDER(safe_reciprocal);
LBANN_LAYER_DEFAULT_BUILDER(sign);
LBANN_LAYER_DEFAULT_BUILDER(sin);
LBANN_LAYER_DEFAULT_BUILDER(sinh);
LBANN_LAYER_DEFAULT_BUILDER(sqrt);
LBANN_LAYER_DEFAULT_BUILDER(square);
LBANN_LAYER_DEFAULT_BUILDER(squared_difference);
LBANN_LAYER_DEFAULT_BUILDER(subtract);
LBANN_LAYER_DEFAULT_BUILDER(tan);
LBANN_LAYER_DEFAULT_BUILDER(tanh);

#define PROTO_DEVICE(T,D)                               \
  LBANN_LAYER_BUILDER_ETI(abs, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(acos, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(acosh, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(add, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(asin, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(asinh, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(atan, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(atanh, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(ceil, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(clamp, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(cos, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(cosh, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(divide, T, D);                \
  LBANN_LAYER_BUILDER_ETI(equal, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(exp, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(expm1, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(floor, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(greater, T, D);               \
  LBANN_LAYER_BUILDER_ETI(greater_equal, T, D);         \
  LBANN_LAYER_BUILDER_ETI(erf, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(erfinv, T, D);                \
  LBANN_LAYER_BUILDER_ETI(less, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(less_equal, T, D);            \
  LBANN_LAYER_BUILDER_ETI(log, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(log1p, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(logical_and, T, D);           \
  LBANN_LAYER_BUILDER_ETI(logical_not, T, D);           \
  LBANN_LAYER_BUILDER_ETI(logical_or, T, D);            \
  LBANN_LAYER_BUILDER_ETI(logical_xor, T, D);           \
  LBANN_LAYER_BUILDER_ETI(matmul, T, D);                \
  LBANN_LAYER_BUILDER_ETI(max, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(min, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(mod, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(multiply, T, D);              \
  LBANN_LAYER_BUILDER_ETI(negative, T, D);              \
  LBANN_LAYER_BUILDER_ETI(not_equal, T, D);             \
  LBANN_LAYER_BUILDER_ETI(pow, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(reciprocal, T, D);            \
  LBANN_LAYER_BUILDER_ETI(round, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(rsqrt, T, D);                 \
  LBANN_LAYER_BUILDER_ETI(safe_divide, T, D);           \
  LBANN_LAYER_BUILDER_ETI(safe_reciprocal, T, D);       \
  LBANN_LAYER_BUILDER_ETI(sign, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(sin, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(sinh, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(sqrt, T, D);                  \
  LBANN_LAYER_BUILDER_ETI(square, T, D);                \
  LBANN_LAYER_BUILDER_ETI(squared_difference, T, D);    \
  LBANN_LAYER_BUILDER_ETI(subtract, T, D);              \
  LBANN_LAYER_BUILDER_ETI(tan, T, D);                   \
  LBANN_LAYER_BUILDER_ETI(tanh, T, D)
#include <lbann/macros/instantiate_device.hpp>
} // namespace lbann
