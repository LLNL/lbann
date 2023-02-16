////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#include "lbann/layers/loss/loss_layer_builders.hpp"

#include "lbann/layers/loss/categorical_accuracy.hpp"
#include "lbann/layers/loss/cross_entropy.hpp"
#include "lbann/layers/loss/l1_norm.hpp"
#include "lbann/layers/loss/l2_norm2.hpp"
#include "lbann/layers/loss/mean_absolute_error.hpp"
#include "lbann/layers/loss/mean_squared_error.hpp"
#include "lbann/layers/loss/top_k_categorical_accuracy.hpp"

#include "lbann/proto/layers.pb.h"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_categorical_accuracy_layer_from_pbuf(lbann_comm* comm,
                                                  lbann_data::Layer const&)
{
  return std::make_unique<categorical_accuracy_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_cross_entropy_layer_from_pbuf(lbann_comm* comm,
                                           lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.cross_entropy();
  return std::make_unique<cross_entropy_layer<T, L, D>>(comm,
                                                        params.use_labels());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_l1_norm_layer_from_pbuf(lbann_comm* comm, lbann_data::Layer const&)
{
  return std::make_unique<l1_norm_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_l2_norm2_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const&)
{
  return std::make_unique<l2_norm2_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_mean_absolute_error_layer_from_pbuf(lbann_comm* comm,
                                                 lbann_data::Layer const&)
{
  return std::make_unique<mean_absolute_error_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_mean_squared_error_layer_from_pbuf(lbann_comm* comm,
                                                lbann_data::Layer const&)
{
  return std::make_unique<mean_squared_error_layer<T, L, D>>(comm);
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_top_k_categorical_accuracy_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.top_k_categorical_accuracy();
  return std::make_unique<top_k_categorical_accuracy_layer<T, L, D>>(
    comm,
    params.k());
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(categorical_accuracy, T, Device);                    \
  LBANN_LAYER_BUILDER_ETI(cross_entropy, T, Device);                           \
  LBANN_LAYER_BUILDER_ETI(l1_norm, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(l2_norm2, T, Device);                                \
  LBANN_LAYER_BUILDER_ETI(mean_absolute_error, T, Device);                     \
  LBANN_LAYER_BUILDER_ETI(mean_squared_error, T, Device);                      \
  LBANN_LAYER_BUILDER_ETI(top_k_categorical_accuracy, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
