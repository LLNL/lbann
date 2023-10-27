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

#include "lbann/layers/transform/transform_builders.hpp"

#include "lbann/layers/transform/batchwise_reduce_sum.hpp"
#include "lbann/layers/transform/concatenate.hpp"
#include "lbann/layers/transform/crop.hpp"
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/layers/transform/gather.hpp"
#include "lbann/layers/transform/gaussian.hpp"
#include "lbann/layers/transform/in_top_k.hpp"
#include "lbann/layers/transform/permute.hpp"
#include "lbann/layers/transform/reshape.hpp"
#include "lbann/layers/transform/scatter.hpp"
#include "lbann/layers/transform/slice.hpp"
#include "lbann/layers/transform/sort.hpp"
#include "lbann/layers/transform/tessellate.hpp"
#include "lbann/layers/transform/uniform.hpp"
#include "lbann/layers/transform/unpooling.hpp"

#include "lbann/execution_algorithms/execution_context.hpp"

#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer> lbann::build_batchwise_reduce_sum_layer_from_pbuf(
  lbann_comm* /*comm*/,
  lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL)
    return std::make_unique<
      batchwise_reduce_sum_layer<T, data_layout::DATA_PARALLEL, D>>();
  else {
    LBANN_ERROR("Attempted to instantiate layer \"batchwise_reduce_sum\""
                "with Layout=",
                to_string(L),
                ".\n"
                "This layer is only supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_concatenate_layer_from_pbuf(lbann_comm* comm,
                                         lbann_data::Layer const& proto_layer)
{
  return std::make_unique<concatenate_layer<T, L, D>>(
    comm,
    proto_layer.concatenation().axis());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_crop_layer_from_pbuf(lbann_comm* comm,
                                  lbann_data::Layer const& proto_layer)
{
  LBANN_ASSERT_MSG_HAS_FIELD(proto_layer, crop);

  if constexpr (L == lbann::data_layout::DATA_PARALLEL) {
    const auto& params = proto_layer.crop();
    return std::make_unique<
      crop_layer<T, lbann::data_layout::DATA_PARALLEL, D>>(
      comm,
      protobuf::to_vector<int>(params.dims()));
  }
  else {
    (void)comm;
    LBANN_ERROR("Attempted to instantiate \"crop\" layer with "
                "Layout=",
                to_string(L),
                ".\nThis layer is only "
                "supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer> lbann::build_discrete_random_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.discrete_random();
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU) {
    return std::make_unique<
      discrete_random_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>>(
      comm,
      protobuf::to_vector<DataType>(params.values()),
      protobuf::to_vector<int>(params.dims()));
  }
  else {
    (void)comm;
    LBANN_ERROR("discrete random layer is only supported on CPU");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_gather_layer_from_pbuf(lbann_comm* /*comm*/,
                                    lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL) {
    const auto& params = proto_layer.gather();
    return std::make_unique<gather_layer<T, data_layout::DATA_PARALLEL, D>>(
      params.has_axis() ? params.axis().value() : -1);
  }
  else {
    LBANN_ERROR("Attempted to instantiate \"gather\" layer with Layout=",
                to_string(L),
                ".\n"
                "This layer is only supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_gaussian_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.gaussian();
  const auto& dims = protobuf::to_vector<int>(params.neuron_dims());
  double mean = params.mean();
  double stdev = params.stdev();
  if (mean == 0.0 && stdev == 0.0) {
    mean = 0.0;
    stdev = 1.0;
  }
  return std::make_unique<gaussian_layer<T, L, D>>(comm,
                                                   dims,
                                                   El::To<T>(mean),
                                                   El::To<T>(stdev),
                                                   params.training_only());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_in_top_k_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const& proto_layer)
{
  return std::make_unique<in_top_k_layer<T, L, D>>(comm,
                                                   proto_layer.in_top_k().k());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_scatter_layer_from_pbuf(lbann_comm* /*comm*/,
                                     lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL) {
    const auto& params = proto_layer.scatter();
    return std::make_unique<scatter_layer<T, data_layout::DATA_PARALLEL, D>>(
      protobuf::to_vector<int>(params.dims()),
      params.has_axis() ? params.axis().value() : -1);
  }
  else {
    LBANN_ERROR("Attempted to instantiate layer \"scatter\""
                "with Layout=",
                to_string(L),
                ".\n"
                "This layer is only "
                "supported with DATA_PARALLEL data layout.");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_permute_layer_from_pbuf(lbann_comm* /*comm*/,
                                     lbann_data::Layer const& proto_layer)
{
#ifdef LBANN_HAS_TENSOR_PERMUTE
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::GPU) {
    return std::make_unique<PermuteLayer<T>>(
      protobuf::to_vector<int>(proto_layer.permute().axes()));
  }
  else {
    LBANN_ERROR("PermuteLayers are only supported on GPUs and with "
                "DATA_PARALLEL layout.");
    return nullptr;
  }
#else
  LBANN_ERROR(
    "At this time, PermuteLayers are only supported on CUDA platforms "
    "with cuTENSOR or cuTT support, or on ROCm platforms with hipTT support. "
    "Please open an issue "
    "(https://github.com/LLNL/lbann/issues/new) to request additional support. "
    "In the meantime, please use the \"Permute\" module in the LBANN "
    "Python Front-End to generate a correct implementation of this operation "
    "suitable to your build.");
#endif
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_reshape_layer_from_pbuf(lbann_comm* comm,
                                     lbann_data::Layer const& proto) {
    const auto& params = proto.reshape();
    return std::make_unique<reshape_layer<T, L, D>>(
      comm,
      protobuf::to_vector<int>(params.dims()));
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_slice_layer_from_pbuf(lbann_comm* comm,
                                   lbann_data::Layer const& proto_layer)
{
  auto layer = std::make_unique<slice_layer<T, L, D>>(comm);

  const auto& params = proto_layer.slice();
  if (params.get_slice_points_from_reader() != "") {
    const slice_points_mode var =
      slice_points_mode_from_string(params.get_slice_points_from_reader());
    layer->setup_slice_points(params.axis(), true, var);
  }
  else {
    if (params.slice_points_size() < 2) {
      LBANN_ERROR("Failed to get slice points via 'slice_points'.");
      return nullptr;
    }
    layer->setup_slice_points(
      params.axis(),
      protobuf::to_vector<size_t>(params.slice_points()));
  }
  return layer;
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_sort_layer_from_pbuf(lbann_comm* comm,
                                  lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL) {
    return std::make_unique<sort_layer<T, data_layout::DATA_PARALLEL, D>>(
      comm,
      proto_layer.sort().descending());
  }
  else {
    (void)comm;
    LBANN_ERROR("sort layer is only supported with "
                "a data-parallel layout");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_tessellate_layer_from_pbuf(lbann_comm* comm,
                                        lbann_data::Layer const& proto_layer)
{
  return std::make_unique<tessellate_layer<T, L, D>>(
    comm,
    protobuf::to_vector<int>(proto_layer.tessellate().dims()));
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_uniform_layer_from_pbuf(lbann_comm* comm,
                                     lbann_data::Layer const& proto_layer)
{
  const auto& params = proto_layer.uniform();
  const auto& dims = protobuf::to_vector<int>(params.neuron_dims());
  double min = params.min();
  double max = params.max();
  if (min == 0.0 && max == 0.0) {
    min = 0.0;
    max = 1.0;
  }
  return std::make_unique<uniform_layer<T, L, D>>(comm,
                                                  dims,
                                                  El::To<T>(min),
                                                  El::To<T>(max),
                                                  params.training_only());
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_unpooling_layer_from_pbuf(lbann_comm* comm,
                                       lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU) {
    return std::make_unique<
      unpooling_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>>(comm);
  }
  else {
    (void)comm;
    LBANN_ERROR("unpooling layer is only supported with "
                "a data-parallel layout and on CPU");
    return nullptr;
  }
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(batchwise_reduce_sum, T, Device);                    \
  LBANN_LAYER_BUILDER_ETI(concatenate, T, Device);                             \
  LBANN_LAYER_BUILDER_ETI(crop, T, Device);                                    \
  LBANN_LAYER_BUILDER_ETI(discrete_random, T, Device);                         \
  LBANN_LAYER_BUILDER_ETI(gather, T, Device);                                  \
  LBANN_LAYER_BUILDER_ETI(gaussian, T, Device);                                \
  LBANN_LAYER_BUILDER_ETI(in_top_k, T, Device);                                \
  LBANN_LAYER_BUILDER_ETI(permute, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(reshape, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(scatter, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(slice, T, Device);                                   \
  LBANN_LAYER_BUILDER_ETI(sort, T, Device);                                    \
  LBANN_LAYER_BUILDER_ETI(tessellate, T, Device);                              \
  LBANN_LAYER_BUILDER_ETI(uniform, T, Device);                                 \
  LBANN_LAYER_BUILDER_ETI(unpooling, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
