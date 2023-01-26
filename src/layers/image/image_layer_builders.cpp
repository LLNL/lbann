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

#include "lbann/layers/image/image_layer_builders.hpp"

#include "lbann/layers/image/bilinear_resize.hpp"
#include "lbann/layers/image/composite_image_transformation.hpp"
#include "lbann/layers/image/rotation.hpp"
#include "lbann/layers/image/cutout.hpp"

#include <layers.pb.h>

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer> lbann::build_bilinear_resize_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const& proto_layer)
{
  if constexpr (L == data_layout::DATA_PARALLEL) {
    auto const& params = proto_layer.bilinear_resize();
    return std::make_unique<
      bilinear_resize_layer<T, data_layout::DATA_PARALLEL, D>>(comm,
                                                               params.height(),
                                                               params.width());
  }
  LBANN_ERROR(
    "bilinear resize layer is only supported with a data-parallel layout");
  return nullptr;
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_composite_image_transformation_layer_from_pbuf(
  lbann_comm* comm,
  lbann_data::Layer const&)
{
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU) {
    if constexpr (std::is_same_v<T, float>)
      return std::make_unique<
        composite_image_transformation_layer<float,
                                             data_layout::DATA_PARALLEL,
                                             El::Device::CPU>>(comm);
    else if constexpr (std::is_same_v<T, double>)
      return std::make_unique<
        composite_image_transformation_layer<double,
                                             data_layout::DATA_PARALLEL,
                                             El::Device::CPU>>(comm);
    else
      LBANN_ERROR("composite_image_transformation_layer is only supported for "
                  "\"float\" and \"double\".");
  }
  else {
    LBANN_ERROR("composite image transformation layer is only supported with "
                "a data-parallel layout and on CPU");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_rotation_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const&)
{
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU)
    if constexpr (std::is_same_v<T, float>)
      return std::make_unique<
        rotation_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(
        comm);
    else if constexpr (std::is_same_v<T, double>)
      return std::make_unique<
        rotation_layer<double, data_layout::DATA_PARALLEL, El::Device::CPU>>(
        comm);
    else
      LBANN_ERROR(
        "rotation_layer is only supported for \"float\" and \"double\".");
  else {
    LBANN_ERROR("rotation layer is only supported with a data-parallel layout "
                "and on CPU");
    return nullptr;
  }
}

template <typename T, lbann::data_layout L, El::Device D>
std::unique_ptr<lbann::Layer>
lbann::build_cutout_layer_from_pbuf(lbann_comm* comm,
                                      lbann_data::Layer const&)
{
  if constexpr (L == data_layout::DATA_PARALLEL && D == El::Device::CPU)
    if constexpr (std::is_same_v<T, float>)
      return std::make_unique<
        cutout_layer<float, data_layout::DATA_PARALLEL, El::Device::CPU>>(
        comm);
    else if constexpr (std::is_same_v<T, double>)
      return std::make_unique<
        cutout_layer<double, data_layout::DATA_PARALLEL, El::Device::CPU>>(
        comm);
    else
      LBANN_ERROR(
        "cutout_layer is only supported for \"float\" and \"double\".");
  else {
    LBANN_ERROR("cutout layer is only supported with a data-parallel layout "
                "and on CPU");
    return nullptr;
  }
}

namespace lbann {

#define PROTO_DEVICE(T, Device)                                                \
  LBANN_LAYER_BUILDER_ETI(bilinear_resize, T, Device);                         \
  LBANN_LAYER_BUILDER_ETI(composite_image_transformation, T, Device);          \
  LBANN_LAYER_BUILDER_ETI(rotation, T, Device);                                \
  LBANN_LAYER_BUILDER_ETI(cutout, T, Device)

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
