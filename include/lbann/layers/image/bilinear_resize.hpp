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

#ifndef LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_HPP_INCLUDED
#define LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/proto/layers.pb.h"

namespace lbann {

/** @brief Resize image with bilinear interpolation.
 *
 *  Tensors are assumed to be image data in CHW format. Gradients are
 *  not propagated during backprop.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class bilinear_resize_layer : public data_type_layer<TensorDataType>
{
  static_assert(Layout == data_layout::DATA_PARALLEL,
                "bilinear_resize_layer only supports DATA_PARALLEL");

public:
  bilinear_resize_layer(lbann_comm* comm, El::Int height, El::Int width)
    : data_type_layer<TensorDataType>(comm), m_height(height), m_width(width)
  {}

  bilinear_resize_layer* copy() const override
  {
    return new bilinear_resize_layer(*this);
  }

  /** @name Serialization */
  ///@{

  template <typename ArchiveT>
  void serialize(ArchiveT& ar);

  ///@}

  std::string get_type() const override { return "bilinear resize"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
  bool can_run_inplace() const override { return false; }
  int get_backprop_requirements() const override { return ERROR_SIGNALS; }

  void fp_compute() override;

protected:
  /** Add layer specific data to prototext */
  void write_specific_proto(lbann_data::Layer& proto) const final;

  friend class cereal::access;
  bilinear_resize_layer() : bilinear_resize_layer(nullptr, 1, 1) {}

  void setup_dims(DataReaderMetaData& dr_metadata) override;

private:
  /** Output image height.
   *  Data is assumed to be in CHW format.
   */
  El::Int m_height;
  /** Output image width.
   *  Data is assumed to be in CHW format.
   */
  El::Int m_width;
};

template <typename T, data_layout L, El::Device D>
void bilinear_resize_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_bilinear_resize();
  msg->set_height(m_height);
  msg->set_width(m_width);
}

#ifndef LBANN_BILINEAR_RESIZE_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                                                \
  extern template class bilinear_resize_layer<T,                               \
                                              data_layout::DATA_PARALLEL,      \
                                              Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_BILINEAR_RESIZE_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_IMAGE_BILINEAR_RESIZE_HPP_INCLUDED
