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

#ifndef LBANN_LAYERS_LEARNING_CHANNELWISE_FULLY_CONNECTED_HPP_INCLUDED
#define LBANN_LAYERS_LEARNING_CHANNELWISE_FULLY_CONNECTED_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/**
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class channelwise_fully_connected_layer
  : public data_type_layer<TensorDataType> {

  static_assert(Layout == data_layout::DATA_PARALLEL,
                "channelwise_fully_connected layer "
                "only supports data parallel layout");

public:

  channelwise_fully_connected_layer(
    lbann_comm* comm,
    std::vector<size_t> output_channel_dims);

  channelwise_fully_connected_layer(
    const channelwise_fully_connected_layer& other) = default;
  channelwise_fully_connected_layer& operator=(
    const channelwise_fully_connected_layer& other) = default;
  ~channelwise_fully_connected_layer() = default;

  channelwise_fully_connected_layer* copy() const override;
  std::string get_type() const override;
  data_layout get_data_layout() const override;
  El::Device get_device_allocation() const override;

protected:

  void setup_dims() override;
  void setup_data() override;

  void fp_compute() override;
  void bp_compute() override;

private:

  std::vector<size_t> m_output_channel_dims;

};

// Builder function
LBANN_DEFINE_LAYER_BUILDER(channelwise_fully_connected);

#ifndef LBANN_CHANNELWISE_FULLY_CONNECTED_LAYER_INSTANTIATE

#define PROTO_DEVICE(T, Device) \
  extern template class channelwise_fully_connected_layer<T, data_layout::DATA_PARALLEL, Device>

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#undef LBANN_INSTANTIATE_CPU_HALF
#undef LBANN_INSTANTIATE_GPU_HALF

#endif // LBANN_CHANNELWISE_FULLY_CONNECTED_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_LEARNING_CHANNELWISE_FULLY_CONNECTED_HPP_INCLUDED
