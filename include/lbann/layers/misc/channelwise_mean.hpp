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

#ifndef LBANN_LAYERS_MISC_CHANNELWISE_MEAN_HPP_INCLUDED
#define LBANN_LAYERS_MISC_CHANNELWISE_MEAN_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** @todo Replace with more general reduction layer. */
template <data_layout Layout = data_layout::DATA_PARALLEL, El::Device Device = El::Device::CPU>
class channelwise_mean_layer : public Layer {
public:

  channelwise_mean_layer(lbann_comm *comm)
    : Layer(comm) {
    static_assert(Layout == data_layout::DATA_PARALLEL,
                  "channelwise_mean_layer only supports "
                  "data-parallel data layout");
    if (comm->am_trainer_master()) {
      LBANN_WARNING("channelwise_mean_layer is experimental "
                    "and may be deprecated at any time");
    }
  }

  channelwise_mean_layer* copy() const override { return new channelwise_mean_layer(*this); }
  std::string get_type() const override { return "channel-wise mean"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    const auto& input_dims = get_input_dims();
    set_output_dims({input_dims[0]});
  }

  void fp_compute() override;
  void bp_compute() override;

};

} // namespace lbann

#endif // LBANN_LAYERS_MISC_CHANNELWISE_MEAN_HPP_INCLUDED
