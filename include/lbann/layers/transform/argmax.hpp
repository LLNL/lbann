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

#ifndef LBANN_LAYER_ARGMAX_HPP_INCLUDED
#define LBANN_LAYER_ARGMAX_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** @brief argmax of a column vector 
 *  temporarily supports data parallel data layout and CPU device
 *
 *  
 */
template <data_layout Layout, El::Device Device>
class argmax_layer : public transform_layer {
private:

public:

  argmax_layer(lbann_comm *comm,
                std::vector<int> dims)
    : transform_layer(comm) {
    set_output_dims(dims);
    static_assert(Layout == data_layout::DATA_PARALLEL,
                  "argmax layer only supports data parallel layout");
    static_assert(Device == El::Device::CPU,
                  "argmax layer only supports CPU");
    this->m_expected_num_parent_layers = 1;
  }
  argmax_layer* copy() const override { return new argmax_layer(*this); }
  std::string get_type() const override { return "argmax"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

  description get_description() const override {
    auto desc = transform_layer::get_description();
    return desc;
  }

protected:

  void fp_compute() override {
    El::Zero(get_activations());
    // Local data
    //check that local height is 1
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    const auto& local_width = local_input.Width();
    const auto& local_height = local_input.Height();
    CPUMat sample_v;
    for (El::Int col = 0; col < local_width; ++ col) {
      El::LockedView(sample_v, local_input, El::ALL, El::IR(col));
      const auto max_it = std::max_element(sample_v.LockedBuffer(), 
                                           sample_v.LockedBuffer() + local_height);
      const auto max_index = std::distance(sample_v.LockedBuffer(), max_it);
      local_output.Set(0, col, max_index);
    }

  }

};

} // namespace lbann

#endif // LBANN_LAYER_ARGMAX_HPP_INCLUDED
