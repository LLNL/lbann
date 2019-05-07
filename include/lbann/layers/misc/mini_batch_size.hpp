////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC.
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

#ifndef LBANN_LAYERS_MISC_MINI_BATCH_SIZE_HPP_INCLUDED
#define LBANN_LAYERS_MISC_MINI_BATCH_SIZE_HPP_INCLUDED

#include "lbann/layers/layer.hpp"

namespace lbann {

/** @brief Mini-batch size.
 *
 *  Output tensor is a 1D tensor with a single entry containing the
 *  model's current mini-batch size.
 */
template <data_layout Layout = data_layout::DATA_PARALLEL, El::Device Device = El::Device::CPU>
class mini_batch_size_layer : public Layer {
public:

  mini_batch_size_layer(lbann_comm* comm) : Layer(comm) {
    this->m_expected_num_parent_layers = 0;
  }

  mini_batch_size_layer* copy() const override { return new mini_batch_size_layer(*this); }
  std::string get_type() const override { return "mini-batch size"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }

protected:

  void setup_dims() override {
    Layer::setup_dims();
    set_output_dims({1});
  }

  void fp_setup_outputs(El::Int mini_batch_size) override {
    Layer::fp_setup_outputs(mini_batch_size);
    m_mini_batch_size = mini_batch_size;
  }

  void fp_compute() override {
    El::Fill(get_activations(), DataType(m_mini_batch_size));
  }

private:

  /** Mini-batch size. */
  El::Int m_mini_batch_size = 0;

};

} // namespace lbann

#endif // LBANN_LAYERS_MISC_MINI_BATCH_SIZE_HPP_INCLUDED
