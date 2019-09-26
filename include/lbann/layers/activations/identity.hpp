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

#ifndef LBANN_LAYERS_ACTIVATIONS_IDENTITY_HPP_INCLUDED
#define LBANN_LAYERS_ACTIVATIONS_IDENTITY_HPP_INCLUDED

#include "lbann/layers/data_type_layer.hpp"

namespace lbann {

/** @brief Output a tensor view.
 *
 *  Forward and backward prop simply involve setting up tensor views,
 *  and hence are very cheap.
 */
template <typename TensorDataType, data_layout Layout, El::Device Device>
class identity_layer : public data_type_layer<TensorDataType> {
public:
  identity_layer(lbann_comm *comm) : data_type_layer<TensorDataType>(comm) {}
  identity_layer* copy() const override { return new identity_layer(*this); }
  std::string get_type() const override { return "identity"; }
  data_layout get_data_layout() const override { return Layout; }
  El::Device get_device_allocation() const override { return Device; }
protected:
  void setup_dims() override {
    data_type_layer<TensorDataType>::setup_dims();
    this->set_output_dims(this->get_input_dims());
  }
  void fp_setup_outputs(El::Int mini_batch_size) override {
    El::LockedView(this->get_activations(), this->get_prev_activations());
  }
  void bp_setup_gradient_wrt_inputs(El::Int mini_batch_size) override {
    El::LockedView(this->get_error_signals(), this->get_prev_error_signals());
  }
  void fp_compute() override {}
  void bp_compute() override {}
};

#ifndef LBANN_IDENTITY_LAYER_INSTANTIATE
extern template class identity_layer<
  data_layout::DATA_PARALLEL, El::Device::CPU>;
extern template class identity_layer<
  data_layout::MODEL_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
extern template class identity_layer<
  data_layout::DATA_PARALLEL, El::Device::GPU>;
extern template class identity_layer<
  data_layout::MODEL_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU
#endif // LBANN_IDENTITY_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYERS_ACTIVATIONS_IDENTITY_HPP_INCLUDED
