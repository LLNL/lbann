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

#ifndef LBANN_LAYER_SUM_HPP_INCLUDED
#define LBANN_LAYER_SUM_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Sum layer.
 *  This layer performs a weighted sum of input tensors, possibly with
 *  a different scaling factor for each input. If the scaling factors
 *  are not provided, they are all set to one so that this layer
 *  performs a simple sum.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class sum_layer : public transform_layer {
 private:

  /** Scaling term applied to each input tensor.
   *  If these are not provided, the scaling factors are set to one.
   */
  std::vector<DataType> m_scaling_factors;

 public:
  sum_layer(lbann_comm *comm,
            std::vector<DataType> scaling_factors = std::vector<DataType>(),
            cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_scaling_factors(scaling_factors) {

    // Sum layer has no limit on parents
    m_expected_num_parent_layers = -1;

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  sum_layer* copy() const override { return new sum_layer(*this); }
  std::string get_type() const override { return "sum"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
     s << " sum; parents: ";
     for (size_t i=0; i<this->m_parent_layers.size(); i++) {
       s << this->m_parent_layers[i]->get_name() << " " << this->m_parent_layers[i]->get_type() << " ";
     }
     s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

 protected:

  void setup_dims() override {
    transform_layer::setup_dims();
    for (const auto& parent : this->m_parent_layers) {
      const auto& parent_dims = parent->fp_output_dims(this);
      if (m_neuron_dims != parent_dims) {
        std::stringstream err;
        err << __FILE__ << " " << __LINE__ << " :: "
            << "layer " << get_name() << " expects inputs with "
            << "dimensions ";
        for (size_t i = 0; i < m_neuron_dims.size(); ++i) {
          err << (i > 0 ? "x" : "") << m_neuron_dims[i];
        }
        err << ", but layer " << parent->get_name() << " outputs with "
            << "dimensions ";
        for (size_t i = 0; i < parent_dims.size(); ++i) {
          err << (i > 0 ? "x" : "") << parent_dims[i];
        }
        throw lbann_exception(err.str());
      }
    }
  }

  void setup_data() override {
    transform_layer::setup_data();
    if (m_scaling_factors.empty()) {
      m_scaling_factors.assign(get_num_parents(), DataType(1));
    }
    if ((int) m_scaling_factors.size() != get_num_parents()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << get_name() << " has an invalid number of "
          << "scaling factors "
          << "(found " << m_scaling_factors.size() << ", "
          << "but there are " << get_num_parents() << " parent layers)";
      throw lbann_exception(err.str());
    }
  }

  void fp_compute() override {
    auto& output = get_activations();
    if (get_num_parents() > 0) {
      El::Copy(get_prev_activations(0), output);
    } else {
      El::Zero(output);
    }
    for (int i = 1; i < get_num_parents(); ++i) {
      El::Axpy(DataType(1), get_prev_activations(i), output);
    }
  }

  void bp_compute() override {
    const auto& gradient_wrt_output = get_prev_error_signals();
    for (auto* gradient_wrt_input : this->m_error_signals) {
      El::LockedView(*gradient_wrt_input, gradient_wrt_output);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_SUM_HPP_INCLUDED
