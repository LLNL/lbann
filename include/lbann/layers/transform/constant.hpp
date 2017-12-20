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

#ifndef LBANN_LAYER_CONSTANT_HPP_INCLUDED
#define LBANN_LAYER_CONSTANT_HPP_INCLUDED

#include "lbann/layers/transform/transform.hpp"

namespace lbann {

/** Layer with constant output. */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class constant_layer : public transform_layer {

 public:
  /** Constructor. */
  constant_layer(lbann_comm *comm,
                 DataType value,
                 const std::vector<int>& neuron_dims,
                 cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm), m_value(value) {

    // Initialize matrices
    initialize_distributed_matrices();

    // Record neuron dimensions
    this->m_neuron_dims = neuron_dims;
    this->m_num_neuron_dims = neuron_dims.size();
    this->m_num_neurons = std::accumulate(neuron_dims.begin(),
                                          neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Constant layer has no parents
    m_max_num_parent_layers = 0;

  }

  /** Copy constructor. */
  constant_layer(const constant_layer& other) = default;
  /** Copy assignment operator. */
  constant_layer& operator=(const constant_layer& other) = default;
  /** Destructor. */
  ~constant_layer() override = default;
  /** Copy function. */
  constant_layer* copy() const override { return new constant_layer(*this); }

  /** Returns description. */
  std::string get_description() const override {
    std::stringstream s;
     s << "constant_layer  value: " << m_value
       << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

  /** Get layer type. */
  std::string get_type() const override { return "constant"; }

  virtual inline void initialize_distributed_matrices() {
    transform_layer::initialize_distributed_matrices<T_layout>();
  }
  data_layout get_data_layout() const override { return T_layout; }

 protected:

  void setup_dims() override {
    const auto neuron_dims = this->m_neuron_dims;
    transform_layer::setup_dims();
    this->m_neuron_dims = neuron_dims;
    this->m_num_neuron_dims = neuron_dims.size();
    this->m_num_neurons = std::accumulate(neuron_dims.begin(),
                                          neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());
  }

  void setup_data() override {
    transform_layer::setup_data();
    if (m_value != DataType(0)) {
      El::Fill(*m_activations, m_value);
    }
  }

  void fp_compute() override {}
  void bp_compute() override {}

 private:

  /** Constant value. */
  DataType m_value;

};

}  // namespace lbann

#endif  // LBANN_LAYER_CONSTANT_HPP_INCLUDED
