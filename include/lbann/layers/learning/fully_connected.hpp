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

#ifndef LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
#define LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED

#include "lbann/layers/learning/learning.hpp"
#include "lbann/layers/activations/activation.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/variance_scaling_initializers.hpp"
#include <string>
#include <sstream>

namespace lbann {

/** Fully-connected layer.
 *  This layer applies an affine transformation.
 */
template <data_layout T_layout, El::Device Dev>
class fully_connected_layer : public learning_layer {
 private:

  /** Scaling factor for bias term.
   *  If the scaling factor is zero, bias is not applied.
   */
  DataType m_bias_scaling_factor;

  /** Linearity gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the linearity weights (i.e. its matrix weights).
   */
  AbsDistMat* m_linearity_gradient;
  /** Bias weights gradient.
   *  This is this layer's contribution to the objective function
   *  gradient w.r.t. the bias weights.
   */
  AbsDistMat* m_bias_gradient;

  /** Whether the transpose of the linearity matrix is applied. */
  bool m_transpose;

 public:

  fully_connected_layer(lbann_comm *comm,
                        int output_size,  /// @todo Accept a vector
                        bool transpose = false,
                        weights* weight = nullptr,
                        bool has_bias = true)
    : learning_layer(comm),
      m_linearity_gradient(nullptr),
      m_bias_gradient(nullptr),
      m_transpose(transpose) {

    // Initialize output tensor dimensions
    set_output_dims({output_size});

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " fully_connected; num_neurons: "
     + std::to_string(get_output_size())
     + " has_bias: " + std::to_string(this->m_bias_scaling_factor)
     + " dataLayout: " + this->get_data_layout_string(get_data_layout())
     + " device alloc: " + this->get_device_allocation_string(get_device_allocation());
  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning_layer(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor),
    m_transpose(other.m_transpose) {

    // Deep matrix copies
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning_layer::operator=(other);
    m_bias_scaling_factor = other.m_bias_scaling_factor;
    m_transpose = other.m_transpose;

    // Deep matrix copies
    deallocate_matrices();
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

    return *this;
  }

  ~fully_connected_layer() override {
    deallocate_matrices();
  }

  fully_connected_layer* copy() const override {
    return new fully_connected_layer(*this);
  }

  std::string get_type() const override { return "fully connected"; }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  /** Set dimensions of output tensor.
   *  E.g. set the dimensions of an "activations tensor" or the
   *  "neuron dimensions."
   */
  void set_output_dims(std::vector<int> dims) {
    learning_layer::set_output_dims(dims);
  }

 protected:

  void setup_matrices(const El::Grid& grid) override;

  void setup_data() override {
    learning_layer::setup_data();

    // Initialize default weights if none are provided
    if (this->m_weights.size() > 2) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "attempted to setup " << m_name << " with an invalid number of weights";
      throw lbann_exception(err.str());
    }
    this->m_weights.resize(2, nullptr);
    if (this->m_weights[0] == nullptr) {
      auto* w = new weights(get_comm());
      std::unique_ptr<weights_initializer> init(new he_initializer(probability_distribution::gaussian));
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      w->set_name(get_name() + "_linearity_weights");
      w->set_initializer(init);
      w->set_optimizer(opt);
      this->m_weights[0] = w;
      this->m_model->add_weights(w);
    }
    if (this->m_weights[1] == nullptr) {
      auto* w = new weights(get_comm());
      std::unique_ptr<optimizer> opt(m_model->create_optimizer());
      w->set_name(get_name() + "_bias_weights");
      w->set_optimizer(opt);
      this->m_weights[1] = w;
      this->m_model->add_weights(w);
    }
    auto& linearity_weights = *this->m_weights[0];
    auto& bias_weights = *this->m_weights[1];

    // Initialize variance scaling initialization
    auto* cast_initializer
      = dynamic_cast<variance_scaling_initializer*>(linearity_weights.get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(get_input_size());
      cast_initializer->set_fan_out(get_output_size());
    }

    // Setup linearity weights
    auto linearity_dist = get_prev_activations().DistData();
    if (linearity_dist.colDist != El::MC
        || linearity_dist.rowDist != El::MR) {
      linearity_dist.colDist = El::STAR;
      linearity_dist.rowDist = El::STAR;
    }
    if (m_transpose) {
      linearity_weights.set_dims(get_input_dims(), get_output_dims());
    } else {
      linearity_weights.set_dims(get_output_dims(), get_input_dims());
    }
    linearity_weights.set_matrix_distribution(linearity_dist);

    // Setup bias weights
    auto bias_dist = get_activations().DistData();
    bias_dist.rowDist = El::STAR;
    bias_weights.set_dims(get_output_dims());
    bias_weights.set_matrix_distribution(bias_dist);

    // Setup weight gradients
    El::Zeros(*m_linearity_gradient,
              linearity_weights.get_matrix_height(),
              linearity_weights.get_matrix_width());
    El::Zeros(*this->m_bias_gradient,
              bias_weights.get_matrix_height(),
              bias_weights.get_matrix_width());

    // Initialize freeze state
    for (auto&& w : this->m_weights) {
      if (m_frozen) {
        w->freeze();
      } else {
        w->unfreeze();
      }
    }
    for (auto&& w : this->m_weights) {
      if (w->is_frozen() != m_frozen) {
        std::stringstream err;
        err << (m_frozen ? "" : "un") << "frozen "
            << "layer \"" << get_name() << "\" has "
            << (w->is_frozen() ? "" : "un") << "frozen "
            << "weights \"" << w->get_name() << "\"";
        LBANN_ERROR(err.str());
      }
    }
    
  }

  void fp_compute() override;
  void bp_compute() override;

 private:

  /** Deallocate distributed matrices. */
  void deallocate_matrices() {
    if (m_linearity_gradient != nullptr) delete m_linearity_gradient;
    if (m_bias_gradient != nullptr) delete m_bias_gradient;
  }

};

} // namespace lbann

#endif // LBANN_LAYER_FULL_CONNECTED_HPP_INCLUDED
