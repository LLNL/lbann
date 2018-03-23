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
#include "lbann/utils/random.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/models/model.hpp"
#include "lbann/weights/initializer.hpp"
#include "lbann/weights/fan_in_fan_out_initializers.hpp"
#include "lbann/utils/cublas_wrapper.hpp"
#include <string>
#include <sstream>

namespace lbann {

enum class device {CPU, CUDA};

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

#ifdef LBANN_HAS_CUDNN
  /** GPU memory for linearity gradient. */
  cudnn::matrix m_linearity_gradient_d;
  /** GPU memory for bias gradient. */
  cudnn::matrix m_bias_gradient_d;
#endif // __LIB_CUNN

 public:

  fully_connected_layer(lbann_comm *comm,
                        int num_neurons,  // TODO: accept a vector for neuron dims
                        weights* weight = nullptr,
                        bool has_bias = true,
                        cudnn::cudnn_manager *cudnn = nullptr)
    : learning_layer(comm),
      m_linearity_gradient(nullptr),
      m_bias_gradient(nullptr) {

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims.assign(1, this->m_num_neurons);

    // Initialize bias
    m_bias_scaling_factor = has_bias ? DataType(1) : DataType(0);

#ifdef LBANN_HAS_CUDNN
    if (cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
#endif // LBANN_HAS_CUDNN
  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    return std::string {} +
     " fully_connected; num_neurons: "
     + std::to_string(this->m_num_neurons)
     + " has_bias: " + std::to_string(this->m_bias_scaling_factor)
     + " dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

  fully_connected_layer(const fully_connected_layer& other) :
    learning_layer(other),
    m_bias_scaling_factor(other.m_bias_scaling_factor) {

    // Deep matrix copies
    m_linearity_gradient = other.m_linearity_gradient;
    m_bias_gradient = other.m_bias_gradient;
    if (m_linearity_gradient != nullptr) {
      m_linearity_gradient = m_linearity_gradient->Copy();
    }
    if (m_bias_gradient != nullptr) {
      m_bias_gradient = m_bias_gradient->Copy();
    }

#ifdef LBANN_HAS_CUDNN
    m_linearity_gradient_d = other.m_linearity_gradient_d;
    m_bias_gradient_d = other.m_bias_gradient_d;
#endif // LBANN_HAS_CUDNN

  }

  fully_connected_layer& operator=(const fully_connected_layer& other) {
    learning_layer::operator=(other);
    m_bias_scaling_factor = other.m_bias_scaling_factor;

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

  #ifdef LBANN_HAS_CUDNN
    m_linearity_gradient_d = other.m_linearity_gradient_d;
    m_bias_gradient_d = other.m_bias_gradient_d;
  #endif // LBANN_HAS_CUDNN

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

  void setup_matrices(const El::Grid& grid) override;

  void setup_dims() override {
    // Store neuron tensor dimensions
    const int num_neurons = this->m_num_neurons;
    const int num_neuron_dims = this->m_num_neuron_dims;
    const std::vector<int> neuron_dims = this->m_neuron_dims;

    // Initialize previous neuron tensor dimensions
    learning_layer::setup_dims();

    // Initialize neuron tensor dimensions
    this->m_num_neurons = num_neurons;
    this->m_num_neuron_dims = num_neuron_dims;
    this->m_neuron_dims = neuron_dims;
  }

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
      this->m_weights[0] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[0]->set_name(this->m_name + "_linearity_weights");
      this->m_weights[0]->set_initializer(new he_normal_initializer(this->m_comm));
      this->m_weights[0]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[0]);
    }
    if (this->m_weights[1] == nullptr) {
      this->m_weights[1] = new weights(this->m_comm, this->m_cudnn);
      this->m_weights[1]->set_name(this->m_name + "_bias_weights");
      this->m_weights[1]->set_optimizer(m_model->create_optimizer());
      this->m_model->add_weights(this->m_weights[1]);
    }

    // Initialize Glorot or He weight initialization
    auto* cast_initializer
      = dynamic_cast<fan_in_fan_out_initializer*>(&this->m_weights[0]->get_initializer());
    if (cast_initializer != nullptr) {
      cast_initializer->set_fan_in(this->m_num_prev_neurons);
      cast_initializer->set_fan_out(this->m_num_neurons);
    }

    // Setup weights
    // Note: linearity matrix is duplicated across processes unless
    // the data layout is model-parallel.
    El::Distribution col_dist = El::STAR;
    El::Distribution row_dist = El::STAR;
    if (get_data_layout() == data_layout::MODEL_PARALLEL) {
      col_dist = El::MC;
      row_dist = El::MR;
    }
    this->m_weights[0]->setup(this->m_num_neurons,
                              this->m_num_prev_neurons,
                              col_dist, row_dist, Dev);
    this->m_weights[1]->setup(this->m_num_neurons,
                              1,
                              get_activations().DistData().colDist,
                              El::STAR, Dev);

    // Setup weight gradients
    El::Zeros(*this->m_linearity_gradient,
              this->m_weights[0]->get_matrix_height(),
              this->m_weights[0]->get_matrix_width());
    El::Zeros(*this->m_bias_gradient,
              this->m_weights[1]->get_matrix_height(),
              this->m_weights[1]->get_matrix_width());

  }

  void setup_gpu() override {
    learning_layer::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    throw lbann_exception("fully_connected_layer: CUDA not detected");
#else
    m_linearity_gradient_d = cudnn::matrix(m_cudnn,
                                           m_linearity_gradient->Height(),
                                           m_linearity_gradient->Width());
    if(m_bias_scaling_factor != DataType(0)) {
      m_bias_gradient_d = cudnn::matrix(m_cudnn,
                                        m_bias_gradient->Height(),
                                        m_bias_gradient->Width());
    }
#endif // LBANN_HAS_CUDNN
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
