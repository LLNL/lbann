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

#ifndef LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
#define LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED

#include "lbann/layers/regularizers/regularizer.hpp"

namespace lbann {

/** Dropout layer.
 *  Probabilistically drop layer outputs. See:
 *    Srivastava, Nitish, et al. "Dropout: a simple way to prevent
 *    neural networks from overfitting." Journal of Machine Learning
 *    Research 15.1 (2014).
 *  The weights are multiplied by 1/(keep probability) at training
 *  time, as discussed in section 10 of the paper. Keep probabilities
 *  of 0.5 for fully-connected layers and 0.8 for input layers are
 *  good starting points.
 */
template <data_layout T_layout, El::Device Dev>
class dropout : public regularizer_layer {
 public:
  /** Keep units with probabiliy keep_prob. */
  dropout(lbann_comm *comm,
          EvalType keep_prob = EvalType(0.5),
          cudnn::cudnn_manager* cudnn = nullptr)
    : regularizer_layer(comm),
      m_keep_prob(keep_prob) {

  #if defined(LBANN_HAS_CUDNN) && !defined(LBANN_SEQUENTIAL_CONSISTENCY)
    // Initialize GPU memory if using GPU
    /// @todo GPU implementation of dropout with sequential consistency
    if (cudnn != nullptr/* && T_layout == data_layout::DATA_PARALLEL*/) {
      // this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  dropout(const dropout& other) :
    regularizer_layer(other),
    m_keep_prob(other.m_keep_prob),
    m_mask(!other.m_mask? nullptr : other.m_mask->Copy()) {
  #ifdef LBANN_HAS_CUDNN
    m_states = other.m_states;
    m_reserve_space = other.m_reserve_space;
    if (other.m_dropout_cudnn_desc != nullptr) {
      setup_dropout_cudnn_desc();
    }
  #endif // LBANN_HAS_CUDNN
  }

  dropout& operator=(const dropout& other) {
    regularizer_layer::operator=(other);
    m_keep_prob = other.m_keep_prob;
    if (!!other.m_mask) {
      m_mask = std::unique_ptr<AbsDistMat>(other.m_mask->Copy());
    }
  #ifdef LBANN_HAS_CUDNN
    m_states = other.m_states;
    m_reserve_space = other.m_reserve_space;
    if (m_dropout_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyDropoutDescriptor(m_dropout_cudnn_desc));
    }
    m_dropout_cudnn_desc = nullptr;
    if (other.m_dropout_cudnn_desc != nullptr) {
      setup_dropout_cudnn_desc();
    }
  #endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~dropout() override {
  #ifdef LBANN_HAS_CUDNN
    if (m_dropout_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyDropoutDescriptor(m_dropout_cudnn_desc));
    }
  #endif // LBANN_HAS_CUDNN
  }

  dropout* copy() const override { return new dropout(*this); }

  std::string get_type() const override { return "dropout"; }

  std::string get_description() const override {
    return " dropout keep_prob: " + std::to_string(m_keep_prob)
           + " dataLayout: " + get_data_layout_string(get_data_layout());
  }

  void setup_matrices(const El::Grid& grid) override {
    regularizer_layer::setup_matrices(grid);
    m_mask = std::unique_ptr<AbsDistMat>(get_activations().Copy());
  }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  void setup_gpu() override {
    regularizer_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else

    // Allocate work spaces
    size_t size;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(this->m_cudnn->get_handle(0), &size));
    size = (size + sizeof(DataType) - 1) / sizeof(DataType);
    El::Zeros(m_states, size, 1);
    CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(this->m_prev_activations_cudnn_desc, &size));
    size = (size + sizeof(DataType) - 1) / sizeof(DataType);
    El::Zeros(m_reserve_space, size, 1);

    // Initialize cuDNN descriptors
    setup_dropout_cudnn_desc();

  #endif
  }

 protected:

  void fp_compute () override {
    if (using_gpus()) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute () override {
    if (using_gpus()) {
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

 private:

  void fp_setup_data(int mini_batch_size) override {
  #ifdef LBANN_HAS_CUDNN
    if (using_gpus()) {
      // Make sure GPU output is not a view during training
      const auto& mode = this->m_model->get_execution_mode();
      auto& output = get_activations();
      if (mode == execution_mode::training && output.Viewing()) {
        const auto& input = get_prev_activations();
        El::Zeros(output, input.LocalHeight(), input.LocalWidth());
      }
    }
  #endif // LBANN_HAS_CUDNN
    regularizer_layer::fp_setup_data(mini_batch_size);
  }

  void fp_compute_cpu() {

    // Matrices
    const auto& input = get_prev_activations();
    auto& output = get_activations();

    // Do nothing if dropout is disabled
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::LockedView(output, input);
      return;
    }

    // Construct mask matrix
    const DataType scale = 1 / m_keep_prob;
    const auto& height = input.Height();
    const auto& width = input.Width();
    m_mask->Resize(height, width);
#ifdef LBANN_SEQUENTIAL_CONSISTENCY
    bernoulli_fill_procdet(*m_mask, height, width, DataType(m_keep_prob));
    *m_mask *= scale;
#else
    El::EntrywiseMap(*m_mask,
                     (std::function<DataType(const DataType&)>)
                     ([this,scale](const DataType& z)->DataType {
                       auto& gen = get_fast_generator();
                       std::bernoulli_distribution dist(m_keep_prob);
                       return dist(gen) ? scale : DataType(0);
                     }));
#endif // LBANN_SEQUENTIAL_CONSISTENCY

    // Apply mask matrix to get activations
    El::Hadamard(input, *m_mask, output);

  }

  /** Adjust gradients for dropout in backprop. */
  void bp_compute_cpu() {
    const auto& gradient_wrt_output = get_prev_error_signals();
    auto& gradient_wrt_input = get_error_signals();
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::Axpy(DataType(1), gradient_wrt_output, gradient_wrt_input);
    } else {
      El::Hadamard(gradient_wrt_output, *m_mask, *m_mask);
      El::Axpy(DataType(1), *m_mask, gradient_wrt_input);
    }
  }

  void fp_compute_gpu() {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    const auto& input = get_prev_activations();
    auto& output = get_activations();

    // Do nothing if dropout is disabled
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::LockedView(output, input);
      return;
    }

    // Apply dropout on the GPU
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
    CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                               this->m_cudnn->get_stream()));
    CHECK_CUDNN(cudnnDropoutForward(this->m_cudnn->get_handle(),
                                    m_dropout_cudnn_desc,
                                    this->m_prev_activations_cudnn_desc,
                                    input.LockedBuffer(),
                                    this->m_activations_cudnn_desc,
                                    output.Buffer(),
                                    m_reserve_space.Buffer(),
                                    m_reserve_space.Height() * sizeof(DataType)));
  #endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else
    const auto& gradient_wrt_output = get_prev_error_signals();
    auto& gradient_wrt_input = get_error_signals();

    // Copy error signal if dropout is disabled
    /// @todo This is technically incorrect since it overwrites the error signal
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::LockedView(gradient_wrt_input, gradient_wrt_output);
      return;
    }

    // Apply dropout backprop on each GPU
    /// @todo This is technically incorrect since it overwrites the error signal
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
    CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                               this->m_cudnn->get_stream()));
    CHECK_CUDNN(cudnnDropoutBackward(this->m_cudnn->get_handle(),
                                     m_dropout_cudnn_desc,
                                     this->m_prev_error_signals_cudnn_desc,
                                     gradient_wrt_output.LockedBuffer(),
                                     this->m_error_signals_cudnn_desc,
                                     gradient_wrt_input.Buffer(),
                                     m_reserve_space.Buffer(),
                                     m_reserve_space.Height() * sizeof(DataType)));

  #endif // LBANN_HAS_CUDNN
  }

  #ifdef LBANN_HAS_CUDNN
  /** Setup cuDNN dropout descriptors.
   *  It is assumed that m_states_d has already been initialized.
   */
  void setup_dropout_cudnn_desc() {
    CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
    CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(),
                               this->m_cudnn->get_stream()));
    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&m_dropout_cudnn_desc));
    CHECK_CUDNN(cudnnSetDropoutDescriptor(m_dropout_cudnn_desc,
                                          this->m_cudnn->get_handle(),
                                          float(1 - m_keep_prob),
                                          m_states.Buffer(),
                                          m_states.Height() * sizeof(DataType),
                                          get_generator()()));
  }
  #endif // LBANN_HAS_CUDNN

  /** Probability of keeping each unit. */
  EvalType m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  std::unique_ptr<AbsDistMat> m_mask;

  #ifdef LBANN_HAS_CUDNN
  /** Dropout cuDNN descriptor. */
  cudnnDropoutDescriptor_t m_dropout_cudnn_desc = nullptr;
  /** RNG state for cuDNN dropout. */
  GPUMat m_states;
  /** Work space for cuDNN dropout. */
  GPUMat m_reserve_space;
  #endif // LBANN_HAS_CUDNN

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
