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
template <data_layout T_layout>
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
    if (cudnn != nullptr && T_layout == data_layout::DATA_PARALLEL) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN
    
  }

  dropout(const dropout& other) :
    regularizer_layer(other),
    m_keep_prob(other.m_keep_prob),
    m_mask(!other.m_mask? nullptr : other.m_mask->Copy()) {
  #ifdef LBANN_HAS_CUDNN
    m_states_d = other.m_states_d;
    m_reserve_space_d = other.m_reserve_space_d;
    if (!other.m_dropout_cudnn_desc.empty()) {
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
    m_states_d = other.m_states_d;
    m_reserve_space_d = other.m_reserve_space_d;
    for (auto&& desc : m_dropout_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyDropoutDescriptor(desc));
    }
    m_dropout_cudnn_desc.clear();
    if (!other.m_dropout_cudnn_desc.empty()) {
      setup_dropout_cudnn_desc();
    }
  #endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~dropout() override {
  #ifdef LBANN_HAS_CUDNN
    for (auto&& desc : m_dropout_cudnn_desc) {
      CHECK_CUDNN(cudnnDestroyDropoutDescriptor(desc));
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
    if (!this->m_using_gpus) {
      m_mask = std::unique_ptr<AbsDistMat>(get_activations().Copy());
    }
  }
  data_layout get_data_layout() const override { return T_layout; }

  void setup_gpu() override {
    regularizer_layer::setup_gpu();
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else

    // Allocate work spaces
    size_t size;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(this->m_cudnn->get_handle(0), &size));
    size = (size + sizeof(DataType) - 1) / sizeof(DataType);
    m_states_d = cudnn::matrix(m_cudnn, size, 1);
    CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(this->m_prev_activations_cudnn_desc, &size));
    size = (size + sizeof(DataType) - 1) / sizeof(DataType);
    m_reserve_space_d = cudnn::matrix(m_cudnn, size, 1);

    // Initialize cuDNN descriptors
    setup_dropout_cudnn_desc();
    
  #endif
  }

 protected:

  void fp_compute () override {
    if (this->m_using_gpus) {
      fp_compute_gpu();
    } else {
      fp_compute_cpu();
    }
  }

  void bp_compute () override {
    if (this->m_using_gpus) {
      bp_compute_gpu();
    } else {
      bp_compute_cpu();
    }
  }

 private:

  void fp_setup_data(int mini_batch_size) override {
  #ifdef LBANN_HAS_CUDNN
    if (this->m_using_gpus) {
      // Make sure GPU output is not a view during training
      const auto& mode = this->m_model->get_execution_mode();
      auto& output_d = m_activations_d[0];
      if (mode == execution_mode::training && output_d.is_view()) {
        const auto& input_d = m_prev_activations_d[0];
        output_d.clear();
        output_d.resize(input_d.get_height(), input_d.get_width_per_gpu());
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
    const auto& input_d = this->m_prev_activations_d[0];
    auto& output_d = this->m_activations_d[0];
    const auto& num_gpus = this->m_cudnn->get_num_gpus();

    // Do nothing if dropout is disabled
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      output_d.locked_view(input_d);
      return;
    }

    // Apply dropout on each GPU
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnDropoutForward(this->m_cudnn->get_handle(i),
                                      m_dropout_cudnn_desc[i],
                                      this->m_prev_activations_cudnn_desc,
                                      input_d.get_locked_data(i),
                                      this->m_activations_cudnn_desc,
                                      output_d.get_data(i),
                                      m_reserve_space_d.get_data(i),
                                      m_reserve_space_d.get_height() * sizeof(DataType)));
    }

  #endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() {
  #ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
  #else 
    const auto& gradient_wrt_output_d = this->m_prev_error_signals_d[0];
    auto& gradient_wrt_input_d = this->m_error_signals_d[0];
    const auto& num_gpus = this->m_cudnn->get_num_gpus();

    // Add to error signal if dropout is disabled
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      for (int i=0; i<num_gpus; ++i) {
        CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
        cublas::geam(this->m_cudnn->get_cublas_handle(i),
                     CUBLAS_OP_N, CUBLAS_OP_N,
                     gradient_wrt_input_d.get_height(),
                     this->m_mini_batch_size_per_gpu,
                     DataType(1),
                     gradient_wrt_output_d.get_locked_data(i),
                     gradient_wrt_output_d.get_leading_dim(),
                     DataType(1),
                     gradient_wrt_input_d.get_locked_data(i),
                     gradient_wrt_input_d.get_leading_dim(),
                     gradient_wrt_input_d.get_data(i),
                     gradient_wrt_input_d.get_leading_dim());
      }
      return;
    }

    // Apply dropout backprop on each GPU
    /// @todo This is technically incorrect since it overwrites the error signal
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnDropoutBackward(this->m_cudnn->get_handle(i),
                                       m_dropout_cudnn_desc[i],
                                       this->m_prev_error_signals_cudnn_desc,
                                       gradient_wrt_output_d.get_locked_data(i),
                                       this->m_error_signals_cudnn_desc,
                                       gradient_wrt_input_d.get_data(i),
                                       m_reserve_space_d.get_data(i),
                                       m_reserve_space_d.get_height() * sizeof(DataType)));
    }

  #endif // LBANN_HAS_CUDNN
  }

  #ifdef LBANN_HAS_CUDNN
  /** Setup cuDNN dropout descriptors.
   *  It is assumed that m_states_d has already been initialized.
   */
  void setup_dropout_cudnn_desc() {
    const auto& num_gpus = m_cudnn->get_num_gpus();
    m_dropout_cudnn_desc.assign(num_gpus, nullptr);
    for (int i = 0; i < num_gpus; ++i) {
      CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
      CHECK_CUDNN(cudnnSetStream(this->m_cudnn->get_handle(i),
                                 this->m_cudnn->get_stream(i)));
      CHECK_CUDNN(cudnnCreateDropoutDescriptor(&m_dropout_cudnn_desc[i]));
      CHECK_CUDNN(cudnnSetDropoutDescriptor(m_dropout_cudnn_desc[i],
                                            this->m_cudnn->get_handle(i),
                                            float(1 - m_keep_prob),
                                            m_states_d.get_data(i),
                                            m_states_d.get_height() * sizeof(DataType),
                                            get_generator()()));
    }
  }
  #endif // LBANN_HAS_CUDNN

  /** Probability of keeping each unit. */
  EvalType m_keep_prob;
  /** Current dropout mask (a scaled Bernoulli random matrix). */
  std::unique_ptr<AbsDistMat> m_mask;

  #ifdef LBANN_HAS_CUDNN
  /** Dropout cuDNN descriptor. */
  std::vector<cudnnDropoutDescriptor_t> m_dropout_cudnn_desc;
  /** RNG state for cuDNN dropout. */
  cudnn::matrix m_states_d;
  /** Work space for cuDNN dropout. */
  cudnn::matrix m_reserve_space_d;
  #endif // LBANN_HAS_CUDNN

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
