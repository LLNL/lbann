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
#include "lbann/utils/cudnn.hpp"

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
          EvalType keep_prob = EvalType(0.5))
    : regularizer_layer(comm),
      m_keep_prob(keep_prob)
#ifdef LBANN_HAS_CUDNN
    , m_dropout_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {
#if defined(LBANN_HAS_CUDNN) && defined(LBANN_SEQUENTIAL_CONSISTENCY)
    /// @todo GPU implementation of dropout with sequential consistency
    if (Dev == El::Device::GPU && get_comm()->am_model_master()) {
      std::cerr << "Warning: GPU dropout currently does not guarantee "
                << "sequential consistency" << std::endl;
    }
#endif // defined(LBANN_HAS_CUDNN) && defined(LBANN_SEQUENTIAL_CONSISTENCY)
  }

  dropout(const dropout& other)
    : regularizer_layer(other),
      m_keep_prob(other.m_keep_prob),
      m_mask(other.m_mask ? other.m_mask->Copy() : nullptr)
#ifdef LBANN_HAS_CUDNN
    , m_dropout_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc.set_layer(this);
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
    m_mask = other.m_mask ? other.m_mask->Copy() : nullptr;
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
    m_states = other.m_states;
    m_reserve_space = other.m_reserve_space;
    if (other.m_dropout_cudnn_desc != nullptr) {
      setup_dropout_cudnn_desc();
    } else {
      CHECK_CUDNN(cudnnDestroyDropoutDescriptor(m_dropout_cudnn_desc));
      m_dropout_cudnn_desc = nullptr;
    }
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~dropout() override {
#ifdef LBANN_HAS_CUDNN
    if (m_dropout_cudnn_desc != nullptr) {
      cudnnDestroyDropoutDescriptor(m_dropout_cudnn_desc);
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

#ifdef HYDROGEN_HAVE_CUB
    // Use CUB GPU memory pool for some matrices
    // Note: Activation matrix owns data during training and is a
    // matrix view during evaluation. To avoid expensive GPU memory
    // allocation and deallocation, we use CUB's GPU memory pool.
    if (Dev == El::Device::GPU) {
      get_local_activations().SetMemoryMode(1);
      get_local_error_signals().SetMemoryMode(1);
      m_reserve_space.SetMemoryMode(1);
    }
#endif // HYDROGEN_HAVE_CUB

    // Initialize cuDNN objects
    setup_dropout_cudnn_desc();

#endif // LBANN_HAVE_CUDNN
  }

 protected:

  void fp_setup_data(int mini_batch_size) override {
    // If needed, reset matrix view without deallocating memory
    // Note: Activation matrix owns data during training and is a
    // matrix view during evaluation.
    const auto& mode = this->m_model->get_execution_mode();
    if (mode == execution_mode::training && m_keep_prob < EvalType(0)) {
      get_activations().Empty(false);
    }
    regularizer_layer::fp_setup_data(mini_batch_size);
  }

  void bp_setup_data(int mini_batch_size) override {
    // If needed, reset matrix view without deallocating memory
    // Note: Activation matrix owns data during training and is a
    // matrix view during evaluation.
    const auto& mode = this->m_model->get_execution_mode();
    if (mode == execution_mode::training && m_keep_prob < EvalType(0)) {
      get_error_signals().Empty(false);
    }
    regularizer_layer::bp_setup_data(mini_batch_size);
  }

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
      El::LockedView(gradient_wrt_input, gradient_wrt_output);
    } else {
      El::Hadamard(gradient_wrt_output, *m_mask, gradient_wrt_input);
    }
  }

  void fp_compute_gpu() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    
    // Matrices
    const auto& input = get_prev_activations();
    const auto& local_input = input.LockedMatrix();
    auto& output = get_activations();
    auto& local_output = output.Matrix();

    // Do nothing if dropout is disabled or there is no local data
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::LockedView(output, input);
      return;
    }
    if (local_input.Height() < 1 && local_input.Width() < 1) { return; }

    // Initialize cuDNN objects
    auto&& input_desc = m_tensors_cudnn_desc.get_prev_activations();
    auto&& output_desc = m_tensors_cudnn_desc.get_activations();
    size_t size;
    CHECK_CUDNN(cudnnDropoutGetReserveSpaceSize(input_desc, &size));
    m_reserve_space.Resize((size + sizeof(DataType) - 1) / sizeof(DataType), 1);

    // Apply dropout on the GPU
    CHECK_CUDNN(cudnnDropoutForward(cudnn::get_handle(),
                                    m_dropout_cudnn_desc,
                                    input_desc,
                                    local_input.LockedBuffer(),
                                    output_desc,
                                    local_output.Buffer(),
                                    m_reserve_space.Buffer(),
                                    m_reserve_space.Height() * sizeof(DataType)));

#endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Matrices
    const auto& gradient_wrt_output = get_prev_error_signals();
    const auto& local_gradient_wrt_output = gradient_wrt_output.LockedMatrix();
    auto& gradient_wrt_input = get_error_signals();
    auto& local_gradient_wrt_input = gradient_wrt_input.Matrix();

    // Copy error signal if dropout is disabled
    const auto& mode = this->m_model->get_execution_mode();
    if (mode != execution_mode::training || m_keep_prob < EvalType(0)) {
      El::LockedView(gradient_wrt_input, gradient_wrt_output);
    } else {
      if (local_gradient_wrt_input.Height() > 0
          && local_gradient_wrt_input.Width() > 0) {
        CHECK_CUDNN(cudnnDropoutBackward(cudnn::get_handle(),
                                         m_dropout_cudnn_desc,
                                         m_tensors_cudnn_desc.get_prev_error_signals(),
                                         local_gradient_wrt_output.LockedBuffer(),
                                         m_tensors_cudnn_desc.get_error_signals(),
                                         local_gradient_wrt_input.Buffer(),
                                         m_reserve_space.Buffer(),
                                         m_reserve_space.Height() * sizeof(DataType)));
      }
    }
#endif // LBANN_HAS_CUDNN
  }

#ifdef LBANN_HAS_CUDNN
  /** Setup cuDNN dropout descriptor and RNG state.
   */
  void setup_dropout_cudnn_desc() {

    // Deallocate dropout descriptor if needed
    if (m_dropout_cudnn_desc != nullptr) {
      CHECK_CUDNN(cudnnDestroyDropoutDescriptor(m_dropout_cudnn_desc));
    }
    m_dropout_cudnn_desc = nullptr;

    // Setup RNG state
    size_t size;
    CHECK_CUDNN(cudnnDropoutGetStatesSize(cudnn::get_handle(), &size));
    m_states.Resize((size + sizeof(DataType) - 1) / sizeof(DataType), 1);

    // Setup dropout descriptor
    CHECK_CUDNN(cudnnCreateDropoutDescriptor(&m_dropout_cudnn_desc));
    CHECK_CUDNN(cudnnSetDropoutDescriptor(m_dropout_cudnn_desc,
                                          cudnn::get_handle(),
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
  cudnnDropoutDescriptor_t m_dropout_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::entrywise_layer_tensor_manager m_tensors_cudnn_desc;
  /** RNG state for cuDNN dropout. */
  GPUMat m_states;
  /** Work space for cuDNN dropout. */
  GPUMat m_reserve_space;
#endif // LBANN_HAS_CUDNN

};

} // namespace lbann

#endif // LBANN_LAYER_REGULARIZER_DROPOUT_HPP_INCLUDED
