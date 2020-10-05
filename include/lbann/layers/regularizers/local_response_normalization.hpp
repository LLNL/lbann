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

#ifndef LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED
#define LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/regularizers/regularizer.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** @brief
 *
 *  See:
 *
 *  Alex Krizhevsky, Ilya Sutskever, and Geoffrey E. Hinton. "ImageNet
 *  classification with deep convolutional neural networks." In
 *  Advances in Neural Information Processing Systems,
 *  pp. 1097-1105. 2012.
 */
template <typename TensorDataType,
          data_layout T_layout = data_layout::DATA_PARALLEL,
          El::Device Dev = El::Device::CPU>
class local_response_normalization_layer : public regularizer_layer<TensorDataType> {
#ifdef LBANN_HAS_CUDNN
  using ScalingType = cudnn::ScalingParamType<TensorDataType>;
#else
  using ScalingType = TensorDataType;
#endif // LBANN_HAS_CUDNN

  static_assert(T_layout == data_layout::DATA_PARALLEL,
                "local_response_normalization only supports DATA_PARALLEL");
public:

  local_response_normalization_layer(lbann_comm *comm,
                                     int window_width,
                                     TensorDataType alpha,
                                     TensorDataType beta,
                                     TensorDataType k)
    : regularizer_layer<TensorDataType>(comm),
      m_window_width(window_width), m_alpha(alpha), m_beta(beta), m_k(k)
#ifdef LBANN_HAS_CUDNN
    , m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  { }

  local_response_normalization_layer(
    const local_response_normalization_layer& other)
    : regularizer_layer<TensorDataType>(other),
      m_window_width(other.m_window_width),
      m_alpha(other.m_alpha),
      m_beta(other.m_beta),
      m_k(other.m_k)
#ifdef LBANN_HAS_CUDNN
    , m_lrn_cudnn_desc(other.m_lrn_cudnn_desc),
      m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  local_response_normalization_layer& operator=(
    const local_response_normalization_layer& other) {
    regularizer_layer<TensorDataType>::operator=(other);
    m_window_width = other.m_window_width;
    m_alpha = other.m_alpha;
    m_beta = other.m_beta;
    m_k = other.m_k;
#ifdef LBANN_HAS_CUDNN
    m_lrn_cudnn_desc = other.m_lrn_cudnn_desc;
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  ~local_response_normalization_layer() override = default;

  local_response_normalization_layer* copy() const override {
    return new local_response_normalization_layer(*this);
  }
  std::string get_type() const override { return "LRN"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  description get_description() const override {
    auto desc = regularizer_layer<TensorDataType>::get_description();
    desc.add("alpha", m_alpha);
    desc.add("beta", m_beta);
    desc.add("k", m_k);
    return desc;
  }

protected:

  void setup_dims(DataReaderMetaData& dr_metadata) override {
    regularizer_layer<TensorDataType>::setup_dims(dr_metadata);
    this->set_output_dims(this->get_input_dims());
  }

  /// Initialize GPU objects
  void setup_gpu() override {
    regularizer_layer<TensorDataType>::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    m_lrn_cudnn_desc.set(m_window_width,
                         m_alpha, m_beta, m_k);
#endif // #ifndef LBANN_HAS_CUDNN
  }

  void fp_compute() override {
    if (this->using_gpus()) {
      fp_compute_cudnn();
    }
    else {
      fp_compute_cpu();
    }
  }

  void bp_compute() override {
    if (this->using_gpus()) {
      bp_compute_cudnn();
    }
    else {
      bp_compute_cpu();
    }
  }

private:

  /** Normalization window width. */
  int m_window_width;
  /** LRN alpha scaling parameter. */
  TensorDataType m_alpha;
  /** LRN beta power parameter. */
  TensorDataType m_beta;
  /** LRN k parameter. */
  TensorDataType m_k;

#ifdef LBANN_HAS_CUDNN
  /** LRN cuDNN descriptor. */
  cudnn::LRNDescriptor m_lrn_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager<TensorDataType> m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

  /// GPU implementation of forward propagation
  void fp_compute_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();
    if (local_input.Height() > 0 && local_input.Width() > 0) {
      const ScalingType zero = El::TypeTraits<ScalingType>::Zero();
      const ScalingType one = El::TypeTraits<ScalingType>::One();
      CHECK_CUDNN(cudnnLRNCrossChannelForward(cudnn::get_handle(),
                                              m_lrn_cudnn_desc,
                                              CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                              &one,
                                              m_tensors_cudnn_desc.get_prev_activations(),
                                              local_input.LockedBuffer(),
                                              &zero,
                                              m_tensors_cudnn_desc.get_activations(),
                                              local_output.Buffer()));
    }
#endif // LBANN_HAS_CUDNN
  }

  /// GPU implementation of backward propagation
  void bp_compute_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    const auto& local_input = this->get_local_prev_activations();
    const auto& local_output = this->get_local_activations();
    const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();
    if (local_input.Height() > 0 && local_input.Width() > 0) {
      const ScalingType zero = El::TypeTraits<ScalingType>::Zero();
      const ScalingType one = El::TypeTraits<ScalingType>::One();
      CHECK_CUDNN(cudnnLRNCrossChannelBackward(cudnn::get_handle(),
                                               m_lrn_cudnn_desc,
                                               CUDNN_LRN_CROSS_CHANNEL_DIM1,
                                               &one,
                                               m_tensors_cudnn_desc.get_prev_activations(),
                                               local_input.LockedBuffer(),
                                               m_tensors_cudnn_desc.get_prev_error_signals(),
                                               local_gradient_wrt_output.LockedBuffer(),
                                               m_tensors_cudnn_desc.get_activations(),
                                               local_output.LockedBuffer(),
                                               &zero,
                                               m_tensors_cudnn_desc.get_error_signals(),
                                               local_gradient_wrt_input.Buffer()));
    }
#endif // LBANN_HAS_CUDNN
  }

  /// CPU implementation of forward propagation
  void fp_compute_cpu() {

    // Local matrices
    const auto& local_input = this->get_local_prev_activations();
    auto& local_output = this->get_local_activations();

    // Matrix parameters
    const int local_width = local_input.Width();
    const TensorDataType* input_buffer = local_input.LockedBuffer();
    const int input_ldim = local_input.LDim();
    TensorDataType* output_buffer = local_output.Buffer();
    const int output_ldim = local_output.LDim();

    // Get LRN parameters
    const auto& output_dims = this->get_output_dims();
    const int num_channels = output_dims[0];
    const int num_per_channel = this->get_output_size() / num_channels;

    // Check if LRN is using default beta parameter
    const bool default_beta = (std::fabs((m_beta - El::To<TensorDataType>(0.75))
                                         / El::To<TensorDataType>(0.75))
                               < 2 * std::numeric_limits<DataType>::epsilon());

    ////////////////////////////////////////////////////////////////
    // activations(i) = prev_activations(i) / scale_factor(i) ^ beta
    // scale_factor(i)
    //   = k + alpha * sum( prev_activations(j) ^ 2 )
    // Note: The sum is over entries in the normalization window.
    ////////////////////////////////////////////////////////////////

    // Iterate through blocks in channels of each data sample
    const int max_block_size = 16;
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (int sample = 0; sample < local_width; ++sample) {
      for (int block_start = 0;
          block_start < num_per_channel;
          block_start += max_block_size) {
        const int block_size = std::min(max_block_size,
                                        num_per_channel - block_start);
        TensorDataType workspace[max_block_size];

        // Iterate through channels
        for (int channel = 0; channel < num_channels; ++channel) {
          const int window_start = std::max(channel - m_window_width / 2, 0);
          const int window_end = std::min(channel + m_window_width / 2, num_channels - 1);

          // Compute sum of squares in workspace
          std::fill(workspace, workspace + block_size, El::TypeTraits<TensorDataType>::Zero());
          for (int window_pos = window_start; window_pos <= window_end; ++window_pos) {
            for (int block_pos = 0; block_pos < block_size; ++block_pos) {
              const int index = block_start + block_pos + window_pos * num_per_channel;
              const TensorDataType input_entry = input_buffer[index + sample * input_ldim];
              workspace[block_pos] += input_entry * input_entry;
            }
          }

          // Compute 1 / (k + alpha * sum(x^2) ) in workspace
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            workspace[block_pos] = El::TypeTraits<TensorDataType>::One()
              / (m_k + m_alpha * workspace[block_pos]);
          }

          // Compute output
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            const int index = block_start + block_pos + channel * num_per_channel;
            const TensorDataType scale_factor = workspace[block_pos];
            const TensorDataType input_entry = input_buffer[index + sample * input_ldim];
            TensorDataType& output_entry = output_buffer[index + sample * output_ldim];
            if (default_beta) { // Special case when beta = 0.75
              output_entry = (input_entry
                              * El::Sqrt(scale_factor * El::Sqrt(scale_factor)));
            }
            else {
              output_entry = input_entry * El::Pow(scale_factor, m_beta);
            }
          }

        }

      }
    }

  }

  /// CPU implementation of backward propagation
  void bp_compute_cpu() {

    // Get local matrices
    const auto& local_input = this->get_local_prev_activations();
    const auto& local_output = this->get_local_activations();
    const auto& local_gradient_wrt_output = this->get_local_prev_error_signals();
    auto& local_gradient_wrt_input = this->get_local_error_signals();

    // Get matrix buffers
    const int local_width = local_input.Width();
    const TensorDataType* input_buffer = local_input.LockedBuffer();
    const int input_ldim = local_input.LDim();
    const TensorDataType* output_buffer = local_output.LockedBuffer();
    const int output_ldim = local_output.LDim();
    const TensorDataType* gradient_wrt_output_buffer = local_gradient_wrt_output.LockedBuffer();
    const int gradient_wrt_output_ldim = local_gradient_wrt_output.LDim();
    TensorDataType* gradient_wrt_input_buffer = local_gradient_wrt_input.Buffer();
    const int gradient_wrt_input_ldim = local_gradient_wrt_input.LDim();

    // Get LRN parameters
    const auto& output_dims = this->get_output_dims();
    const int num_channels = output_dims[0];
    const int num_per_channel = this->get_output_size() / num_channels;

    // Check if LRN is using default beta parameter
    const bool default_beta = (std::fabs((m_beta - El::To<TensorDataType>(0.75))
                                         / El::To<TensorDataType>(0.75))
                               < El::To<TensorDataType>(2) * std::numeric_limits<TensorDataType>::epsilon());

    ////////////////////////////////////////////////////////////////
    // error_signal(i)
    //   = prev_error_signal(i) / scale_factor(i) ^ beta
    //     - 2 * alpha * beta / window_width * prev_activations(i)
    //       * sum( prev_error_signal(j) * activations(j)
    //              / scale_factor(j) )
    // Note: See comments in fp_linearity_cpu for a definition of
    //   scale_factor. The sum is over entries in the normalization
    //   window.
    ////////////////////////////////////////////////////////////////

    // Iterate through blocks in channels of each data sample
    const int max_block_size = 16;
    LBANN_OMP_PARALLEL_FOR_COLLAPSE2
    for (int sample = 0; sample < local_width; ++sample) {
      for (int block_start = 0;
          block_start < num_per_channel;
          block_start += max_block_size) {
        const int block_size = std::min(max_block_size,
                                        num_per_channel - block_start);
        TensorDataType workspace[max_block_size];

        // Iterate through channels
        for (int channel = 0; channel < num_channels; ++channel) {
          const int window_start = std::max(channel - m_window_width / 2, 0);
          const int window_end = std::min(channel + m_window_width / 2, num_channels - 1);

          // Compute sum of squares in workspace
          std::fill(workspace, workspace + block_size, El::TypeTraits<TensorDataType>::Zero());
          for (int window_pos = window_start; window_pos <= window_end; ++window_pos) {
            for (int block_pos = 0; block_pos < block_size; ++block_pos) {
              const int index = block_start + block_pos + window_pos * num_per_channel;
              const TensorDataType input_entry = input_buffer[index + sample * input_ldim];
              workspace[block_pos] += input_entry * input_entry;
            }
          }

          // Compute 1 / (k + alpha * sum(x^2) ) in workspace
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            workspace[block_pos] = El::TypeTraits<TensorDataType>::One()
              / (m_k + m_alpha * workspace[block_pos]);
          }

          // Compute error signal contribution for current entry
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            const int index = block_start + block_pos + channel * num_per_channel;
            const TensorDataType scale_factor = workspace[block_pos];
            const TensorDataType gradient_wrt_output_entry
              = gradient_wrt_output_buffer[index + sample * gradient_wrt_output_ldim];
            TensorDataType& gradient_wrt_input_entry
              = gradient_wrt_input_buffer[index + sample * gradient_wrt_input_ldim];
            if (default_beta) { // Special case when beta = 0.75
              gradient_wrt_input_entry
                = gradient_wrt_output_entry * El::Sqrt(scale_factor * El::Sqrt(scale_factor));
            }
            else {
              gradient_wrt_input_entry
                = gradient_wrt_output_entry * El::Pow(scale_factor, m_beta);
            }
          }

          // Compute y * dy / (k + alpha * sum(x^2) ) in workspace
          for (int block_pos = 0; block_pos < block_size; ++block_pos) {
            const int index = block_start + block_pos + channel * num_per_channel;
            const TensorDataType output_entry = output_buffer[index + sample * output_ldim];
            const TensorDataType gradient_wrt_output_entry
              = gradient_wrt_output_buffer[index + sample * gradient_wrt_output_ldim];
            workspace[block_pos] = (El::To<TensorDataType>(-2) * m_alpha * m_beta * workspace[block_pos]
                                    * output_entry * gradient_wrt_output_entry);
          }

          // Compute error signal contribution for entries in window
          for (int window_pos = window_start; window_pos <= window_end; ++window_pos) {
            for (int block_pos = 0; block_pos < block_size; ++block_pos) {
              const int index = block_start + block_pos + window_pos * num_per_channel;
              const TensorDataType input_entry = input_buffer[index + sample * input_ldim];
              gradient_wrt_input_buffer[index + sample * gradient_wrt_input_ldim]
                += workspace[block_pos] * input_entry;
            }
          }

        }

      }
    }

  }

};

LBANN_DEFINE_LAYER_BUILDER(local_response_normalization);

#ifndef LBANN_LOCAL_RESPONSE_NORMALIZATION_LAYER_INSTANTIATE
#define PROTO_DEVICE(T, Device)                             \
  extern template class local_response_normalization_layer< \
    T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"
#undef PROTO_DEVICE
#endif // LBANN_LOCAL_RESPONSE_NORMALIZATION_LAYER_INSTANTIATE

} // namespace lbann

#endif // LBANN_LAYER_LOCAL_RESPONSE_NORMALIZATION_HPP_INCLUDED
