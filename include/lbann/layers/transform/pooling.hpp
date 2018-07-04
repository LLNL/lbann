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

#ifndef LBANN_LAYER_POOLING_HPP_INCLUDED
#define LBANN_LAYER_POOLING_HPP_INCLUDED

#include <utility>
#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/cudnn.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

// Forward declaration
template <data_layout T_layout, El::Device Dev>
class unpooling_layer;

/** Pooling layer. */
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class pooling_layer : public transform_layer {
 private:

  /** Pooling mode. */
  const pool_mode m_pool_mode;

  /** Pooling window dimensions. */
  std::vector<int> m_pool_dims;
  /** Size of pooling window. */
  int m_pool_size;
  /** Pooling padding. */
  std::vector<int> m_pads;
  /** Pooling strides. */
  std::vector<int> m_strides;

  /** Input indices for max pooling.
   *  Each entry corresponds to a local entry in the activations
   *  matrix. The entry gives the index of the maximum entry within
   *  the pooling window.
   */
  std::vector<int> m_max_pool_indices;

#ifdef LBANN_HAS_CUDNN
  /** Pooling descriptor. */
  cudnnPoolingDescriptor_t m_pooling_cudnn_desc;
  /** Tensor cuDNN descriptors. */
  cudnn::data_parallel_layer_tensor_manager m_tensors_cudnn_desc;
#endif // LBANN_HAS_CUDNN

  friend class unpooling_layer<T_layout, Dev>;

public:

  pooling_layer(lbann_comm *comm,
                int num_data_dims,
                int pool_dim,
                int pad,
                int stride,
                pool_mode mode)
    : pooling_layer(comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, pool_dim),
                    std::vector<int>(num_data_dims, pad),
                    std::vector<int>(num_data_dims, stride),
                    mode) {}

  pooling_layer(lbann_comm *comm,
                int num_data_dims,
                std::vector<int> pool_dims,
                std::vector<int> pads,
                std::vector<int> strides,
                pool_mode mode)
    : transform_layer(comm),
      m_pool_mode(mode),
      m_pool_dims(pool_dims),
      m_pads(pads),
      m_strides(strides)
#ifdef LBANN_HAS_CUDNN
    , m_pooling_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(this)
#endif // LBANN_HAS_CUDNN
  {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "pooling only supports DATA_PARALLEL");

    // Initialize input dimensions and pooling parameters
    m_pool_size = std::accumulate(m_pool_dims.begin(),
                                  m_pool_dims.end(),
                                  1,
                                  std::multiplies<int>());

  }

  pooling_layer(const pooling_layer& other)
    : transform_layer(other),
      m_pool_mode(other.m_pool_mode),
      m_pool_dims(other.m_pool_dims),
      m_pool_size(other.m_pool_size),
      m_pads(other.m_pads),
      m_strides(other.m_strides),
      m_max_pool_indices(other.m_max_pool_indices)
#ifdef LBANN_HAS_CUDNN
    , m_pooling_cudnn_desc(nullptr),
      m_tensors_cudnn_desc(other.m_tensors_cudnn_desc)
#endif // LBANN_HAS_CUDNN
  {
#ifdef LBANN_HAS_CUDNN
    copy_pooling_cudnn_desc(other.m_pooling_cudnn_desc, m_pooling_cudnn_desc);
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
  }

  pooling_layer& operator=(const pooling_layer& other){
    transform_layer::operator=(other);
    m_pool_mode = other.m_pool_mode;
    m_pool_dims = other.m_pool_dims;
    m_pool_size = other.m_pool_size;
    m_pads = other.m_pads;
    m_strides = other.m_strides;
    m_max_pool_indices = other.m_max_pool_indices;
#ifdef LBANN_HAS_CUDNN
    copy_pooling_cudnn_desc(other.m_pooling_cudnn_desc, m_pooling_cudnn_desc);
    m_tensors_cudnn_desc = other.m_tensors_cudnn_desc;
    m_tensors_cudnn_desc.set_layer(this);
#endif // LBANN_HAS_CUDNN
    return *this;
  }

  pooling_layer* copy() const override { return new pooling_layer(*this); }
  std::string get_type() const override { return "pooling"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " pooling; num_data_dims: "
    + std::to_string(m_pool_dims.size()) + " pool_dims: ";
    for (size_t h=0; h<this->m_pool_dims.size(); h++) {
      s << this->m_pool_dims[h] << " ";
    }
    s << " pads: ";
    for (size_t h=0; h<this->m_pads.size(); h++) {
      s << this->m_pads[h] << " ";
    }
    s << " strides: ";
    for (size_t h=0; h<this->m_strides.size(); h++) {
      s << this->m_strides[h] << " ";
    }
    s << " pool_mode: " << get_pool_mode_name(this->m_pool_mode);
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    s << " device alloc: " + this->get_device_allocation_string(get_device_allocation());

    return s.str();
  }

  ~pooling_layer() {
#ifdef LBANN_HAS_CUDNN
    if (m_pooling_cudnn_desc != nullptr) {
      cudnnDestroyPoolingDescriptor(m_pooling_cudnn_desc);
    }
#endif // LBANN_HAS_CUDNN
  }

  void setup_dims() override {

    // Initialize previous neuron tensor dimensions
    transform_layer::setup_dims();

    // Initialize neuron tensor dimensions
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2 * m_pads[i] - m_pool_dims[i] + 1);
      this->m_neuron_dims[i+1] = ((effective_dim + m_strides[i] - 1)
                                  / m_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  /// Initialize GPU objects
  void setup_gpu() override {
    transform_layer::setup_gpu();
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else

    // Set pooling descriptor
    cudnnPoolingMode_t cudnn_pool_mode;
    switch(m_pool_mode) {
    case pool_mode::max:
    #ifndef LBANN_DETERMINISTIC
      cudnn_pool_mode = CUDNN_POOLING_MAX; break;
    #else
      cudnn_pool_mode = CUDNN_POOLING_MAX_DETERMINISTIC; break;
    #endif
    case pool_mode::average:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_INCLUDE_PADDING; break;
    case pool_mode::average_no_pad:
      cudnn_pool_mode = CUDNN_POOLING_AVERAGE_COUNT_EXCLUDE_PADDING; break;
    default:
      std::stringstream err;
      err << "no GPU implementation for pooling mode " << static_cast<int>(m_pool_mode);
      LBANN_ERROR(err.str());
      cudnn_pool_mode = CUDNN_POOLING_MAX;
    }
    CHECK_CUDNN(cudnnCreatePoolingDescriptor(&m_pooling_cudnn_desc));
    CHECK_CUDNN(cudnnSetPoolingNdDescriptor(m_pooling_cudnn_desc,
                                            cudnn_pool_mode,
                                            CUDNN_PROPAGATE_NAN,
                                            m_pool_dims.size(),
                                            m_pool_dims.data(),
                                            m_pads.data(),
                                            m_strides.data()));

#endif // #ifndef LBANN_HAS_CUDNN
  }

  protected:

  void fp_compute() override {
    if(this->using_gpus()) {
      fp_compute_cudnn();
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
      bp_compute_cudnn();
    } else {
      bp_compute_im2col();
    }
  }

 private:

  /// Pooling forward propagation with cuDNN
  void fp_compute_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();
    if (local_input.Height() > 0 && local_input.Width() > 0) {
      const DataType zero = DataType(0);
      const DataType one = DataType(1);
      CHECK_CUDNN(cudnnPoolingForward(cudnn::get_handle(),
                                      m_pooling_cudnn_desc,
                                      &one,
                                      m_tensors_cudnn_desc.get_prev_activations(),
                                      local_input.LockedBuffer(),
                                      &zero,
                                      m_tensors_cudnn_desc.get_activations(),
                                      local_output.Buffer()));
    }
#endif // #ifndef LBANN_HAS_CUDNN
  }

  /// Pooling backward propagation with cuDNN
  void bp_compute_cudnn() {
#ifndef LBANN_HAS_CUDNN
    LBANN_ERROR("cuDNN not detected");
#else
    const auto& local_input = get_local_prev_activations();
    const auto& local_output = get_local_activations();
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();
    if (local_input.Height() > 0 && local_input.Width() > 0) {

      // Useful constants
      const DataType one = DataType(1);
      const DataType zero = DataType(0);

      // Perform backprop on GPU
      CHECK_CUDNN(cudnnPoolingBackward(cudnn::get_handle(),
                                       m_pooling_cudnn_desc,
                                       &one,
                                       m_tensors_cudnn_desc.get_activations(),
                                       local_output.LockedBuffer(),
                                       m_tensors_cudnn_desc.get_prev_error_signals(),
                                       local_gradient_wrt_output.LockedBuffer(),
                                       m_tensors_cudnn_desc.get_prev_activations(),
                                       local_input.LockedBuffer(),
                                       &zero,
                                       m_tensors_cudnn_desc.get_error_signals(),
                                       local_gradient_wrt_input.Buffer()));

    }
#endif // #ifndef LBANN_HAS_CUDNN
  }

  /// Pooling forward propagation with im2col
  void fp_compute_im2col() {
    if(m_pool_mode != pool_mode::max && m_pool_mode != pool_mode::average) {
      LBANN_ERROR("CPU pooling layer only supports max and average pooling");
    }

    // Local matrices
    const auto& local_input = get_local_prev_activations();
    auto& local_output = get_local_activations();

    // Pool parameters
    const int local_width = local_input.Width();
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_channels;

    // Initialize max pool indices if needed
    if(m_pool_mode == pool_mode::max) {
      m_max_pool_indices.assign(this->m_num_neurons * local_width, 0);
    }

    // Initialize matrices
    DMat<Dev> im2col_mat(m_pool_size * num_channels, num_per_output_channel);
    DMat<Dev> input_mat;

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Construct im2col matrix from input
      El::LockedView(input_mat, local_input,
                     El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             &this->m_prev_neuron_dims[1],
             m_pads.data(),
             m_pool_dims.data(),
             m_strides.data());

      if(m_pool_mode == pool_mode::max) {
        // Apply max pooling
        DataType *output_buffer = local_output.Buffer(0, sample);
        int *indices_buffer = &m_max_pool_indices[sample * this->m_num_neurons];
#pragma omp taskloop default(shared)
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            DataType max_entry = im2col_buffer[0];
            int max_index = 0;
            for(int i = 1; i < m_pool_size; ++i) {
              const DataType current_entry = im2col_buffer[i];
              if(current_entry > max_entry) {
                max_entry = current_entry;
                max_index = i;
              }
            }
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = max_entry;
            indices_buffer[output_index] = max_index;
          }
        }
      }

      if(m_pool_mode == pool_mode::average) {
        // Apply average pooling
        DataType *output_buffer = local_output.Buffer(0, sample);
#pragma omp taskloop default(shared)
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            const DataType *im2col_buffer
              = im2col_mat.LockedBuffer(channel*m_pool_size, j);
            DataType output_entry = 0;
            for(int i = 0; i < m_pool_size; ++i) {
              output_entry += im2col_buffer[i];
            }
            output_entry /= m_pool_size;
            const int output_index = j + channel * num_per_output_channel;
            output_buffer[output_index] = output_entry;
          }
        }
      }

    }

  }

  /// Pooling forward propagation with im2col
  void bp_compute_im2col() {
    if(m_pool_mode != pool_mode::max && m_pool_mode != pool_mode::average) {
      LBANN_ERROR("CPU pooling layer only supports max and average pooling");
    }

    // Local matrices
    const auto& local_gradient_wrt_output = get_local_prev_error_signals();
    auto& local_gradient_wrt_input = get_local_error_signals();

    // Pool parameters
    const int local_width = local_gradient_wrt_output.Width();
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_input_channel = this->m_num_neurons / num_channels;

    // Initialize matrices
    CPUMat im2col_mat(m_pool_size * num_channels, num_per_input_channel);
    CPUMat gradient_wrt_input_col;

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Compute gradient w.r.t. im2col matrix for max pooling
      if(m_pool_mode == pool_mode::max) {

        // Clear im2col matrix
        El::Zero(im2col_mat);

        // Copy previous error signal to im2col matrix entries
        // corresponding to max
        const DataType *gradient_wrt_output_buffer
          = local_gradient_wrt_output.LockedBuffer(0, sample);
        const int *indices_buffer
          = &m_max_pool_indices[sample * this->m_num_neurons];
#pragma omp taskloop default(shared)
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_input_channel; ++j) {
            const int input_index = j + channel * num_per_input_channel;
            const int max_index = indices_buffer[input_index];
            DataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            im2col_buffer[max_index]
              = gradient_wrt_output_buffer[input_index];
          }
        }

      }

      // Compute gradient w.r.t. im2col matrix for average pooling
      if(m_pool_mode == pool_mode::average) {
        const DataType *gradient_wrt_output_buffer
          = local_gradient_wrt_output.LockedBuffer(0, sample);
#pragma omp taskloop default(shared)
        for(int channel = 0; channel < num_channels; ++channel) {
          for(int j = 0; j < num_per_input_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(channel*m_pool_size, j);
            const int input_index = j + channel * num_per_input_channel;
            const DataType output_entry
              = gradient_wrt_output_buffer[input_index] / m_pool_size;
            for(int i = 0; i < m_pool_size; ++i) {
              im2col_buffer[i] = output_entry;
            }
          }
        }

      }

      // Compute error signal (i.e. gradient w.r.t. input)
      El::View(gradient_wrt_input_col, local_gradient_wrt_input,
               El::ALL, El::IR(sample));
      col2im(im2col_mat,
             gradient_wrt_input_col,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             &this->m_prev_neuron_dims[1],
             m_pads.data(),
             m_pool_dims.data(),
             m_strides.data());

    }

  }

#ifdef LBANN_HAS_CUDNN
  /** Copy pooling cuDNN descriptor. */
  static void copy_pooling_cudnn_desc(const cudnnPoolingDescriptor_t& src,
                                      cudnnPoolingDescriptor_t& dst) {

    // Create or destroy descriptor if needed
    if(src != nullptr && dst == nullptr) {
        CHECK_CUDNN(cudnnCreatePoolingDescriptor(&dst));
    }
    else if(src == nullptr && dst != nullptr) {
        CHECK_CUDNN(cudnnDestroyPoolingDescriptor(dst));
        dst = nullptr;
    }

    // Copy descriptor data if needed
    if(src != nullptr) {
        cudnnPoolingMode_t mode;
        cudnnNanPropagation_t nan_propagation;
        int num_dims;
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                nullptr,
                                                nullptr,
                                                nullptr));
        std::vector<int> dims(num_dims), pads(num_dims), strides(num_dims);
        CHECK_CUDNN(cudnnGetPoolingNdDescriptor(src,
                                                0,
                                                &mode,
                                                &nan_propagation,
                                                &num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
        CHECK_CUDNN(cudnnSetPoolingNdDescriptor(dst,
                                                mode,
                                                nan_propagation,
                                                num_dims,
                                                dims.data(),
                                                pads.data(),
                                                strides.data()));
    }

  }
#endif // LBANN_HAS_CUDNN

};

} // namespace lbann

#endif // LBANN_LAYER_POOLING_HPP_INCLUDED
