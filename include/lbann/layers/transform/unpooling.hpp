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
//
// lbann_layer_unpooling .hpp .cpp - Unpooling Layer (max, average)
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_UNPOOLING_HPP_INCLUDED
#define LBANN_LAYER_UNPOOLING_HPP_INCLUDED

#include <vector>
#include "lbann/base.hpp"
#include "lbann/layers/transform/pooling.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/im2col.hpp"

namespace lbann {

/// Unpooling layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class unpooling_layer : public transform {
 private:

  /// Unpooling mode
  const pool_mode m_pool_mode;

  /// Unpooling window dimensions
  std::vector<int> m_pool_dims;
  /// Unpooling padding
  std::vector<int> m_pool_pads;
  /// Unpooling strides
  std::vector<int> m_pool_strides;
  /// Size of unpooling window
  int m_pool_size;
 
  /// max pooling mask
  ElMat* m_max_pool_mask; 


 public:

  unpooling_layer(int index,
                lbann_comm *comm,
                int num_data_dims,
                int pool_dim,
                int pool_pad,
                int pool_stride,
                pool_mode _pool_mode,
                pooling_layer<T_layout>* p_layer,
                cudnn::cudnn_manager *cudnn = NULL)
    : unpooling_layer(index,
                    comm,
                    num_data_dims,
                    std::vector<int>(num_data_dims, pool_dim).data(),
                    std::vector<int>(num_data_dims, pool_pad).data(),
                    std::vector<int>(num_data_dims, pool_stride).data(),
                    _pool_mode,
                    p_layer,
                    cudnn) {}

  /// Constructor
  unpooling_layer(int index,
                lbann_comm *comm,
                int num_data_dims,
                const int *pool_dims,
                const int *pool_pads,
                const int *pool_strides,
                pool_mode _pool_mode,
                pooling_layer<T_layout>* p_layer,
                cudnn::cudnn_manager *cudnn = NULL)
    : transform(index, comm),
      m_pool_mode(_pool_mode) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "unpooling only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();

    // Pooling layer mask matrices
    m_max_pool_mask = p_layer->get_max_pool_mask();
    // Initialize input dimensions and unpooling parameters
    m_pool_dims.assign(pool_dims, pool_dims+num_data_dims);
    m_pool_size = std::accumulate(m_pool_dims.begin(),
                                  m_pool_dims.end(),
                                  1,
                                  std::multiplies<int>());
    m_pool_pads.assign(pool_pads, pool_pads+num_data_dims);
    m_pool_strides.assign(pool_strides, pool_strides+num_data_dims);

    this->m_using_gpus = false; //GPU not yet supported

  }

  unpooling_layer(const unpooling_layer&) = default;
  unpooling_layer& operator=(const unpooling_layer&) = default;

  /// Destructor
  ~unpooling_layer() {
  }

  unpooling_layer* copy() const { return new unpooling_layer(*this); }

  std::string get_name() const { return "unpooling"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {

    // Initialize previous neuron tensor dimensions
    transform::setup_dims();

    // Initialize neuron tensor dimensions
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      const int effective_dim = (this->m_prev_neuron_dims[i+1]
                                 + 2*m_pool_pads[i] - m_pool_dims[i] + 1);
      this->m_neuron_dims[i+1] = ((effective_dim + m_pool_strides[i] - 1)
                                  / m_pool_strides[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      throw lbann_exception("unpooling_layer fp_compute: GPU version not yet implemented");
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      throw lbann_exception("unpooling_layer bp_compute: GPU version not yet implemented");
    } else {
      bp_compute_im2col();
    }
  }

 private:

  /// Unpooling forward propagation with im2col
  void fp_compute_im2col() {

    // Throw exception if unpooling mode is not max unpooling
    if(m_pool_mode != pool_mode::max) {
      throw lbann_exception("unpooling_layer: CPU unpooling layer only implements max and average unpooling");
    }

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();
    Mat& max_pool_local = m_max_pool_mask->Matrix();

    // Output entries are divided amongst channels
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_channels;

    // Initialize im2col matrix
    Mat im2col_mat(m_pool_size * num_channels, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Construct im2col matrix from input
      const Mat input_mat = El::LockedView(prev_activations_local, El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_pool_pads.data(),
             m_pool_dims.data(),
             m_pool_strides.data());

      // Apply max unpooling
      if(m_pool_mode == pool_mode::max) {
        DataType *output_buffer = activations_local.Buffer(0, sample);
        DataType *mask_buffer = max_pool_local.Buffer(0,sample);
        #pragma omp parallel for collapse(2)
        for(int c = 0; c < num_channels; ++c) {
          for(int j = 0; j < num_per_output_channel; ++j) {
            DataType *im2col_buffer = im2col_mat.Buffer(c*m_pool_size, j);
            const int index = j + c * num_per_output_channel;
            const int mask_index = mask_buffer[index];
            output_buffer[mask_index] = im2col_buffer[index];
          }
        }
      }

    }

  }

  /// Unpooling backward  propagation with im2col
  void bp_compute_im2col() {

    // Throw exception if unpooling mode is not max pooling
    if(m_pool_mode != pool_mode::max) {
      throw lbann_exception("unpooling_layer: CPU unpooling layer only implements max unpooling");
    }

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    const Mat& prev_error_signal_local = this->m_prev_error_signal->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();
    Mat& max_pool_local = m_max_pool_mask->LockedMatrix();

    // Output entries are divided amongst channels
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_neurons / num_channels;

    // Initialize im2col matrix
    Mat im2col_mat(m_pool_size * num_channels, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < prev_activations_local.Width(); ++sample) {

      // Compute gradient w.r.t. im2col matrix for max unpooling
      if(m_pool_mode == pool_mode::max) {

        // Construct im2col matrix from input
        const Mat input_mat = El::LockedView(prev_activations_local, El::ALL, El::IR(sample));
        im2col(input_mat,
               im2col_mat,
               num_channels,
               this->m_num_prev_neuron_dims - 1,
               this->m_prev_neuron_dims.data() + 1,
               m_pool_pads.data(),
               m_pool_dims.data(),
               m_pool_strides.data());

        // Copy previous error signal to im2col matrix entries
        // corresponding to max
        const DataType *prev_error_signal_buffer
          = prev_error_signal_local.LockedBuffer(0, sample);
        DataType *mask_buffer = max_pool_local.LockedBuffer(0,sample);
        #pragma omp parallel for collapse(2)
        for(int j = 0; j < num_per_output_channel; ++j) {
          for(int c = 0; c < num_channels; ++c) {
            DataType *im2col_buffer = im2col_mat.Buffer(c*m_pool_size, j);
            const int index = j + c * num_per_output_channel;
            const int mask_index = mask_buffer[index];
            im2col_buffer[index] = prev_error_signal_buffer[mask_index];

          }
        }

      }

      // Compute error signal (i.e. gradient w.r.t. input)
      Mat output_mat = El::View(error_signal_local, El::ALL, El::IR(sample));
      col2im(im2col_mat,
             output_mat,
             num_channels,
             this->m_num_prev_neuron_dims - 1,
             this->m_prev_neuron_dims.data() + 1,
             m_pool_pads.data(),
             m_pool_dims.data(),
             m_pool_strides.data());

    }

  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_POOLING_HPP_INCLUDED
