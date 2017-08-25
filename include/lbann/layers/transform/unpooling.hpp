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
 
  /// Corresponding pooling layer
  pooling_layer<T_layout>* m_pooling_layer;

 public:

  /// Constructor
  unpooling_layer(int index,
                  lbann_comm *comm,
                  pooling_layer<T_layout>* p_layer)
    : transform(index, comm),
      m_pooling_layer(p_layer) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "unpooling only supports DATA_PARALLEL");

    // Setup the data distribution
    initialize_distributed_matrices();

    // Check that pooling layer is valid
    if(m_pooling_layer->m_pool_mode != pool_mode::max) {
      throw lbann_exception("unpooling_layer: currently only max unpooling layer is implemented");
    }
    if(m_pooling_layer->using_gpus()) {
      throw lbann_exception("unpooling_layer: GPU version not yet implemented");
    }

  }

  unpooling_layer(const unpooling_layer&) = default;
  unpooling_layer& operator=(const unpooling_layer&) = default;
  ~unpooling_layer() = default;

  unpooling_layer* copy() const { return new unpooling_layer(*this); }

  std::string get_name() const { return "unpooling"; }

  virtual inline void initialize_distributed_matrices() {
    transform::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  void setup_dims() {

    // Initialize previous neuron tensor dimensions
    transform::setup_dims();

    // Check that previous neuron tensor is valid
    if(this->m_num_prev_neurons != m_pooling_layer->m_num_neurons
       || this->m_num_prev_neuron_dims != m_pooling_layer->m_num_neuron_dims
       || this->m_prev_neuron_dims != m_pooling_layer->m_neuron_dims) {
      throw lbann_exception("unpooling_layer: previous neuron tensor must match neuron tensor of corresponding pooling layer");
    }

    // Initialize neuron tensor based on corresponding pooling layer
    this->m_num_neurons = m_pooling_layer->m_num_prev_neurons;
    this->m_num_neuron_dims = m_pooling_layer->m_num_prev_neuron_dims;
    this->m_neuron_dims = m_pooling_layer->m_prev_neuron_dims;

  }

  protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      throw lbann_exception("unpooling_layer: GPU version not yet implemented");
    } else {
      fp_compute_im2col();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      throw lbann_exception("unpooling_layer: GPU version not yet implemented");
    } else {
      bp_compute_im2col();
    }
  }

 private:

  /// Unpooling forward propagation with im2col
  void fp_compute_im2col() {

    // Get local matrices
    const Mat& prev_activations_local = this->m_prev_activations->LockedMatrix();
    Mat& activations_local = this->m_activations_v->Matrix();

    // Get parameters
    const int local_width = prev_activations_local.Width();
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_input_channel = this->m_num_prev_neurons / num_channels;
    const int pool_size = m_pooling_layer->m_pool_size;

    // Initialize im2col matrix
    Mat im2col_mat(pool_size * num_channels, num_per_input_channel);

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Clear im2col matrix
      El::Zero(im2col_mat);

      // Populate im2col matrix
      const DataType *prev_activations_buffer
        = prev_activations_local.LockedBuffer(0, sample);
      const int *indices_buffer
        = &m_pooling_layer->m_max_pool_indices[sample * this->m_num_prev_neurons];
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        for(int j = 0; j < num_per_input_channel; ++j) {
          const int input_index = j + channel * num_per_input_channel;
          const int max_index = indices_buffer[input_index];
          DataType *im2col_buffer
            = im2col_mat.Buffer(channel * pool_size, j);
          im2col_buffer[max_index]
            = prev_activations_buffer[input_index];
        }
      }

      // Convert im2col matrix to output matrix
      Mat output_mat = El::View(activations_local, El::ALL, El::IR(sample));
      col2im(im2col_mat,
             output_mat,
             num_channels,
             this->m_num_neuron_dims - 1,
             this->m_neuron_dims.data() + 1,
             m_pooling_layer->m_pool_pads.data(),
             m_pooling_layer->m_pool_dims.data(),
             m_pooling_layer->m_pool_strides.data(),
             static_cast<const DataType&(*)(const DataType&,const DataType&)>(&std::max<DataType>));

    }

  }

  /// Unpooling backward propagation with im2col
  void bp_compute_im2col() {

    // Get local matrices
    const Mat& prev_error_signal_local = this->m_prev_error_signal->LockedMatrix();
    Mat& error_signal_local = this->m_error_signal_v->Matrix();

    // Get parameters
    const int local_width = prev_error_signal_local.Width();
    const int num_channels = this->m_prev_neuron_dims[0];
    const int num_per_output_channel = this->m_num_prev_neurons / num_channels;
    const int pool_size = m_pooling_layer->m_pool_size;

    // Initialize im2col matrix
    Mat im2col_mat(pool_size * num_channels, num_per_output_channel);

    // Iterate through data samples
    for(int sample = 0; sample < local_width; ++sample) {

      // Construct im2col matrix from input
      const Mat input_mat = El::LockedView(prev_error_signal_local,
                                           El::ALL, El::IR(sample));
      im2col(input_mat,
             im2col_mat,
             num_channels,
             this->m_num_neuron_dims - 1,
             this->m_neuron_dims.data() + 1,
             m_pooling_layer->m_pool_pads.data(),
             m_pooling_layer->m_pool_dims.data(),
             m_pooling_layer->m_pool_strides.data());

      // Propagate error signal based on pooling layer
      DataType *output_buffer = error_signal_local.Buffer(0, sample);
      const int *indices_buffer
        = &m_pooling_layer->m_max_pool_indices[sample * this->m_num_prev_neurons];
      #pragma omp parallel for
      for(int channel = 0; channel < num_channels; ++channel) {
        for(int j = 0; j < num_per_output_channel; ++j) {
          const int output_index = j + channel * num_per_output_channel;
          const int max_index = indices_buffer[output_index];
          DataType *im2col_buffer
            = im2col_mat.Buffer(channel * pool_size, j);
          output_buffer[output_index] = im2col_buffer[max_index];
        }
      }

    }

  }

};

template<> inline void unpooling_layer<data_layout::MODEL_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::MODEL_PARALLEL>();
}

template<> inline void unpooling_layer<data_layout::DATA_PARALLEL>::initialize_distributed_matrices() {
  transform::initialize_distributed_matrices<data_layout::DATA_PARALLEL>();
}

}  // namespace lbann

#endif  // LBANN_LAYER_POOLING_HPP_INCLUDED
