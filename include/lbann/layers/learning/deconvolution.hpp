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
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/// Deconvolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class deconvolution_layer : public base_convolution_layer {
 private:

  friend class lbann_callback_imcomm;

  public:

  deconvolution_layer(int index,
                    lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    int conv_dim,
                    int conv_pad,
                    int conv_stride,
                    weight_initialization init,
                    optimizer *opt,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = NULL)
    : base_convolution_layer(index,
                          comm,
                          num_data_dims,
                          num_output_channels,
                          conv_dim,
                          conv_pad,
                          conv_stride,
                          init,
                          opt,
                          has_bias,
                          cudnn) {}

  deconvolution_layer(int index,
                    lbann_comm *comm,
                    int num_data_dims,
                    int num_output_channels,
                    const int *conv_dims,
                    const int *conv_pads,
                    const int *conv_strides,
                    weight_initialization init,
                    optimizer *opt,
                    bool has_bias = true,
                    cudnn::cudnn_manager *cudnn = NULL)
    : base_convolution_layer(index,
                          comm,
                          num_data_dims,
                          num_output_channels,
                          conv_dims,
                          conv_pads,
                          conv_strides,
                          init,
                          opt,
                          has_bias,
                          cudnn) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "deconvolution only supports DATA_PARALLEL");
    // Setup the data distribution
    initialize_distributed_matrices();
  }

  deconvolution_layer(const deconvolution_layer& other) :
    base_convolution_layer(other) {
  }

  deconvolution_layer& operator=(const deconvolution_layer& other) {
    base_convolution_layer::operator=(other);
    return *this;
  }

  ~deconvolution_layer() {
   //delete matrice in this class
  }

  deconvolution_layer* copy() const { return new deconvolution_layer(*this); }

  std::string get_name() const { return "deconvolution"; }

  void initialize_distributed_matrices() {
    base_convolution_layer::initialize_distributed_matrices<T_layout>();
  }
  virtual data_layout get_data_layout() const { return T_layout; }

  bool reverse_dimensions() { return true; }

  void setup_dims() {
    // Store neuron tensor dimensions
    const std::vector<int> neuron_dims = this->m_neuron_dims;
    
    // Initialize previous neuron tensor dimensions
    base_convolution_layer::setup_dims();

    // Check if previous neuron tensor dimensions are valid
    #ifdef LBANN_DEBUG
    if(this->m_num_neuron_dims != (int) neuron_dims.size()) {
      throw lbann_exception("deconvolution_layer: neuron tensor dimensions are unexpected");
    }
    #endif

    // Initialize neuron tensor dimensions
    this->m_neuron_dims = neuron_dims;
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      //Use Caffe formula stride*(input-1) + filter -2*pad
      this->m_neuron_dims[i+1] = this->m_conv_strides[i]*(this->m_prev_neuron_dims[i+1] - 1)
                                 + this->m_conv_dims[i] - 2*this->m_conv_pads[i];
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Get size of deconvolutional filters
    this->m_conv_size = std::accumulate(this->m_conv_dims.begin(),
                                  this->m_conv_dims.end(),
                                  this->m_prev_neuron_dims[0] * this->m_neuron_dims[0],
                                  std::multiplies<int>());
  }

  void setup_data() {
    base_convolution_layer::setup_data();

  }

  void setup_views() {
    base_convolution_layer::setup_views();
  }


 protected:

  void fp_compute() {
    if(this->m_using_gpus) {
      bp_compute_cudnn();
    } else {
      bp_compute_im2col();
    }
  }

  void bp_compute() {
    if(this->m_using_gpus) {
      fp_compute_cudnn();
    } else {
      fp_compute_im2col();
    }
  }



};
}

#endif // LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
