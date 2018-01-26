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

#ifndef LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
#define LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/learning/base_convolution.hpp"
#include "lbann/layers/layer.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/random.hpp"
#include "lbann/utils/timer.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/// Deconvolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class deconvolution_layer : public base_convolution_layer {
 private:

  friend class lbann_callback_imcomm;

  public:

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      int conv_dim,
                      int pad,
                      int stride,
                      bool has_bias = true,
                      cudnn::cudnn_manager *cudnn = nullptr)
    : deconvolution_layer(comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim),
                          std::vector<int>(num_data_dims, pad),
                          std::vector<int>(num_data_dims, stride),
                          has_bias,
                          cudnn) {}

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      std::vector<int> conv_dims,
                      std::vector<int> pads,
                      std::vector<int> strides,
                      bool has_bias = true,
                      cudnn::cudnn_manager *cudnn = nullptr)
    : base_convolution_layer(comm,
                             num_data_dims,
                             num_output_channels,
                             conv_dims,
                             pads,
                             strides,
                             has_bias,
                             cudnn) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");

    // Use GPUs if cuDNN manager is available
    if(this->m_cudnn != nullptr) {
      this->m_using_gpus = true;
    }

  }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " deconvolution; conv_dims: ";
    for (size_t h=0; h<this->m_kernel_dims.size(); h++) {
      s << this->m_kernel_dims[h] << " ";
    }
    s << " pads: ";
    for (size_t h=0; h<this->m_pads.size(); h++) {
      s << this->m_pads[h] << " ";
    }
    s << " strides: ";
    for (size_t h=0; h<this->m_strides.size(); h++) {
      s << this->m_strides[h] << " ";
    }
    s << " num_output_channels: " << this->m_neuron_dims[0]
      << " has_bias: " << this->m_bias_scaling_factor
      << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  deconvolution_layer* copy() const override { return new deconvolution_layer(*this); }

  std::string get_type() const override { return "deconvolution"; }

  data_layout get_data_layout() const override { return T_layout; }

  void setup_dims() override {

    // Initialize previous neuron tensor dimensions
    base_convolution_layer::setup_dims();

    // Initialize deconvolution kernel dimensions
    // Note that unlike the convolutional kernel, the previous layer's
    // number of channels is now the leading position -- keep in mind
    // that deconvolution is the transpose of a convolution
    this->m_kernel_dims.insert(this->m_kernel_dims.begin(),
                               this->m_prev_neuron_dims[0]);

    // Check if previous neuron tensor dimensions are valid
  #ifdef LBANN_DEBUG
    if(this->m_num_neuron_dims != (int) this->m_kernel_dims.size() - 1) {
      throw lbann_exception("deconvolution_layer: previous neuron tensor dimensions are unexpected");
    }
  #endif

    // Initialize neuron tensor dimensions
    this->m_neuron_dims[0] = this->m_kernel_dims[1];
    for(int i=0; i<this->m_num_neuron_dims-1; ++i) {
      this->m_neuron_dims[i+1]
        = ((this->m_prev_neuron_dims[i+1]-1) * this->m_strides[i]
           + this->m_kernel_dims[i+2] - 2*this->m_pads[i]);
    }
    this->m_num_neurons = std::accumulate(this->m_neuron_dims.begin(),
                                          this->m_neuron_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Get size of convolutional kernel
    this->m_kernel_size = std::accumulate(m_kernel_dims.begin(),
                                          m_kernel_dims.end(),
                                          1,
                                          std::multiplies<int>());

  }

  void setup_data() override {
    base_convolution_layer::setup_data();
    this->m_weights[0]->setup(m_kernel_size / this->m_prev_neuron_dims[0],
                              this->m_prev_neuron_dims[0],
                              El::STAR, El::STAR);
    El::Zeros(this->m_kernel_gradient,
              this->m_weights[0]->get_height(),
              this->m_weights[0]->get_width());
  }

 protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
      apply_transposed_convolution_cudnn(true);
      apply_bias_cudnn();
    } else {
      apply_transposed_convolution_im2col(true);
      apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
      apply_convolution_cudnn(false);
      compute_gradients_cudnn(true);
    } else {
      apply_convolution_im2col(false);
      compute_gradients_im2col(true);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
