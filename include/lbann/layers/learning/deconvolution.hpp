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
#include "lbann/utils/exception.hpp"

namespace lbann {

// Forward declaration.
class lbann_callback_imcomm;

/// Deconvolution layer
template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class deconvolution_layer : public base_convolution_layer<Dev> {
private:

  friend class lbann_callback_imcomm;

public:

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      int conv_dim,
                      int pad,
                      int stride,
                      int dilation,
                      int groups,
                      bool has_bias = true)
    : deconvolution_layer(comm,
                          num_data_dims,
                          num_output_channels,
                          std::vector<int>(num_data_dims, conv_dim),
                          std::vector<int>(num_data_dims, pad),
                          std::vector<int>(num_data_dims, stride),
                          std::vector<int>(num_data_dims, dilation),
                          groups,
                          has_bias) {}

  deconvolution_layer(lbann_comm *comm,
                      int num_data_dims,
                      int num_output_channels,
                      std::vector<int> conv_dims,
                      std::vector<int> pads,
                      std::vector<int> strides,
                      std::vector<int> dilations,
                      int groups,
                      bool has_bias = true)
    : base_convolution_layer<Dev>(comm,
                                  num_data_dims,
                                  num_output_channels,
                                  conv_dims,
                                  pads,
                                  strides,
                                  dilations,
                                  groups,
                                  has_bias) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "convolution only supports DATA_PARALLEL");

  }

  deconvolution_layer* copy() const override { return new deconvolution_layer(*this); }

  std::string get_type() const override { return "deconvolution"; }

  data_layout get_data_layout() const override { return T_layout; }

  El::Device get_device_allocation() const override { return Dev; }

  void setup_dims() override {
    base_convolution_layer<Dev>::setup_dims();

    // Get tensor dimensions
    auto& kernel_dims = this->m_kernel_dims;
    const auto& input_dims = this->get_input_dims();
    auto output_dims = input_dims;

    bool nonunit_dilation = false;
    for (const auto& d : this->m_dilations) {
      if (d != 1) {
        nonunit_dilation = true;
        break;
      }
    }
    if (nonunit_dilation) {
      std::stringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << " does not support dilated convolutions";
      LBANN_ERROR(err.str());
    }
    if (this->m_num_groups != 1) {
      std::stringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << " does not support grouped convolutions";
      LBANN_ERROR(err.str());
    }
    // Initialize deconvolution kernel dimensions
    // Note: Unlike the convolutional kernel, the previous layer's
    // number of channels is now the leading position -- keep in mind
    // that deconvolution is the transpose of a convolution.
    kernel_dims.insert(kernel_dims.begin(), input_dims[0]);
    this->m_kernel_size = std::accumulate(kernel_dims.begin(),
                                          kernel_dims.end(),
                                          1,
                                          std::multiplies<int>());

    // Check if input tensor dimensions are valid
    if (input_dims.size() != kernel_dims.size() - 1) {
      std::stringstream err;
      err << this->get_type() << " layer \"" << this->get_name() << "\" "
          << "has an input tensor with "
          << input_dims.size() << " dimensions "
          << "and a convolution kernel with "
          << kernel_dims.size() << " dimensions";
      LBANN_ERROR(err.str());
    }

    // Initialize output tensor dimensions
    output_dims[0] = kernel_dims[1];
    for (size_t i = 0; i < output_dims.size() - 1; ++i) {
      const auto& stride = this->m_strides[i];
      const auto& pad = this->m_pads[i];
      output_dims[i+1] = ((input_dims[i+1] - 1) * stride
                          + kernel_dims[i+2] - 2 * pad);
    }
    this->set_output_dims(output_dims);

  }

protected:

  void fp_compute() override {
    if(this->using_gpus()) {
      base_convolution_layer<Dev>::apply_transposed_convolution_cudnn(true);
      base_convolution_layer<Dev>::apply_bias_cudnn();
    } else {
      base_convolution_layer<Dev>::apply_transposed_convolution_im2col(true);
      base_convolution_layer<Dev>::apply_bias_cpu();
    }
  }

  void bp_compute() override {
    if(this->using_gpus()) {
      base_convolution_layer<Dev>::compute_gradients_cudnn(true);
      base_convolution_layer<Dev>::apply_convolution_cudnn(false);
    } else {
      base_convolution_layer<Dev>::compute_gradients_im2col(true);
      base_convolution_layer<Dev>::apply_convolution_im2col(false);
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_DECONVOLUTION_HPP_INCLUDED
