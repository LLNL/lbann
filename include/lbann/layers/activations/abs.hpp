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

#ifndef LBANN_LAYER_ACTIVATION_ABS_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_ABS_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"

namespace lbann {

#ifdef LBANN_HAS_GPU
namespace abs_cuda {
  void fp(int height, int width,
          const DataType* input,
          int input_leading_dim,
          DataType* output,
          int output_leading_dim);
  void bp(int height, int width,
          const DataType* input,
          int input_leading_dim,
          const DataType* gradient_wrt_output,
          int gradient_wrt_output_leading_dim,
          DataType* gradient_wrt_input,
          int gradient_wrt_input_leading_dim);
} // namespace abs_cuda
#endif // LBANN_HAS_GPU

/** Absolute value. */
template <data_layout T_layout, El::Device Dev>
class abs_layer : public entrywise_activation_layer {
 public:
  abs_layer(lbann_comm *comm) : entrywise_activation_layer(comm) { }
  abs_layer* copy() const override { return new abs_layer(*this); }
  std::string get_type() const override { return "abs"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }
  std::string get_description() const override {
    return std::string {}
      + " abs" + " dataLayout: "
      + this->get_data_layout_string(get_data_layout());
  }

 protected:

  DataType activation(DataType x) const override {
    return std::abs(x);
  }

  DataType activation_derivative(DataType x) const override {
    if (x > DataType(0)) {
      return 1;
    } else if (x < DataType(0)) {
      return -1;
    } else {
      return 0;
    }
  }

  void fp_compute_gpu() override {
#ifndef LBANN_HAS_GPU
    LBANN_ERROR("CUDA not detected");
#else
    abs_cuda::fp(get_output_size(),
                     get_prev_activations().LocalWidth(),
                     get_prev_activations().LockedBuffer(),
                     get_prev_activations().LDim(),
                     get_activations().Buffer(),
                     get_activations().LDim());
#endif // LBANN_HAS_GPU
  }

  void bp_compute_gpu() override {
#ifndef LBANN_HAS_GPU
    LBANN_ERROR("CUDA not detected");
#else
    abs_cuda::bp(get_output_size(),
                 get_prev_activations().LocalWidth(),
                 get_prev_activations().LockedBuffer(),
                 get_prev_activations().LDim(),
                 get_prev_error_signals().LockedBuffer(),
                 get_prev_error_signals().LDim(),
                 get_error_signals().Buffer(),
                 get_error_signals().LDim());
#endif // LBANN_HAS_GPU
  }
  
};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_ABS_HPP_INCLUDED
