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

#ifndef LBANN_LAYER_ACTIVATION_SIGMOID_HPP_INCLUDED
#define LBANN_LAYER_ACTIVATION_SIGMOID_HPP_INCLUDED

#include "lbann/layers/activations/activation.hpp"
#include "lbann/utils/cudnn_wrapper.hpp"

// Output is strictly in (0,1) to avoid numerical issues
#define LBANN_ENABLE_SIGMOID_CUTOFF

namespace lbann {

#ifdef LBANN_HAS_CUDNN
namespace sigmoid_cuda {
  void fp(cudnn::cudnn_manager& cudnn,
          int height,
          int width_per_gpu,
          const std::vector<DataType*>& input,
          int input_leading_dim,
          std::vector<DataType*>& output,
          int output_leading_dim,
          DataType cutoff);
  void bp(cudnn::cudnn_manager& cudnn,
          int height, int width_per_gpu,
          const std::vector<DataType*>& input,
          int input_leading_dim,
          const std::vector<DataType*>& gradient_wrt_output,
          int gradient_wrt_output_leading_dim,
          std::vector<DataType*>& gradient_wrt_input,
          int gradient_wrt_input_leading_dim,
          DataType cutoff);
} // namespace sigmoid_cuda
#endif // LBANN_HAS_CUDNN

/** Sigmoid activation function.
 *  See https://en.wikipedia.org/wiki/Sigmoid_function
 */
template <data_layout T_layout>
class sigmoid_layer : public entrywise_activation_layer {

 private:

  /** Cutoff value for inputs.
   *  If sigmoid cutoff is enabled, this cutoff value ensures that the
   *  output is strictly in (0,1).
   */
  DataType m_cutoff;

 public:
  sigmoid_layer(lbann_comm *comm,
                cudnn::cudnn_manager *cudnn = nullptr)
    : entrywise_activation_layer(comm) {

    // Compute cutoff value to ensure output is in (0,1)
    const DataType eps = std::numeric_limits<DataType>::epsilon();
    m_cutoff = std::log(DataType(1) - eps) - std::log(eps);

  #ifdef LBANN_HAS_CUDNN
    // Activate GPU if needed
    if (cudnn && T_layout == data_layout::DATA_PARALLEL) {
      this->m_cudnn = cudnn;
      this->m_using_gpus = true;
    }
  #endif // LBANN_HAS_CUDNN

  }

  sigmoid_layer* copy() const override { return new sigmoid_layer(*this); }
  std::string get_type() const override { return "sigmoid"; }
  data_layout get_data_layout() const override { return T_layout; }

  std::string get_description() const override {
    return std::string{} +
     " sigmoid dataLayout: " + this->get_data_layout_string(get_data_layout());
  }

 protected:

  DataType activation(DataType x) const override {
  #ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    // Ensure -m_cutoff <= x <= m_cutoff
    x = std::min(std::max(x, -m_cutoff), m_cutoff);
  #endif // LBANN_ENABLE_SIGMOID_CUTOFF
    return 1 / (DataType(1) + std::exp(-x));
  }

  DataType activation_derivative(DataType x) const override {
  #ifdef LBANN_ENABLE_SIGMOID_CUTOFF
    if (x < -m_cutoff || x > m_cutoff) { return DataType(0); }
  #endif // LBANN_ENABLE_SIGMOID_CUTOFF
    const auto sigx = activation(x);
    return sigx * (DataType(1) - sigx);
  }

  void fp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("sigmoid_layer: cuDNN not detected");
  #else
    sigmoid_cuda::fp(*m_cudnn,
                     get_num_neurons(),
                     m_mini_batch_size_per_gpu,
                     m_prev_activations_d[0].get_locked_data(),
                     m_prev_activations_d[0].get_leading_dim(),
                     m_activations_d[0].get_data(),
                     m_activations_d[0].get_leading_dim(),
                     m_cutoff);
  #endif // LBANN_HAS_CUDNN
  }

  void bp_compute_gpu() override {
  #ifndef LBANN_HAS_CUDNN
    throw lbann_exception("sigmoid_layer: cuDNN not detected");
  #else
    sigmoid_cuda::bp(*m_cudnn,
                     get_num_neurons(),
                     m_mini_batch_size_per_gpu,
                     m_prev_activations_d[0].get_locked_data(),
                     m_prev_activations_d[0].get_leading_dim(),
                     m_prev_error_signals_d[0].get_locked_data(),
                     m_prev_error_signals_d[0].get_leading_dim(),
                     m_error_signals_d[0].get_data(),
                     m_error_signals_d[0].get_leading_dim(),
                     m_cutoff);
  #endif // LBANN_HAS_CUDNN
  }

};

} // namespace lbann

#endif // LBANN_LAYER_ACTIVATION_SIGMOID_HPP_INCLUDED
