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

#ifndef LBANN_LAYER_SUM_HPP_INCLUDED
#define LBANN_LAYER_SUM_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/** Sum layer.
 *  This layer adds input tensors.
 */
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class sum_layer : public transform_layer {
 private:

  /** Scaling term applied to each input tensor.
   *  If these are not provided, the scaling factors are set to one.
   */
  std::vector<DataType> m_scaling_factors;

 public:
  sum_layer(lbann_comm *comm,
            std::vector<DataType> scaling_factors = std::vector<DataType>(),
            cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_scaling_factors(scaling_factors) {

    // Sum layer has no limit on parents
    m_expected_num_parent_layers = -1;

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  sum_layer* copy() const override { return new sum_layer(*this); }
  std::string get_type() const override { return "sum"; }
  data_layout get_data_layout() const override { return T_layout; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
     s << " sum; parents: ";
     for (size_t i=0; i<this->m_parent_layers.size(); i++) {
       s << this->m_parent_layers[i]->get_name() << " " << this->m_parent_layers[i]->get_type() << " ";
     }
     s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
     return s.str();
  }

 protected:

  void setup_data() {
    transform_layer::setup_data();
    if (m_scaling_factors.empty()) {
      m_scaling_factors.assign(get_num_parents(), DataType(1));
    }
    if ((int) m_scaling_factors.size() != get_num_parents()) {
      std::stringstream err;
      err << __FILE__ << " " << __LINE__ << " :: "
          << "layer " << get_name() << " has an invalid number of "
          << "scaling factors "
          << "(found " << m_scaling_factors.size() << ", "
          << "but there are " << get_num_parents() << " parent layers)";
      throw lbann_exception(err.str());
    }
  }

  void fp_compute() override {
    if(this->m_using_gpus) {
  #ifndef LBANN_HAS_CUDNN
      throw lbann_exception("sum_layer: cuDNN not detected");
  #else
      const int num_gpus = m_cudnn->get_num_gpus();
      auto& output_d = this->m_activations_d[0];
      this->m_cudnn->clear_on_gpus(output_d.get_data(),
                                   output_d.get_height(),
                                   this->m_mini_batch_size_per_gpu,
                                   output_d.get_leading_dim());
      for (int parent = 0; parent < get_num_parents(); ++parent) {
        const auto& input_d = this->m_prev_activations_d[parent];
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(gpu)));
          cublas::geam(this->m_cudnn->get_cublas_handle(gpu),
                       CUBLAS_OP_N,
                       CUBLAS_OP_N,
                       input_d.get_height(),
                       this->m_mini_batch_size_per_gpu,
                       m_scaling_factors[parent],
                       input_d.get_locked_data(gpu),
                       input_d.get_leading_dim(),
                       DataType(1),
                       output_d.get_locked_data(gpu),
                       output_d.get_leading_dim(),
                       output_d.get_data(gpu),
                       output_d.get_leading_dim());
        }
      }
  #endif // LBANN_HAS_CUDNN
    } else {
      auto& output = get_activations();
      El::Zero(output);
      for (int parent = 0; parent < get_num_parents(); ++parent) {
        const auto& input = get_prev_activations(parent);
        El::Axpy(m_scaling_factors[parent], input, output);
      }
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
  #ifndef LBANN_HAS_CUDNN
      throw lbann_exception("sum_layer: cuDNN not detected");
  #else
      const int num_gpus = m_cudnn->get_num_gpus();
      const auto& gradient_wrt_output_d = m_prev_error_signals_d[0];
      for (int parent = 0; parent < get_num_parents(); ++parent) {
        auto& gradient_wrt_input_d = this->m_error_signals_d[parent];
        for (int gpu = 0; gpu < num_gpus; ++gpu) {
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(gpu)));
          cublas::geam(this->m_cudnn->get_cublas_handle(gpu),
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       gradient_wrt_input_d.get_height(),
                       this->m_mini_batch_size_per_gpu,
                       m_scaling_factors[parent],
                       gradient_wrt_output_d.get_locked_data(gpu),
                       gradient_wrt_output_d.get_leading_dim(),
                       DataType(1),
                       gradient_wrt_input_d.get_locked_data(gpu),
                       gradient_wrt_input_d.get_leading_dim(),
                       gradient_wrt_input_d.get_data(gpu),
                       gradient_wrt_input_d.get_leading_dim());
        }
      }
  #endif // LBANN_HAS_CUDNN
    } else {
      const auto& gradient_wrt_output = get_prev_error_signals();
      for (int parent = 0; parent < get_num_parents(); ++parent) {
        auto& gradient_wrt_input = get_error_signals(parent);
        El::Axpy(m_scaling_factors[parent], gradient_wrt_output, gradient_wrt_input);
      }
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_SUM_HPP_INCLUDED
