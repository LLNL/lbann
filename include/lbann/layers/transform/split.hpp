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

#ifndef LBANN_LAYER_SPLIT_HPP_INCLUDED
#define LBANN_LAYER_SPLIT_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/cublas_wrapper.hpp"

namespace lbann {

/** Split layer.
 *  This layer can accommodate an arbitrary number of outputs.
 */
  template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class split_layer : public transform_layer {
 private:

 public:

  split_layer(lbann_comm *comm,
              cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm) {

    // Split layer has no limit on children
    m_expected_num_child_layers = -1;

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  split_layer* copy() const override { return new split_layer(*this); }
  std::string get_type() const override { return "split"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

  /** Returns description of ctor params */
  std::string get_description() const override {
    std::stringstream s;
    s << " split; children: ";
    for (size_t h=0; h<this->m_child_layers.size(); h++) {
      s << this->m_child_layers[h]->get_name() << " " << this->m_child_layers[h]->get_type() << " ";
    }
    s << " dataLayout: " << this->get_data_layout_string(get_data_layout());
    return s.str();
  }

  protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
  #ifndef LBANN_HAS_CUDNN
      throw lbann_exception("split_layer: cuDNN not detected");
  #else
      const auto& input_d = this->m_prev_activations_d[0];
      for (auto& output_d : this->m_activations_d ) {
        output_d.locked_view(input_d);
      }
  #endif // LBANN_HAS_CUDNN
    } else {
      const auto& input = get_prev_activations();
      for (auto& output : this->m_activations) {
        El::LockedView(*output, input);
      }
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
  #ifndef LBANN_HAS_CUDNN
      throw lbann_exception("split_layer: cuDNN not detected");
  #else
      const int num_gpus = m_cudnn->get_num_gpus();
      auto& gradient_wrt_input_d = m_error_signals_d[0];
      for (const auto& gradient_wrt_output_d : this->m_prev_error_signals_d) {
        for (int i=0; i<num_gpus; ++i) {
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu(i)));
          cublas::geam(this->m_cudnn->get_cublas_handle(i),
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       gradient_wrt_input_d.get_height(),
                       this->m_mini_batch_size_per_gpu,
                       DataType(1),
                       gradient_wrt_output_d.get_locked_data(i),
                       gradient_wrt_output_d.get_leading_dim(),
                       DataType(1),
                       gradient_wrt_input_d.get_locked_data(i),
                       gradient_wrt_input_d.get_leading_dim(),
                       gradient_wrt_input_d.get_data(i),
                       gradient_wrt_input_d.get_leading_dim());
        }
      }
  #endif // LBANN_HAS_CUDNN
    } else {
      auto& gradient_wrt_input = get_error_signals();
      for (const auto& gradient_wrt_output : this->m_prev_error_signals) {
        El::Axpy(DataType(1), *gradient_wrt_output, gradient_wrt_input);
      }
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_SPLIT_HPP_INCLUDED
