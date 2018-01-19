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
// sum.hpp - Sum layer
////////////////////////////////////////////////////////////////////////////////

#ifndef LBANN_LAYER_SUM_HPP_INCLUDED
#define LBANN_LAYER_SUM_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"
#include "lbann/utils/exception.hpp"

namespace lbann {

/// Sum layer
template <data_layout T_layout = data_layout::DATA_PARALLEL>
class sum_layer : public transform_layer {
 private:

 public:
  /// Constructor
  sum_layer(lbann_comm *comm,
            cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm) {

    // Sum layer has no limit on parents
    m_expected_num_parent_layers = -1;

  #ifdef __LIB_CUDNN
    // Initialize GPU if available
    if(cudnn) {
      this->m_using_gpus = true;
      this->m_cudnn = cudnn;
    }
  #endif // __LIB_CUDNN

  }

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

  sum_layer* copy() const override { return new sum_layer(*this); }

  std::string get_type() const override { return "sum"; }

  data_layout get_data_layout() const override { return T_layout; }

  protected:

  void fp_compute() override {
    if(this->m_using_gpus) {
  #ifndef __LIB_CUDNN
      throw lbann_exception("sum_layer: cuDNN not detected");
  #else
      const int num_gpus = m_cudnn->get_num_gpus();
      auto& output_d = this->m_activations_d[0];
      output_d.zero();
      for (const auto& input_d : this->m_prev_activations_d ) {
        for (int i=0; i<num_gpus; ++i) {
          CHECK_CUBLAS(cublas::geam(this->m_cudnn->get_cublas_handle(i),
                                    CUBLAS_OP_N,
                                    CUBLAS_OP_N,
                                    input_d.get_height(),
                                    this->m_mini_batch_size_per_gpu,
                                    DataType(1),
                                    input_d.get_locked_data(i),
                                    input_d.get_leading_dim(),
                                    DataType(1),
                                    output_d.get_locked_data(i),
                                    output_d.get_leading_dim(),
                                    output_d.get_data(i),
                                    output_d.get_leading_dim()));
        }
      }
  #endif // __LIB_CUDNN
    } else {
      auto& output = get_activations();
      El::Zero(output);
      for (const auto& input : this->m_prev_activations) {
        El::Axpy(DataType(1), *input, output);
      }
    }
  }

  void bp_compute() override {
    if(this->m_using_gpus) {
  #ifndef __LIB_CUDNN
      throw lbann_exception("sum_layer: cuDNN not detected");
  #else
      const int num_gpus = m_cudnn->get_num_gpus();
      const auto& gradient_wrt_output_d = m_prev_error_signals_d[0];
      for (auto& gradient_wrt_input_d : this->m_error_signals_d) {
        for (int i=0; i<num_gpus; ++i) {
          CHECK_CUBLAS(cublas::geam(this->m_cudnn->get_cublas_handle(i),
                                    CUBLAS_OP_N, CUBLAS_OP_N,
                                    gradient_wrt_output_d.get_height(),
                                    this->m_mini_batch_size_per_gpu,
                                    DataType(1),
                                    gradient_wrt_output_d.get_locked_data(i),
                                    gradient_wrt_output_d.get_leading_dim(),
                                    DataType(1),
                                    gradient_wrt_input_d.get_locked_data(i),
                                    gradient_wrt_input_d.get_leading_dim(),
                                    gradient_wrt_input_d.get_data(i),
                                    gradient_wrt_input_d.get_leading_dim()));
        }
      }
  #endif // __LIB_CUDNN
    } else {
      const auto& gradient_wrt_output = get_prev_error_signals();
      for (auto& gradient_wrt_input : this->m_error_signals) {
        El::Axpy(DataType(1), gradient_wrt_output, *gradient_wrt_input);
      }
    }
  }

};

}  // namespace lbann

#endif  // LBANN_LAYER_SUM_HPP_INCLUDED
