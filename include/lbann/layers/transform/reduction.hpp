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

#ifndef LBANN_LAYER_REDUCTION_HPP_INCLUDED
#define LBANN_LAYER_REDUCTION_HPP_INCLUDED

#include <vector>
#include "lbann/layers/transform/transform.hpp"

namespace lbann {

enum class reduction_mode {INVALID, SUM, AVERAGE};

/** Reduction layer. */
  template <data_layout T_layout = data_layout::DATA_PARALLEL, El::Device Dev = El::Device::CPU>
class reduction_layer : public transform_layer {
 private:

  /** Reduction mode. */
  const reduction_mode m_mode;

 public:

  reduction_layer(lbann_comm *comm,
                  reduction_mode mode,
                  cudnn::cudnn_manager *cudnn = nullptr)
    : transform_layer(comm),
      m_mode(mode) {
    static_assert(T_layout == data_layout::DATA_PARALLEL,
                  "reduction currently only supports DATA_PARALLEL");
    if (mode == reduction_mode::INVALID) {
      LBANN_ERROR("invalid reduction mode");
    }

    // Initialize neuron tensor dimensions
    this->m_num_neurons = 1;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims = {1};

  #ifdef LBANN_HAS_CUDNN
    // Initialize GPU if available
    if (cudnn != nullptr) {
      this->m_cudnn = cudnn;
    }
  #endif // LBANN_HAS_CUDNN

  }

  reduction_layer* copy() const override { return new reduction_layer(*this); }
  std::string get_type() const override { return "reduction"; }
  data_layout get_data_layout() const override { return T_layout; }
  El::Device get_device_allocation() const override { return Dev; }

 protected:

  void setup_dims() override {
    transform_layer::setup_dims();
    this->m_num_neurons = 1;
    this->m_num_neuron_dims = 1;
    this->m_neuron_dims = {1};
  }

  void fp_compute() override {
    if(using_gpus()) {
    #ifndef LBANN_HAS_CUDNN
      LBANN_ERROR("cuDNN not detected");
    #else

      // GPU data
      const auto& input = get_prev_activations();
      auto& output = get_activations();
      const auto& input_size = get_num_prev_neurons();
      const auto& mini_batch_size = m_mini_batch_size_per_gpu;
      const auto& input_ldim = input.LDim();

      // Stop early if possible
      if (mini_batch_size == 0) { return; }

      // Apply reduction on GPU
      switch (m_mode) {
      case reduction_mode::SUM:
        {
          GPUMat ones;
          El::Ones(ones, input_size, 1);
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
          cublas::gemv(this->m_cudnn->get_cublas_handle(),
                       CUBLAS_OP_T,
                       input_size, mini_batch_size,
                       DataType(1),
                       input.LockedBuffer(), input_ldim,
                       ones.LockedBuffer(), 1,
                       DataType(0),
                       output.Buffer(), 1);
        }
        break;
      case reduction_mode::AVERAGE:
        {
          GPUMat ones;
          El::Ones(ones, input_size, 1);
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
          cublas::gemv(this->m_cudnn->get_cublas_handle(),
                       CUBLAS_OP_T,
                       input_size, mini_batch_size,
                       DataType(1) / input_size,
                       input.LockedBuffer(), input_ldim,
                       ones.LockedBuffer(), 1,
                       DataType(0),
                       output.Buffer(), 1);
        }
        break;
      default:
        LBANN_ERROR("invalid reduction mode");
      }
    #endif // LBANN_HAS_CUDNN
    } else {
      // Apply reduction on CPU
      const auto& local_input = get_local_prev_activations();
      auto& local_output = get_local_activations();
      switch (m_mode) {
      case reduction_mode::SUM:
        El::ColumnSum(local_input, local_output);
        break;
      case reduction_mode::AVERAGE:
        El::ColumnSum(local_input, local_output);
        local_output *= DataType(1) / get_num_prev_neurons();
        break;
      default:
        LBANN_ERROR("invalid reduction mode");
      }
    }
  }

  void bp_compute() override {
    if(using_gpus()) {
    #ifndef LBANN_HAS_CUDNN
      LBANN_ERROR("cuDNN not detected");
    #else

      // GPU data
      const auto& gradient_wrt_output = get_prev_error_signals();
      auto& gradient_wrt_input = get_error_signals();
      const auto& input_size = get_num_prev_neurons();
      const auto& mini_batch_size = m_mini_batch_size_per_gpu;
      const auto& gradient_wrt_input_ldim = gradient_wrt_input.LDim();

      // Stop early if possible
      if (mini_batch_size == 0) { return; }

      // Apply reduction on GPU
      switch (m_mode) {
      case reduction_mode::SUM:
        {
          GPUMat ones;
          El::Ones(ones, input_size, 1);
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
          cublas::gemm(this->m_cudnn->get_cublas_handle(),
                       CUBLAS_OP_N, CUBLAS_OP_N,
                       input_size, mini_batch_size, 1,
                       DataType(1),
                       ones.LockedBuffer(), input_size,
                       gradient_wrt_output.LockedBuffer(), 1,
                       DataType(0),
                       gradient_wrt_input.Buffer(),
                       gradient_wrt_input_ldim);
        }
        break;
      case reduction_mode::AVERAGE:
        {
          GPUMat ones;
          El::Ones(ones, input_size, 1);
          CHECK_CUDA(cudaSetDevice(this->m_cudnn->get_gpu()));
          cublas::gemm(this->m_cudnn->get_cublas_handle(),
                       CUBLAS_OP_N, CUBLAS_OP_T,
                       input_size, mini_batch_size, 1,
                       DataType(1) / mini_batch_size,
                       ones.LockedBuffer(), input_size,
                       gradient_wrt_output.LockedBuffer(), 1,
                       DataType(0),
                       gradient_wrt_input.Buffer(),
                       gradient_wrt_input_ldim);
        }
        break;
      default:
        LBANN_ERROR("invalid reduction mode");
      }

   #endif // LBANN_HAS_CUDNN
    } else {
      // Apply reduction on CPU
      const auto& local_gradient_wrt_output = get_local_prev_error_signals();
      auto& local_gradient_wrt_input = get_local_error_signals();
      switch (m_mode) {
      case reduction_mode::SUM:
        El::IndexDependentMap(local_gradient_wrt_input,
                              (std::function<DataType(El::Int,El::Int,const DataType&)>)
                              ([this,&local_gradient_wrt_output]
                               (El::Int r, El::Int c,const DataType& dx)
                               ->DataType {
                                return dx + local_gradient_wrt_output(0, c);
                              }));
        break;
      case reduction_mode::AVERAGE:
        {
          const DataType scale = DataType(1) / get_num_prev_neurons();
          El::IndexDependentMap(local_gradient_wrt_input,
                                (std::function<DataType(El::Int,El::Int,const DataType&)>)
                                ([this,&local_gradient_wrt_output,scale]
                                 (El::Int r, El::Int c,const DataType& dx)
                                 ->DataType {
                                  return dx + scale * local_gradient_wrt_output(0, c);
                                }));
        }
        break;
      default:
        LBANN_ERROR("invalid reduction mode");
      }
    }
  }

};

} // namespace lbann

#endif // LBANN_LAYER_REDUCTION_HPP_INCLUDED
