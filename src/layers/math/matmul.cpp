////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2019, Lawrence Livermore National Security, LLC.
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

#define LBANN_MATMUL_LAYER_INSTANTIATE
#include "lbann/layers/math/matmul.hpp"
#ifdef LBANN_HAS_GPU
#include "lbann/utils/cublas.hpp"
#endif // LBANN_HAS_GPU

namespace lbann {

template <typename TensorDataType>
void fp_compute_impl(matmul_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::CPU>& l,
                     bool transpose_input0,
                     bool transpose_input1) {

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input0 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& local_output = dynamic_cast<LocalMat&>(l.get_local_activations());
  const auto& local_mini_batch_size = local_input0.Width();

  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();
  const El::Int input0_height = *(input0_dims.rbegin()+1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_height = *(input1_dims.rbegin()+1);
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin()+1);
  const El::Int output_width = *(output_dims.rbegin());

  // Compute matrix multiplication for each mini-batch sample
  // Note: Elemental matrices are in Fortran layout while LBANN
  // tensors are in C layout.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    LocalMat input0_v, input1_v, output_v;
    input0_v.LockedAttach(input0_width, input0_height,
                          local_input0.LockedBuffer(0,i), input0_width);
    input1_v.LockedAttach(input1_width, input1_height,
                          local_input1.LockedBuffer(0,i), input1_width);
    output_v.Attach(output_width, output_height,
                    local_output.Buffer(0,i), output_width);
    El::Gemm(transpose_input1 ? El::TRANSPOSE : El::NORMAL,
             transpose_input0 ? El::TRANSPOSE : El::NORMAL,
             DataType{1}, input1_v, input0_v,
             DataType{0}, output_v);
  }

}

template <typename TensorDataType>
void bp_compute_impl(matmul_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::CPU>& l,
                     bool transpose_input0,
                     bool transpose_input1) {

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::CPU>;
  const auto& local_input0 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals(0));
  auto& local_input0_grad = dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& local_input1_grad = dynamic_cast<LocalMat&>(l.get_local_error_signals(1));
  const auto& local_mini_batch_size = local_input0.Width();

  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();
  const El::Int input0_height = *(input0_dims.rbegin()+1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_height = *(input1_dims.rbegin()+1);
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin()+1);
  const El::Int output_width = *(output_dims.rbegin());

  // Compute gradients for each mini-batch sample
  // Note: Elemental matrices are in Fortran layout while LBANN
  // tensors are in C layout.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    LocalMat input0_v, input1_v, output_grad_v, input0_grad_v, input1_grad_v;
    input0_v.LockedAttach(input0_width, input0_height,
                          local_input0.LockedBuffer(0,i), input0_width);
    input1_v.LockedAttach(input1_width, input1_height,
                          local_input1.LockedBuffer(0,i), input1_width);
    output_grad_v.LockedAttach(output_width, output_height,
                               local_output_grad.LockedBuffer(0,i), output_width);
    input0_grad_v.Attach(input0_width, input0_height,
                         local_input0_grad.Buffer(0,i), input0_width);
    input1_grad_v.Attach(input1_width, input1_height,
                         local_input1_grad.Buffer(0,i), input1_width);
    if (transpose_input0) {
      El::Gemm(El::TRANSPOSE,
               transpose_input1 ? El::TRANSPOSE : El::NORMAL,
               DataType{1}, output_grad_v, input1_v,
               DataType{0}, input0_grad_v);
    }
    else {
      El::Gemm(transpose_input1 ? El::NORMAL : El::TRANSPOSE,
               El::NORMAL,
               DataType{1}, input1_v, output_grad_v,
               DataType{0}, input0_grad_v);
    }
    if (transpose_input1) {
      El::Gemm(transpose_input0 ? El::TRANSPOSE : El::NORMAL,
               El::TRANSPOSE,
               DataType{1}, input0_v, output_grad_v,
               DataType{0}, input1_grad_v);
    }
    else {
      El::Gemm(El::NORMAL,
               transpose_input0 ? El::NORMAL : El::TRANSPOSE,
               DataType{1}, output_grad_v, input0_v,
               DataType{0}, input1_grad_v);
    }
  }

}

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void fp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l,
                     bool transpose_input0,
                     bool transpose_input1) {

  // Local data
  const auto& local_input0 = dynamic_cast<const GPUMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const GPUMat&>(l.get_local_prev_activations(1));
  auto& local_output = dynamic_cast<GPUMat&>(l.get_local_activations());
  const auto& local_mini_batch_size = local_input0.Width();

  // Return immediately if nothing needs to be done
  if (local_mini_batch_size < 1) { return; }

  // Matrix dimensions
  const auto output_dims = l.get_output_dims();
  const auto input0_dims = l.get_input_dims(0);
  const El::Int m = *(output_dims.rbegin()+1);
  const El::Int n = *(output_dims.rbegin());
  const El::Int k = *(input0_dims.rbegin());

  // Compute matrix multiplication for each mini-batch sample
  // Note: cuBLAS expects matrices in Fortran layout while LBANN
  // tensors are in C layout.
  auto&& handle = El::GPUManager::cuBLASHandle();
  cublas::gemm_strided_batched(
    handle, CUBLAS_OP_N, CUBLAS_OP_N, n, m, k,
    DataType{1},
    local_input1.LockedBuffer(), n, local_input1.LDim(),
    local_input0.LockedBuffer(), k, local_input0.LDim(),
    DataType{0},
    local_output.Buffer(), n, local_output.LDim(),
    local_mini_batch_size);

}
#endif // LBANN_HAS_GPU

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void bp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l,
                     bool transpose_input0,
                     bool transpose_input1) {

  // Local data
  const auto& local_input0 = dynamic_cast<const GPUMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const GPUMat&>(l.get_local_prev_activations(1));
  const auto& local_output_grad = dynamic_cast<const GPUMat&>(l.get_local_prev_error_signals());
  auto& local_input0_grad = dynamic_cast<GPUMat&>(l.get_local_error_signals(0));
  auto& local_input1_grad = dynamic_cast<GPUMat&>(l.get_local_error_signals(1));
  const auto& local_mini_batch_size = local_input0.Width();

  // Return immediately if nothing needs to be done
  if (local_mini_batch_size < 1) { return; }

  // Matrix dimensions
  const auto output_dims = l.get_output_dims();
  const auto input0_dims = l.get_input_dims(0);
  const El::Int m = *(output_dims.rbegin()+1);
  const El::Int n = *(output_dims.rbegin());
  const El::Int k = *(input0_dims.rbegin());

  // Compute gradients for each mini-batch sample
  // Note: cuBLAS expects matrices in Fortran layout while LBANN
  // tensors are in C layout.
  auto&& handle = El::GPUManager::cuBLASHandle();
  cublas::gemm_strided_batched(
    handle, CUBLAS_OP_T, CUBLAS_OP_N, k, m, n,
    DataType{1},
    local_input1.LockedBuffer(), n, local_input1.LDim(),
    local_output_grad.LockedBuffer(), n, local_output_grad.LDim(),
    DataType{0},
    local_input0_grad.Buffer(), k, local_input0_grad.LDim(),
    local_mini_batch_size);
  cublas::gemm_strided_batched(
    handle, CUBLAS_OP_N, CUBLAS_OP_T, n, k, m,
    DataType{1},
    local_output_grad.LockedBuffer(), n, local_output_grad.LDim(),
    local_input0.LockedBuffer(), k, local_input0.LDim(),
    DataType{0},
    local_input1_grad.Buffer(), n, local_input1_grad.LDim(),
    local_mini_batch_size);

}
#endif // LBANN_HAS_GPU

template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::fp_compute() {
  fp_compute_impl(*this, m_transpose_a, m_transpose_b);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::bp_compute() {
  bp_compute_impl(*this, m_transpose_a, m_transpose_b);
}

// Explicit instantiation
template class matmul_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class matmul_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
