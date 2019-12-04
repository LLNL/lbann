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
void fp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::CPU>& l) {

  // Local data
  const auto& local_input0 = dynamic_cast<const CPUMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const CPUMat&>(l.get_local_prev_activations(1));
  auto& local_output = dynamic_cast<CPUMat&>(l.get_local_activations());
  const auto& local_mini_batch_size = local_input0.Width();

  // Matrix dimensions
  const auto output_dims = l.get_output_dims();
  const auto input0_dims = l.get_input_dims(0);
  const El::Int m = *(output_dims.rbegin()+1);
  const El::Int n = *(output_dims.rbegin());
  const El::Int k = *(input0_dims.rbegin());

  // Compute matrix multiplication for each mini-batch sample
  // Note: Elemental matrices are in Fortran layout while LBANN
  // tensors are in C layout.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    CPUMat input0_v, input1_v, output_v;
    input0_v.LockedAttach(k, m, local_input0.LockedBuffer(0,i), k);
    input1_v.LockedAttach(n, k, local_input1.LockedBuffer(0,i), n);
    output_v.Attach(n, m, local_output.Buffer(0,i), n);
    El::Gemm(El::NORMAL, El::NORMAL,
             DataType{1}, input1_v, input0_v,
             DataType{0}, output_v);
  }

}

template <typename TensorDataType>
void bp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::CPU>& l) {

  // Local data
  const auto& local_input0 = dynamic_cast<const CPUMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const CPUMat&>(l.get_local_prev_activations(1));
  const auto& local_output_grad = dynamic_cast<const CPUMat&>(l.get_local_prev_error_signals());
  auto& local_input0_grad = dynamic_cast<CPUMat&>(l.get_local_error_signals(0));
  auto& local_input1_grad = dynamic_cast<CPUMat&>(l.get_local_error_signals(1));
  const auto& local_mini_batch_size = local_input0.Width();

  // Matrix dimensions
  const auto output_dims = l.get_output_dims();
  const auto input0_dims = l.get_input_dims(0);
  const El::Int m = *(output_dims.rbegin()+1);
  const El::Int n = *(output_dims.rbegin());
  const El::Int k = *(input0_dims.rbegin());

  // Compute gradients for each mini-batch sample
  // Note: Elemental matrices are in Fortran layout while LBANN
  // tensors are in C layout.
  LBANN_OMP_PARALLEL_FOR
  for (El::Int i = 0; i < local_mini_batch_size; ++i) {
    CPUMat input0_v, input1_v, output_grad_v, input0_grad_v, input1_grad_v;
    input0_v.LockedAttach(k, m, local_input0.LockedBuffer(0,i), k);
    input1_v.LockedAttach(n, k, local_input1.LockedBuffer(0,i), n);
    output_grad_v.LockedAttach(n, m, local_output_grad.LockedBuffer(0,i), n);
    input0_grad_v.Attach(k, m, local_input0_grad.Buffer(0,i), k);
    input1_grad_v.Attach(n, k, local_input1_grad.Buffer(0,i), n);
    El::Gemm(El::TRANSPOSE, El::NORMAL,
             DataType{1}, input1_v, output_grad_v,
             DataType{0}, input0_grad_v);
    El::Gemm(El::NORMAL, El::TRANSPOSE,
             DataType{1}, output_grad_v, input0_v,
             DataType{0}, input1_grad_v);
  }

}

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void fp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

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
void bp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l) {

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
  fp_compute_impl(*this);
}
template <typename TensorDataType, data_layout Layout, El::Device Device>
void matmul_layer<TensorDataType, Layout, Device>::bp_compute() {
  bp_compute_impl(*this);
}

// Explicit instantiation
template class matmul_layer<DataType, data_layout::DATA_PARALLEL, El::Device::CPU>;
#ifdef LBANN_HAS_GPU
template class matmul_layer<DataType, data_layout::DATA_PARALLEL, El::Device::GPU>;
#endif // LBANN_HAS_GPU

} // namespace lbann
