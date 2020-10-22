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
#include "lbann/utils/gpu/helpers.hpp"
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
             El::TypeTraits<TensorDataType>::One(), input1_v, input0_v,
             El::TypeTraits<TensorDataType>::Zero(), output_v);
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
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
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
               El::TypeTraits<TensorDataType>::One(), output_grad_v, input1_v,
               El::TypeTraits<TensorDataType>::Zero(), input0_grad_v);
    }
    else {
      El::Gemm(transpose_input1 ? El::NORMAL : El::TRANSPOSE,
               El::NORMAL,
               El::TypeTraits<TensorDataType>::One(), input1_v, output_grad_v,
               El::TypeTraits<TensorDataType>::Zero(), input0_grad_v);
    }
    if (transpose_input1) {
      El::Gemm(transpose_input0 ? El::TRANSPOSE : El::NORMAL,
               El::TRANSPOSE,
               El::TypeTraits<TensorDataType>::One(), input0_v, output_grad_v,
               El::TypeTraits<TensorDataType>::Zero(), input1_grad_v);
    }
    else {
      El::Gemm(El::NORMAL,
               transpose_input0 ? El::NORMAL : El::TRANSPOSE,
               El::TypeTraits<TensorDataType>::One(), output_grad_v, input0_v,
               El::TypeTraits<TensorDataType>::Zero(), input1_grad_v);
    }
  }

}

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void fp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l,
                     bool transpose_input0,
                     bool transpose_input1) {

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input0 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  auto& local_output = dynamic_cast<LocalMat&>(l.get_local_activations());
  const auto& local_mini_batch_size = local_input0.Width();

  // Return immediately if nothing needs to be done
  if (local_mini_batch_size < 1) { return; }

  // Matrix dimensions
  const auto input0_dims = l.get_input_dims(0);
  const auto input1_dims = l.get_input_dims(1);
  const auto output_dims = l.get_output_dims();
  const El::Int input0_height = *(input0_dims.rbegin()+1);
  const El::Int input0_width = *(input0_dims.rbegin());
  const El::Int input1_width = *(input1_dims.rbegin());
  const El::Int output_height = *(output_dims.rbegin()+1);
  const El::Int output_width = *(output_dims.rbegin());

  // Compute matrix multiplication for each mini-batch sample
  // Note: cuBLAS expects matrices in Fortran layout while LBANN
  // tensors are in C layout.
  {
    using namespace hydrogen;
    auto multisync = MakeMultiSync(gpu::get_sync_info(local_output),
                                   gpu::get_sync_info(local_input0),
                                   gpu::get_sync_info(local_input1));
    gpu_blas::GemmStridedBatched(
      transpose_input1 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
      transpose_input0 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
      output_width,
      output_height,
      transpose_input0 ? input0_height : input0_width,
      El::TypeTraits<TensorDataType>::One(),
      local_input1.LockedBuffer(), input1_width, local_input1.LDim(),
      local_input0.LockedBuffer(), input0_width, local_input0.LDim(),
      El::TypeTraits<TensorDataType>::Zero(),
      local_output.Buffer(), output_width, local_output.LDim(),
      local_mini_batch_size,
      multisync);
  }
}
#endif // LBANN_HAS_GPU

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void bp_compute_impl(matmul_layer<TensorDataType, data_layout::DATA_PARALLEL,El::Device::GPU>& l,
                     bool transpose_input0,
                     bool transpose_input1) {

  // Local data
  using LocalMat = El::Matrix<TensorDataType, El::Device::GPU>;
  const auto& local_input0 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(0));
  const auto& local_input1 = dynamic_cast<const LocalMat&>(l.get_local_prev_activations(1));
  const auto& local_output_grad = dynamic_cast<const LocalMat&>(l.get_local_prev_error_signals());
  auto& local_input0_grad = dynamic_cast<LocalMat&>(l.get_local_error_signals(0));
  auto& local_input1_grad = dynamic_cast<LocalMat&>(l.get_local_error_signals(1));
  const auto& local_mini_batch_size = local_input0.Width();

  // Return immediately if nothing needs to be done
  if (local_mini_batch_size < 1) { return; }

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
  // Note: cuBLAS expects matrices in Fortran layout while LBANN
  // tensors are in C layout.
  {
    using namespace hydrogen;
    // The SyncInfo of the C matrix leads the way!
    auto multisync = MakeMultiSync(gpu::get_sync_info(local_input0_grad),
                                   gpu::get_sync_info(local_input1),
                                   gpu::get_sync_info(local_output_grad));

    if (transpose_input0) {
      gpu_blas::GemmStridedBatched(
        TransposeMode::TRANSPOSE,
        transpose_input1 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
        input0_width, input0_height, output_width,
        El::TypeTraits<TensorDataType>::One(),
        local_output_grad.LockedBuffer(), output_width, local_output_grad.LDim(),
        local_input1.LockedBuffer(), input1_width, local_input1.LDim(),
        El::TypeTraits<TensorDataType>::Zero(),
        local_input0_grad.Buffer(), input0_width, local_input0_grad.LDim(),
        local_mini_batch_size,
        multisync);
    }
    else {
      gpu_blas::GemmStridedBatched(
        transpose_input1 ? TransposeMode::NORMAL : TransposeMode::TRANSPOSE,
        TransposeMode::NORMAL,
        input0_width, input0_height, output_width,
        El::TypeTraits<TensorDataType>::One(),
        local_input1.LockedBuffer(), input1_width, local_input1.LDim(),
        local_output_grad.LockedBuffer(), output_width, local_output_grad.LDim(),
        El::TypeTraits<TensorDataType>::Zero(),
        local_input0_grad.Buffer(), input0_width, local_input0_grad.LDim(),
        local_mini_batch_size,
        multisync);
    }
  }
  {
    using namespace hydrogen;
    // The SyncInfo of the C matrix leads the way!
    auto multisync = MakeMultiSync(gpu::get_sync_info(local_input1_grad),
                                   gpu::get_sync_info(local_input0),
                                   gpu::get_sync_info(local_output_grad));
    if (transpose_input1) {
      gpu_blas::GemmStridedBatched(
        transpose_input0 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
        TransposeMode::TRANSPOSE,
        input1_width, input1_height, output_height,
        El::TypeTraits<TensorDataType>::One(),
        local_input0.LockedBuffer(), input0_width, local_input0.LDim(),
        local_output_grad.LockedBuffer(), output_width, local_output_grad.LDim(),
        El::TypeTraits<TensorDataType>::Zero(),
        local_input1_grad.Buffer(), input1_width, local_input1_grad.LDim(),
        local_mini_batch_size,
        multisync);
    }
    else {
      gpu_blas::GemmStridedBatched(
        TransposeMode::NORMAL,
        transpose_input0 ? TransposeMode::NORMAL : TransposeMode::TRANSPOSE,
        input1_width, input1_height, output_height,
        El::TypeTraits<TensorDataType>::One(),
        local_output_grad.LockedBuffer(), output_width, local_output_grad.LDim(),
        local_input0.LockedBuffer(), input0_width, local_input0.LDim(),
        El::TypeTraits<TensorDataType>::Zero(),
        local_input1_grad.Buffer(), input1_width, local_input1_grad.LDim(),
        local_mini_batch_size,
        multisync);
    }
  }
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
#define PROTO_DEVICE(T, Device) \
  template class matmul_layer<T, data_layout::DATA_PARALLEL, Device>

#include "lbann/macros/instantiate_device.hpp"

} // namespace lbann
