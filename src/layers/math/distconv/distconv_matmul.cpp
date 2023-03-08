////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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
#define LBANN_LAYERS_MATH_DISTCONV_MATMUL_INSTANTIATE
#include "lbann/layers/math/distconv/distconv_matmul.hpp"
#include "lbann/base.hpp"
#include "lbann/utils/distconv.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace distconv {

template <typename Backend, typename DataType>
template <typename Allocator>
int MatMul<Backend, DataType>::forward(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& input_0,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& input_1,
  tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& output,
  const bool transpose_0,
  const bool transpose_1)
{
  if (input_0.get_local_size() == 0 || input_1.get_local_size() == 0 ||
      output.get_local_size() == 0) {
    return 0; // no op for emptry inputs
  }
  const auto& input_0_dims = input_0.get_local_shape();
  const auto& input_1_dims = input_1.get_local_shape();
  const auto& output_dims = output.get_local_shape();

  const auto input_0_width = input_0_dims[0];
  const auto input_0_height = input_0_dims[1];
  const auto input_1_width = input_1_dims[0];
  const auto input_1_height = input_1_dims[1];
  const auto output_width = output_dims[0];
  const auto output_height = output_dims[1];

  const auto mat_depth = input_0_dims[2];
  const auto local_mini_batch_size = input_0_dims[3];

  const auto num_matrices = mat_depth * local_mini_batch_size;
  const auto input_0_stride = input_0_height * input_0_width;
  const auto input_1_stride = input_1_height * input_1_width;
  const auto output_stride = output_height * output_width;

  // Check if buffer is not null possibly

  if (input_0.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "input 0 buffer is null";
  }

  if (input_1.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "input 1 buffer is null";
  }

  if (output.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "output buffer is null";
  }

  // Convert to Hydrogen matrices for GEMM
  El::Matrix<DataType, El::Device::GPU> input_0_mat(input_0_stride,
                                                    num_matrices,
                                                    input_0.get_buffer(),
                                                    input_0_stride);

  El::Matrix<DataType, El::Device::GPU> input_1_mat(input_1_stride,
                                                    num_matrices,
                                                    input_1.get_buffer(),
                                                    input_1_stride);

  El::Matrix<DataType, El::Device::GPU> output_mat(output_stride,
                                                   num_matrices,
                                                   output.get_buffer(),
                                                   output_stride);
  {
    using namespace hydrogen;
    auto multisync = MakeMultiSync(El::SyncInfoFromMatrix(output_mat),
                                   El::SyncInfoFromMatrix(input_0_mat),
                                   El::SyncInfoFromMatrix(input_1_mat));
    gpu_blas::GemmStridedBatched(
      transpose_1 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
      transpose_0 ? TransposeMode::TRANSPOSE : TransposeMode::NORMAL,
      output_width,
      output_height,
      transpose_0 ? input_0_height : input_0_width,
      El::TypeTraits<DataType>::One(),
      input_1_mat.LockedBuffer(),
      input_1_width,
      input_1_stride,
      input_0_mat.LockedBuffer(),
      input_0_width,
      input_0_stride,
      El::TypeTraits<DataType>::Zero(),
      output_mat.Buffer(),
      output_width,
      output_stride,
      num_matrices,
      multisync);
  }
  return 0;
}

template <typename Backend, typename DataType>
template <typename Allocator>
int MatMul<Backend, DataType>::backward(
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& input_0,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& input_1,
  const tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& output_grad,
  tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& input_0_grad,
  tensor::Tensor<DataType, tensor::LocaleMPI, Allocator>& input_1_grad,
  bool transpose_0,
  bool transpose_1)
{
  if (input_0.get_local_size() == 0 || input_1.get_local_size() == 0 ||
      output_grad.get_local_size() == 0 || input_0_grad.get_local_size() == 0 ||
      input_0_grad.get_local_size() == 0) {
    return 0; // no op for emptry inputs
  }

  const auto& input_0_dims = input_0.get_local_shape();
  const auto& input_1_dims = input_1.get_local_shape();
  const auto& output_grad_dims = output_grad.get_local_shape();

  const auto input_0_width = input_0_dims[0];
  const auto input_0_height = input_0_dims[1];
  const auto input_1_width = input_1_dims[0];
  const auto input_1_height = input_1_dims[1];
  const auto output_width = output_grad_dims[0];
  const auto output_height = output_grad_dims[1];

  const auto mat_depth = input_0_dims[2];
  const auto local_mini_batch_size = input_0_dims[3];

  const auto num_matrices = mat_depth * local_mini_batch_size;
  const auto input_0_stride = input_0_height * input_0_width;
  const auto input_1_stride = input_1_height * input_1_width;
  const auto output_stride = output_height * output_width;

  // Check if buffer is not null possibly

  if (input_0.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "input 0 buffer is null";
  }

  if (input_1.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "input 1 buffer is null";
  }

  if (output_grad.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "output grad buffer is null";
  }

  if (input_0_grad.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "input 0 grad buffer is null";
  }

  if (input_1_grad.get_buffer() == nullptr) {
    util::MPIRootPrintStreamInfo() << "input 1 grad buffer is null";
  }

  // Convert to Hydrogen matrices for GEMM
  El::Matrix<DataType, El::Device::GPU> input_0_mat(input_0_stride,
                                                    num_matrices,
                                                    input_0.get_buffer(),
                                                    input_0_stride);

  El::Matrix<DataType, El::Device::GPU> input_1_mat(input_1_stride,
                                                    num_matrices,
                                                    input_1.get_buffer(),
                                                    input_1_stride);

  El::Matrix<DataType, El::Device::GPU> output_grad_mat(
    output_stride,
    num_matrices,
    output_grad.get_buffer(),
    output_stride);

  El::Matrix<DataType, El::Device::GPU> input_0_grad_mat(
    input_0_stride,
    num_matrices,
    input_0_grad.get_buffer(),
    input_0_stride);

  El::Matrix<DataType, El::Device::GPU> input_1_grad_mat(
    input_1_stride,
    num_matrices,
    input_1_grad.get_buffer(),
    input_1_stride);
  {
    using namespace hydrogen;
    auto multisync = MakeMultiSync(El::SyncInfoFromMatrix(input_0_grad_mat),
                                   El::SyncInfoFromMatrix(input_1_mat),
                                   El::SyncInfoFromMatrix(output_grad_mat));
    if (transpose_0) {
      gpu_blas::GemmStridedBatched(TransposeMode::TRANSPOSE,
                                   transpose_1 ? TransposeMode::TRANSPOSE
                                               : TransposeMode::NORMAL,
                                   input_0_width,
                                   input_0_height,
                                   output_width,
                                   El::TypeTraits<DataType>::One(),
                                   output_grad_mat.LockedBuffer(),
                                   output_width,
                                   output_stride,
                                   input_1_mat.LockedBuffer(),
                                   input_1_width,
                                   input_1_stride,
                                   El::TypeTraits<DataType>::Zero(),
                                   input_0_grad_mat.Buffer(),
                                   input_0_width,
                                   input_0_stride,
                                   num_matrices,
                                   multisync);
    }
    else {
      gpu_blas::GemmStridedBatched(transpose_1 ? TransposeMode::NORMAL
                                               : TransposeMode::TRANSPOSE,
                                   TransposeMode::NORMAL,
                                   input_0_width,
                                   input_0_height,
                                   output_width,
                                   El::TypeTraits<DataType>::One(),
                                   input_1_mat.LockedBuffer(),
                                   input_1_width,
                                   input_1_stride,
                                   output_grad_mat.LockedBuffer(),
                                   output_width,
                                   output_stride,
                                   El::TypeTraits<DataType>::Zero(),
                                   input_0_grad_mat.Buffer(),
                                   input_0_width,
                                   input_0_stride,
                                   num_matrices,
                                   multisync);
    }
  }
  {
    using namespace hydrogen;
    auto multisync = MakeMultiSync(El::SyncInfoFromMatrix(input_1_grad_mat),
                                   El::SyncInfoFromMatrix(input_0_mat),
                                   El::SyncInfoFromMatrix(output_grad_mat));
    if (transpose_1) {
      gpu_blas::GemmStridedBatched(transpose_0 ? TransposeMode::TRANSPOSE
                                               : TransposeMode::NORMAL,
                                   TransposeMode::TRANSPOSE,
                                   input_1_width,
                                   input_1_height,
                                   output_height,
                                   El::TypeTraits<DataType>::One(),
                                   input_0_mat.LockedBuffer(),
                                   input_0_width,
                                   input_0_stride,
                                   output_grad_mat.LockedBuffer(),
                                   output_width,
                                   output_stride,
                                   El::TypeTraits<DataType>::Zero(),
                                   input_1_grad_mat.Buffer(),
                                   input_1_width,
                                   input_1_stride,
                                   num_matrices,
                                   multisync);
    }
    else {
      gpu_blas::GemmStridedBatched(TransposeMode::NORMAL,
                                   transpose_0 ? TransposeMode::NORMAL
                                               : TransposeMode::TRANSPOSE,
                                   input_1_width,
                                   input_1_height,
                                   output_height,
                                   El::TypeTraits<DataType>::One(),
                                   output_grad_mat.LockedBuffer(),
                                   output_width,
                                   output_stride,
                                   input_0_mat.LockedBuffer(),
                                   input_0_width,
                                   input_0_stride,
                                   El::TypeTraits<DataType>::Zero(),
                                   input_1_grad_mat.Buffer(),
                                   input_1_width,
                                   input_1_stride,
                                   num_matrices,
                                   multisync);
    }
  }
  return 1;
}
// Explicit template instantiation

#define ETI(T, Backend)                                                        \
  template class MatMul<Backend, T>;                                           \
  template int MatMul<Backend, T>::forward<tensor::CUDAAllocator>(             \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      input_0,                                                                 \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      input_1,                                                                 \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& output_0,     \
    const bool transpose_0,                                                    \
    const bool transpose_1);                                                   \
  template int MatMul<Backend, T>::backward<tensor::CUDAAllocator>(            \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      input_0,                                                                 \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      input_1,                                                                 \
    const tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>&         \
      output_grad,                                                             \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input_grad_0, \
    tensor::Tensor<T, tensor::LocaleMPI, tensor::CUDAAllocator>& input_grad_1, \
    const bool transpose_0,                                                    \
    const bool transpose_1);

ETI(float, BackendDNNLib)
ETI(double, BackendDNNLib)
#undef ETI
} // namespace distconv
