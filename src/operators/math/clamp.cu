////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#define LBANN_CLAMP_OPERATOR_INSTANTIATE
#include "lbann/operators/math/clamp.hpp"
#include "lbann/utils/gpu/sync_info_helpers.hpp"

namespace lbann {

namespace {

/** CUDA kernel for forward prop computation. */
template <typename TensorDataType>
__global__ void fp_kernel(TensorDataType min,
                          TensorDataType max,
                          El::Int height,
                          El::Int width,
                          const TensorDataType* __restrict__ input,
                          El::Int input_ldim,
                          TensorDataType* __restrict__ output,
                          El::Int output_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    auto& y = output[row + col * output_ldim];
    if (x <= min)      { y = min; }
    else if (x >= max) { y = max; }
    else              { y = x;   }
  }
}

/** GPU kernel for backprop computation. */
template <typename TensorDataType>
__global__ void bp_kernel(TensorDataType min,
                          TensorDataType max,
                          El::Int height,
                          El::Int width,
                          const TensorDataType* __restrict__ input,
                          El::Int input_ldim,
                          const TensorDataType* __restrict__ gradient_wrt_output,
                          El::Int gradient_wrt_output_ldim,
                          TensorDataType* __restrict__ gradient_wrt_input,
                          El::Int gradient_wrt_input_ldim) {
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int size = height * width;
  const El::Int num_threads = blockDim.x * gridDim.x;
  for (El::Int pos = gid; pos < size; pos += num_threads) {
    const auto& row = pos % height;
    const auto& col = pos / height;
    const auto& x = input[row + col * input_ldim];
    const auto& dy = gradient_wrt_output[row + col * gradient_wrt_output_ldim];
    auto& dx = gradient_wrt_input[row + col * gradient_wrt_input_ldim];
    dx = (x <= min || x >= max) ? TensorDataType(0.f) : dy;
  }
}

/** Local forward prop computation. */
template <typename TensorDataType>
void local_fp(TensorDataType min,
              TensorDataType max,
              const El::AbstractMatrix<TensorDataType>& input,
              El::AbstractMatrix<TensorDataType>& output) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  // TODO: HIP/ROCM notes
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch GPU kernel
  if (grid_dim > 0) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(input),
                                       gpu::get_sync_info(output));
    hydrogen::gpu::LaunchKernel(
      fp_kernel<TensorDataType>,
      grid_dim, block_dim, 0, multisync,
      min, max, height, width,
      input.LockedBuffer(), input.LDim(),
      output.Buffer(), output.LDim());
  }

}

/** Local backprop computation. */
template <typename TensorDataType>
void local_bp(TensorDataType min,
              TensorDataType max,
              const El::AbstractMatrix<TensorDataType>& input,
              const El::AbstractMatrix<TensorDataType>& gradient_wrt_output,
              El::AbstractMatrix<TensorDataType>& gradient_wrt_input) {

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  // TODO: HIP/ROCM notes
  const El::Int height = input.Height();
  const El::Int width = input.Width();
  const El::Int block_dim = 256;
  El::Int grid_dim = (height * width + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(unsigned int)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch GPU kernel
  if (grid_dim > 0) {
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(gradient_wrt_output),
                                       gpu::get_sync_info(gradient_wrt_input));
    hydrogen::gpu::LaunchKernel(
      bp_kernel<TensorDataType>,
      grid_dim, block_dim, 0, multisync,
      min, max, height, width,
      input.LockedBuffer(), input.LDim(),
      gradient_wrt_output.LockedBuffer(), gradient_wrt_output.LDim(),
      gradient_wrt_input.Buffer(), gradient_wrt_input.LDim());
  }

}

} // namespace

template <typename TensorDataType>
void ClampOperator<TensorDataType>::fp_compute_local(GPUMatrixType const& input,
                                                     GPUMatrixType& output) const {
  local_fp(this->m_min, this->m_max,
           input,
           output);
}

template <typename TensorDataType>
void ClampOperator<TensorDataType>::bp_compute_local(GPUMatrixType const& input,
                                                     GPUMatrixType const& gradient_wrt_output,
                                                     GPUMatrixType& gradient_wrt_input) const {
  local_bp(this->m_min, this->m_max,
           input,
           gradient_wrt_output,
           gradient_wrt_input);
}

#define PROTO(T)                                     \
  template class ClampOperator<T>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

} // namespace lbann
