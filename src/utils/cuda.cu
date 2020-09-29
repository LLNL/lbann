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

#include "lbann/utils/cuda.hpp"

#ifdef LBANN_HAS_GPU

namespace lbann {
namespace cuda {

// -------------------------------------------------------------
// Utilities for CUDA events
// -------------------------------------------------------------

event_wrapper::event_wrapper() : m_event(nullptr), m_stream(0) {
  CHECK_CUDA(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
}

event_wrapper::event_wrapper(const event_wrapper& other)
  : m_event(nullptr), m_stream(other.m_stream) {
  CHECK_CUDA(cudaEventCreateWithFlags(&m_event, cudaEventDisableTiming));
  if (!other.query()) { record(m_stream); }
}

event_wrapper& event_wrapper::operator=(const event_wrapper& other) {
  m_stream = other.m_stream;
  if (!other.query()) { record(m_stream); }
  return *this;
}

event_wrapper::~event_wrapper() {
  cudaEventDestroy(m_event);
}

void event_wrapper::record(cudaStream_t stream) {
  m_stream = stream;
  CHECK_CUDA(cudaEventRecord(m_event, m_stream));
}

bool event_wrapper::query() const {
  const auto& status = cudaEventQuery(m_event);
  switch (status) {
  case cudaSuccess:       return true;
  case cudaErrorNotReady: return false;
  default:
    CHECK_CUDA(status);
    return false;
  }
}

void event_wrapper::synchronize() {
  CHECK_CUDA(cudaEventSynchronize(m_event));
}

cudaEvent_t& event_wrapper::get_event() { return m_event; }

// -------------------------------------------------------------
// Helper functions for tensor operations
// -------------------------------------------------------------

namespace {

using int4 = cuda::array<int, 4>;

/**
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (dim[3] / bdimx) x (dim[2] / bdimy) x (dim[1] / bdimx)
 */
template <typename TensorDataType>
__global__ void copy_4d_kernel(
  int4 dims,
  const TensorDataType* __restrict__ input,
  int4 input_strides,
  TensorDataType* __restrict__ output,
  int4 output_strides) {

  // Indices
  const auto& gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const auto& gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const auto& nthreadsx = gridDim.x * blockDim.x;
  const auto& nthreadsy = gridDim.y * blockDim.y;
  const auto& nthreadsz = gridDim.z * blockDim.z;

  for (int i0=0; i0<dims[0]; ++i0) {
    for (int i1=gidz; i1<dims[1]; i1+=nthreadsz) {
      for (int i2=gidy; i2<dims[2]; i2+=nthreadsy) {
        for (int i3=gidx; i3<dims[3]; i3+=nthreadsx) {
          const auto& x = input[i0 * input_strides[0]
                                + i1 * input_strides[1]
                                + i2 * input_strides[2]
                                + i3 * input_strides[3]];
          auto& y = output[i0 * output_strides[0]
                           + i1 * output_strides[1]
                           + i2 * output_strides[2]
                           + i3 * output_strides[3]];
          y = x;
        }
      }
    }
  }

}

} // namespace <anon>

template <typename TensorDataType>
void copy_tensor(
  cudaStream_t stream,
  const std::vector<size_t>& dims,
  const TensorDataType* input,
  const std::vector<size_t>& input_strides,
  TensorDataType* output,
  const std::vector<size_t>& output_strides) {

  // Check inputs
  if (dims.empty() || dims.size() > 4) {
    LBANN_ERROR("invalid number of tensor dimensions (",dims.size(),")");
  }
  if (dims.size() != input_strides.size()) {
    LBANN_ERROR(
      "number of input strides (",input_strides.size(),") ",
      "does not match number of tensor dimensions (",dims.size(),")");
  }
  if (dims.size() != output_strides.size()) {
    LBANN_ERROR(
      "number of output strides (",output_strides.size(),") ",
      "does not match number of tensor dimensions (",dims.size(),")");
  }

  // Pad tensor dimensions to 4D
  std::vector<int>
    rdims(dims.rbegin(), dims.rend()),
    input_rstrides(input_strides.rbegin(), input_strides.rend()),
    output_rstrides(output_strides.rbegin(), output_strides.rend());
  rdims.resize(4, 1);
  input_rstrides.resize(4, input_rstrides.back());
  output_rstrides.resize(4, output_rstrides.back());

  // Launch CUDA kernel
  const auto size = std::accumulate(
    dims.begin(), dims.end(), 1, std::multiplies<int>());
  if (size > 0) {
    constexpr size_t block_size = 64;
    dim3 block_dims, grid_dims;
    block_dims.x = block_size;
    block_dims.y = 1;
    block_dims.z = 1;
    grid_dims.x = (rdims[0] + block_dims.x - 1) / block_dims.x;
    grid_dims.y = (rdims[1] + block_dims.y - 1) / block_dims.y;
    grid_dims.z = (rdims[2] + block_dims.z - 1) / block_dims.z;
    grid_dims.y = El::Min(grid_dims.y, 65535);
    grid_dims.z = El::Min(grid_dims.z, 65535);
    copy_4d_kernel<<<grid_dims, block_dims, 0, stream>>>(
      {rdims[3], rdims[2], rdims[1], rdims[0]},
      input,
      {input_rstrides[3], input_rstrides[2],
          input_rstrides[1], input_rstrides[0]},
      output,
      {output_rstrides[3], output_rstrides[2],
          output_rstrides[1], output_rstrides[0]});
  }

}

#if defined(LBANN_HAS_HALF) && defined(LBANN_HAS_GPU_HALF)
template <>
void copy_tensor<cpu_fp16>(
  cudaStream_t stream,
  const std::vector<size_t>& dims,
  const cpu_fp16* input,
  const std::vector<size_t>& input_strides,
  cpu_fp16* output,
  const std::vector<size_t>& output_strides) {
  copy_tensor<fp16>(
    stream,
    dims,
    reinterpret_cast<const fp16*>(input),
    input_strides,
    reinterpret_cast<fp16*>(output),
    output_strides);
}
#endif // defined(LBANN_HAS_HALF) && defined(LBANN_HAS_GPU_HALF)

// Explicit template instantiation
#define PROTO(T)                                \
  template void copy_tensor<T>(                 \
    cudaStream_t stream,                        \
    const std::vector<size_t>& dims,            \
    const T* input,                             \
    const std::vector<size_t>& input_strides,   \
    T* output,                                  \
    const std::vector<size_t>& output_strides);
#define LBANN_INSTANTIATE_GPU_HALF
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

} // namespace cuda
} // namespace lbann

#endif // LBANN_HAS_GPU
