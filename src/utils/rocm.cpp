////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2023, Lawrence Livermore National Security, LLC.
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

#include "lbann/utils/dim_helpers.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#ifdef LBANN_HAS_ROCM

namespace lbann {
namespace gpu_lib {

// -------------------------------------------------------------
// Device properties
// -------------------------------------------------------------

dim3 max_grid_dims()
{
  static dim3 max_grid_dims_(0, 0, 0);
  if (max_grid_dims_.x == 0) {
    int device = 0;
    hipDeviceProp_t prop;
    CHECK_ROCM(hipGetDevice(&device));
    CHECK_ROCM(hipGetDeviceProperties(&prop, device));
    max_grid_dims_.x = prop.maxGridSize[0];
    max_grid_dims_.y = prop.maxGridSize[1];
    max_grid_dims_.z = prop.maxGridSize[2];
    if (max_grid_dims_.x == 0) {
      LBANN_ERROR("Could not setup max HIP grid size");
    }
  }
  return max_grid_dims_;
}

} // namespace gpu_lib
} // namespace lbann

namespace lbann {
namespace rocm {

// -------------------------------------------------------------
// event_wrapper
// -------------------------------------------------------------

event_wrapper::event_wrapper() : m_event(nullptr), m_stream(0)
{
#if HIP_VERSION < 50600000
  CHECK_ROCM(hipEventCreateWithFlags(&m_event, hipEventDisableTiming));
#else
  CHECK_ROCM(hipEventCreateWithFlags(&m_event,
                                     hipEventDisableTiming |
                                       hipEventDisableSystemFence));
#endif
}

event_wrapper::event_wrapper(const event_wrapper& other)
  : m_event(nullptr), m_stream(other.m_stream)
{
#if HIP_VERSION < 50600000
  CHECK_ROCM(hipEventCreateWithFlags(&m_event, hipEventDisableTiming));
#else
  CHECK_ROCM(hipEventCreateWithFlags(&m_event,
                                     hipEventDisableTiming |
                                       hipEventDisableSystemFence));
#endif
  if (!other.query()) {
    record(m_stream);
  }
}

event_wrapper& event_wrapper::operator=(const event_wrapper& other)
{
  m_stream = other.m_stream;
  if (!other.query()) {
    record(m_stream);
  }
  return *this;
}

event_wrapper::~event_wrapper() { static_cast<void>(hipEventDestroy(m_event)); }

void event_wrapper::record(hipStream_t stream)
{
  m_stream = stream;
  CHECK_ROCM(hipEventRecord(m_event, m_stream));
}

bool event_wrapper::query() const
{
  const auto& status = hipEventQuery(m_event);
  switch (status) {
  case hipSuccess:
    return true;
  case hipErrorNotReady:
    return false;
  default:
    CHECK_ROCM(status);
    return false;
  }
}

void event_wrapper::synchronize() { CHECK_ROCM(hipEventSynchronize(m_event)); }

hipEvent_t& event_wrapper::get_event() { return m_event; }

// -------------------------------------------------------------
// Helper functions for tensor operations
// -------------------------------------------------------------

namespace {

using int4 = gpu_lib::array<int, 4>;

/**
 *  Block dimensions: bdimx x bdimy x bdimz
 *
 *  Grid dimensions: (dim[3] / bdimx) x (dim[2] / bdimy) x (dim[1] / bdimx)
 */
template <typename TensorDataType>
__global__ void copy_4d_kernel(int4 dims,
                               const TensorDataType* __restrict__ input,
                               int4 input_strides,
                               TensorDataType* __restrict__ output,
                               int4 output_strides)
{

  // Indices
  const auto& gidx = threadIdx.x + blockIdx.x * blockDim.x;
  const auto& gidy = threadIdx.y + blockIdx.y * blockDim.y;
  const auto& gidz = threadIdx.z + blockIdx.z * blockDim.z;
  const auto& nthreadsx = gridDim.x * blockDim.x;
  const auto& nthreadsy = gridDim.y * blockDim.y;
  const auto& nthreadsz = gridDim.z * blockDim.z;

  for (int i0 = 0; i0 < dims[0]; ++i0) {
    for (int i1 = gidz; i1 < dims[1]; i1 += nthreadsz) {
      for (int i2 = gidy; i2 < dims[2]; i2 += nthreadsy) {
        for (int i3 = gidx; i3 < dims[3]; i3 += nthreadsx) {
          const auto& x = input[i0 * input_strides[0] + i1 * input_strides[1] +
                                i2 * input_strides[2] + i3 * input_strides[3]];
          auto& y = output[i0 * output_strides[0] + i1 * output_strides[1] +
                           i2 * output_strides[2] + i3 * output_strides[3]];
          y = x;
        }
      }
    }
  }
}

} // namespace

template <typename TensorDataType>
void copy_tensor(hipStream_t stream,
                 const std::vector<size_t>& dims,
                 const TensorDataType* input,
                 const std::vector<size_t>& input_strides,
                 TensorDataType* output,
                 const std::vector<size_t>& output_strides)
{

  // Check inputs
  if (dims.empty() || dims.size() > 4) {
    LBANN_ERROR("invalid number of tensor dimensions (", dims.size(), ")");
  }
  if (dims.size() != input_strides.size()) {
    LBANN_ERROR("number of input strides (",
                input_strides.size(),
                ") ",
                "does not match number of tensor dimensions (",
                dims.size(),
                ")");
  }
  if (dims.size() != output_strides.size()) {
    LBANN_ERROR("number of output strides (",
                output_strides.size(),
                ") ",
                "does not match number of tensor dimensions (",
                dims.size(),
                ")");
  }

  // Pad tensor dimensions to 4D
  std::vector<int> rdims(dims.rbegin(), dims.rend()),
    input_rstrides(input_strides.rbegin(), input_strides.rend()),
    output_rstrides(output_strides.rbegin(), output_strides.rend());
  rdims.resize(4, 1);
  input_rstrides.resize(4, input_rstrides.back());
  output_rstrides.resize(4, output_rstrides.back());

  // Launch HIP kernel
  const auto size = get_linear_size(dims);
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
    hipLaunchKernelGGL(copy_4d_kernel,
                       dim3(grid_dims),
                       dim3(block_dims),
                       0,
                       stream,
                       {rdims[3], rdims[2], rdims[1], rdims[0]},
                       input,
                       {input_rstrides[3],
                        input_rstrides[2],
                        input_rstrides[1],
                        input_rstrides[0]},
                       output,
                       {output_rstrides[3],
                        output_rstrides[2],
                        output_rstrides[1],
                        output_rstrides[0]});
  }
}

#if defined(LBANN_HAS_HALF) && defined(LBANN_HAS_GPU_HALF)
template <>
void copy_tensor<cpu_fp16>(hipStream_t stream,
                           const std::vector<size_t>& dims,
                           const cpu_fp16* input,
                           const std::vector<size_t>& input_strides,
                           cpu_fp16* output,
                           const std::vector<size_t>& output_strides)
{
  copy_tensor<fp16>(stream,
                    dims,
                    reinterpret_cast<const fp16*>(input),
                    input_strides,
                    reinterpret_cast<fp16*>(output),
                    output_strides);
}
#endif // defined(LBANN_HAS_HALF) && defined(LBANN_HAS_GPU_HALF)

// Explicit template instantiation
#define PROTO(T)                                                               \
  template void copy_tensor<T>(hipStream_t stream,                             \
                               const std::vector<size_t>& dims,                \
                               const T* input,                                 \
                               const std::vector<size_t>& input_strides,       \
                               T* output,                                      \
                               const std::vector<size_t>& output_strides);
#define LBANN_INSTANTIATE_GPU_HALF
#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

void mem_copy_async(void* output,
                    const void* input,
                    const size_t count,
                    hipMemcpyKind kind,
                    hipStream_t stream)
{
  CHECK_ROCM(hipMemcpyAsync(output, input, count, kind, stream));
}

} // namespace rocm
} // namespace lbann
#endif // LBANN_HAS_ROCM
