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

#define LBANN_ONE_HOT_LAYER_INSTANTIATE
#include "lbann/layers/misc/one_hot.hpp"
#include "lbann/utils/gpu/helpers.hpp"

namespace lbann {

namespace {

/**
 *  On input, output is assumed to be filled with zeros.
 *
 *  Block dimensions: bsize x 1 x 1
 *
 *  Grid dimensions: (width / bsize) x 1 x 1
 */
template <typename TensorDataType>
__global__ void fp_kernel(unsigned long long height,
                          unsigned long long width,
                          const TensorDataType* __restrict__ indices,
                          unsigned long long indices_stride,
                          TensorDataType* __restrict__ output,
                          unsigned long long output_ldim) {
  const unsigned long long gid = threadIdx.x + blockIdx.x * blockDim.x;
  const unsigned long long nthreads = blockDim.x * gridDim.x;
  for (unsigned long long col = gid; col < width; col += nthreads) {
    const auto& ind = indices[col*indices_stride];
    if (TensorDataType(0.f) <= ind && ind < TensorDataType(height)) {
      const unsigned long long row = static_cast<unsigned long long>(ind);
      output[row+col*output_ldim] = TensorDataType(1.f);
    }
  }
}

} // namespace <anon>

template <typename TensorDataType, data_layout Layout, El::Device Device>
void one_hot_layer<TensorDataType, Layout, Device>::fp_compute() {

  using GPUMatType = El::Matrix<TensorDataType, El::Device::GPU>;

  // Local matrices
  const auto& local_input =
    dynamic_cast<const GPUMatType&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<GPUMatType&>(this->get_local_activations());

  // Populate one-hot vectors
  El::Zero(local_output);
  if (!local_output.IsEmpty()) {
    const size_t local_height = local_output.Height();
    const size_t local_width = local_output.Width();
    constexpr size_t block_size = 64;
    const size_t grid_size = (local_width + block_size - 1) / block_size;
    auto multisync = El::MakeMultiSync(gpu::get_sync_info(local_input),
                                       gpu::get_sync_info(local_output));
    hydrogen::gpu::LaunchKernel(
      fp_kernel<TensorDataType>,
      grid_size, block_size, 0, multisync,
      local_height,
      local_width,
      local_input.LockedBuffer(),
      local_input.LDim(),
      local_output.Buffer(),
      local_output.LDim());
  }

}

#define PROTO(T)                     \
  template class one_hot_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
