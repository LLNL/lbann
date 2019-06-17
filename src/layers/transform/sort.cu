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

#include "lbann/layers/transform/sort.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"

#include <thrust/system/cuda/execution_policy.h>
#include <thrust/sequence.h>
#include <thrust/sort.h>
#include <thrust/scatter.h>
#include <thrust/device_ptr.h>

namespace lbann {

template <>
void sort_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::fp_compute() {

  // Local matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();
  auto& local_indices = *m_indices;
  const auto& local_height = local_input.Height();
  const auto& local_width = local_input.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  cuda::thrust::allocator<> alloc(stream);

  // Sort each matrix column
  El::Copy(local_input, local_output);
  for (El::Int col = 0; col < local_width; ++col) {
    ::thrust::device_ptr<DataType> vals(local_output.Buffer(0, col));
    ::thrust::device_ptr<El::Int> inds(local_indices.Buffer(0, col));
    ::thrust::sequence(thrust::cuda::par(alloc).on(stream),
                       inds, inds + local_height);
    if (m_descending) {
      ::thrust::sort_by_key(thrust::cuda::par(alloc).on(stream),
                            vals, vals + local_height, inds,
                            ::thrust::greater<DataType>());
    } else {
      ::thrust::sort_by_key(thrust::cuda::par(alloc).on(stream),
                            vals, vals + local_height, inds,
                            ::thrust::less<DataType>());
    }
  }

}

template <>
void sort_layer<data_layout::DATA_PARALLEL, El::Device::GPU>
     ::bp_compute() {

  // Local matrices
  const auto& local_gradient_wrt_output = get_local_prev_error_signals();
  auto& local_gradient_wrt_input = get_local_error_signals();
  const auto& local_indices = *m_indices;
  const auto& local_height = local_gradient_wrt_input.Height();
  const auto& local_width = local_gradient_wrt_input.Width();

  // GPU objects
  auto&& stream = El::GPUManager::Stream();
  cuda::thrust::allocator<> alloc(stream);

  // Scatter gradients based on sorted indices
  for (El::Int col = 0; col < local_width; ++col) {
    const ::thrust::device_ptr<const El::Int> inds(m_indices->LockedBuffer(0, col));
    const ::thrust::device_ptr<const DataType> grad_wrt_out(local_gradient_wrt_output.LockedBuffer(0, col));
    ::thrust::device_ptr<DataType> grad_wrt_in(local_gradient_wrt_input.Buffer(0, col));
    ::thrust::scatter(thrust::cuda::par(alloc).on(stream),
                      grad_wrt_out, grad_wrt_out + local_height, inds,
                      grad_wrt_in);
  }

}

} // namespace lbann
