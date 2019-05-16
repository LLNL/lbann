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

#include "lbann/layers/image/bilinear_resize.hpp"
#include "lbann/utils/cuda.hpp"

namespace lbann {

namespace {

template <int block_size>
__global__ void fp_kernel(El::Int num_samples,
                          El::Int num_channels,
                          El::Int input_height,
                          El::Int input_width,
                          const DataType* __restrict__ input,
                          El::Int input_ldim,
                          El::Int output_height,
                          El::Int output_width,
                          DataType* __restrict__ output,
                          El::Int output_ldim) {

  // Useful constants
  constexpr DataType half = 0.5;
  constexpr DataType one = 1;
  const El::Int gid = threadIdx.x + blockIdx.x * blockDim.x;
  const El::Int num_threads = blockDim.x * gridDim.x;

  // Stride between interpolation points
  const auto& x_stride = static_cast<DataType>(input_width) / output_width;
  const auto& y_stride = static_cast<DataType>(input_height) / output_height;

  const auto& size = (num_samples * num_channels
                      * output_height * output_width);
  for (El::Int pos = gid; pos < size; pos += num_threads) {

    // Indices
    const auto& sample = pos / (num_channels * output_height * output_width);
    const auto& channel = (pos / (output_height * output_width)) % num_channels;
    const auto& output_row = (pos / output_width) % output_height;
    const auto& output_col = pos % output_width;

    // Interpolation point
    const auto& x = (output_col + half) * x_stride;
    const auto& y = (output_row + half) * y_stride;

    // Find input pixels near interpolation point
    const auto input_col = static_cast<El::Int>(cuda::floor(x - half));
    const auto& input_col0 = cuda::max(input_col, El::Int(0));
    const auto& input_col1 = cuda::min(input_col+1, input_width-1);
    const auto input_row = static_cast<El::Int>(cuda::floor(y - half));
    const auto& input_row0 = cuda::max(input_row, El::Int(0));
    const auto& input_row1 = cuda::min(input_row+1, input_height-1);

    // Interpolation point relative to input pixel centers
    const auto& unit_x = x - (input_col + half);
    const auto& unit_y = y - (input_row + half);

    // Input and output pixels
    const auto& pixel00 = input[sample * input_ldim
                                + channel * input_height * input_width
                                + input_row0 * input_width
                                + input_col0];
    const auto& pixel01 = input[sample * input_ldim
                                + channel * input_height * input_width
                                + input_row0 * input_width
                                + input_col1];
    const auto& pixel10 = input[sample * input_ldim
                                + channel * input_height * input_width
                                + input_row1 * input_width
                                + input_col0];
    const auto& pixel11 = input[sample * input_ldim
                                + channel * input_height * input_width
                                + input_row1 * input_width
                                + input_col1];
    auto& result = output[sample * output_ldim
                          + channel * output_height * output_width
                          + output_row * output_width
                          + output_col];

    // Bilinear interpolation
    result = (pixel00 * (one - unit_x) * (one - unit_y)
              + pixel01 * unit_x * (one - unit_y)
              + pixel10 * (one - unit_x) * unit_y
              + pixel11 * unit_x * unit_y);

  }

}

}


template <>
void bilinear_resize_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {

  // Matrices
  const auto& local_input = get_local_prev_activations();
  auto& local_output = get_local_activations();

  // Dimensions
  const auto& input_dims = get_input_dims();
  const auto& num_dims = input_dims.size();
  const auto& num_samples = local_input.Width();
  const El::Int num_channels = std::accumulate(input_dims.begin(),
                                               input_dims.end()-2,
                                               1,
                                               std::multiplies<int>());
  const El::Int input_height = input_dims[num_dims-2];
  const El::Int input_width = input_dims[num_dims-1];

  // Get CUDA grid dimensions
  // Note: Maximum CUDA grid dimension is 2^32-1
  // (https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#features-and-technical-specifications).
  const El::Int size = local_output.Height() * local_output.Width();
  constexpr El::Int block_dim = 256;
  El::Int grid_dim = (size + block_dim - 1) / block_dim;
  if (sizeof(El::Int) > sizeof(uint32_t)
      && grid_dim > std::numeric_limits<uint32_t>::max()) {
    grid_dim = std::numeric_limits<uint32_t>::max();
  }

  // Launch CUDA kernel
  if (grid_dim > 0) {
    fp_kernel<block_dim>
      <<<grid_dim, block_dim, 0, El::GPUManager::Stream()>>>(
        num_samples, num_channels,
        input_height, input_width,
        local_input.LockedBuffer(), local_input.LDim(),
        m_height, m_width,
        local_output.Buffer(), local_output.LDim());
  }

}

} // namespace lbann
