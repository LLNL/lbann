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

#define LBANN_UTILS_IM2COL_GPU_INSTANTIATE
#include "lbann/utils/im2col.hpp"
#include <cassert>

namespace lbann {

template <typename TensorDataType>
__global__ void im2col_2d_kernel(
    const TensorDataType *__restrict__ input_buffer,
    TensorDataType *__restrict__ output_buffer,
    const int input_dim_x, const int input_dim_y,
    const int input_pad_x, const int input_pad_y,
    const int num_channels,
    const int window_dim_x, const int window_dim_y,
    const int offset_stride_x, const int offset_stride_y,
    const int offset_start_x, const int offset_start_y,
    const int offset_end_x, const int offset_end_y,
    const int offset_num_x, const int offset_num_y,
    const size_t output_height, const size_t output_num) {

  // TODO: Use size_t for indices appropriately

  const size_t gid = threadIdx.x + blockIdx.x * blockDim.x;
  if(gid < output_num) {
    const auto window_pos_x = gid%window_dim_x;
    const auto window_pos_y = (gid/window_dim_x)%window_dim_y;
    const auto window_dim_prod = window_dim_x*window_dim_y;
    const auto channel = (gid/window_dim_prod)%num_channels;
    const auto offset_x = (gid/window_dim_prod/num_channels)%offset_num_x;
    const auto offset_y = (gid/window_dim_prod/num_channels/offset_num_x)%offset_num_y;
    const auto sample = gid/window_dim_prod/num_channels/offset_num_x/offset_num_y;

    // Get input entry
    const int offset_pos_y = offset_start_y + offset_y * offset_stride_y;
    const int offset_pos_x = offset_start_x + offset_x * offset_stride_x;
    const int input_pos_y = offset_pos_y + window_pos_y;
    const int input_pos_x = offset_pos_x + window_pos_x;
    const int input_index = (input_pos_x
                             + input_pos_y * input_dim_x
                             + channel * input_dim_x * input_dim_y
                             + sample * num_channels * input_dim_x * input_dim_y);
    const bool input_pos_valid = (0 <= input_pos_y
                                  && input_pos_y < input_dim_y
                                  && 0 <= input_pos_x
                                  && input_pos_x < input_dim_x);

    // Get output entry
    const int output_row = (window_pos_x
                            + window_pos_y * window_dim_x
                            + channel * window_dim_x * window_dim_y);
    const int output_col = (offset_x
                            + offset_y * offset_num_x
                            + sample * offset_num_x * offset_num_y);
    const int output_index = output_row + output_col * output_height;

    // Copy input entry to output entry if valid
    output_buffer[output_index]
        = input_pos_valid ? input_buffer[input_index] : TensorDataType(0.);

  }
}

template <typename TensorDataType>
void im2col(const El::Matrix<TensorDataType, El::Device::GPU>& im,
            El::Matrix<TensorDataType, El::Device::GPU>& col,
            const int num_channels,
            const int im_num_dims,
            const int * im_dims,
            const int * im_pads,
            const int * window_dims,
            const int * window_strides,
            const El::SyncInfo<El::Device::GPU>& sync_info) {

  // Input and output parameters
  const size_t num_samples = im.Width();

  // im2col parameters
  std::vector<size_t> offset_start(im_num_dims);
  std::vector<size_t> offset_end(im_num_dims);
  std::vector<size_t> offset_stride(im_num_dims);
  std::vector<size_t> offset_num(im_num_dims);
  for(int d = 0; d < im_num_dims; ++d) {
    offset_start[d] = -im_pads[d];
    offset_end[d] = im_dims[d] + im_pads[d] - window_dims[d] + 1;
    offset_stride[d] = window_strides[d];
    offset_num[d] = (offset_end[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
  }

  // Compute the output size and resize col
  const std::vector<int> window_dims_v(window_dims, window_dims+im_num_dims);
  const size_t output_height =
      num_channels * std::accumulate(window_dims_v.begin(), window_dims_v.end(), 1, std::multiplies<size_t>());
  const size_t output_width =
      std::accumulate(offset_num.begin(), offset_num.end(), 1, std::multiplies<size_t>()) * num_samples;
  const size_t output_num = output_height * output_width;
  assert(col.Height() == output_height);
  assert(col.Width() == output_width);

  const TensorDataType *__restrict__ im_buffer = im.LockedBuffer();
  TensorDataType *__restrict__ col_buffer = col.Buffer();

  if(im_num_dims == 2) {
    constexpr size_t block_size = 256;
    const size_t grid_size = (output_num + block_size - 1) / block_size;
    if (grid_size > 0) {
      hydrogen::gpu::LaunchKernel(
        im2col_2d_kernel<TensorDataType>,
        grid_size, block_size, 0, sync_info,
        im.LockedBuffer(), col.Buffer(),
        im_dims[0], im_dims[1],
        im_pads[0], im_pads[1],
        num_channels,
        window_dims[0], window_dims[1],
        offset_stride[0], offset_stride[1],
        offset_start[0], offset_start[1],
        offset_end[0], offset_end[1],
        offset_num[0], offset_num[1],
        output_height, output_num);
    }
  } else {
    std::cerr << "im2col on GPU only accepts 2D layers." << std::endl;
    abort();
  }
}

#define PROTO(T)                                        \
  template void im2col<T>(                              \
      const El::Matrix<T, El::Device::GPU>& im,         \
      El::Matrix<T, El::Device::GPU>& col,              \
      const int num_channels,                           \
      const int im_num_dims,                            \
      const int * im_dims,                              \
      const int * im_pads,                              \
      const int * window_dims,                          \
      const int * window_strides,                       \
      const El::SyncInfo<El::Device::GPU>& sync_info)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace lbann
