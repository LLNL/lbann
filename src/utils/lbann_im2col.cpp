////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2016, Lawrence Livermore National Security, LLC. 
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

#include "lbann/utils/lbann_im2col.hpp"

using namespace El;

namespace lbann {

void im2col(const Mat& im,
            Mat& col,
            const std::vector<Int>& im_dims,
            const std::vector<Int>& im_pads,
            Int num_im_channels,
            const std::vector<Int>& window_dims,
            const std::vector<Int>& window_strides) {

  // Input and output parameters
  const size_t num_channels = num_im_channels;
  const size_t num_dims = im_dims.size();
  const size_t output_height = col.Height();
  const size_t output_width = col.Width();
  const DataType* __restrict__ input_buffer = im.LockedBuffer();
  DataType* __restrict__ output_buffer = col.Buffer();

  // Call optimized routine if data is 2D
  if(num_dims == 2) {
    im2col_2d(input_buffer, output_buffer,
              im_dims[1], im_dims[0], im_pads[1], im_pads[0], num_channels,
              window_dims[1], window_dims[0],
              window_strides[1], window_strides[0]);
    return;
  }

  // im2col parameters
  std::vector<size_t> input_dim(num_dims);
  std::vector<size_t> window_dim(num_dims);
  std::vector<size_t> offset_start(num_dims);
  std::vector<size_t> offset_end(num_dims);
  std::vector<size_t> offset_stride(num_dims);
  std::vector<size_t> offset_num(num_dims);
  for(size_t d = 0; d < num_dims; ++d) {
    input_dim[d] = im_dims[d];
    window_dim[d] = window_dims[d];
    offset_start[d] = -im_pads[d];
    offset_end[d] = im_dims[d] + im_pads[d] - window_dims[d] + 1;
    offset_stride[d] = window_strides[d];
    offset_num[d] = (offset_end[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
  }

  // Iterate through output matrix columns
#pragma omp parallel for
  for(size_t output_col = 0; output_col < output_width; ++output_col) {

    // Initialize arrays
    std::vector<size_t> offset_pos(num_dims);
    std::vector<size_t> window_pos(num_dims);

    // Get position of current offset
    size_t output_col_remainder = output_col;
    for(size_t d = num_dims; d-- > 0; ) {
      const size_t offset = output_col_remainder % offset_num[d];
      offset_pos[d] = offset_start[d] + offset * offset_stride[d];
      output_col_remainder /= offset_num[d];
    }

    // Iterate through output matrix entries
    for(size_t output_row = 0; output_row < output_height; ++output_row) {
      const size_t output_index = output_row + output_col * output_height;

      // Get position in window and channel
      size_t output_row_remainder = output_row;
      for(size_t d = num_dims; d-- > 0; ) {
        window_pos[d] = output_row_remainder % window_dim[d];
        output_row_remainder /= window_dim[d];
      }
      const size_t channel = output_row_remainder;

      // Get input matrix entry
      bool input_pos_valid = true;
      size_t input_index = channel;
      for(size_t d = 0; d < num_dims; ++d) {
        const size_t input_pos = offset_pos[d] + window_pos[d];
        input_pos_valid = input_pos_valid && 0 <= input_pos && input_pos < input_dim[d];
        input_index = input_pos + input_index * input_dim[d];
      }

      // Copy input matrix entry to output matrix if valid
      output_buffer[output_index]
        = input_pos_valid ? input_buffer[input_index] : DataType(0);

    }
  }

}

void col2im(const Mat& col,
            Mat& im,
            const std::vector<Int>& im_dims,
            const std::vector<Int>& im_pads,
            Int num_im_channels,
            const std::vector<Int>& window_dims,
            const std::vector<Int>& window_strides) {

  // Input and output parameters
  const size_t num_channels = num_im_channels;
  const size_t num_dims = im_dims.size();
  const size_t input_height = col.Height();
  const size_t input_width = col.Width();
  const size_t output_size = im.Height() * im.Width();
  const DataType* __restrict__ input_buffer = col.LockedBuffer();
  DataType* __restrict__ output_buffer = im.Buffer();

  // Call optimized routine if data is 2D
  if(num_dims == 2) {
    col2im_2d(input_buffer, output_buffer,
              im_dims[1], im_dims[0], im_pads[1], im_pads[0], num_channels,
              window_dims[1], window_dims[0],
              window_strides[1], window_strides[0]);
    return;
  }

  // col2im parameters
  std::vector<size_t> output_dim(num_dims);
  std::vector<size_t> window_dim(num_dims);
  std::vector<size_t> offset_start(num_dims);
  std::vector<size_t> offset_end(num_dims);
  std::vector<size_t> offset_stride(num_dims);
  std::vector<size_t> offset_num(num_dims);
  for(size_t d = 0; d < num_dims; ++d) {
    output_dim[d] = im_dims[d];
    window_dim[d] = window_dims[d];
    offset_start[d] = -im_pads[d];
    offset_end[d] = im_dims[d] + im_pads[d] - window_dims[d] + 1;
    offset_stride[d] = window_strides[d];
    offset_num[d] = (offset_end[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
  }

  // Iterate through output entries
#pragma omp parallel for
  for(size_t output_index = 0; output_index < output_size; ++output_index) {
    
    // Initialize arrays
    std::vector<size_t> output_pos(num_dims);
    std::vector<size_t> first_offset(num_dims);
    std::vector<size_t> last_offset(num_dims);
    std::vector<size_t> offset(num_dims);

    // Get position of output entry
    size_t output_index_remainder = output_index;
    for(size_t d = num_dims; d-- > 0; ) {
      output_pos[d] = output_index_remainder % output_dim[d];
      output_index_remainder /= output_dim[d];
    }
    const size_t channel = output_index_remainder;

    // Initialize output entry
    DataType output_entry = 0;

    // Get window offsets containing output entry
    for(size_t d = 0; d < num_dims; ++d) {
      first_offset[d] = (output_pos[d] - offset_start[d] - window_dim[d]) / offset_stride[d] + 1;
      first_offset[d] = Max(first_offset[d], 0);
      last_offset[d] = (output_pos[d] - offset_start[d] + offset_stride[d] - 1) / offset_stride[d];
      last_offset[d] = Min(last_offset[d], offset_end[d] - 1);
      offset[d] = first_offset[d];
    }

    // Iterate through window offsets containing output entry
    while(offset[0] < last_offset[0]) {
      
      // Get input entry corresponding to input entry
      size_t input_col = 0;
      size_t input_row = channel;
      for(size_t d = 0; d < num_dims; ++d) {
        const size_t window_pos = output_pos[d] - (offset_start[d] + offset[d] * offset_stride[d]);
        input_col = offset[d] + input_col * offset_num[d];
        input_row = window_pos + input_row * window_dim[d];
      }
      const size_t input_index = input_row + input_col * input_height;

      // Add input entry to output entry
      output_entry += input_buffer[input_index];

      // Move to next window offset
      ++offset[num_dims-1];
      for(size_t d = num_dims; d-- > 1; ) {
        if(offset[d] >= last_offset[d]) {
          offset[d] = first_offset[d];
          ++offset[d-1];
        }
      }

    }

    // Update output entry
    output_buffer[output_index] = output_entry;

  }

}

void im2col_2d(const DataType* __restrict__ input_buffer,
               DataType* __restrict__ output_buffer,
               size_t input_dim_x,
               size_t input_dim_y,
               size_t input_pad_x,
               size_t input_pad_y,
               size_t num_channels,
               size_t window_dim_x,
               size_t window_dim_y,
               size_t offset_stride_x,
               size_t offset_stride_y) {

  // im2col parameters
  const size_t offset_start_x = -input_pad_x;
  const size_t offset_start_y = -input_pad_y;
  const size_t offset_end_x = input_dim_x + input_pad_x - window_dim_x + 1;
  const size_t offset_end_y = input_dim_y + input_pad_y - window_dim_y + 1;
  const size_t offset_num_x = (offset_end_x - offset_start_x + offset_stride_x - 1) / offset_stride_x;
  const size_t offset_num_y = (offset_end_y - offset_start_y + offset_stride_y - 1) / offset_stride_y;
  const size_t output_height = num_channels * window_dim_x * window_dim_y;
  const size_t output_width = offset_num_x * offset_num_y;

  // Iterate through output matrix entries
#pragma omp parallel for collapse(5)
  for(size_t offset_y = 0; offset_y < offset_num_y; ++offset_y) {
    for(size_t offset_x = 0; offset_x < offset_num_x; ++offset_x) {
      for(size_t channel = 0; channel < num_channels; ++channel) {
        for(size_t window_pos_y = 0;
            window_pos_y < window_dim_y;
            ++window_pos_y) {
          for(size_t window_pos_x = 0;
              window_pos_x < window_dim_x;
              ++window_pos_x) {

            // Get input entry
            const size_t offset_pos_y = offset_start_y + offset_y * offset_stride_y;
            const size_t offset_pos_x = offset_start_x + offset_x * offset_stride_x;
            const size_t input_pos_y = offset_pos_y + window_pos_y;
            const size_t input_pos_x = offset_pos_x + window_pos_x;
            const size_t input_index = (input_pos_x
                                        + input_pos_y * input_dim_x
                                        + channel * input_dim_x * input_dim_y);
            const bool input_pos_valid = (0 <= input_pos_y
                                          && input_pos_y < input_dim_y
                                          && 0 <= input_pos_x
                                          && input_pos_x < input_dim_x);

            // Get output entry
            const size_t output_row = (window_pos_x
                                       + window_pos_y * window_dim_x
                                       + channel * window_dim_x * window_dim_y);
            const size_t output_col = offset_x + offset_y * offset_num_x;
            const size_t output_index = output_row + output_col * output_height;

            // Copy input entry to output entry if valid
            output_buffer[output_index]
              = input_pos_valid ? input_buffer[input_index] : DataType(0);
            
          }
        }
      }
    }
  }

}

void col2im_2d(const DataType* __restrict__ input_buffer,
               DataType* __restrict__ output_buffer,
               size_t output_dim_x,
               size_t output_dim_y,
               size_t output_pad_x,
               size_t output_pad_y,
               size_t num_channels,
               size_t window_dim_x,
               size_t window_dim_y,
               size_t offset_stride_x,
               size_t offset_stride_y) {

  // col2im parameters
  const size_t offset_start_x = -output_pad_x;
  const size_t offset_start_y = -output_pad_y;
  const size_t offset_end_x = output_dim_x + output_pad_x - window_dim_x + 1;
  const size_t offset_end_y = output_dim_y + output_pad_y - window_dim_y + 1;
  const size_t offset_num_x = (offset_end_x - offset_start_x + offset_stride_x - 1) / offset_stride_x;
  const size_t offset_num_y = (offset_end_y - offset_start_y + offset_stride_y - 1) / offset_stride_y;
  const size_t input_height = num_channels * window_dim_x * window_dim_y;
  const size_t input_width = offset_num_x * offset_num_y;
  const size_t output_size = num_channels * output_dim_x * output_dim_y;

  // Iterate through output entries
#pragma omp parallel for collapse(3)
  for(size_t channel = 0; channel < num_channels; ++channel) {
    for(size_t output_pos_y = 0;
        output_pos_y < output_dim_y;
        ++output_pos_y) {
      for(size_t output_pos_x = 0;
          output_pos_x < output_dim_x;
          ++output_pos_x) {

        // Get output entry
        const size_t output_index = (output_pos_x
                                     + output_pos_y * output_dim_x
                                     + channel * output_dim_x * output_dim_y);
        DataType output_entry = 0;

        // Get window offsets containing output entry
        const size_t offset_x_lower = (output_pos_x - offset_start_x - window_dim_x) / offset_stride_x + 1;
        const size_t offset_y_lower = (output_pos_y - offset_start_y - window_dim_y) / offset_stride_y + 1;
        const size_t offset_x_upper = (output_pos_x - offset_start_x + offset_stride_x - 1) / offset_stride_x;
        const size_t offset_y_upper = (output_pos_y - offset_start_y + offset_stride_y - 1) / offset_stride_y;
        const size_t first_offset_x = Max(offset_x_lower, 0);
        const size_t first_offset_y = Max(offset_y_lower, 0);
        const size_t last_offset_x = Min(offset_x_upper, offset_end_x - 1);
        const size_t last_offset_y = Min(offset_y_upper, offset_end_y - 1);

        // Iterate through window offsets
        for(size_t offset_y = first_offset_y;
            offset_y <= last_offset_y;
            ++offset_y) {
          const size_t window_pos_y = output_pos_y - (offset_start_y + offset_y * offset_stride_y);
          for(size_t offset_x = first_offset_x;
              offset_x <= last_offset_x;
              ++offset_x) {
            const size_t window_pos_x = output_pos_x - (offset_start_x + offset_x * offset_stride_x);

            // Get input entry
            const size_t input_row = (window_pos_x
                                      + window_pos_y * window_dim_x
                                      + channel * window_dim_x * window_dim_y);
            const size_t input_col = offset_x + offset_y * offset_num_x;
            const size_t input_index = input_row + input_col * input_height;

            // Add input entry to output entry
            output_entry += input_buffer[input_index];

          }          
        }

        // Update output entry
        output_buffer[output_index] = output_entry;

      }
    }
  }

}

}  // namespace lbann
