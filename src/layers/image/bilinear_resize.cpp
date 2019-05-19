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

namespace lbann {

template <>
void bilinear_resize_layer<data_layout::DATA_PARALLEL, El::Device::CPU>::fp_compute() {

  // Useful constants
  constexpr DataType half = 0.5;
  constexpr DataType one = 1;

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

  // Perform bilinear interpolation for each output pixel
  const auto& x_stride = static_cast<DataType>(input_width) / m_width;
  const auto& y_stride = static_cast<DataType>(input_height) / m_height;
  LBANN_OMP_PARALLEL_FOR_COLLAPSE4
  for (El::Int sample = 0; sample < num_samples; ++sample) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      for (El::Int output_row = 0; output_row < m_height; ++output_row) {
        for (El::Int output_col = 0; output_col < m_width; ++output_col) {

          // Interpolation point
          const auto& x = (output_col + half) * x_stride;
          const auto& y = (output_row + half) * y_stride;

          // Find input pixels near interpolation point
          const auto input_col = static_cast<El::Int>(std::floor(x - half));
          const auto& input_col0 = std::max(input_col, El::Int(0));
          const auto& input_col1 = std::min(input_col+1, input_width-1);
          const auto input_row = static_cast<El::Int>(std::floor(y - half));
          const auto& input_row0 = std::max(input_row, El::Int(0));
          const auto& input_row1 = std::min(input_row+1, input_height-1);

          // Interpolation point relative to input pixel centers
          const auto& unit_x = x - (input_col + half);
          const auto& unit_y = y - (input_row + half);

          // Input and output pixels
          const auto& pixel00 = local_input(channel * input_height * input_width
                                            + input_row0 * input_width
                                            + input_col0,
                                            sample);
          const auto& pixel01 = local_input(channel * input_height * input_width
                                            + input_row0 * input_width
                                            + input_col1,
                                            sample);
          const auto& pixel10 = local_input(channel * input_height * input_width
                                            + input_row1 * input_width
                                            + input_col0,
                                            sample);
          const auto& pixel11 = local_input(channel * input_height * input_width
                                            + input_row1 * input_width
                                            + input_col1,
                                            sample);
          auto& result = local_output(channel * m_height * m_width
                                      + output_row * m_width
                                      + output_col,
                                      sample);

          // Bilinear interpolation
          result = (pixel00 * (one - unit_x) * (one - unit_y)
                    + pixel01 * unit_x * (one - unit_y)
                    + pixel10 * (one - unit_x) * unit_y
                    + pixel11 * unit_x * unit_y);

        }
      }
    }
  }

}

} // namespace lbann
