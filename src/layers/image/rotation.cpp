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

#define LBANN_ROTATION_LAYER_INSTANTIATE
#include "lbann/layers/image/rotation.hpp"

#include <math.h>  

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void rotation_layer<TensorDataType, Layout, Device>::fp_compute() {

  // Useful constants
  constexpr DataType Pi = M_PI;
  constexpr DataType degree = 180;
  constexpr DataType half = 0.5;
  
  // Input and output tensors
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();

  // Tensor dimensions
  const auto& input_dims = this->get_input_dims(0);
  const El::Int num_dims = input_dims.size();
  const auto& num_samples = local_input.Width();
  const El::Int num_channels = std::accumulate(input_dims.begin(),
                                               input_dims.end()-2,
                                               1,
                                               std::multiplies<int>());
  const El::Int input_height = input_dims[num_dims-2];
  const El::Int input_width = input_dims[num_dims-1];

  // Get rotation angle
  const auto& angles = this->get_prev_activations(1);
	
  // Perform rotation for each input pixel based on the center pixel
  LBANN_OMP_PARALLEL_FOR_COLLAPSE4
  for (El::Int sample = 0; sample < num_samples; ++sample) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      for (El::Int output_row = 0; output_row < input_height; ++output_row) {
        for (El::Int output_col = 0; output_col < input_width; ++output_col) {	

          // Convert to rad
          const auto& angle = angles.Get(0, sample);
          const auto& angle_rad = angle * Pi / degree;

          // Get center pixel for rotation
          const El::Int row_center = input_width/2;
          const El::Int col_center = input_height/2;

          // Rotate point relative to input pixel centers
          const auto& rotated_row =  (output_row - row_center) * cos(angle_rad) - (output_col - col_center) * sin(angle_rad) + row_center;
          const auto& rotated_col = (output_row - row_center) * sin(angle_rad) + (output_col - col_center) * cos(angle_rad) + col_center;

          // Find input pixels near rotation point
          auto input_col = static_cast<El::Int>(std::floor(rotated_col - half));
          auto input_row = static_cast<El::Int>(std::floor(rotated_row - half));

          // Input and output pixels
          auto& pixel_output = local_output(channel * input_height * input_width
                                                + output_row * input_width
                                                + output_col,
                                                sample);

	  if((input_row > 0 && input_row < input_height) && (input_col > 0 && input_col < input_width)){
          	auto& pixel_input = local_input(channel * input_height * input_width
                                       	     	  + input_row * input_width
                                           	  + input_col,
                                                  sample);

          	pixel_output = pixel_input;
	  }
	  else {
          	pixel_output = zero;
	  }	 
        }
      }
    }
  }
}

template <typename TensorDataType, data_layout T_layout, El::Device Dev>
void rotation_layer<TensorDataType, T_layout, Dev>::bp_compute() {

}


#define PROTO(T) \
  template class rotation_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"
#undef PROTO

} // namespace lbann
