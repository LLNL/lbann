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

#define LBANN_COMPOSITE_IMAGE_TRANSFORMATION_LAYER_INSTANTIATE
#include "lbann/layers/image/composite_image_transformation.hpp"
#include "lbann/proto/datatype_helpers.hpp"
#include "lbann/utils/exception.hpp"

#include "lbann/proto/layers.pb.h"
#include <math.h>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void composite_image_transformation_layer<TensorDataType, Layout, Device>::
  setup_dims(DataReaderMetaData& dr_metadata)
{
  data_type_layer<TensorDataType>::setup_dims(dr_metadata);

  // Get input dimensions
  auto dims = this->get_input_dims(0);

  // Check that dimensions are valid
  if (dims.size() != 3) {
    std::ostringstream ss;
    for (size_t i = 0; i < dims.size(); ++i) {
      ss << (i > 0 ? " x " : "") << dims[i];
    }
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "expects a 3D input in CHW format, ",
                "but input dimensions are ",
                ss.str());
  }
}

template <typename T, data_layout L, El::Device D>
void composite_image_transformation_layer<T, L, D>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<T>);
  proto.mutable_composite_image_transformation();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void composite_image_transformation_layer<TensorDataType, Layout, Device>::
  fp_compute()
{

  // Useful constants
  constexpr DataType Pi = M_PI;
  constexpr DataType degree = 180;
  constexpr DataType zero = 0;
  constexpr DataType one = 1;

  // Input and output tensors
  const auto& local_input = this->get_local_prev_activations();
  auto& local_output = this->get_local_activations();

  // Tensor dimensions
  const auto& input_dims = this->get_input_dims(0);
  const auto& num_samples = local_input.Width();
  const El::Int num_channels = input_dims[0];
  const El::Int input_height = input_dims[1];
  const El::Int input_width = input_dims[2];

  // Get rotation angle
  const auto& angles = this->get_local_prev_activations(1);

  // Get shear factor
  const auto& shears = this->get_local_prev_activations(2);

  // Get translation factor
  const auto& translations = this->get_local_prev_activations(3);

  // Perform rotation for each input pixel based on the center pixel
  LBANN_OMP_PARALLEL_FOR_COLLAPSE4
  for (El::Int sample = 0; sample < num_samples; ++sample) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      for (El::Int output_row = 0; output_row < input_height; ++output_row) {
        for (El::Int output_col = 0; output_col < input_width; ++output_col) {

          // Convert to rad
          const auto& angle = angles.Get(0, sample);
          const auto& angle_rad = angle * Pi / degree;

          // Shear factor
          const auto& shear_X = shears.Get(0, sample);
          const auto& shear_Y = shears.Get(1, sample);

          // Translation factor
          const auto& translate_X = translations.Get(0, sample);
          const auto& translate_Y = translations.Get(1, sample);

          // Get center pixel for rotation
          const El::Int col_center = input_width / 2;
          const El::Int row_center = input_height / 2;

          // Rotate point relative to input pixel centers
          const auto& rotated_col = (output_row - row_center) * sin(angle_rad) +
                                    (output_col - col_center) * cos(angle_rad) +
                                    col_center;
          const auto& rotated_row = (output_row - row_center) * cos(angle_rad) -
                                    (output_col - col_center) * sin(angle_rad) +
                                    row_center;

          // Shear the rotated point
          const auto& shear_col = rotated_col + shear_X * rotated_row;
          const auto& shear_row = rotated_row + shear_Y * rotated_col;

          // Translate the shear point
          const auto& translated_col = shear_col + translate_X;
          const auto& translated_row = shear_row + translate_Y;

          // Find input pixels near output point
          const auto input_col =
            static_cast<El::Int>(std::floor(translated_col));
          const auto input_row =
            static_cast<El::Int>(std::floor(translated_row));

          // Input and output pixels
          auto& pixel_output =
            local_output(channel * input_height * input_width +
                           output_row * input_width + output_col,
                         sample);

          if ((input_row >= 0 && input_row < input_height - 1) &&
              (input_col >= 0 && input_col < input_width - 1)) {

            const El::Int input_col0 = std::max(input_col, El::Int(0));
            const El::Int input_col1 = std::min(input_col + 1, input_width - 1);

            const El::Int input_row0 = std::max(input_row, El::Int(0));
            const El::Int input_row1 =
              std::min(input_row + 1, input_height - 1);

            // Point relative to input pixel centers
            const auto& unit_col = translated_col - input_col;
            const auto& unit_row = translated_row - input_row;

            auto& pixel00 = local_input(channel * input_height * input_width +
                                          input_row0 * input_width + input_col0,
                                        sample);

            auto& pixel01 = local_input(channel * input_height * input_width +
                                          input_row0 * input_width + input_col1,
                                        sample);

            auto& pixel10 = local_input(channel * input_height * input_width +
                                          input_row1 * input_width + input_col0,
                                        sample);

            auto& pixel11 = local_input(channel * input_height * input_width +
                                          input_row1 * input_width + input_col1,
                                        sample);

            // Bilinear interpolation
            pixel_output = (pixel00 * (one - unit_col) * (one - unit_row) +
                            pixel01 * unit_col * (one - unit_row) +
                            pixel10 * (one - unit_col) * unit_row +
                            pixel11 * unit_col * unit_row);
          }
          else {
            pixel_output = zero;
          }
        }
      }
    }
  }
}

#define PROTO(T)                                                               \
  template class composite_image_transformation_layer<                         \
    T,                                                                         \
    data_layout::DATA_PARALLEL,                                                \
    El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
