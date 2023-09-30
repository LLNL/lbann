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

#define LBANN_CUTOUT_LAYER_INSTANTIATE
#include "lbann/layers/image/cutout.hpp"
#include "lbann/proto/datatype_helpers.hpp"

#include "lbann/proto/layers.pb.h"
#include "lbann/proto/lbann.pb.h"

#include "lbann/utils/exception.hpp"

#include <algorithm>
#include <math.h>

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void cutout_layer<TensorDataType, Layout, Device>::setup_dims()
{
  data_type_layer<TensorDataType>::setup_dims();

  // Get input dimensions
  auto dims = this->get_input_dims(0);
  const auto& cutout_length = this->get_input_dims(1);

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
  if (cutout_length.size() > 1 || cutout_length[0] != 1) {
    std::ostringstream ss;
    for (size_t i = 0; i < cutout_length.size(); ++i) {
      ss << (i > 0 ? " x " : "") << cutout_length[i];
    }
    LBANN_ERROR(this->get_type(),
                " layer \"",
                this->get_name(),
                "\" ",
                "expects a scalar input for the cutout length, ",
                "but input dimensions are ",
                ss.str());
  }
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void cutout_layer<TensorDataType, Layout, Device>::write_specific_proto(
  lbann_data::Layer& proto) const
{
  proto.set_datatype(proto::ProtoDataType<TensorDataType>);
  proto.mutable_cutout();
}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void cutout_layer<TensorDataType, Layout, Device>::fp_compute()
{

  // Useful constants
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

  // Get cutout length
  const auto& cutouts = this->get_local_prev_activations(1);

  // RNG
  std::random_device rd;
  std::mt19937 gen(rd());
  std::uniform_real_distribution<DataType> uni(zero, one);

  const El::Int col_center = uni(gen) * input_width;
  const El::Int row_center = uni(gen) * input_height;

  // Perform cutout
  LBANN_OMP_PARALLEL_FOR_COLLAPSE4
  for (El::Int sample = 0; sample < num_samples; ++sample) {
    for (El::Int channel = 0; channel < num_channels; ++channel) {
      for (El::Int output_row = 0; output_row < input_height; ++output_row) {
        for (El::Int output_col = 0; output_col < input_width; ++output_col) {

          const auto& cutout = cutouts.Get(0, sample);

          const El::Int col_start =
            std::max(static_cast<El::Int>(col_center - cutout / 2), El::Int(0));
          const El::Int col_end =
            std::min(static_cast<El::Int>(col_start + cutout), input_width - 1);

          const El::Int row_start =
            std::max(static_cast<El::Int>(row_center - cutout / 2), El::Int(0));
          const El::Int row_end =
            std::min(static_cast<El::Int>(row_start + cutout),
                     input_height - 1);

          // Find input pixels
          const auto input_col = output_col;
          const auto input_row = output_row;

          // Input and output pixels
          auto& pixel_output =
            local_output(channel * input_height * input_width +
                           input_row * input_width + input_col,
                         sample);

          if ((input_col >= col_start && input_col < col_end) &&
              (input_row >= row_start && input_row < row_end)) {
            pixel_output = zero;
          }
          else {
            pixel_output = local_input(channel * input_height * input_width +
                                         input_row * input_width + input_col,
                                       sample);
          }
        }
      }
    }
  }
}

#define PROTO(T)                                                               \
  template class cutout_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
