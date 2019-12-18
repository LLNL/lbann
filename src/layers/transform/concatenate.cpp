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

#define LBANN_CONCATENATE_LAYER_INSTANTIATE
#include "lbann/layers/transform/concatenate.hpp"

namespace lbann {

namespace {

/** @brief Concatenate 4D tensors. */
template <typename T>
void concat4d(
  size_t concat_dim,
  const std::vector<const T*>& input_buffer_list,
  const std::vector<std::array<size_t,4>>& input_dims_list,
  const std::vector<std::array<size_t,4>>& input_strides_list,
  T* output_buffer,
  const std::array<size_t,4>& output_strides) {

  // Compute offset corresponding to each input tensor
  std::vector<size_t> output_offset_list;
  output_offset_list.push_back(0);
  for (const auto& input_dims : input_dims_list) {
    auto offset = output_offset_list.back();
    offset += input_dims[concat_dim] * output_strides[concat_dim];
    output_offset_list.push_back(offset);
  }

  // Iterate through input tensors
  for (size_t j=0; j<input_buffer_list.size(); ++j) {
    const auto& input_buffer = input_buffer_list[j];
    const auto& input_dims = input_dims_list[j];
    const auto& input_strides = input_strides_list[j];
    const auto& output_offset = output_offset_list[j];

    // Copy input tensor to corresponding position in output tensor
    LBANN_OMP_PARALLEL_FOR_COLLAPSE4
    for (size_t i0=0; i0<input_dims[0]; ++i0) {
      for (size_t i1=0; i1<input_dims[1]; ++i1) {
        for (size_t i2=0; i2<input_dims[2]; ++i2) {
          for (size_t i3=0; i3<input_dims[3]; ++i3) {
            const auto& x = input_buffer[i0 * input_strides[0]
                                         + i1 * input_strides[1]
                                         + i2 * input_strides[2]
                                         + i3 * input_strides[3]];
            auto& y = output_buffer[output_offset
                                    + i0 * output_strides[0]
                                    + i1 * output_strides[1]
                                    + i2 * output_strides[2]
                                    + i3 * output_strides[3]];
            y = x;
          }
        }
      }
    }

  }

}

} // namespace <anon>

template <typename TensorDataType>
void fp_compute_impl(
  concatenate_layer<TensorDataType,data_layout::MODEL_PARALLEL,El::Device::CPU>& l,
  size_t concat_dim) {

  // Stack Elemental matrices on top of each other
  // Note: Assume each mini-batch sample is flat.
  auto& output = l.get_activations();
  std::unique_ptr<El::AbstractDistMatrix<TensorDataType>> output_v(
    output.Construct(output.Grid(), output.Root()));
  size_t offset = 0;
  for (size_t i=0; i<static_cast<size_t>(l.get_num_parents()); ++i) {
    const auto& input = l.get_prev_activations(i);
    El::View(*output_v, output,
             El::IR(offset, offset+input.Height()), El::ALL);
    El::Copy(input, *output_v);
    offset += input.Height();
  }

}

template <typename TensorDataType>
void fp_compute_impl(
  concatenate_layer<TensorDataType,data_layout::DATA_PARALLEL,El::Device::CPU>& l,
  size_t concat_dim) {

  // Check that number of dimensions is valid
  /// @todo Support tensors with arbitrary number of dimensions
  const size_t num_dims = l.get_output_dims().size();
  if (num_dims > 3) {
    LBANN_ERROR(l.get_type()," layer \"",l.get_name(),"\" ",
                "is operating on ",num_dims,"-D tensors, ",
                "but only 3-D tensors are currently supported");
  }

  // Get dimensions and strides for each input tensor
  std::vector<const TensorDataType*> input_buffer_list;
  std::vector<std::array<size_t,4>> input_dims_list, input_strides_list;
  for (size_t i=0; i<static_cast<size_t>(l.get_num_parents()); ++i) {
    const auto& input = l.get_prev_activations(i);
    const auto& input_dims = l.get_input_dims(i);

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(input_dims.rbegin(), input_dims.rend());
    std::vector<size_t> rstrides(input_dims.size(), 1);
    for (size_t d=1; d<input_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(input.LocalWidth());
    rstrides.push_back(input.LDim());

    // Pad tensor dimensions to 4D
    while (rdims.size() < 4) {
      rdims.push_back(1);
      rstrides.push_back(rstrides.back());
    }

    input_buffer_list.push_back(input.LockedBuffer());
    input_dims_list.push_back({rdims[3], rdims[2], rdims[1], rdims[0]});
    input_strides_list.push_back(
      {rstrides[3], rstrides[2], rstrides[1], rstrides[0]});
  }

  // Get strides for output tensor
  std::array<size_t,4> output_strides;
  auto& output = l.get_activations();
  {
    const auto& output_dims = l.get_output_dims();

    // Construct dimensions and strides in reverse order
    // Note: Assume each mini-batch sample is fully packed.
    std::vector<size_t> rdims(output_dims.rbegin(), output_dims.rend());
    std::vector<size_t> rstrides(output_dims.size(), 1);
    for (size_t d=1; d<output_dims.size(); ++d) {
      rstrides[d] = rdims[d-1] * rstrides[d-1];
    }
    rdims.push_back(output.LocalWidth());
    rstrides.push_back(output.LDim());

    // Pad tensor dimensions to 4D
    while (rdims.size() < 4) {
      rdims.push_back(1);
      rstrides.push_back(rstrides.back());
    }

    output_strides = {rstrides[3], rstrides[2], rstrides[1], rstrides[0]};
  }

  // Perform concatenation on 4D tensors
  concat4d<TensorDataType>(
    concat_dim + (4-num_dims),
    input_buffer_list,
    input_dims_list,
    input_strides_list,
    output.Buffer(),
    output_strides);

}

template <typename TensorDataType, data_layout Layout, El::Device Device>
void concatenate_layer<TensorDataType,Layout,Device>::fp_compute() {

  // Just make a view if there is one input
  if (this->get_num_parents() == 1) {
    El::LockedView(this->get_activations(), this->get_prev_activations(0));
    return;
  }

  // Perform concatenation
  fp_compute_impl(*this, m_concat_dim);

}

// Explicit instantiation
#define PROTO(T)                                        \
  template class concatenate_layer<                     \
    T, data_layout::DATA_PARALLEL, El::Device::CPU>;    \
  template class concatenate_layer<                     \
    T, data_layout::MODEL_PARALLEL, El::Device::CPU>;

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
