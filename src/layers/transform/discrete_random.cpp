////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2022, Lawrence Livermore National Security, LLC.
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

#define LBANN_DISCRETE_RANDOM_LAYER_INSTANTIATE
#include "lbann/layers/transform/discrete_random.hpp"
#include "lbann/execution_algorithms/execution_context.hpp"

#include "lbann/utils/protobuf.hpp"

#include "lbann/proto/layers.pb.h"

#include "lbann/proto/datatype_helpers.hpp"


namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void discrete_random_layer<TensorDataType,Layout,Device>::fp_compute() {

  // Input and output matrices
  const auto& input = this->get_prev_activations();
  const auto& local_input = input.LockedMatrix();
  auto& output = this->get_activations();
  auto& local_output = output.Matrix();
  const int num_values = m_values.size();
  const auto& num_outputs = local_output.Height();
  const auto& width = input.Width();
  const auto& local_width = input.LocalWidth();

  // Initialize random numbers
  const auto& mode =
    this->m_model->get_execution_context().get_execution_mode();
  if (mode == execution_mode::training) {
    uniform_fill(output, 1, width, TensorDataType(0.5), TensorDataType(0.5));
  }

  // Process each mini-batch sample
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& input_ptr = local_input.LockedBuffer(0, col);
    const auto& output_ptr = local_output.Buffer(0, col);
    if (mode == execution_mode::training) {
      // Sample outputs from probability distribution
      std::vector<DataType> cdf(num_values);
      std::partial_sum(input_ptr, input_ptr + num_values, cdf.begin());
      for (El::Int row = 0; row < num_outputs; ++row) {
        const int index =
          (std::lower_bound(cdf.begin(), cdf.end(), local_output(row, col)) -
           cdf.begin());
        local_output(row, col) = m_values[index];
      }
    }
    else {
      // Fill output with mode of probability distribution
      const int index =
        (std::max_element(input_ptr, input_ptr + num_values) - input_ptr);
      std::fill_n(output_ptr, num_outputs, m_values[index]);
    }
  }
}

template <typename T, data_layout L, El::Device D>
void discrete_random_layer<T,L,D>::write_specific_proto(lbann_data::Layer& proto) const {
  proto.set_datatype(proto::ProtoDataType<T>);
  auto* msg = proto.mutable_discrete_random();
  protobuf::assign_to_repeated(*msg->mutable_values(), m_values);
  protobuf::assign_to_repeated(*msg->mutable_dims(), this->get_output_dims());
}

#define PROTO(T)                                                               \
  template class discrete_random_layer<T,                                      \
                                       data_layout::DATA_PARALLEL,             \
                                       El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
