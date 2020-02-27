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

#define LBANN_ONE_HOT_LAYER_INSTANTIATE
#include "lbann/layers/misc/one_hot.hpp"

namespace lbann {

template <typename TensorDataType, data_layout Layout, El::Device Device>
void one_hot_layer<TensorDataType, Layout, Device>::fp_compute() {
  using CPUMatType = El::Matrix<TensorDataType, El::Device::CPU>;

  // Local matrices
  const auto& local_input =
    dynamic_cast<const CPUMatType&>(this->get_local_prev_activations());
  auto& local_output = dynamic_cast<CPUMatType&>(this->get_local_activations());
  const El::Int local_height = local_output.Height();
  const El::Int local_width = local_output.Width();

  // Populate one-hot vectors
  El::Zero(local_output);
  LBANN_OMP_PARALLEL_FOR
  for (El::Int col = 0; col < local_width; ++col) {
    const auto& ind = local_input(0, col);
    if (El::TypeTraits<TensorDataType>::Zero() <= ind && ind < TensorDataType(local_height)) {
      const El::Int row = static_cast<El::Int>(ind);
      local_output(row, col) = El::TypeTraits<TensorDataType>::One();
    }
  }

}

#define PROTO(T)                     \
  template class one_hot_layer<T, data_layout::DATA_PARALLEL, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
