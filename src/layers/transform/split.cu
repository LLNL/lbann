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

#define LBANN_SPLIT_LAYER_INSTANTIATE
#include "lbann/layers/transform/split.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/tensor/algorithms_cuda.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

LBANN_LAYER_DEFAULT_BUILDER(split)

#define PROTO(T)                                                        \
  template class split_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class split_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>; \
  LBANN_LAYER_BUILDER_ETI(split, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

#ifdef LBANN_HAS_DISTCONV
namespace {
template <typename TensorDataType>
struct accumulate_op {
  __device__ void operator()(TensorDataType &x, const TensorDataType &y) const {
    x += y;
  }
};

template <typename TensorDataType>
struct sum_op {
  __device__ void operator()(TensorDataType &x, const TensorDataType &y,
                             const TensorDataType &z) const {
    x = y + z;
  }
};
} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void split_distconv_adapter<TensorDataType, Layout, Dev>::bp_compute() {
  if (Layout != data_layout::DATA_PARALLEL) {
    LBANN_ERROR("Distconv not supported");
  }
  auto &error_signals = this->get_error_signals(0);
  switch (this->layer().get_num_children()) {
    case 0:
      error_signals.zero(default_hydrogen_stream());
      break;
    case 1:
      dc::tensor::Copy(error_signals,
                       this->get_prev_error_signals(0),
                       default_hydrogen_stream());
      break;
    case 2:
      dc::tensor::Transform(error_signals,
                            this->get_prev_error_signals(0),
                            this->get_prev_error_signals(1),
                            sum_op<TensorDataType>(),
                            default_hydrogen_stream());
      break;
    default:
      dc::tensor::Copy(error_signals,
                       this->get_prev_error_signals(1),
                       default_hydrogen_stream());
      for (int i = 1; i < this->layer().get_num_children(); ++i) {
        const auto &prev_error = this->get_prev_error_signals(i);
        dc::tensor::Transform(error_signals,
                              prev_error,
                              accumulate_op<TensorDataType>(),
                              default_hydrogen_stream());
      }
  }
  return;
}

#define PROTO(T)                                                        \
  template class split_distconv_adapter<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class split_distconv_adapter<T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#endif // LBANN_HAS_DISTCONV

} // namespace lbann
