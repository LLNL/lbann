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

#define LBANN_SPLIT_LAYER_INSTANTIATE
#include "lbann/layers/transform/split.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"
#include "lbann/utils/memory.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace {

template <typename TensorDataType>
struct accumulate {
  __device__ void operator()(TensorDataType &x, const TensorDataType &y) const {
    x += y;
  }
};

template <typename TensorDataType>
struct accumulate2 {
  __device__ void operator()(TensorDataType &x, const TensorDataType &y,
                             const TensorDataType &z) const {
    x = y + z;
  }
};

template <typename SplitAdapter>
void bp_compute_distconv(SplitAdapter &dc, int num_children) {
  using TensorDataType = typename SplitAdapter::TensorDevType::data_type;
  auto &error_signals = dc.get_error_signals(0);
  switch (num_children) {
    case 0:
      dc::MPIPrintStreamInfo() << "No parent for this sum layer";
      error_signals.zero(El::GPUManager::Stream());
      break;
    case 1:
      dc::tensor::Copy(error_signals,
                       dc.get_prev_error_signals(0),
                       El::GPUManager::Stream());
      break;
    case 2:
      dc::tensor::Transform(error_signals,
                            dc.get_prev_error_signals(0),
                            dc.get_prev_error_signals(1),
                            accumulate2<TensorDataType>(),
                            dc::get_backend().get_stream());
      break;
    default:
      dc::tensor::Copy(error_signals,
                       dc.get_prev_error_signals(1),
                       El::GPUManager::Stream());
      for (int i = 1; i < num_children; ++i) {
        const auto &prev_error = dc.get_prev_error_signals(i);
        dc::tensor::Transform(error_signals, prev_error,
                              accumulate<TensorDataType>(),
                              dc::get_backend().get_stream());
      }
  }
}
} // namespace
#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void split_layer<TensorDataType, Layout, Dev>::bp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    assert_always(Layout == data_layout::DATA_PARALLEL);
    bp_compute_distconv(this->dc(), this->get_num_children());
    this->dc().copy_out_error_signals();
    return;
  }
#endif // LBANN_HAS_DISTCONV
  auto& gradient_wrt_input = this->get_error_signals();
  if (this->get_num_children() > 0) {
    El::Copy(this->get_prev_error_signals(0), gradient_wrt_input);
  } else {
    El::Zero(gradient_wrt_input);
  }
  for (int i = 1; i < this->get_num_children(); ++i) {
    El::Axpy(TensorDataType{1}, this->get_prev_error_signals(i),
             gradient_wrt_input);
  }
}

LBANN_LAYER_DEFAULT_BUILDER(split)

#define PROTO(T)                                                        \
  template class split_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class split_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>; \
  LBANN_LAYER_BUILDER_ETI(split, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
