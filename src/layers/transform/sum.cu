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

#define LBANN_SUM_LAYER_INSTANTIATE
#include "lbann/layers/transform/sum.hpp"
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
  __device__ void operator()(TensorDataType &x, TensorDataType &y) const {
    x += y;
  }
};

template <typename TensorDataType>
struct accumulate2 {
  __device__ void operator()(TensorDataType &x, TensorDataType &y, TensorDataType &z) const {
    x = y + z;
  }
};

template <typename TensorDataType, template<typename> class SumDistConvLayer>
void fp_compute_distconv(int num_parents, SumDistConvLayer<TensorDataType> &dc) {
  auto activations = dc.get_activations();
  switch (num_parents) {
    case 0:
      activations.zero(El::GPUManager::Stream());
      break;
    case 1:
      dc::tensor::Copy(activations, dc.get_prev_activations(),
                       El::GPUManager::Stream());
      break;
    case 2:
      // Optimization for layers with 2 parents (e.g.,
      // Resnet50). Avoids loading destination tensors multiple times
      dc.get_prev_activations(1).set_outermost_dimension(
          activations.get_shape()[-1]);
      dc::tensor::Transform(activations,
                            dc.get_prev_activations(0),
                            dc.get_prev_activations(1),
                            accumulate2<TensorDataType>(),
                            dc::get_backend().get_stream());
      break;
    default:
      for (int i = 0; i < num_parents; ++i) {
        auto &prev_activations = dc.get_prev_activations(i);
        prev_activations.set_outermost_dimension(activations.get_shape()[-1]);
        if (i == 0) {
          dc::tensor::Copy(activations, prev_activations,
                           El::GPUManager::Stream());
        } else {
          distconv::tensor::Transform(activations, prev_activations,
                                      accumulate<TensorDataType>(),
                                      dc::get_backend().get_stream());
        }
      }
  }
}
} // namespace
#endif // LBANN_HAS_DISTCONV

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void sum_layer<TensorDataType, Layout, Dev>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    fp_compute_distconv(
        this->get_num_parents(), this->dc());
    return;
  }
#endif
  auto& output = this->get_activations();
  El::Copy(this->get_prev_activations(0), output);
  for (int i = 1; i < this->get_num_parents(); ++i) {
    El::Axpy(TensorDataType{1}, this->get_prev_activations(i), output);
  }
}

LBANN_LAYER_DEFAULT_BUILDER(sum)

#define PROTO(T)                                                        \
  template class sum_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class sum_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>; \
  LBANN_LAYER_BUILDER_ETI(sum, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

} // namespace lbann
