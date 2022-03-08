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

#define LBANN_SUM_LAYER_INSTANTIATE
#include "lbann/layers/transform/sum.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/tensor/algorithms_cuda.hpp"
#endif // LBANN_HAS_DISTCONV

namespace lbann {

LBANN_LAYER_DEFAULT_BUILDER(sum)

#define PROTO(T)                                                        \
  template class sum_layer<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class sum_layer<T, data_layout::MODEL_PARALLEL, El::Device::GPU>; \
  LBANN_LAYER_BUILDER_ETI(sum, T, El::Device::GPU)

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"
#undef PROTO

#ifdef LBANN_HAS_DISTCONV
namespace {
template <typename TensorDataType>
struct accumulate_op {
  __device__ void operator()(TensorDataType &x, TensorDataType &y) const {
    x += y;
  }
};

template <typename TensorDataType>
struct sum_op {
  __device__ void operator()(TensorDataType &x, TensorDataType &y, TensorDataType &z) const {
    x = y + z;
  }
};
} // namespace

template <typename TensorDataType, data_layout Layout, El::Device Dev>
void sum_distconv_adapter<TensorDataType, Layout, Dev>::fp_compute() {
  auto &activations = this->get_activations();
  switch (this->layer().get_num_parents()) {
    case 0:
      activations.zero(hydrogen::cuda::GetDefaultStream());
      break;
    case 1:
      dc::tensor::Copy(activations, this->get_prev_activations(),
                       hydrogen::cuda::GetDefaultStream());
      break;
    case 2:
      // Optimization for layers with 2 parents (e.g.,
      // Resnet50). Avoids loading destination tensors multiple times
      this->get_prev_activations(1).set_outermost_dimension(
          activations.get_shape()[-1]);
      dc::tensor::Transform(activations,
                            this->get_prev_activations(0),
                            this->get_prev_activations(1),
                            sum_op<TensorDataType>(),
                            hydrogen::cuda::GetDefaultStream());
      break;
    default:
      for (int i = 0; i < this->layer().get_num_parents(); ++i) {
        auto &prev_activations = this->get_prev_activations(i);
        prev_activations.set_outermost_dimension(activations.get_shape()[-1]);
        if (i == 0) {
          dc::tensor::Copy(activations, prev_activations,
                           hydrogen::cuda::GetDefaultStream());
        } else {
          distconv::tensor::Transform(activations, prev_activations,
                                      accumulate_op<TensorDataType>(),
                                      hydrogen::cuda::GetDefaultStream());
        }
      }
  }
}

#define PROTO(T)                                                        \
  template class sum_distconv_adapter<T, data_layout::DATA_PARALLEL, El::Device::GPU>; \
  template class sum_distconv_adapter<T, data_layout::MODEL_PARALLEL, El::Device::GPU>

#include "lbann/macros/instantiate.hpp"
#endif // LBANN_HAS_DISTCONV

} // namespace lbann
