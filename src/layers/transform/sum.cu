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

#include "lbann/layers/transform/sum.hpp"
#include "lbann/utils/cuda.hpp"
#include "lbann/utils/exception.hpp"

#ifdef LBANN_HAS_DISTCONV
#include "distconv/tensor/algorithms_cuda.hpp"
#include "distconv/util/util_mpi.hpp"
#endif

namespace lbann {

#ifdef LBANN_HAS_DISTCONV
namespace {

template <typename DataType>
struct accumulate {
  __device__ void operator()(DataType &x, DataType &y) const {
    x += y;
  }
};

template <typename DataType>
struct accumulate2 {
  __device__ void operator()(DataType &x, DataType &y, DataType &z) const {
    x = y + z;
  }
};

void fp_compute_distconv(dc::TensorDev &activations,
                         dc::TensorDev &prev_activations,
                         std::vector<dc::TensorDev> &prev_activations_siblings,
                         int num_parents) {
  switch (num_parents) {
    case 0:
      dc::MPIPrintStreamDebug() << "No parent for sum layer";
      activations.zero();
      break;
    case 1:
      dc::MPIPrintStreamDebug() << "Just one parent for sum layer";
      dc::tensor::Copy(activations, prev_activations);
      break;
    case 2:
      // Optimization for layers with 2 parents (e.g.,
      // Resnet50). Avoids loading destination tensors multiple times
      prev_activations_siblings.at(0).set_outermost_dimension(
          activations.get_shape()[-1]);
      dc::tensor::Transform(activations, prev_activations,
                            prev_activations_siblings.at(0),
                            accumulate2<DataType>(),
                            dc::get_backend().get_stream());
      break;
    default:
      dc::tensor::Copy(activations, prev_activations);      
      for (auto &p: prev_activations_siblings) {
        p.set_outermost_dimension(activations.get_shape()[-1]);
        distconv::tensor::Transform(activations, p, accumulate<DataType>(),
                                    dc::get_backend().get_stream());
      }
  }
}
} // namespace
#endif // LBANN_HAS_DISTCONV

template <>
void sum_layer<data_layout::DATA_PARALLEL, El::Device::GPU>::fp_compute() {
#ifdef LBANN_HAS_DISTCONV
  if (this->distconv_enabled()) {
    fp_compute_distconv(m_activations_t, m_prev_activations_t,
                        m_prev_activations_siblings, get_num_parents());
    copy_out_activations();
    if (!early_terminate_last_iteration()) {
      return;
    }
  }
#endif
  // Same as the generic fp_compute
  auto& output = get_activations();
  if (get_num_parents() < 1) {
    El::Zero(output);
  } else {
    El::Copy(get_prev_activations(0), output);
    for (int i = 1; i < get_num_parents(); ++i) {
      El::Axpy(DataType(1), get_prev_activations(i), output);
    }
  }
}

} // namespace lbann
