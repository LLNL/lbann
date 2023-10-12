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

#include "lbann/utils/amp.hpp"
#include "lbann/utils/exception.hpp"

#include <cmath>

namespace lbann {
namespace amp {

template <typename TensorDataType>
bool is_finite_and_unscale(El::AbstractDistMatrix<TensorDataType>& grads,
                           EvalType scale)
{
  switch (grads.GetLocalDevice()) {
  case El::Device::CPU:
    return is_finite_and_unscale_cpu(grads, scale);
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    return is_finite_and_unscale_gpu(grads, scale);
#endif
  default:
    LBANN_ERROR("Unsupported device type: ",
                static_cast<int>(grads.GetLocalDevice()));
  }
}

template <typename TensorDataType>
bool is_finite_and_unscale_cpu(El::AbstractDistMatrix<TensorDataType>& grads,
                               EvalType scale)
{
  const auto inv_scale = El::To<TensorDataType>(EvalType{1} / scale);
  auto* __restrict__ buf = grads.Buffer();

  bool is_finite = true;

  if (grads.Contiguous()) {
    const size_t local_size = grads.LocalHeight() * grads.LocalWidth();
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(&& : is_finite))
    for (size_t i = 0; i < local_size; ++i) {
      auto& val = buf[i];
      if (!std::isfinite(val)) {
        is_finite = false;
      }
      val *= inv_scale;
    }
  }
  else {
    const size_t ldim = grads.LDim();
    const size_t width = grads.LocalWidth();
    const size_t height = grads.LocalHeight();
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(&& : is_finite) collapse(2))
    for (size_t col = 0; col < width; ++col) {
      for (size_t row = 0; row < height; ++row) {
        auto& val = buf[row + col*ldim];
        if (!std::isfinite(val)) {
          is_finite = false;
        }
        val *= inv_scale;
      }
    }
  }

  return is_finite;
}

#define PROTO(T)                                                               \
  template bool is_finite_and_unscale<T>(El::AbstractDistMatrix<T>&, EvalType);

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

}  // namespace amp
}  // namespace lbann
