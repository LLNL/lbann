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
#include "lbann/optimizers/optimizer.hpp"
#include "lbann/utils/exception.hpp"

#include <cmath>

#ifdef LBANN_HAS_HALF
namespace {
bool isfinite(__half x) { return std::isfinite((float)x); }
} // namespace
#endif
using std::isfinite;

namespace lbann {
namespace amp {

template <typename TensorDataType>
void is_finite_and_unscale(El::AbstractDistMatrix<TensorDataType>& grads,
                           EvalType scale,
                           float* is_finite_cpu,
                           float* is_finite_gpu)
{
  switch (grads.GetLocalDevice()) {
  case El::Device::CPU:
    is_finite_and_unscale_cpu(grads, scale, is_finite_cpu);
    break;
#ifdef LBANN_HAS_GPU
  case El::Device::GPU:
    is_finite_and_unscale_gpu(grads, scale, is_finite_gpu);
    break;
#endif
  default:
    LBANN_ERROR("Unsupported device type: ",
                static_cast<int>(grads.GetLocalDevice()));
  }
}

bool is_finite_and_unscale_all(std::vector<optimizer*> optimizers,
                               EvalType scale)
{
  // Keep two separate pointers for CPU and GPU finiteness.
  float is_finite_cpu = 1.0f;
  float* is_finite_cpu_p = &is_finite_cpu;
#ifdef LBANN_HAS_GPU
  El::Matrix<float, El::Device::GPU> is_finite_mat;
#ifdef HYDROGEN_HAVE_CUB
  is_finite_mat.SetMemoryMode(1); // Use CUB memory pool.
#endif
  is_finite_mat.Resize(1, 1);
  El::Fill(is_finite_mat, El::TypeTraits<float>::One());
  float* is_finite_gpu_p = is_finite_mat.Buffer();
  // TODO: We should probably use a sync object to ensure GPU
  // computations are synchronized with this buffer creation, but I
  // don't see a good way to plumb that through right now.
#else
  float* is_finite_gpu_p = nullptr;
#endif

  for (auto&& opt : optimizers) {
    auto grads = opt->get_raw_gradients();
    for (auto&& grad_r : grads) {
      auto& grad = grad_r.get();
      // Attempt to convert from a BaseDistMatrix to an AbstractDistMatrix.
      if (auto* ptr_f = dynamic_cast<El::AbstractDistMatrix<float>*>(&grad)) {
        is_finite_and_unscale(*ptr_f, scale, is_finite_cpu_p, is_finite_gpu_p);
      }
      else if (auto* ptr_d =
                 dynamic_cast<El::AbstractDistMatrix<double>*>(&grad)) {
        is_finite_and_unscale(*ptr_d, scale, is_finite_cpu_p, is_finite_gpu_p);
      }
#ifdef LBANN_HAS_HALF
      else if (auto* ptr_cpufp16 =
                 dynamic_cast<El::AbstractDistMatrix<cpu_fp16>*>(&grad)) {
        is_finite_and_unscale(*ptr_cpufp16,
                              scale,
                              is_finite_cpu_p,
                              is_finite_gpu_p);
      }
#endif
#ifdef LBANN_HAS_GPU_FP16
      else if (auto* ptr_fp16 =
                 dynamic_cast<El::AbstractDistMatrix<fp16>*>(&grad)) {
        is_finite_and_unscale(*ptr_fp16,
                              scale,
                              is_finite_cpu_p,
                              is_finite_gpu_p);
      }
#endif
      else {
        LBANN_ERROR("Could not determine gradient type");
      }
    }
  }

  bool is_finite = is_finite_cpu == 1.0f;
#ifdef LBANN_HAS_GPU
#pragma GCC diagnostic push
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
  is_finite &= (is_finite_mat.Get(0, 0) == 1.0f);
#pragma GCC diagnostic pop
#endif
  return is_finite;
}

template <typename TensorDataType>
void is_finite_and_unscale_cpu(El::AbstractDistMatrix<TensorDataType>& grads,
                               EvalType scale,
                               float* is_finite_p)
{
  const auto inv_scale = El::To<TensorDataType>(EvalType{1} / scale);
  auto* __restrict__ buf = grads.Buffer();

  bool is_finite = true;

  if (grads.Contiguous()) {
    const size_t local_size = grads.LocalHeight() * grads.LocalWidth();
    LBANN_OMP_PARALLEL_FOR_ARGS(reduction(&& : is_finite))
    for (size_t i = 0; i < local_size; ++i) {
      auto& val = buf[i];
      if (!isfinite(val)) {
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
        auto& val = buf[row + col * ldim];
        if (!isfinite(val)) {
          is_finite = false;
        }
        val *= inv_scale;
      }
    }
  }

  if (!is_finite) {
    *is_finite_p = 0.0f;
  }
}

#define PROTO(T)                                                               \
  template void is_finite_and_unscale<T>(El::AbstractDistMatrix<T>&,           \
                                         EvalType,                             \
                                         float*,                               \
                                         float*);

#define LBANN_INSTANTIATE_CPU_HALF
#define LBANN_INSTANTIATE_GPU_HALF
#define LBANN_INSTANTIATE_DOUBLE
#include "lbann/macros/instantiate.hpp"

} // namespace amp
} // namespace lbann
