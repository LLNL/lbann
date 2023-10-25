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

#ifndef LBANN_UTILS_AMP_HPP
#define LBANN_UTILS_AMP_HPP

#include "lbann/base.hpp"

#include <vector>

namespace lbann {

// Forward-declare this.
class optimizer;

namespace amp {

/** Check gradients for invalid values and unscale them.
 *  Will set the value pointed to by is_finite_cpu or is_finite_gpu to
 *  zero if a non-finite (inf or NaN) value is found; otherwise this
 *  will not modify them.
 *  If is_finite_cpu/gpu is set to zero, grads should not be used.
 *
 * @todo This involves a CPU<->GPU sync which could be avoided by
 * fusing these checks into optimizers.
 *
 * @todo is_finite_cpu/gpu is kind of a hack to share a single buffer
 * across many invocations.
 */
template <typename TensorDataType>
void is_finite_and_unscale(
  El::AbstractDistMatrix<TensorDataType>& grads,
  EvalType scale,
  float* is_finite_cpu,
  float* is_finite_gpu);

/** Apply is_finite_and_unscale to all gradients in optimizers. This
 *  can be faster than applying it individually.
 */
bool is_finite_and_unscale_all(
  std::vector<optimizer*> optimizers,
  EvalType scale);

template <typename TensorDataType>
void is_finite_and_unscale_cpu(
  El::AbstractDistMatrix<TensorDataType>& grads,
  EvalType scale,
  float* is_finite);

#ifdef LBANN_HAS_GPU
template <typename TensorDataType>
void is_finite_and_unscale_gpu(
  El::AbstractDistMatrix<TensorDataType>& grads,
  EvalType scale,
  float* is_finite);
#endif

}  // namespace amp
}  // namespace lbann

#endif  // LBANN_UTILS_AMP_HPP
