////////////////////////////////////////////////////////////////////////////////
// Copyright (c) 2014-2021, Lawrence Livermore National Security, LLC.
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

#include "lbann/operators/math/abs.hpp"

#include "lbann/base.hpp"
#include "lbann/utils/gpu/helpers.hpp"

#include "common.cuh"

namespace lbann {

namespace {

template <typename DataT>
struct AbsOpImpl {
  using ComplexT = thrust::complex<DataT>;

  inline __device__ DataT operator()(DataT const& x) const {
    return gpu_lib::abs(x);
  }
  inline __device__ DataT operator()(ComplexT const& x) const {
    return thrust::abs(x);
  }
  inline __device__ DataT operator()(DataT const& x, DataT const& dy) const {
    return (x > (DataT) 0.
            ? dy
            : (x < (DataT) 0.
               ? -dy
               : (DataT) 0.));
  }
  inline __device__ ComplexT operator()(ComplexT const& x,
                                        DataT const& dy) const {
    return (x == ComplexT(0.f)
            ? ComplexT(0.f)
            : thrust::conj(x * (dy / thrust::abs(x))));
  }
};// struct AbsOpImpl

} // namespace

template <typename DataT, El::Device Device>
void AbsOperator<DataT, Device>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  auto const& input = inputs.front().data();
  auto& output = outputs.front().data();
  El::EntrywiseMap(input,
                   output,
                   AbsOpImpl<El::Base<DataT>>{});
}

template <typename DataT, El::Device Device>
void AbsOperator<DataT, Device>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<ConstLocalOutputTensorType> grads_wrt_outputs,
  std::vector<LocalInputTensorType> grads_wrt_inputs) const
{
  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(grads_wrt_outputs.size() == 1);
  LBANN_ASSERT_DEBUG(grads_wrt_inputs.size() == 1);
  auto const& input = inputs.front().data();
  auto const& grad_wrt_output = grads_wrt_outputs.front().data();
  auto& grad_wrt_input = grads_wrt_inputs.front().data();
  internal::EntrywiseZipInto(input,
                             grad_wrt_output,
                             grad_wrt_input,
                             AbsOpImpl<El::Base<DataT>>{});
}

#define PROTO(T) template class AbsOperator<T, El::Device::GPU>

#define LBANN_INSTANTIATE_GPU_HALF
#include "lbann/macros/instantiate.hpp"

#undef LBANN_INSTANTIATE_GPU_HALF
#undef PROTO
#define PROTO(T) template class AbsOperator<El::Complex<T>, El::Device::GPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
