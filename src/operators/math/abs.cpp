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

#include "lbann/operators/math/abs.hpp"

#include "common.hpp"
#include <hydrogen/meta/TypeTraits.hpp>

namespace lbann {

namespace {

template <typename RealT>
void do_abs_bp(El::Matrix<RealT, El::Device::CPU> const& x,
               El::Matrix<RealT, El::Device::CPU> const& gradient_wrt_output,
               El::Matrix<RealT, El::Device::CPU>& gradient_wrt_input)
{
  internal::EntrywiseZipInto(x,
                             gradient_wrt_output,
                             gradient_wrt_input,
                             [](RealT const& x_, RealT const& dy_) {
                               return (x_ > El::TypeTraits<RealT>::Zero()
                                         ? dy_
                                         : (x_ < El::TypeTraits<RealT>::Zero()
                                              ? -dy_
                                              : El::TypeTraits<RealT>::Zero()));
                             });
}

template <typename RealT>
void do_abs_bp(
  El::Matrix<El::Complex<RealT>, El::Device::CPU> const& x,
  El::Matrix<RealT, El::Device::CPU> const& gradient_wrt_output,
  El::Matrix<El::Complex<RealT>, El::Device::CPU>& gradient_wrt_input)
{
  using ComplexT = El::Complex<RealT>;
  internal::EntrywiseZipInto(
    x,
    gradient_wrt_output,
    gradient_wrt_input,
    [](ComplexT const& e, RealT dy) {
      return (e == ComplexT{0} ? ComplexT{0} : El::Conj(e * (dy / El::Abs(e))));
    });
}

} // namespace

template <typename DataT, El::Device D>
void AbsOperator<DataT, D>::fp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<LocalOutputTensorType> outputs) const
{
  using CType = DataT;
  using RType = El::Base<DataT>;

  LBANN_ASSERT_DEBUG(inputs.size() == 1);
  LBANN_ASSERT_DEBUG(outputs.size() == 1);
  auto const& input = inputs.front().data();
  auto& output = outputs.front().data();
  El::EntrywiseMap(input,
                   output,
                   std::function<RType(CType const&)>(
                     [](CType const& x) { return El::Abs(x); }));
}

template <typename DataT, El::Device D>
void AbsOperator<DataT, D>::bp_compute_local(
  std::vector<ConstLocalInputTensorType> inputs,
  std::vector<ConstLocalOutputTensorType> gradient_wrt_outputs,
  std::vector<LocalInputTensorType> gradient_wrt_inputs) const
{
  LBANN_ASSERT(inputs.size() == 1 && gradient_wrt_inputs.size() == 1);

  auto const& input = inputs.front().data();
  auto const& grad_wrt_output = gradient_wrt_outputs.front().data();
  auto& grad_wrt_input = gradient_wrt_inputs.front().data();
  do_abs_bp(input, grad_wrt_output, grad_wrt_input);
}

#define PROTO(T) template class AbsOperator<T, El::Device::CPU>

#define LBANN_INSTANTIATE_CPU_HALF
#include "lbann/macros/instantiate.hpp"

#undef LBANN_INSTANTIATE_CPU_HALF
#undef PROTO
#define PROTO(T) template class AbsOperator<El::Complex<T>, El::Device::CPU>

#include "lbann/macros/instantiate.hpp"

} // namespace lbann
